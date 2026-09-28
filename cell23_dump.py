# =====================================================================
# CELL 17 — Shared temporal encoder, shared heads, and the RSTE core
# =====================================================================
class TemporalEncoder(nn.Module):
    """Dilated causal conv stack over the W-step window, applied per bus.

    Shared by EVERY architecture so temporal capacity is held exactly constant
    and the benchmark isolates the spatial operator.
    """

    def __init__(self, c_in=CHANNELS, d=192, n_layers=3):
        super().__init__()
        self.proj = nn.Linear(c_in, d)
        self.convs = nn.ModuleList([
            nn.Conv1d(d, d, 3, padding=2 ** (i + 1), dilation=2 ** (i + 1))
            for i in range(n_layers)])
        self.norms = nn.ModuleList([nn.GroupNorm(1, d) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d)

    def forward(self, x):                       # x: (B, W, N, C)
        B, W, N, C = x.shape
        h = self.proj(x)                        # (B,W,N,d)
        h = h.permute(0, 2, 3, 1).reshape(B * N, -1, W)
        for cv, nm in zip(self.convs, self.norms):
            y = cv(h)[..., :W]
            h = h + F.gelu(nm(y))               # residual, causal-cropped
        h = h.mean(-1).reshape(B, N, -1)        # (B,N,d)
        return self.ln(h)


class Heads(nn.Module):
    """Detection / localization / repair. Shared across every architecture."""

    def __init__(self, d=192, c_out=CHANNELS):
        super().__init__()
        self.loc = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        # +2 inputs: the top-k localisation statistic and its spread. See
        # forward() for why the detector should read the localisation head.
        self.det = nn.Sequential(nn.Linear(2 * d + 2, d), nn.GELU(),
                                 nn.Linear(d, 1))
        self.rep = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, c_out))

    @staticmethod
    def _topk_pool(x, dim=1):
        """Mean of the top-k entries, where k tracks the measured attack support.

        A max (k=1) is maximally sensitive to a single outlier bus, which is what
        gives a detector a heavy right tail on CLEAN windows: one unusual bus
        representation produces a high score, the 1-in-1000 false-alarm threshold
        is dragged up, and recall at that operating point collapses.

        Part 5 measures the actual attack support -- 5.5 buses on IEEE 14-bus,
        20.7 on IEEE 118-bus, never one -- so aggregating several buses is also
        the physically correct statistic, not merely the more robust one.
        """
        n = x.shape[dim]
        k = max(2, n // 6)                       # ~ measured buses-hit fraction
        k = min(k, n)
        return x.topk(k, dim=dim).values.mean(dim)

    def forward(self, h, P_range=None):         # h: (B,N,d)
        loc = self.loc(h).squeeze(-1)                       # (B,N)
        # Detector reads the localisation head. If an attack lights up k buses,
        # "are the top-k per-bus scores high, and are they separated from the
        # rest?" is a sufficient statistic for detection under exactly the
        # assumption the localisation head is trained on. Coupling them shares
        # that assumption instead of learning it twice.
        _n = loc.shape[1]
        _k = min(max(2, _n // 6), _n)
        _top = loc.topk(_k, dim=1).values                   # (B,k)
        loc_feat = torch.stack([_top.mean(1),
                                _top.mean(1) - loc.mean(1)], dim=-1)   # (B,2)
        pooled = torch.cat([h.mean(1), self._topk_pool(h, 1),
                            loc_feat.to(h.dtype)], -1)      # (B, 2d+2)
        det = self.det(pooled).squeeze(-1)                  # (B,)
        rep = self.rep(h)                                   # (B,N,C)
        if P_range is not None:
            # Upgrade 2: the injection channel of a physically realisable frame
            # lies in range(H_inj). Anything orthogonal to it is precisely what a
            # bad-data test would reject, so it cannot be part of a correct
            # repair. Project it away rather than asking the network to learn to
            # avoid it.
            inj = rep[:, :, 2]                              # (B,N)
            proj = torch.einsum("ij,bj->bi", P_range.to(inj.dtype), inj)
            rep = torch.cat([rep[:, :, :2], proj.unsqueeze(-1),
                             rep[:, :, 3:]], dim=-1)
        return det, loc, rep


# ---------------------------------------------------------------------
#  SCEPTRE -- Recursive Separator-Tree Encoder            (CONTRIBUTION C1)
# ---------------------------------------------------------------------
class RSTE(nn.Module):
    """Weight-tied recursive encoder over a nested-dissection separator tree.

    Parameter count is INDEPENDENT of N and of the tree shape, because Merge,
    Bcast, Gate and LeafOut are the same tensors at every level and every node.
    """

    def __init__(self, d=192, n_sweeps=2, n_blocks=1):
        super().__init__()
        self.d = d; self.n_sweeps = n_sweeps
        mk = lambda i, o: nn.Sequential(nn.Linear(i, 2 * d), nn.GELU(),
                                        nn.Linear(2 * d, o))
        self.merge = mk(3 * d, d)
        self.bcast = mk(2 * d, d)
        self.gate = nn.Sequential(nn.Linear(2 * d, d), nn.Sigmoid())
        self.leaf_out = nn.Linear(2 * d, d)
        self.ln_u = nn.LayerNorm(d); self.ln_d = nn.LayerNorm(d)
        # optional extra weight-tied refinement blocks (still N-independent)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(),
                          nn.Linear(2 * d, d)) for _ in range(n_blocks)])

    def forward(self, hb, plan):                # hb: (B,N,d)
        # Under bf16 autocast, Linear returns bf16 while LayerNorm stays fp32,
        # so index_add/index_copy would see mismatched scalar types. Pin one
        # working dtype for the whole sweep and cast at every boundary.
        dt = hb.dtype
        B, N, d = hb.shape
        hn = hb.new_zeros(B, plan.n_nodes, d)

        # ---- leaves <- mean over their buses ----
        leafh = hb.new_zeros(B, plan.n_leaf, d)
        leafh = leafh.index_add(1, plan.l2b_leaf, hb[:, plan.l2b_bus, :].to(dt))
        leafh = leafh / plan.leaf_cnt[None, :, None]
        hn = hn.index_copy(1, plan.leaf_ids, self.ln_u(leafh).to(dt))

        # ---- bottom-up sweep (deepest level first) ----
        for s in plan.up:
            hl = hn[:, s["lo"], :]; hh = hn[:, s["hi"], :]
            sep = hb.new_zeros(B, s["m"], d)
            if s["sep_row"].numel() > 0:
                sep = sep.index_add(1, s["sep_row"], hb[:, s["sep_col"], :].to(dt))
                sep = sep / s["nsep"][None, :, None]
            h = self.ln_u(self.merge(torch.cat([hl, hh, sep], -1)))
            hn = hn.index_copy(1, s["ids"], h.to(dt))

        # ---- top-down sweeps: a truncated fixed-point iteration ----
        for _ in range(self.n_sweeps):
            for s in plan.down:
                h = hn[:, s["ids"], :]; ph = hn[:, s["par"], :]
                cat = torch.cat([ph, h], -1)
                h = self.ln_d(h + self.gate(cat) * self.bcast(cat))
                hn = hn.index_copy(1, s["ids"], h.to(dt))

        # ---- scatter leaf states back to their buses ----
        leaf_state = hn[:, plan.leaf_ids, :][:, plan.l2b_leaf, :]
        upd = self.leaf_out(torch.cat([hb[:, plan.l2b_bus, :], leaf_state], -1))
        out = hb.index_add(1, plan.l2b_bus, upd.to(dt))
        for blk in self.blocks:
            out = out + blk(out).to(dt)
        return out, hn


# ---------------------------------------------------------------------
#  Benchmark spatial operators
# ---------------------------------------------------------------------
class FlatAttention(nn.Module):
    """Full pairwise self-attention over buses. `bias='hop'` -> Graphormer."""

    def __init__(self, d=192, heads=6, layers=3, bias=None, n_bias=20):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict(dict(
                attn=nn.MultiheadAttention(d, heads, batch_first=True),
                ln1=nn.LayerNorm(d), ln2=nn.LayerNorm(d),
                ff=nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))))
            for _ in range(layers)])
        self.bias_kind = bias
        if bias == "hop":
            self.hop_bias = nn.Parameter(torch.zeros(n_bias))   # init 0 == flat

    def forward(self, h, ctx):
        mask = None
        if self.bias_kind == "hop":
            hop = ctx["hop"].clamp(max=self.hop_bias.numel() - 1)
            mask = self.hop_bias[hop]                            # (N,N) additive
        elif self.bias_kind == "adj":
            mask = ctx["adj_mask"]
        for L in self.layers:
            x = L["ln1"](h)
            a, _ = L["attn"](x, x, x, attn_mask=mask, need_weights=False)
            h = h + a
            h = h + L["ff"](L["ln2"](h))
        return h, h.mean(1)


class GCNStack(nn.Module):
    """Symmetric-normalised graph convolution (Kipf & Welling)."""

    def __init__(self, d=192, layers=3):
        super().__init__()
        self.ls = nn.ModuleList([nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                               nn.Linear(2 * d, d))
                                 for _ in range(layers)])
        self.ns = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])

    def forward(self, h, ctx):
        A = ctx["adj_norm"]
        for lin, ln in zip(self.ls, self.ns):
            h = ln(h + lin(torch.einsum("ij,bjd->bid", A, h)))
        return h, h.mean(1)


class GATStack(nn.Module):
    """Graph attention, hard-masked to 1-hop neighbourhoods."""

    def __init__(self, d=192, heads=6, layers=3):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict(dict(attn=nn.MultiheadAttention(d, heads, batch_first=True),
                               ln=nn.LayerNorm(d),
                               ff=nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                                nn.Linear(2 * d, d))))
            for _ in range(layers)])

    def forward(self, h, ctx):
        m = ctx["adj_mask"]
        for L in self.layers:
            x = L["ln"](h)
            a, _ = L["attn"](x, x, x, attn_mask=m, need_weights=False)
            h = h + a + L["ff"](x)
        return h, h.mean(1)


class S4Lite(nn.Module):
    """Diagonal selective state-space scan over the bus sequence (Mamba family).

    The scan is a Python loop over N, which is genuinely how a naive SSM behaves
    without a fused kernel. It is the honest cost of this architecture here.
    """

    def __init__(self, d=192, layers=3, state=24):
        super().__init__()
        self.state = state
        self.A = nn.ParameterList([nn.Parameter(-torch.rand(d, state) - 0.5) for _ in range(layers)])
        self.Bp = nn.ParameterList([nn.Parameter(torch.randn(d, state) * 0.1) for _ in range(layers)])
        self.Cp = nn.ParameterList([nn.Parameter(torch.randn(d, state) * 0.1) for _ in range(layers)])
        self.dt = nn.ParameterList([nn.Parameter(torch.zeros(d)) for _ in range(layers)])
        self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.ff = nn.ModuleList([nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                               nn.Linear(2 * d, d)) for _ in range(layers)])

    def forward(self, h, ctx):
        B, N, d = h.shape
        for k in range(len(self.ln)):
            x = self.ln[k](h)
            dt = F.softplus(self.dt[k])[None, :, None]
            Ab = torch.exp(self.A[k][None] * dt)                       # (1,d,s)
            ys = []
            s = x.new_zeros(B, d, self.state)
            Bd = self.Bp[k][None] * dt
            for n in range(N):
                s = Ab * s + Bd * x[:, n, :, None]
                ys.append((s * self.Cp[k][None]).sum(-1))
            h = h + torch.stack(ys, 1) + self.ff[k](x)
        return h, h.mean(1)


class KANLayer(nn.Module):
    """Kolmogorov-Arnold layer: learnable RBF-spline edge functions."""

    def __init__(self, d_in, d_out, n_basis=8):
        super().__init__()
        self.n = n_basis
        self.register_buffer("centres", torch.linspace(-2.5, 2.5, n_basis))
        self.w = nn.Parameter(torch.randn(d_in, d_out, n_basis)
                              * (1.0 / math.sqrt(d_in * n_basis)))
        self.wb = nn.Parameter(torch.randn(d_in, d_out) * (1.0 / math.sqrt(d_in)))

    def forward(self, x):                       # (..., d_in)
        z = (x[..., None] - self.centres) / 0.7
        phi = torch.exp(-z * z)                                     # RBF basis
        return (torch.einsum("...ik,iok->...o", phi, self.w)
                + torch.einsum("...i,io->...o", F.silu(x), self.wb))


class KANStack(nn.Module):
    def __init__(self, d=192, layers=3):
        super().__init__()
        self.mix = nn.ModuleList([nn.Linear(d, d) for _ in range(layers)])
        self.kan = nn.ModuleList([KANLayer(d, d) for _ in range(layers)])
        self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])

    def forward(self, h, ctx):
        A = ctx["adj_norm"]
        for mix, kan, ln in zip(self.mix, self.kan, self.ln):
            h = ln(h + kan(mix(torch.einsum("ij,bjd->bid", A, h))))
        return h, h.mean(1)


class BiLSTMStack(nn.Module):
    def __init__(self, d=192, layers=2):
        super().__init__()
        self.rnn = nn.LSTM(d, d // 2, layers, batch_first=True, bidirectional=True)
        self.ln = nn.LayerNorm(d)

    def forward(self, h, ctx):
        y, _ = self.rnn(h)
        return self.ln(h + y), h.mean(1)


class MLPStack(nn.Module):
    """No spatial mixing at all -- the topology-free control condition."""

    def __init__(self, d=192, layers=3):
        super().__init__()
        self.net = nn.ModuleList([nn.Sequential(nn.LayerNorm(d),
                                                nn.Linear(d, 2 * d), nn.GELU(),
                                                nn.Linear(2 * d, d))
                                  for _ in range(layers)])

    def forward(self, h, ctx):
        for blk in self.net:
            h = h + blk(h)
        return h, h.mean(1)


# ---------------------------------------------------------------------
class Detector(nn.Module):
    """Temporal encoder + swappable spatial operator + shared heads."""

    def __init__(self, kind="sceptre", d=None, c_in=CHANNELS, n_sweeps=2,
                 n_layers=None, spatial_override=None):
        super().__init__()
        d = d or BUDGET.d_model
        n_layers = n_layers or BUDGET.n_layers
        self.kind = kind; self.d = d
        self.temporal = TemporalEncoder(c_in, d, n_layers)
        self.heads = Heads(d, c_in)
        self.spatial = spatial_override or {
            "sceptre":     lambda: RSTE(d, n_sweeps, n_blocks=n_layers - 1),
            "msa3e":       lambda: FlatAttention(d, 6, n_layers, bias=None),
            "transformer": lambda: FlatAttention(d, 6, n_layers, bias=None),
            "graphormer":  lambda: FlatAttention(d, 6, n_layers, bias="hop"),
            "gat":         lambda: GATStack(d, 6, n_layers),
            "gcn":         lambda: GCNStack(d, n_layers),
            "s4":          lambda: S4Lite(d, n_layers),
            "kan":         lambda: KANStack(d, n_layers),
            "bilstm":      lambda: BiLSTMStack(d, 2),
            "mlp":         lambda: MLPStack(d, n_layers),
        }[kind]()
        # a learned positional table is what makes flat models size-BOUND
        self.needs_pos = kind in ("transformer", "msa3e", "graphormer",
                                  "bilstm", "mlp", "s4")
        self.pos = None

    def bind(self, plan, ctx, n_bus, grid=None):
        """Attach grid-specific buffers. Flat models must allocate a positional
        table of size N, so they cannot transfer to a different N.

        `grid` supplies the range projector for Upgrade 2. It is a BUFFER, not a
        parameter -- it is determined by the topology, carries no learned weights,
        and is swapped when the model is re-bound to another system. The
        parameter count therefore stays independent of N.
        """
        self.plan, self.ctx = plan, ctx
        self.P_range = None
        if USE_PROJREP and grid is not None and "P_inj_perp" in grid:
            dev = next(self.parameters()).device
            eye = np.eye(grid["P_inj_perp"].shape[0], dtype=np.float32)
            self.P_range = torch.tensor(eye - grid["P_inj_perp"],
                                        dtype=torch.float32, device=dev)
        if self.needs_pos and (self.pos is None or self.pos.shape[0] != n_bus):
            dev = next(self.parameters()).device
            self.pos = nn.Parameter(torch.zeros(n_bus, self.d, device=dev))
            nn.init.normal_(self.pos, std=0.02)
        return self

    def forward(self, x):
        h = self.temporal(x)
        if self.needs_pos and self.pos is not None:
            h = h + self.pos[None]
        if self.kind == "sceptre":
            h, _ = self.spatial(h, self.plan)
        else:
            h, _ = self.spatial(h, self.ctx)
        return self.heads(h, getattr(self, "P_range", None))

    def n_params(self):
        n = sum(p.numel() for p in self.parameters())
        if self.pos is not None:
            n += self.pos.numel()
        return n


def make_ctx(case, device=DEVICE):
    """Grid-derived buffers the baselines need (adjacency, hop matrix)."""
    g = GRIDS[case]; N = g["n"]
    A = np.zeros((N, N), dtype=np.float32)
    for (f, t) in g["edges"]:
        A[f, t] = 1.0; A[t, f] = 1.0
    A_hat = A + np.eye(N, dtype=np.float32)
    dg = A_hat.sum(1); dinv = 1.0 / np.sqrt(dg)
    A_norm = dinv[:, None] * A_hat * dinv[None, :]
    mask = torch.tensor(A_hat == 0, device=device)             # True = blocked
    mask = mask.masked_fill(torch.eye(N, dtype=torch.bool, device=device), False)
    return dict(adj_norm=torch.tensor(A_norm, device=device),
                adj_mask=mask,
                hop=torch.tensor(g["hop"], dtype=torch.long, device=device))


CTX = {c: make_ctx(c) for c in CASES}

MODELS = ["sceptre", "msa3e", "transformer", "graphormer", "gat", "gcn",
          "s4", "kan", "bilstm", "mlp"]
PRETTY = {"sceptre": "SCEPTRE (RSTE)", "msa3e": "MSA3E [base paper]",
          "transformer": "ST-Transformer", "graphormer": "Graphormer",
          "gat": "GAT", "gcn": "GCN", "s4": "S4 / Mamba-lite",
          "kan": "KAN", "bilstm": "BiLSTM", "mlp": "MLP (no topology)"}

hdr = (f"{'model':<22}{'params @ case14':>18}{'params @ case118':>19}"
       f"{'N-independent?':>17}{'fwd ms @14':>13}{'fwd ms @118':>14}")
print(hdr); print("-" * len(hdr))
PARAM_TABLE, LAT_TABLE = {}, {}
for m in MODELS:
    row = []
    for c in ["case14", "case118"]:
        mdl = Detector(m).to(DEVICE).bind(PLANS[c], CTX[c], GRIDS[c]["n"], GRIDS[c]).eval()
        npar = mdl.n_params()
        xb = torch.randn(64, WINDOW, GRIDS[c]["n"], CHANNELS, device=DEVICE)
        with torch.no_grad(), torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=USE_AMP):
            for _ in range(3): mdl(xb)
            if DEVICE == "cuda": torch.cuda.synchronize()
            t1 = time.time()
            for _ in range(10): mdl(xb)
            if DEVICE == "cuda": torch.cuda.synchronize()
        row.append((npar, (time.time() - t1) / 10 * 1000))
        del mdl, xb
        if DEVICE == "cuda": torch.cuda.empty_cache()
    PARAM_TABLE[m] = (row[0][0], row[1][0])
    LAT_TABLE[m] = (row[0][1], row[1][1])
    inv = "YES" if row[0][0] == row[1][0] else "no"
    print(f"{PRETTY[m]:<22}{row[0][0]:>18,}{row[1][0]:>19,}{inv:>17}"
          f"{row[0][1]:>13.2f}{row[1][1]:>14.2f}")
print("-" * len(hdr))
_inv = [m for m in MODELS if PARAM_TABLE[m][0] == PARAM_TABLE[m][1]]
print()
print(textwrap.fill(
    f"INFERENCE. {len(_inv)} of the {len(MODELS)} architectures are "
    f"parameter-invariant in N ({', '.join(PRETTY[m] for m in _inv)}). The other "
    f"{len(MODELS)-len(_inv)} are not, because a learned positional table with N "
    "rows is the only way a flat sequence model can tell one bus from another. "
    "That distinction is not a detail: it decides, in Part 10, which models can "
    "even be EVALUATED on a different bus count. Only SCEPTRE combines "
    "N-invariance with a full-grid receptive field -- GCN, GAT and KAN are also "
    "N-invariant but remain hop-limited.", 79))
print()
print(textwrap.fill(
    f"On cost: every arm is now in the "
    f"{min(PARAM_TABLE[m][0] for m in MODELS)/1e6:.1f}-"
    f"{max(PARAM_TABLE[m][0] for m in MODELS)/1e6:.1f} M parameter range at "
    f"d = {BUDGET.d_model}, which is roughly 20x the previous revision and enough "
    "to actually load this GPU. Latency is measured on a 64-window batch under "
    "bf16 autocast.", 79))