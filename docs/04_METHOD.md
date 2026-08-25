# Method — formal statement

Notation: `N` buses, `M = N + |E|` measurements, window length `W`, channel count
`C = 4` (`|V|, θ, P_inj, Q_inj`), model width `d`.

---

## 1. The stealth subspace

DC state estimation solves `z = Hx + e` with `H ∈ R^{M×(N−1)}` the measurement
Jacobian for `z = [P_inj ; P_flow]` against bus angles. The weighted-least-squares
residual is

```
    r(z) = (I − H H⁺) z ,          H⁺ = (HᵀH)⁻¹Hᵀ
```

For any `c ∈ R^{N−1}`, `H c ∈ range(H) = ker(I − H H⁺)`, hence

```
    r(z + Hc) = r(z)          exactly.
```

The classical χ² bad-data test is therefore blind to `a = Hc` **by construction**.
Cell 2 verifies this numerically on all four systems (relative change ≈ 10⁻¹⁶,
i.e. machine epsilon); Cell 9 measures the contrast against the base paper's
multiplicative attack, which scores up to `1.8 × 10⁴ ×` the χ² threshold.

This is the threat model. Everything downstream defends against `a = Hc`, not
against the base paper's `V(1 + m g)`.

---

## 2. C1 — the nested-dissection separator tree

### 2.1 Construction

Given the susceptance-weighted adjacency `W` (`W_ij = b_ij` for each line or
transformer), recursively:

1. Form the normalised Laplacian of the induced subgraph,
   `L̃ = D^{−1/2}(D − W)D^{−1/2}`.
2. Take the Fiedler vector (eigenvector of the second-smallest eigenvalue),
   rescale by `D^{−1/2}`, and split at its median → partitions `L`, `R`.
3. The **separator** `S` is the set of vertices incident to an edge crossing the
   cut. Removing `S` renders `L` and `R` conditionally independent in the DC
   model.
4. Recurse on `L` and `R` until singletons.

This is George's 1973 nested dissection, the ordering behind sparse Cholesky
factorisation, applied to the electrical network rather than to a mesh.

### 2.2 Measured properties

| system | `N` | tree nodes | depth | `⌈log₂ N⌉` | ratio | root `|S|` |
|---|---|---|---|---|---|---|
| case14 | 14 | 27 | 4 | 4 | 1.00 | 5 |
| case30 | 30 | 59 | 6 | 5 | 1.20 | 8 |
| case57 | 57 | 113 | 7 | 6 | 1.17 | 14 |
| case118 | 118 | 235 | 8 | 7 | 1.14 | 15 |

Depth tracks `⌈log₂ N⌉` within ~1.2×, so the `O(log N)` claim holds empirically
and not merely asymptotically.

### 2.3 RSTE — Recursive Separator-Tree Encoder

Bus features `h_b ∈ R^{N×d}` come from a temporal encoder shared by every
baseline. Then:

**Leaf initialisation.** `h_v = LN( mean_{b ∈ v} h_b )` for each leaf `v`.

**Bottom-up sweep** (deepest level first), for every internal node `p` with
children `lo, hi` and separator `S_p`:

```
    h_p = LN( Merge( [ h_lo ; h_hi ; mean_{b ∈ S_p} h_b ] ) )
```

**Top-down sweep** (root first), repeated `K` times:

```
    g   = σ( Gate( [h_parent ; h_v] ) )
    h_v = LN( h_v + g ⊙ Bcast( [h_parent ; h_v] ) )
```

**Scatter.** Leaf states are written back to their buses through a learned
`LeafOut([h_b ; h_leaf])`.

`Merge`, `Bcast`, `Gate`, `LeafOut` and both LayerNorms are **shared across every
level and every node**. Consequences:

* **Parameter count is independent of `N`** — measured at 116 614 for case14 and
  case118 alike.
* **Repeating the top-down sweep is a truncated fixed-point iteration** on the
  tree — a deep-equilibrium-style construction, but over a spatial decomposition
  rather than over time. Ablated in Part 13 at `K ∈ {0,1,2,3}`.
* **Receptive field.** Any two buses communicate through their lowest common
  ancestor, so information travels at most `2·depth = O(log N)` tree hops. A
  `k`-layer GNN reaches `k` graph hops, and the diameters above are 5/6/12/14.

### 2.4 Complexity

One up-down pass evaluates each of the `2N−1` nodes a constant number of times,
so cost is `O(N d²)` per sweep against `O(N² d)` for dense attention — but the
operative property is the *kernel-launch* count, which is `O(depth) = O(log N)`
because the schedule is compiled level-order (`TreePlan`).

---

## 3. C2 — HALO

### 3.1 The construction

The separator tree is already a segment tree over the bus set. For each node `v`
define the statistic `S_v = max_{b ∈ v} s_b`, with `s_b` the detector's per-bus
localization logit.

**Split-conformal calibration.** On clean calibration windows collect the null
distribution `{S_v^{(i)}}`. For a test window,

```
    p_v = ( 1 + #{ i : S_v^{(i)} ≥ S_v^{test} } ) / ( n_cal + 1 )
```

This is a valid finite-sample p-value under exchangeability of the clean windows,
with no distributional assumption on the detector's scores.

**Descent.** Test the root. Only descend into a child whose family-wise
Benjamini–Hochberg test at level `q` rejects. Buses are flagged when a leaf is
rejected.

**FDR.** Testing a node only after its parent is rejected is Yekutieli's
hierarchical testing structure; BH within each family of siblings then controls
FDR over the tree. The notebook **measures** the realised FDR (Figure 11c) rather
than resting on the theory.

### 3.2 The exchangeability caveat — read this before quoting any FDR number

The calibration set is clean windows, so the p-value is valid for

> *H₀(v): the bus set of node `v` contains no tampered measurement.*

It is **not** valid per-bus. A null-space attack `a = Hc` perturbs **every** bus
simultaneously — that is what makes it stealthy — so an *untampered* bus inside
an *attacked* window is not exchangeable with a bus drawn from a clean window.
Its score is elevated by its neighbours' corruption.

Consequence, measured and reported in the notebook: **the realised per-bus FDR
exceeds the nominal `q`, for HALO and for the flat test alike.** The two
procedures share the same scores and the same `q`, so the *comparison* between
them is unaffected; the absolute level is not a calibration claim and must not be
quoted as one.

Closing this properly needs either a per-bus null that conditions on the attack
being present elsewhere, or a restatement of the target as subtree-level FDR.
Both are open.

### 3.3 What it buys

1. **Cost — established.** An attack on `k` buses is isolated in `O(k log N)`
   node evaluations instead of `O(N)`. Measured across `N = 14 … 118`.
2. **Multiplicity — structural.** The correction applies to families of size 2,
   not to `N` simultaneous hypotheses, so the per-test threshold does not shrink
   as the grid grows.
3. **Power — a trade, not a free lunch.** Early termination at an unrejected
   internal node forfeits every attacked bus beneath it. Whether that costs F1
   relative to the flat test is measured per system, and reported either way.

---

## 4. C3 — NL-LFC

Plant state `(f_i, P_m,i, P_g,i, P_r,i, δ_i, P_tie,i, V_i)`, `i ∈ {1,2}`.

| # | Mechanism | Form |
|---|---|---|
| 1 | AC power flow | Newton–Raphson; surrogate verified on held-out solves, Cell 9 |
| 2 | ZIP load | `P_L = P_0(a_z V² + a_i V + a_p)`, `(a_z,a_i,a_p) = (0.4,0.3,0.3)` |
| 3 | Tie-line | `P_tie = (V_1V_2/X) sin(δ_1 − δ_2)` |
| 4 | Dead band | backlash with memory, half-width `GDB = 6×10⁻⁴` p.u. |
| 5 | Rate limit | `|dP_m/dt| ≤ GRC = 1.7×10⁻³` p.u./s |
| 6 | Reheat | `dP_r = (P_g − P_r)/T_t`, `dP_m = (P_r + K_r T_t dP_r − P_m)/T_r` |
| 7 | Renewables | Ornstein–Uhlenbeck × diurnal envelope |
| 8 | Regimes | 3-state Markov chain over `{0.85, 1.00, 1.15}` |

`linear_mode=True` degrades to the base paper's plant exactly, so each effect can
be ablated with one flag.

**Channel:** piecewise-constant latency (0–2 steps, re-drawn every ~40 steps),
1 % instrument noise, 12-bit quantization, 1 % dropout. The channel is applied
**before** the attack, because an adversary corrupts the telemetry that arrives —
this ordering is what keeps the label attached to the evidence in the tensor.

### The AC response surface

Per-step Newton–Raphson inside an RL loop is infeasible. A full second-order
basis in the driver vector `u = [ΔP_L,1, ΔP_L,2, ΔP_res,1, ΔP_res,2]` is fitted to
true NR solutions over a ±40 % load / ±25 % renewable envelope, and **verified on
held-out true solutions**. Reported honestly: `R²_linear ≥ 0.997`, so the map is
dominated by its linear part; the claim is that the residual the DC model
discards is 10–30× larger than it needs to be, and that residual is the scale at
which a stealth attack operates.

---

## 5. C4 — SSC: Stealth-Stratified Certification

### 5.1 The certificate

For every generated attack vector `a` we compute, and **store with the sample**:

| symbol | definition | meaning |
|---|---|---|
| `σ` | `1 − ‖P_⊥ a‖ / ‖a‖` | **stealth index**; `1` = invisible to χ², `0` = fully visible |
| `Δχ²` | `(‖P_⊥(z+a)‖² − ‖P_⊥z‖²)/σ_n²` | exact displacement of the classical statistic |
| `‖a‖₂` | — | attack energy |
| `μ` | `|supp(a)|` | **attacker cost**: meters that must be compromised |
| `‖c‖₂` | `‖H⁺a‖` | **attacker payoff**: how far the state estimate moves |
| `k` | `‖c‖₀` | compromised substation neighbourhoods |
| `ε` | — | the attacker's relative network-model error |

None of these depends on any model. They are properties of the attack vector and
the grid, which is what makes them comparable across papers and across systems.

### 5.2 Sparse synthesis, and the attacker-cost axis

If `c = e_i`, then `a = He_i` is the `i`-th column of `H`, whose support is exactly
the injection meters at bus `i` and its neighbours plus the flow meters on
incident branches — **one substation neighbourhood**. Taking `‖c‖₀ = k` corresponds
to compromising `k` such neighbourhoods.

Measured cost, at `k = 1`:

| system | total meters | meters needed | % of system |
|---|---|---|---|
| case14 | 34 | 6.6 | 19.4 % |
| case30 | 71 | 6.4 | 9.0 % |
| case57 | 137 | 6.9 | 5.0 % |
| case118 | 304 | 8.3 | **2.7 %** |

The curve is strongly sublinear in `k`: most of an attacker's reach is bought with
the first few substations.

### 5.3 Why σ varies at all — the omniscient-attacker assumption

If the attacker builds `a = Hc` with the operator's **exact** `H`, then `σ ≡ 1` and
the stratification collapses to a point. We measured exactly that on a first pass:
100 % of samples in one stratum.

That degeneracy is not a coding artefact — it is the **omniscient-attacker
assumption** the literature makes without comment. A real adversary's Jacobian
`H̃` carries a relative error `ε`, and the mismatch leaks out of the null space:

```
    ‖P_⊥ a‖ = ‖P_⊥ (H̃ − H) c‖ > 0     whenever   H̃ ≠ H
```

So `σ` becomes a monotone measure of **adversary model fidelity**. Measured on
case14:

| `ε` | who this is | `σ` |
|---|---|---|
| 0 | textbook omniscient attacker | exactly 1 |
| 0.05 | good reconnaissance | ≈ 0.99 |
| 0.2 | partial network knowledge | ≈ 0.97 |
| 0.8 | public-data attacker | ≈ 0.89 |
| 1.5 | little more than a guess | ≈ 0.82 |
| 3.0 | topology largely wrong | ≈ 0.71 |

### 5.4 Strata and reporting rule

Boundaries are **fixed and round**, anchored to the measured `σ ↔ ε` mapping, and
**not fitted to our own distribution**:

```
    S0  σ = 1              ε = 0            omniscient
    S1  0.98 ≤ σ < 1       ε ≈ 0.04–0.12    good reconnaissance
    S2  0.95 ≤ σ < 0.98    ε ≈ 0.12–0.30    partial knowledge
    S3  0.85 ≤ σ < 0.95    ε ≈ 0.30–1.20    public-data attacker
    S4  σ < 0.85           ε > 1.2          topology largely wrong
```

**Reporting rule.** Evaluate on *all clean windows plus the attacked windows in the
stratum*, with the decision **threshold frozen** at the full-test value.

> Freezing the threshold matters. Re-optimising it per stratum lets the model
> trade precision for recall differently in each slice and reports a number no
> deployed system could achieve — a deployed detector has *one* threshold and does
> not know which stratum it is in. It inflates the hard strata substantially.

### 5.5 Honest limitation

The boundaries are reusable, but the *population* of each stratum is a property of
our generator. Anyone comparing against us must report their own stratum
composition; `manifest.json` makes that possible but cannot enforce it.

---

## 6. C5 — the blind-spot certificate

For severity `m`, draw directions `c` uniformly on the unit sphere of `R^{N−1}`
and set `a = Hc·m`. Define

```
    ρ(m) = Pr[ a invisible to the χ² test  AND  below the detector's threshold ]
```

estimated by Monte Carlo with **Clopper–Pearson** intervals. Then

```
    m*(ε) = min { m : ρ(m') < ε  for every m' ≥ m }
```

The "for every larger `m`" quantifier is required because `ρ` is **not**
guaranteed monotone — the measurement shows a blind *band*, not a blind floor,
and a first-crossing definition would be misleading.

Below `m*`, detection is not what keeps the plant safe; the CBF shield is.

---

## 7. C6 — conformal-in-the-loop CBF

**Safe set.** `h_ace,i = ACE_max² − ACE_i² ≥ 0`, `h_f,i = f_max² − f_i² ≥ 0`.

**Barrier condition.** `ḣ ≥ −α h`, affine in `u = ΔP_c`, so the filter is a
minimal-norm projection solved in closed form.

**The conformal coupling.** `ACE_max ← max(ACE_max − κ·w_i, 0.25·ACE_max)` with
`w_i` the calibrated interval width from the Mondrian conformal layer. Large
uncertainty ⇒ the shield reserves more room; small uncertainty ⇒ it gets out of
the way. Because `ΔP_tie` appears in both areas' barriers, area `i`'s width also
shrinks area `j`'s admissible action — there is no single-agent analogue.

**Calibration.**
* *Mondrian split conformal*, grouped by the detector's own flag — an observable
  the controller has at run time, so the conditional guarantee is usable online.
* *Adaptive conformal* (Gibbs–Candès): `α_{t+1} = α_t + γ(ᾱ − err_t)`, which
  restores coverage under attacker drift with no retraining.

**Stackelberg self-play.** A Gaussian policy over `(p_attack, magnitude, mix)` is
trained by REINFORCE against the live defender in alternating best-response
rounds. The reported statistic is the **equilibrium gap**: the improvement one
further best-response budget buys the attacker. It is non-zero here, so the pair
is reported as *not converged*.

**Three defects an earlier revision had, and the fixes.** All three produced
plausible-looking but meaningless numbers, and each is worth recording.

1. *The safe set was saturated.* `ACE_max = 0.05` was violated by the undefended
   plant on **68 % of steps** — not a safe set but a constraint the plant lives
   outside of, making the barrier active everywhere and the experiment
   uninformative. **Fix:** measure the undefended excursion distribution and place
   the bound at its 88th percentile. *General lesson: if the baseline violates the
   constraint most of the time, the experiment cannot detect an improvement.*
2. *The controller had no integral action.* Real LFC secondary control is **PI**;
   proportional-only leaves a steady-state ACE offset that swamps the transient
   the shield acts on. **Fix:** `u = −K_p·ACE − K_i∫ACE`.
3. *The control-authority model was wrong by ~50×.* The barrier's `g(x)` used the
   effect of `u` over one 0.1 s step, but a setpoint change reaches the shaft only
   after the governor and turbine lags — so the filter was formally active and
   numerically inert (identical trajectories to five decimals, paired differences
   ~1e-8). **Fix:** reason over `T_eff = T_g + T_t = 0.7 s`.

After the three fixes, the shield's effect resolves positively on every magnitude
metric at `n = 120` paired episodes, and the conformal margin beats the plain
shield on all of them.

**Paired evaluation.** Attack timing, direction, sensor noise and disturbance are
drawn *before* the episode and replayed identically through every configuration
(`EpisodeSchedule`). Differences are taken **within episode**, so `n` = episodes,
not seeds. This is the fix for the predecessor project's largest measurement
defect, where the two arms diverged after the first shielded action and were
therefore never actually paired.
