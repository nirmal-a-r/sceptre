# Novelty — what is actually new, and how confident we are

This file exists so a reviewer (or the team) can see, per claim, **what is
category-new, what is a strong extension, and what is reused**. Overstating any
of these is the fastest way to lose a paper.

---

## The one-sentence claim

> **A power network should be encoded the way a sparse solver factorises it —
> by recursive dissection into separators — not the way a social network is
> encoded, by passing messages along edges.**

Everything else follows: the separator tree gives an `O(log N)` receptive field
(fixing the measured GNN bottleneck), weight tying across levels gives
`N`-independent parameters (giving zero-shot transfer across bus counts), and the
same tree doubles as a segment tree for `O(k log N)` conformal attack
localization.

---

## Claim-by-claim audit

### C1 — RSTE: recursive separator-tree encoder  ·  **CATEGORY-NEW**

**What is new.** Using **nested dissection / elimination-tree structure** — the
classical data structure behind sparse Cholesky factorisation — as a *neural
inductive bias* for power-grid telemetry.

**Evidence it is unimplemented.** A literature search for
`"nested dissection" OR "elimination tree" OR "separator tree"` combined with
neural networks, FDI detection and power systems returned no prior work; the hits
are decision-tree / gradient-boosting papers, which are unrelated. Search terms
are recorded in `02_LITERATURE.md` so the claim can be challenged directly.

**What is reused.** Nested dissection itself (George, 1973). Weight-tied recursive
networks over trees (Socher *et al.*, TreeLSTM lineage). Fixed-point / deep
equilibrium layers, including a power-systems precedent (`arXiv:2405.06827`).
**None of these is claimed as new.** The composition is.

**Sub-claims, and how each is measured.**

| | claim | measurement |
|---|---|---|
| C1a | beats nine baselines on detection/localization | Fig. 8, 3 seeds, paired t per baseline |
| C1b | parameters independent of `N` | identical weight tensors at case14 and case118 (Part 6; exact count printed by the notebook) |
| C1c | zero-shot transfer 14 → 118 buses | Fig. 10 — identical weight tensors, no gradient steps |
| C1d | receptive field `O(log N)` | tree depth 4/6/7/8 vs `⌈log₂N⌉` 4/5/6/7 (Part 2) |
| C1e | the *nested dissection* is what matters | ablation vs a random balanced tree of equal depth (Fig. 15) |

**The honest weak point — and it is a real one.** C1a **does not hold on
IEEE 14-bus.** At the benchmark operating point a plain GCN matches or beats
SCEPTRE on both detection and per-bus localization. That is the expected result,
not a defect: at `N = 14` a 3-layer message-passing stack already reaches ~89 % of
the grid, so the receptive-field bottleneck that motivates a separator tree simply
does not bind.

We also verified this is not an artefact of the difficulty we chose. A separate
calibration sweep found a harder operating point — severity `(0.10, 0.80)` — at
which **SCEPTRE still learns (F1 0.72) while GCN and MLP collapse to the
all-positive floor (0.66)**. We did **not** adopt it, because a benchmark in which
half the baselines are degenerate compares nothing, and adopting it would have
been tuning the task to flatter the method. The hard regime is reported instead as
a *stratum* in Part 8, where it belongs.

**Therefore: do not lead with C1a.** Lead with C1b/C1c/C2 — properties no flat
baseline can exhibit *at all*, and that message-passing baselines lose
monotonically as `N` grows. A manuscript that opens with the case14 detection
table opens with the weakest evidence in the project.

---

### C2 — HALO: hierarchical conformal localization  ·  **NARROWED — see below**

> **Downgraded from CATEGORY-NEW.** An earlier draft of this file called C2
> category-new. A later literature search found Boyaci *et al.* (IEEE Trans.
> Smart Grid, 2022), who perform **joint GNN detection and localization** of
> stealth FDI and state they are the first to do so. The claim is corrected here
> rather than defended.

**What is NOT new, and must not be claimed.** Using learned graph structure to
localize an FDI attack to specific buses. Boyaci *et al.* [15] do this with
ARMA graph filters; Qu *et al.* [4] do it with a graph wavelet CNN.

**What remains new.** Reusing the separator tree as a **segment tree** and
localizing by descent, with a split-conformal p-value per node and
Benjamini–Hochberg control within each sibling family. Two specific properties
follow, and neither is provided by the prior art above:

* **Cost class.** `O(k log N)` node evaluations rather than `O(N)` per-bus
  scores. Both prior methods score every bus.
* **Multiplicity control.** The correction applies to families of size 2 rather
  than to `N` simultaneous hypotheses, so power does not decay as the grid grows.
  Neither prior method applies any multiplicity correction at all.

**Nearest prior art.** Boyaci *et al.* 2022 [15] — GNN joint detection and
localization, flat per-bus. Qu *et al.* 2024 [4] — graph wavelet CNN
localization, flat per-bus. Conformal + FDR exists for link prediction, and
hierarchical conformal testing exists generically. The *combination on a
power-network separator tree, with the cost class and the FDR guarantee*, is what
is new — and that is how the manuscript phrases it.

**The honest weak points — two, both material.**

1. **The per-bus FDR target is not met, and cannot be with this calibration.**
   The conformal p-values are valid for "this *subtree* contains no tampered
   bus", because they are calibrated on clean windows. They are not valid
   per-bus: a null-space attack `a = Hc` perturbs *every* bus at once, so an
   untampered bus inside an attacked window is not exchangeable with a bus from
   a clean window. Measured per-bus FDR therefore exceeds nominal `q` — **for
   HALO and for the flat baseline equally**. The comparison between the two
   survives (same scores, same `q`); the absolute level is not a calibration
   claim. Fixing this is open work: either a per-bus null that conditions on the
   attack being present elsewhere, or restating the target as subtree-level FDR.
2. **FDR control under hierarchical testing is cited, not proved here.** The
   notebook *measures* the realised rate. A theorem for this estimator would
   strengthen the paper.

**And a scope point.** The cost result (`O(k log N)` vs `O(N)`) is solid and
measured across four systems. Whether HALO also *gains* power over the flat test
is system-dependent and reported per system — early termination at an unrejected
internal node forfeits everything beneath it, so it is a trade, not a free lunch.

---

### C3 — NL-LFC benchmark  ·  **STRONG EXTENSION**

Each of the eight mechanisms is standard in power-systems modelling. What is new
is assembling them into an **FDI-detection benchmark with per-bus attack labels**,
and providing a one-flag reduction (`linear_mode=True`) back to the base paper's
plant so every effect is ablatable.

**Do not claim** that "the grid is highly nonlinear". Cell 7 reports
`R²_linear ≥ 0.997`: the AC map is dominated by its linear part over the sampled
envelope. The defensible claim is narrower and better — *the residual the DC model
discards is 10–30× larger than it needs to be, and that residual is precisely the
scale at which a stealth attack operates.*

---

### C4 — SSC: Stealth-Stratified Certification  ·  **CATEGORY-NEW (this revision)**

**What is new.** Every attacked sample in the dataset ships a **physics-derived
certificate** — stealth index `σ`, χ² displacement, attack energy, meter support
`μ`, state displacement `‖c‖`, sparsity `k`, and the attacker's model error `ε`.
None of these depends on any model; they are properties of the attack vector and
the grid.

**Why it matters.** Every published FDI dataset labels a sample with **one bit**,
and papers report **one F1** on a test set built by their own generator. That
number is not comparable across papers, because F1 depends overwhelmingly on how
hard the author made the attacks — and nobody reports that. SSC makes the
difficulty axis explicit and fixed, so results become a **curve** `F1(σ)` that two
groups can compare at matched difficulty.

**The sub-finding that makes `σ` non-trivial.** If the attacker uses the operator's
exact `H`, then `σ ≡ 1` and the stratification collapses to a point — we measured
exactly that on a first pass, 100 % of samples in one stratum. That degeneracy *is*
the **omniscient-attacker assumption** the literature makes without comment, made
visible. Relaxing it (the attacker builds `a = H̃c` from an imperfect network
model) makes `σ` a continuous, monotone measure of **adversary model fidelity**.
We have not seen that axis reported before.

**Second sub-finding, and it is operationally alarming.** Because `a = He_i` is the
`i`-th column of `H`, a stealth attack needs only the meters in **one substation
neighbourhood**. Measured: on IEEE 118-bus that is **8.3 of 304 meters — 2.7 % of
the system**. The cost curve is strongly sublinear, so most of an attacker's reach
is bought with the first few substations. This is directly actionable (it says
which substations to harden) and we have not found it reported.

**What is reused.** Split conformal prediction; the `a = Hc` stealth construction
(Liu–Ning–Reiter 2011). **The composition — per-sample physics certificates
shipped with the data, and stratified reporting over them — is what is claimed.**

**Honest weak point.** The strata boundaries are fixed and physically anchored,
but the *population* of each stratum is a property of our generator. A reader
comparing against us must report their stratum composition too, which the
`manifest.json` makes possible but does not enforce.

---

### C5 — certified blind-spot volume  ·  **NOVEL, INHERITED**

Carried over from the predecessor project and re-measured on the nonlinear plant.
The idea — report a property of the *(physics, detector)* pair instead of a
test-set score — is genuinely unusual in this literature.

**Honest caveats, both already in the notebook.** `ρ(m)` is **not monotone** in
severity; the sweep shows a blind *band*, so `m*` carries an "for every larger m"
quantifier rather than a first-crossing definition. Monte-Carlo spread at low
severity is wide, which is why Clopper–Pearson intervals are plotted rather than
point estimates.

---

### C6 — Stackelberg attacker + conformal CBF  ·  **NARROWED — see below**

> **Two corrections to an earlier draft of this section.**
>
> 1. It cited `02_LITERATURE.md` B6 for "conformal prediction has already reached
>    grid security". **That entry has been retracted** — the specific paper it
>    named could not be verified. Conformal prediction *is* established in
>    power-systems ML (price forecasting, PV bidding, optimality-gap bounding),
>    which is enough for the point being made, but the FDI-specific citation was
>    not real and is gone.
> 2. It did not know about Mohamed, Kundur & Khalaf (arXiv:2303.02197, 2023),
>    who apply a **control barrier function shield to LFC under attack** — the
>    same mechanism, the same control loop, the same purpose.

**What is NOT new, and must not be claimed.** A CBF safety filter on the AGC
command to bound frequency and ACE excursions under a compromised signal. That is
Mohamed *et al.* [16].

**What remains new.** Three things, and the manuscript claims exactly these:

* **The safe set is calibrated, not assumed.** It is placed at the 88th
  percentile of the *undefended* plant's own excursion distribution. This is not
  a detail: our first attempt used a hand-chosen bound that the plant violated on
  most steps, which made the barrier inactive-by-saturation and caused the
  experiment to measure nothing (§6.6 records the defect and the fix).
* **The margin is conformally tightened.** The width enters **twice** — it
  tightens the CBF safe set *and* enters the MARL reward — so the shield's
  conservatism is tied to a finite-sample guarantee rather than to a constant
  someone picked.
* **The attacker is trained against the live defender**, and the **equilibrium
  gap is reported** rather than convergence being assumed.

The tie-line coupling point also stands: one area's uncertainty constrains the
other area's admissible action, which has no single-agent analogue.

**Honest weak point.** The gap is non-zero: the pair is **not** converged. Report
it as an instruction to run more rounds, not as a result.

---

### C7 — the four propositions  ·  **KNOWN MACHINERY, NEW COROLLARY**

Added in `SCEPTRE_v2.ipynb`, Part 19.

**What is NOT new, and must not be claimed.** Receptive-field limits on
message-passing networks are textbook graph-learning theory: an MPNN is bounded
by its `K`-hop dependency cone, and more sharply by `1`-Weisfeiler-Leman. Prop. 1
is a three-line consequence of the triangle inequality on graph distance. We
present it as such.

**What appears to be unclaimed.** The corollary *in this domain*. A stealthy FDI
attack is constructed as `a = Hc`, so its signature is a correlation between
buses that may sit far apart on the network. Prop. 1 then says a shallow
message-passing detector cannot represent that correlation **at all** — not
poorly, not with more training, but not within the hypothesis class. No
GNN-based FDI paper in `02_LITERATURE.md` states this, and several build
detectors at depths well under the bound.

This converts Figure 4 from an observation ("a 2-layer GNN reaches 9 % of IEEE
118-bus") into a limitation that a reviewer cannot answer with *"then use more
layers"* — because the required depth scales with the graph diameter, and depth
costs parameters, which is precisely what the separator tree avoids (Prop. 2).

**The strongest evidence that it is not post-hoc.** The theory predicts its own
null result. On IEEE 14-bus the diameter is 5, so `K = 3` already satisfies the
Prop. 1 bound, and the propositions therefore predict **no** structural advantage
for SCEPTRE on that system. Figure 8 shows it tied with three baselines there. A
rationalisation constructed after the fact would have predicted a win everywhere.

**Honest weak point.** Prop. 4's exchangeability condition is the paper's most
fragile assumption and is stated as such in the section itself: it holds for the
subtree null `H_v`, and *not* for a per-bus null inside an attacked subtree. That
is why §6.4 reports FDR at both levels rather than only the flattering one.

---

## What was dropped, and why

| Dropped | Reason |
|---|---|
| **Graph attention / GNN front end** | Pre-registered test failed in 4/4 cells at n = 5 seeds (`03_PREREGISTRATION.md`). Receptive-field bottleneck measured and reproduced as Fig. 4(d). |
| **Graph-Mamba (hard 1-hop mask + SSM)** | Same test; lost decisively (ΔRMSE t = +5.55). Retained only as the `gat` / `s4` benchmark arms. |
| **MSU/ORNL "real data" objective** | Feature-space dataset with no multi-area LFC ground truth. Listing it as external validity overstated what it can deliver. Recorded as open, not claimed. |
| **LLM supervisor (N5 of the predecessor)** | Out of scope for this paper; it was never in the causal path to the plant and adds a dependency without adding a testable claim. |

Dropping these **on recorded evidence** is itself a contribution. A reviewer who
sees a pre-registration, a failed criterion, and the replacement built around the
diagnosis will trust the surviving claims more, not less.

---

## The framing that should open the manuscript

Not *"we propose a new architecture."* Rather:

> The FDI-detection literature reports high accuracy against attacks that a 1970s
> χ² bad-data test already catches — we measure the base paper's attack at up to
> 1.8 × 10⁴ times that threshold. Against attacks constrained to the residual null
> space, `a = Hc`, the χ² statistic does not move at all, and the learned
> detectors that replaced it inherit a topology encoding that loses information as
> the grid grows: a 2-layer message-passing stack sees 57 % of IEEE 14-bus and
> 9 % of IEEE 118-bus. We encode the network the way a sparse solver factorises
> it instead — by recursive dissection — and obtain a detector whose parameters do
> not depend on the bus count, whose receptive field is logarithmic in it, and
> which localizes an attack in `O(k log N)` calibrated probes.

That paragraph is the paper.
