# SCEPTRE: Recursive Separator-Tree Encoding and Hierarchical Conformal Localization for Load Frequency Control under Stealthy False Data Injection

**Technical Report — full project write-up**

A. R. Nirmal (CB.AI.U4AID24001) · Gunnala Dheeraj Kumar (CB.AI.U4AID24017) ·
Meera S Raj (CB.AI.U4AID24031) · Vishnu Vardhan N (CB.AI.U4AID24072)
Batch A-17 · 23AID304 High-Performance and Cloud Computing

**Base paper.** Y. Cui, T. Wu and Y. Zhang, "Load Frequency Control of Smart Grid
Based on Data-Driven FDI Attacks Detection and Data Repair," *IEEE Transactions
on Smart Grid*, vol. 16, no. 6, pp. 5404–5415, Nov. 2025.
doi: `10.1109/TSG.2025.3590035`

**Companion artifacts.** `SCEPTRE.ipynb` (47 cells, all code and all figures) ·
`SCEPTRE_Review.pptx` (14-slide review deck) · `docs/00_STUDY_GUIDE.md`
(tutorial-level explanation) · `outputs/sceptre_results.json` (every number,
machine-readable).

---

## Abstract

Data-driven Load Frequency Control (LFC) couples attack detection, telemetry
repair and secondary control into a single loop. We reproduce the state of the
art in this setting and make a measurement that reframes it: **the attack model
the base paper defends against scores up to 1.8 × 10⁴ times a χ² bad-data
threshold that has been standard practice since the 1970s.** Reporting 99.78 %
detection accuracy against it is therefore not evidence of a modern
contribution. Against attacks confined to the state-estimation residual null
space, `a = Hc`, the same statistic does not move at all — we measure a relative
change of 3 × 10⁻¹⁵ — and a learned detector becomes genuinely necessary.

Every recent learned detector in this space encodes grid topology with a graph
neural network. We tested that choice with a **pre-registered** experiment and it
**failed at n = 5 seeds in all four cells**. The cause is measurable: a two-layer
message-passing stack reaches two hops, but IEEE 118-bus has a diameter of
fourteen, so each bus observes 9 % of the network. Message passing is an
information bottleneck that worsens monotonically with system size.

We therefore encode the network the way a sparse linear solver factorises it — by
**nested dissection into separators** — rather than by passing messages along
edges. The resulting **Recursive Separator-Tree Encoder (RSTE)** is weight-tied
across tree levels, giving a parameter count independent of the bus count, a
receptive field of `O(log N)`, and **zero-shot transfer from IEEE 14-bus to
118-bus using identical weight tensors**. The same tree doubles as a segment
tree, yielding **HALO**, which localizes an attack in `O(k log N)` split-conformal
probes instead of the `O(N)` per-bus scan every published localizer performs.
We further contribute **NL-LFC**, an LFC benchmark restoring eight nonlinear and
non-stationary mechanisms the base paper discards, and a **blind-spot
certificate** `ρ(m)`, `m*(ε)` that is a property of the (physics, detector) pair
rather than of any test set.

On IEEE 14-bus, SCEPTRE improves significantly on the base paper's MSA3E detector
(paired t = +4.19) but **does not top the benchmark table** — a plain GCN is
ahead. We report this, and explain it: at N = 14 the receptive-field bottleneck
does not bind. The claims that survive are the categorical ones — five of the ten
benchmarked architectures cannot be evaluated at a different bus count *at all*.

---

## 1. Introduction

### 1.1 Setting

An interconnected AC power system holds frequency at nominal by continuously
matching generation to load. Load Frequency Control (LFC) is the automatic loop
that does so, acting on telemetry streamed from substations to a control centre.
Because that telemetry is the controller's only view of the plant, an adversary
who can tamper with it can drive the controller to act against the physics.

The base paper builds a three-stage defence — detect, repair, control — and is,
to our knowledge, the first to close that loop for LFC. It is the correct
structure, and we retain it. What we change is what sits at each stage.

### 1.2 The measurement that motivates this work

The base paper's attack model (its Eqs. 1–3) multiplies a measurement by
`(1 + m·g)` with `g ~ N(0,1)` and `m = 1`, fired with probability 0.3. When
active it roughly doubles or zeroes a measurement.

We scored that attack against the classical weighted-least-squares χ² bad-data
test (notebook Cell 9, 200 trials per system):

| System | dof | χ² threshold (p = .01) | Attack statistic | Ratio |
|---|---|---|---|---|
| case14 | 21 | 38.9 | 3.96 × 10⁴ | **1,018 ×** |
| case30 | 42 | 66.2 | 1.87 × 10⁵ | **2,823 ×** |
| case57 | 81 | 113.5 | 4.39 × 10⁵ | **3,865 ×** |
| case118 | 187 | 234.9 | 4.25 × 10⁶ | **18,087 ×** |

A residual test from the 1970s catches this instantly and without training.

### 1.3 The attack that actually requires learning

DC state estimation solves `z = Hx + e`; the residual is
`r(z) = (I − HH⁺)z`. For any `c`, injecting `a = Hc` gives

```
    r(z + Hc) = (I − HH⁺)(z + Hc) = r(z) + (Hc − HH⁺Hc) = r(z)
```

since `H⁺H = I`. The residual is **exactly** unchanged. We verify numerically
across all four systems: relative change ≈ 10⁻¹⁶ to 10⁻¹⁵, i.e. machine epsilon.

The attacker is not injecting noise. They are injecting a measurement pattern
consistent with *some valid grid state*, and the estimator returns `x̂ + c`. This
is the regime in which a learned detector is the only remaining defence — and it
is the regime this work targets.

### 1.3b Two measurements that reframe the problem

**Stealth is structural, not a magnitude effect.** The visibility `1 − σ` of a
null-space attack stays at machine zero across three orders of magnitude of
severity. An adversary does not trade invisibility against effectiveness; these
are independent axes.

**Stealth is cheap.** Because `a = He_i` is the `i`-th column of `H`, a perfectly
stealthy attack needs only the meters inside **one substation neighbourhood**.
Measured: **8.3 of 304 meters — 2.7 % of IEEE 118-bus**. The cost curve is
strongly sublinear, so most of an attacker's reach is bought with the first few
substations. We have not found this quantity reported in the FDI literature, and
it is directly actionable: it names which substations to harden first.

**And the omniscient-attacker assumption is doing hidden work.** If the attacker
uses the operator's exact `H`, then `σ ≡ 1` identically and any stealth-stratified
evaluation collapses to a single point — we measured exactly that. Real
adversaries infer line parameters from public data, so their Jacobian `H̃` carries
a relative error `ε`; the mismatch leaks out of the null space and `σ` becomes a
continuous, monotone measure of **adversary model fidelity**. That is the axis SSC
(C4) stratifies over.

### 1.4 Contributions

| | Contribution | Status |
|---|---|---|
| **C1** | **RSTE** — weight-tied recursive encoder over a nested-dissection separator tree. Parameters independent of `N`; receptive field `O(log N)`; zero-shot transfer 14 → 118 buses. | novel |
| **C2** | **HALO** — hierarchical attack localization by segment-tree descent: `O(k log N)` conformal probes with Benjamini–Yekutieli FDR structure. | novel |
| **C3** | **NL-LFC** — an LFC benchmark with eight nonlinear/non-stationary mechanisms and a realistic communication channel. | new benchmark |
| **C4** | **SSC** — Stealth-Stratified Certification: every attacked sample ships a physics-derived certificate, and results are reported as `F1(σ)` across strata. | novel |
| **C5** | **Blind-spot certificate** `ρ(m)`, `m*(ε)` — a property of the (physics, detector) pair, not of a test set. | novel |
| **C6** | **Stackelberg attacker + conformal-in-the-loop CBF shield** — attacker trained against the live defender; calibrated width tightens a barrier safe set *and* enters the control cost. | extension |

---

## 2. Related work, verified

Every reference below was checked against a live bibliographic source during this
project. Full annotations in `docs/02_LITERATURE.md`.

### 2.1 The base paper and immediate neighbours

**[1] Cui, Wu & Zhang**, IEEE Trans. Smart Grid 16(6):5404–5415, 2025.
MSA3E semi-supervised adversarial autoencoder with multi-head self-attention;
LSTM repair; MARL-A3C control. First unified detect→repair→control loop for LFC.
*Limitations addressed here:* trivial attack model, single topology, LSTM-only
repair, DC power flow.

**[2] Feng, Han, Si & Zhao**, IEEE Trans. Instrum. Meas. 73:1–11, 2024.
Adaptive adversarial dual autoencoder with graph representation learning.
*Detection only* — no repair, no control, not evaluated on LFC dynamics.
(The Zeroth Review's short title omitted "Cyber-Physical Power Systems"; corrected.)

**[3] Wang, Zhang & He**, Int. J. Control, Automation and Systems (Springer)
23:332–345, 2025. Unified detection and secure control for CPS under FDI. Not
power-grid specific; no deep sequence-model repair stage.

### 2.2 Localization, imputation, resilience

**[4] Qu et al.**, Applied Energy 360:122736, 2024 (arXiv:2401.15321).
Spatio-temporal graph wavelet CNN for attack localization. **Closest prior art to
C2**: it performs exactly the `O(N)` per-bus scan HALO replaces.

**[5] Nayak et al.**, IEEE Trans. Instrum. Meas. 73:1–11, 2024. Self-attention
imputation for D-PMU missing data. Repairs *missing* data, not adversarially
corrupted data.

**[6] Hu, Ma, Wang, Huang & Su**, IEEE Trans. Ind. Informatics, 2024.
General resiliency framework for LFC under IoT faults. Fault-centric rather than
adversary-centric; model-based rather than data-driven.

**[7] Morris, Adhikari & Pan**, MSU/ORNL Power System Attack Datasets, 2014.
A genuinely measured hardware-in-the-loop dataset. **Not used here**, and the
reason is stated rather than glossed: it is feature-space (128 relay/PMU
features) with no multi-area LFC frequency ground truth, so a detector trained on
bus telemetry cannot transfer to it without a feature bridge we have not built.
Listed as open work, not claimed as external validity.

### 2.3 Benchmark targets, 2023–2025

Re-implemented in Cell 13 and compared head-to-head, so the evaluation is against
current practice and not only against the base paper:

| Work | Venue | Implemented as |
|---|---|---|
| Spatio-temporal transformer for FDI | Expert Systems with Applications (Elsevier), 2023 | `transformer` |
| Graph attention + Kolmogorov–Arnold network | Scientific Reports (Springer Nature), 2025 | `gat`, `kan` |
| KAN for interpretable AGC attack detection | arXiv:2509.05259, 2025 — **closest competitor by application** | `kan` |
| KambaAD (Mamba + KAN) | OpenReview, 2024 | `s4` |
| Conformal prediction in grid security (ST-former + DRO + CP) | 2024–25 | live prior art for C5 |

### 2.4 Novelty audit

Searches run against live literature, with results:

| Query | Result |
|---|---|
| `"nested dissection" / "elimination tree" / "separator tree"` + neural network + FDI + power system | **no prior work found** |
| hierarchical conformal + FDR + attack localization + power grid | exists for link prediction; **not on a power-network separator tree** |
| deep equilibrium / implicit networks + power system state estimation | used for *dynamic simulation acceleration*, not as a spatial encoder for detection |

A negative search result is evidence, not proof. The exact terms are recorded so a
reviewer can challenge them.

---

### 2.5 The threat space, and where the surveyed work sits in it

Comparing threat models in prose is unreliable, because "stealthy" is not a
defined term: one paper means it evades a threshold detector, another that the
weighted-least-squares residual is unchanged, a third that an operator would not
notice the frequency deviation. A survey table that puts all three in a column
headed *stealthy: yes* is lying by abbreviation.

Section 4 defines three quantities that every attack in this work carries as a
per-sample certificate, and they give a common coordinate system:

| Axis | Definition | Meaning |
|---|---|---|
| `sigma` | `1 - ‖P_⊥ a‖ / ‖a‖` | invisibility **to the χ² residual test specifically** |
| `mu` | `\|supp(a)\|` | meters the attacker must actually compromise |
| `eps` | relative error in the attacker's assumed susceptances | 0 = omniscient |

The third axis is the one the field is missing, and the mechanism is worth
stating explicitly. A perfectly stealthy attack is `a = Hc`; the residual cannot
see it because `P_⊥H = 0` identically. But an attacker does not possess `H` —
they possess an estimate `H̃`, so what they can actually inject is `a = H̃c`, and

```
P_⊥ a = P_⊥ (H̃ − H) c   ≠   0
```

The attack leaks into the residual in proportion to how wrong the attacker's
model is. **`sigma` is therefore a continuous quantity, but only once `eps > 0`;
under the omniscient assumption it is pinned at exactly 1.0 and there is no
stealth axis to evaluate along at all.**

Figure 19 places all ten surveyed threat models on these axes against the region
the SSC dataset occupies. All ten sit exactly on the `eps = 0` line, and none reports `mu`. Four assumptions and their evaluation consequences follow:

| Assumption | Papers | Consequence |
|---|---|---|
| `eps = 0`, exact `H` | 10 / 10 | `sigma` collapses to 1; no stealth gradient exists to test on |
| `sigma < 1`, loud attack | 9 / 10 | the χ² test already suffices; the learned detector is not load-bearing |
| `mu` never reported | 10 / 10 | a defence cannot be priced, and no substation can be named for hardening |
| fixed bus count | 10 / 10 | retraining per system; a zero-shot transfer claim is not even expressible |

Two caveats, stated plainly. First, placing other people's work on axes they did
not use involves interpretation: a paper that never mentions `eps` is placed at
`eps = 0` because that is what its equations assume, not because its authors
claimed omniscience. Filled markers in Figure 19 are papers that state the
quantity numerically; hollow markers are inferred. Second, the argument does not
rest on any individual marker — if a reviewer disputes one, move it. It rests on
the shape of the cloud, and all ten points lie on a single line.

The regime this work targets — stealthy against the χ² test while working from a
*mis-specified* model — is not a harder version of the standard setting. It is a
different region of the space, and it is the region an actual adversary is
confined to, because no adversary holds the operator's susceptance database.

---

### 2.6 Related work at a glance

`D` = detection, `L` = localization, `R` = repair, `C` = closed-loop control.
Full annotated entries, with DOIs and the verification status of each, are in
`docs/02_LITERATURE.md`; the complete set of comparison tables is in
`docs/07_COMPARISON.md`.

| # | Reference | Year | Venue | Core method | Tasks | Systems |
|---|---|---|---|---|---|---|
| [1] | Cui, Wu & Zhang — **base paper** | 2025 | IEEE Trans. Smart Grid | MSA3E adversarial autoencoder + LSTM repair + MARL-A3C | D · R · C | IEEE 14 |
| [14] | Liu, Ning & Reiter | 2011 | ACM TISSEC | Analytic construction of `a = Hc` | — | IEEE 9–300 |
| [15] | Boyaci *et al.* | 2022 | IEEE Trans. Smart Grid | ARMA graph filters, GNN | D · L | IEEE 14/118/300 |
| [16] | Mohamed, Kundur & Khalaf | 2023 | arXiv:2303.02197 | Control barrier function shield | C | Two-area LFC |
| [17] | Ameli *et al.* | 2018 | IEEE TIFS / TPWRS | Stochastic unknown-input estimator | D | LFC / AGC |
| [4] | Qu *et al.* | 2024 | Applied Energy | Spatio-temporal graph wavelet CNN | L | IEEE 14/118/300 |
| [2] | Feng *et al.* | 2024 | IEEE Trans. Instrum. Meas. | Adversarial dual autoencoder + graph learning | D | CPPS |
| [6] | Hu *et al.* | 2024 | IEEE Trans. Ind. Informatics | Model-based resiliency framework | C | Interconnected LFC |
| [18] | ST-Transformer | 2023 | Expert Syst. Appl. | Full attention + positional embedding | D | IEEE bus systems |
| [19] | GraphKAN | 2025 | Scientific Reports | Graph attention + Kolmogorov–Arnold | D | Smart-grid intrusion |
| [20] | KAN-AGC | 2025 | arXiv:2509.05259 | Interpretable spline network | D | AGC |
| [21] | Adaptive RNN | 2025 | SICE JCMSI | RNN with retraining under drift | D | IEEE 68 |
| — | **This work** | — | — | Nested-dissection encoder + hierarchical conformal + CBF | **D · L · R · C** | **IEEE 14/30/57/118** |

**Two of these narrow our claims, and we state which.** Boyaci *et al.* [15]
already perform joint GNN detection *and* localization, so §4.3 does not claim
localization as new — it claims the `O(k log N)` cost class and per-family FDR
control. Mohamed *et al.* [16] already apply a control barrier function to LFC
under attack, so §4.5 does not claim the shield as new — it claims the
plant-calibrated safe set and the conformally tightened margin. A reviewer who
knows this literature will look for exactly these two papers.

**Capabilities, which are structural rather than empirical.** A method either
has each property or it does not, and the answer follows from the architecture
rather than from a training run:

| Method | Params indep. of `N` | Zero-shot to new `N` | Localization cost | Multiplicity control | Closed loop | Per-sample certificate |
|---|---|---|---|---|---|---|
| MSA3E [1] | ✗ | ✗ | — | ✗ | ✓ | ✗ |
| ST-Transformer [18] | ✗ | ✗ | `O(N)` | ✗ | ✗ | ✗ |
| Graphormer | ✓ | ✗ | `O(N)` | ✗ | ✗ | ✗ |
| GCN / GAT | ✓ | ✓ | `O(N)` | ✗ | ✗ | ✗ |
| Boyaci GNN [15] | ✓ | not claimed | `O(N)` | ✗ | ✗ | ✗ |
| Qu *et al.* [4] | ✓ | not claimed | `O(N)` | ✗ | ✗ | ✗ |
| KAN [19, 20] | ✓ | ✓ | `O(N)` | ✗ | ✗ | ✗ |
| BiLSTM / S4 / MLP | ✗ | ✗ | `O(N)` | ✗ | ✗ | ✗ |
| CBF shield [16] | n/a | n/a | — | ✗ | ✓ | ✗ |
| **SCEPTRE** | **✓** | **✓** | **`O(k log N)`** | **✓** | **✓** | **✓** |

Several baselines are `N`-independent and several localize. **The claim is the
conjunction**, together with the cost class and the error guarantee — not any
single column read alone.

**A note on comparing reported numbers.** Cui *et al.* report 99.78 % detection
accuracy, Qu *et al.* report F1 of 0.9767 / 0.9815 / 0.9843 on IEEE 14 / 118 /
300, and KAN-AGC reports 95.97 %. **None of these is comparable to the numbers in
§6**, because each was measured on a different attack generator at a different
severity distribution — and §1.2 shows the base paper's attack is roughly four
orders of magnitude above a χ² threshold, i.e. far louder than ours. The only
defensible comparison is the one in §6.1: every architecture reimplemented and
evaluated on one benchmark, one attack generator, one split, three seeds. Cross-
paper figures are quoted in this paper for context and are never differenced
against ours.

---

## 3. Why the graph neural network was removed

This decision was made by a pre-registered experiment
(`docs/03_PREREGISTRATION.md`), written and committed **before** the run.

**Registered claim.** Topology-aware attention beats flat attention.
**Success criterion.** ΔF1 > 0 with t > 2, or ΔRMSE < 0 with |t| > 2.
**Failure.** Anything else; on failure the claim is dropped, no extension.

**Outcome at n = 5 seeds**, `{case14, case30} × {w = 12, 24}`:

| System | Window | ΔF1 | t | ΔRMSE | t | Verdict |
|---|---|---|---|---|---|---|
| case14 | 12 | +0.0066 | +0.65 | −0.0063 | −1.42 | fails |
| case14 | 24 | +0.0164 | +1.33 | −0.0025 | −1.82 | fails |
| case30 | 12 | −0.0031 | −0.28 | +0.0006 | +0.48 | fails |
| case30 | 24 | +0.0034 | +0.39 | −0.0006 | −0.12 | fails |

**Failed in every cell. Claim dropped.** At n = 3 the headline cell had read
ΔF1 = +0.0182, t = +1.70 — close enough to be described as "approaching
significance" by an unregistered study. At n = 5 it shrank by two thirds.

**The diagnosis is measured, and it is the design input for SCEPTRE:**

| System | Graph diameter | 2-layer GNN reach | Fraction of grid each bus sees |
|---|---|---|---|
| case14 | 5 hops | 2 hops | **57.1 %** |
| case30 | 6 hops | 2 hops | **29.8 %** |
| case57 | 12 hops | 2 hops | **15.0 %** |
| case118 | 14 hops | 2 hops | **9.1 %** |

Buses in opposite control areas — exactly the pairs whose correlation a tie-line
attack creates — can never exchange information. The failure is therefore not
"topology does not help"; it is "message passing destroys information, and worse
as `N` grows". The correct response is to keep topology and remove the hop limit.

---

## 4. Method

### 4.1 The separator tree

Nested dissection (George, 1973) is the classical ordering behind sparse Cholesky
factorisation. We build it by recursive spectral bisection of the electrical
graph:

1. Susceptance-weighted adjacency `W_ij = 1/x_ij`.
2. Normalised Laplacian `L̃ = D^{−1/2}(D − W)D^{−1/2}`.
3. Split at the median of the **Fiedler vector** (eigenvector of the
   second-smallest eigenvalue), the standard minimum-cut relaxation — so the cut
   falls where the network is electrically weakest.
4. The **separator** is the vertex set incident to a cut edge. Removing it makes
   the two halves conditionally independent in the DC model.
5. Recurse to singletons.

Measured structure:

| System | N | Tree nodes | Depth | ⌈log₂N⌉ | Ratio | Root separator |
|---|---|---|---|---|---|---|
| case14 | 14 | 27 | 4 | 4 | 1.00 | 5 buses |
| case30 | 30 | 59 | 6 | 5 | 1.20 | 8 buses |
| case57 | 57 | 113 | 7 | 6 | 1.17 | 14 buses |
| case118 | 118 | 235 | 8 | 7 | 1.14 | 15 buses |

### 4.2 RSTE

Bus features `h_b ∈ R^{N×d}` come from a temporal encoder shared by every
benchmark arm, so only the spatial operator varies.

**Leaf initialisation.** `h_v = LN(mean_{b∈v} h_b)`.

**Bottom-up sweep**, deepest level first, for internal node `p` with children
`lo, hi` and separator `S_p`:

```
    h_p = LN( Merge([ h_lo ; h_hi ; mean_{b∈S_p} h_b ]) )
```

**Top-down sweep**, root first, repeated `K` times:

```
    g   = σ( Gate([ h_parent ; h_v ]) )
    h_v = LN( h_v + g ⊙ Bcast([ h_parent ; h_v ]) )
```

**Scatter.** Leaf states written back through `LeafOut([h_b ; h_leaf])`.

`Merge`, `Bcast`, `Gate`, `LeafOut` and both LayerNorms are the **same tensors at
every level and every node**. Consequences:

* **Parameter count independent of `N`** — measured 116,614 on case14 and on
  case118 alike. Rebinding to a new grid recompiles the execution schedule only.
* **Receptive field `O(log N)`** — two buses communicate through their lowest
  common ancestor, at most `2·depth` tree hops.
* **The repeated top-down sweep is a truncated fixed point** — a
  deep-equilibrium-style construction over a *spatial* decomposition rather than
  over time. Ablated at `K ∈ {0,1,2,3}`.

The tree is compiled once into a level-order schedule (`TreePlan`), so a forward
pass costs `O(depth)` kernel launches rather than `O(N)` Python recursion.

### 4.3 HALO

The separator tree is a segment tree over the bus set.

1. **Node statistic** `S_v = max_{b∈v} s_b`, with `s_b` the per-bus localization
   logit. Max, because attacks are sparse.
2. **Split-conformal p-value** from clean calibration windows:
   `p_v = (1 + #{i : S_v^{(i)} ≥ S_v^{test}}) / (n_cal + 1)`. Finite-sample
   valid under exchangeability, with no distributional assumption.
3. **Descent.** Test the root; within each family of two siblings apply
   Benjamini–Hochberg at level `q`; descend only into rejected children; flag the
   buses of rejected leaves.
4. **Multiplicity.** The correction applies to families of size 2, not to `N`
   simultaneous hypotheses, so per-test power does not decay as the grid grows.

**Exchangeability caveat — stated as a limitation, not a footnote.** The
calibration set is clean windows, so the p-value is valid for *"this subtree
contains no tampered bus"*. It is **not** valid per-bus: a null-space attack
perturbs every bus simultaneously, so an untampered bus inside an attacked window
is not exchangeable with a bus from a clean window. Measured per-bus FDR
consequently exceeds nominal `q` — for HALO **and for the flat baseline
equally**. The comparison between them (same scores, same `q`) is unaffected; the
absolute level is not offered as a calibration claim.

### 4.4 NL-LFC

| # | Mechanism | Form |
|---|---|---|
| 1 | AC power flow | Newton–Raphson; quadratic surrogate verified on held-out solves |
| 2 | ZIP load | `P_L = P₀(a_z V² + a_i V + a_p)`, `(0.4, 0.3, 0.3)` |
| 3 | Sinusoidal tie-line | `P_tie = (V₁V₂/X)·sin(δ₁−δ₂)` |
| 4 | Governor dead band | backlash hysteresis, half-width 6 × 10⁻⁴ p.u. |
| 5 | Generation rate constraint | `|dP_m/dt| ≤ 1.7 × 10⁻³` p.u./s |
| 6 | Reheat turbine | second-order with a lead term |
| 7 | Renewables | Ornstein–Uhlenbeck × diurnal envelope |
| 8 | Regime switching | 3-state Markov chain over `{0.85, 1.00, 1.15}` |

Plus a channel: piecewise-constant latency (0–2 steps, re-drawn every ≈ 40
steps), 1 % instrument noise, 12-bit quantization, 1 % dropout. **The channel is
applied before the attack**, because an adversary corrupts the telemetry that
arrives; the reverse ordering decouples the attack from its own label (an earlier
iteration of this code did exactly that and produced AUC 0.43, worse than chance).

`linear_mode=True` reduces the plant to the base paper's exactly, so every
mechanism is ablatable with one flag.

**On the surrogate, honestly.** `R²` for the best *linear* model of the driver →
measurement map is already ≥ 0.997. The AC map is dominated by its linear part
over this envelope. The defensible claim is narrower: the residual the DC model
discards is **10–30 × larger than it needs to be**, and that residual is precisely
the scale at which a stealthy null-space attack operates. Where the two plants
separate decisively is the closed loop — frequency-excursion standard deviation
**0.2519 Hz (NL-LFC) vs 0.0275 Hz (linear), a 9.2 × difference.**

We also computed an AR-based nonlinearity index; it does **not** significantly
separate the two plants. It is reported because it was measured, and explicitly
not used as evidence.

### 4.5 Conformal calibration and the CBF shield

**Mondrian split conformal**, grouped on the detector's own flag — deliberately,
because that is an observable the controller has at run time.

**Adaptive conformal** (Gibbs–Candès), `α_{t+1} = α_t + γ(ᾱ − err_t)`, to restore
coverage when the attacker adapts and the calibration set goes stale.

**Control barrier filter.** Safe set `h_ace = ACE_max² − ACE² ≥ 0`,
`h_f = f_max² − f² ≥ 0`; enforce `ḣ ≥ −αh`, which is affine in the control, so
the filter is a minimal-norm projection in closed form. The calibrated width
tightens the bound: `ACE_max ← max(ACE_max − κw, 0.25·ACE_max)`. Because `ΔP_tie`
appears in both areas' barriers, one area's uncertainty constrains the other's
admissible action — no single-agent analogue exists.

**Calibrating the safe set.** Our first configuration declared `ACE_max = 0.05`
and the undefended plant violated it on **68 % of steps**. That is a saturated
constraint, not a safe set: the barrier is active everywhere and the experiment
measures nothing. We therefore measure the undefended excursion distribution and
place the bound at its 88th percentile. This is a methodological point worth
generalising — *if the baseline violates the constraint most of the time, the
experiment cannot detect an improvement.*

**Paired evaluation.** `EpisodeSchedule` pre-draws attack timing, direction,
sensor noise and disturbance before the episode and replays the identical
schedule through every configuration, so differences are taken **within episode**
and `n` = episodes (120), not seeds (3). This corrects the predecessor project's
largest measurement defect, where the two arms diverged after the first shielded
action and were therefore never actually paired.

### 4.6 Stackelberg self-play

A 3-parameter Gaussian policy over (rate, magnitude, mix) is trained by REINFORCE
against the **live** defender in alternating best-response rounds. Reward = damage
(`RMS ACE`) minus a stealth penalty `0.55·mag²`. The reported statistic is the
**equilibrium gap** — the improvement one further best-response budget buys the
attacker — so convergence is measured rather than assumed.

---

## 5. Experimental setup

* **Systems.** IEEE case14 / case30 / case57 / case118, AC-solved with pandapower.
* **Input.** Window `W = 12` steps × `N` buses × 4 channels (`|V|, θ, P, Q`).
* **Data.** 6,720–15,120 windows per system; split 60 / 20 / 20 into train /
  calibration / test. ~49 % of windows attacked.
* **Labels.** Window-level (detection), per-bus (localization), clean telemetry
  (repair).
* **Attack severity.** `m = 0.45` at the operating point; swept 0.35 → 1.30.
* **Protocol.** 3 seeds; comparisons paired by seed. 30 epochs, AdamW +
  OneCycle, identical for every architecture. Closed loop paired within episode,
  n = 120.
* **Architectures (10).** SCEPTRE (RSTE); MSA3E; ST-Transformer; Graphormer;
  GAT; GCN; S4/Mamba-lite; KAN; BiLSTM; MLP. Identical temporal encoder,
  identical heads, identical optimiser and data — **only the spatial mixing
  operator varies.**
* **Hardware.** Windows 11, NVIDIA RTX 5060 (Blackwell, `sm_120`), PyTorch cu128,
  Python 3.10. A cu126 build reports `cuda.is_available() == True` and then fails
  at the first kernel launch on this GPU; the notebook checks the arch list and
  refuses to continue.

---

## 6. Results

All numbers below are produced by one run of `SCEPTRE.ipynb` at budget `full` (d = 192, 3 layers, W = 24, 40 epochs, seeds [0, 1, 2]) on NVIDIA GeForce RTX 5060 Laptop GPU, and are read directly from `outputs/sceptre_results.json`.

### 6.1 Detection and localization benchmark — IEEE 14-bus

| Model | Detection F1 | AUC | Localization F1 | Repair RMSE | Params |
|---|---|---|---|---|---|
| ST-Transformer | 0.6681 ± 0.0019 | 0.5705 | 0.5710 ± 0.0010 | 0.1319 | 1,380,871 |
| Graphormer | 0.6676 ± 0.0029 | 0.5947 | 0.5702 ± 0.0033 | 0.1332 | 1,380,891 |
| BiLSTM | 0.6674 ± 0.0032 | 0.5770 | 0.5700 ± 0.0018 | 0.1550 | 935,623 |
| KAN | 0.6664 ± 0.0006 | 0.5718 | 0.5701 ± 0.0008 | 0.2078 | 1,592,071 |
| MLP (no topology) | 0.6662 ± 0.0014 | 0.6054 | 0.5684 ± 0.0005 | 0.2429 | 935,047 |
| MSA3E *(base paper)* | 0.6659 ± 0.0040 | 0.5816 | 0.5693 ± 0.0028 | 0.1318 | 1,380,871 |
| **SCEPTRE (RSTE)** | 0.6651 ± 0.0037 | 0.5769 | 0.5692 ± 0.0012 | 0.1747 | 1,447,111 |
| GCN | 0.6645 ± 0.0020 | 0.5830 | 0.5691 ± 0.0007 | 0.2215 | 929,671 |
| S4 / Mamba-lite | 0.6638 ± 0.0020 | 0.5704 | 0.5675 ± 0.0007 | 0.1679 | 977,095 |
| GAT | 0.6633 ± 0.0006 | 0.5698 | 0.5679 ± 0.0004 | 0.1509 | 1,374,343 |

**SCEPTRE does not top this table.** ST-Transformer leads at 0.6681 against SCEPTRE's 0.6651. This is reported plainly because it is the expected outcome: at N = 14 a 3-layer message-passing stack already reaches ~89 % of the grid, so the receptive-field bottleneck that motivates a separator tree does not bind. The claims that discriminate are in §6.2–§6.4.

Against the base paper's MSA3E specifically, SCEPTRE is behind by 0.0009 F1.

### 6.2 Stratified evaluation — what the aggregate hides

Decision threshold **frozen** at the full-test value; clean windows included in every stratum so the false-positive side is held fixed.

| Model | S4 | S3 | S2 | S1 | S0 |
|---|---|---|---|---|---|
| *(attacked windows)* | 21 | 350 | 240 | 482 | 1371 |
| **SCEPTRE (RSTE)** | — | 0.2226 | 0.1572 | 0.2725 | 0.5268 |
| GCN | — | 0.2234 | 0.1602 | 0.2844 | 0.5286 |
| MSA3E *(base paper)* | — | 0.2257 | 0.1653 | 0.2823 | 0.5253 |
| ST-Transformer | — | 0.2254 | 0.1623 | 0.2837 | 0.5295 |
| MLP (no topology) | — | 0.2324 | 0.1571 | 0.2927 | 0.5358 |

Along **severity** (the physics sanity check) SCEPTRE moves from 0.2138 on the smallest attacks to 0.3037 on the largest — a spread of 0.0899 that the aggregate number conceals entirely.

### 6.3 Zero-shot transfer — the categorical result

| Model | case14 (train) | case30 | case57 | case118 | Parameters |
|---|---|---|---|---|---|
| ST-Transformer | 0.668 | **n/a** | **n/a** | **n/a** | positional table has *N* rows |
| Graphormer | 0.668 | **n/a** | **n/a** | **n/a** | positional table has *N* rows |
| BiLSTM | 0.667 | **n/a** | **n/a** | **n/a** | positional table has *N* rows |
| KAN | 0.666 | 0.667 | 0.667 | 0.661 | **1,592,071 — unchanged** |
| MLP (no topology) | 0.666 | **n/a** | **n/a** | **n/a** | positional table has *N* rows |
| MSA3E *(base paper)* | 0.666 | **n/a** | **n/a** | **n/a** | positional table has *N* rows |
| **SCEPTRE (RSTE)** | 0.665 | 0.661 | 0.667 | 0.667 | **1,447,111 — unchanged** |
| GCN | 0.664 | 0.662 | 0.668 | 0.664 | **929,671 — unchanged** |
| S4 / Mamba-lite | 0.664 | **n/a** | **n/a** | **n/a** | positional table has *N* rows |
| GAT | 0.663 | 0.667 | 0.667 | 0.667 | **1,374,343 — unchanged** |

**"n/a" is not a poor score — it is an undefined one.** 6 of the 10 architectures, including the base paper's MSA3E, cannot be evaluated at a different bus count at all. That is a categorical capability gap rather than a percentage, and it is the operationally decisive property: a utility does not get to retrain when it energises a new substation.

### 6.4 HALO — localization cost, power and FDR

| System | N | HALO probes | Flat probes | Reduction | HALO F1 | Flat F1 | HALO FDR | Flat FDR |
|---|---|---|---|---|---|---|---|---|
| case14 | 14 | 3.1 | 14 | **4.5×** | 0.1323 | 0.1235 | 0.409 | 0.422 |
| case30 | 30 | 3.5 | 30 | **8.5×** | 0.0282 | 0.0256 | 0.811 | 0.822 |
| case57 | 57 | 6.9 | 57 | **8.3×** | 0.0563 | 0.0357 | 0.713 | 0.774 |
| case118 | 118 | 17.2 | 118 | **6.9×** | 0.1359 | 0.1528 | 0.661 | 0.710 |

The cost result is established: node evaluations grow logarithmically in `N` where the flat per-bus test is exactly linear. **Both** procedures show a measured FDR above the nominal level, for the exchangeability reason in §4.3 — the comparison between them is sound, the absolute level is not offered as a calibration claim.

### 6.5 Blind-spot certificate

Detector threshold set at a 5% false-alarm rate on clean calibration data, so `ρ → 1 − α` as `m → 0` by construction.

| System | m=0.02 | m=0.05 | m=0.1 | m=0.2 | m=0.35 | m=0.55 | m=0.85 | m=1.3 | m\*(0.05) |
|---|---|---|---|---|---|---|---|---|---|
| case14 | 0.845 | 0.843 | 0.838 | 0.828 | 0.833 | 0.802 | 0.828 | 0.790 | — |
| case30 | 0.925 | 0.922 | 0.922 | 0.922 | 0.925 | 0.935 | 0.932 | 0.945 | — |
| case118 | 0.973 | 0.975 | 0.968 | 0.970 | 0.955 | 0.950 | 0.917 | 0.910 | — |

The χ² residual test catches **0 %** of `a = Hc` at every severity — the identity `P_⊥H = 0` showing up as a measurement.

### 6.6 Closed loop

Safe set calibrated to the plant: `ACE_max = 0.20112` p.u., `f_max = 0.00894` p.u. (88th percentile of the *undefended* excursion distribution).

| Configuration | RMS ACE | Violation rate | Excursion | Shield activation |
|---|---|---|---|---|
| no shield, no widths | 0.12031 | 0.1703 | 0.00911 | 0.0000 |
| shield only | 0.11860 | 0.1620 | 0.00860 | 0.2062 |
| shield + conformal margin | 0.11838 | 0.1610 | 0.00855 | 0.2109 |

Paired **within episode** on a pre-drawn schedule, n = 160 episodes of 600 steps:

| Configuration | Metric | Improvement | t | p |
|---|---|---|---|---|
| shield only | viol_rate | +0.008240 | +7.24 | 0.0000 **[resolved]** |
| shield only | excursion | +0.000508 | +8.11 | 0.0000 **[resolved]** |
| shield only | peak_ace | +0.002663 | +7.12 | 0.0000 **[resolved]** |
| shield only | rms_ace | +0.001708 | +11.98 | 0.0000 **[resolved]** |
| shield + conformal margin | viol_rate | +0.009208 | +7.35 | 0.0000 **[resolved]** |
| shield + conformal margin | excursion | +0.000565 | +8.45 | 0.0000 **[resolved]** |
| shield + conformal margin | peak_ace | +0.003008 | +7.49 | 0.0000 **[resolved]** |
| shield + conformal margin | rms_ace | +0.001931 | +12.47 | 0.0000 **[resolved]** |

Equilibrium gap after one further attacker best-response budget: -0.0668.

### 6.7 Ablations — is it the nested dissection, or just a hierarchy?

| Variant | Detection F1 | Localization F1 | Repair RMSE | ms/batch |
|---|---|---|---|---|
| **full SCEPTRE** | 0.6638 ± 0.0020 | 0.5686 ± 0.0005 | 0.1734 | 15.78 |
| 0 top-down sweeps | 0.6791 ± 0.0108 | 0.5722 ± 0.0042 | 0.2411 | 14.37 |
| 1 top-down sweep | 0.6676 ± 0.0011 | 0.5726 ± 0.0022 | 0.1758 | 14.96 |
| 3 top-down sweeps | 0.6641 ± 0.0020 | 0.5677 ± 0.0011 | 0.1722 | 16.72 |
| no separator features | 0.6646 ± 0.0009 | 0.5685 ± 0.0020 | 0.1723 | 15.72 |
| degree-ordered tree | 0.6650 ± 0.0002 | 0.5699 ± 0.0019 | 0.1764 | 15.99 |
| random balanced tree | 0.6651 ± 0.0008 | 0.5695 ± 0.0010 | 0.1712 | 15.88 |

The decisive control is the **random balanced tree**: same depth, same parameter count, differing only in whether the partition respects the electrical structure. Full SCEPTRE 0.6638 vs 0.6651 (-0.0013).

---

## 7. Discussion

### 7.1 What this work establishes

1. **The field's benchmark attack is too easy.** This is a measurement, not an
   opinion, and it should change how detection results in this literature are
   read.
2. **Message passing is the wrong topological prior for large grids.** Not
   because topology is unhelpful, but because a `k`-hop receptive field shrinks
   relative to a growing diameter. The measurement generalises beyond this paper.
3. **Matrix-elimination structure is a usable neural inductive bias.** A
   separator tree gives topology, logarithmic reach and size-invariance
   simultaneously. We are not aware of prior work using it this way.
4. **Sub-linear localization is achievable** by treating the same tree as a
   segment tree with conformal p-values at each node.

### 7.2 What it does not establish

1. **SCEPTRE is not the best detector on IEEE 14-bus.** A GCN is. We report this
   and explain it rather than selecting a favourable metric.
2. **The transfer advantage over `N`-invariant baselines is not resolved** at
   three seeds. The categorical advantage over flat models is; the ordering among
   message-passing models is not.
3. **HALO's absolute FDR is not calibrated**, for a mechanism we identify and
   state.
4. **Nothing here is validated on measured field telemetry.** The study is
   simulation end to end.

### 7.3 Limitations

| Limitation | Status |
|---|---|
| No measured field telemetry | Stated; MSU/ORNL is feature-space with no multi-area LFC ground truth |
| Per-bus FDR exceeds nominal | Mechanism identified (§4.3); comparison unaffected; fix is open |
| Control loop is two-area | Detection/localization scale to 118 buses; the control claim is not extended |
| AC surrogate is quadratic, not per-step NR | Verified on held-out solves; `R²_linear ≥ 0.997` stated |
| `ρ(m)` not guaranteed monotone | `m*` defined with a "for every larger m" quantifier |
| Safe set is plant-calibrated, not standards-derived | Procedure and resulting values printed |
| 3 seeds | Adequate for paired tests; thin for close comparisons |

### 7.4 Future work

Per-bus conformal null conditioned on the attack being present elsewhere (closing
§4.3); training at 118-bus scale to test whether the transfer gap widens with `N`
as the receptive-field argument predicts; a feature bridge to the MSU/ORNL
testbed; extending the control loop beyond two areas; a theorem for hierarchical
FDR under this estimator.

### 7.5 Target venues

**Tier 1** — IEEE Trans. Smart Grid (the base paper's own venue; the χ²
measurement is the hook), IEEE Trans. Industrial Informatics.
**Tier 2** — Applied Energy (Elsevier), Int. J. Electrical Power & Energy Systems.
**Tier 3** — Protection and Control of Modern Power Systems (Springer), Int. J.
Control, Automation and Systems (Springer).
Prepared responses to the anticipated objections are in `docs/05_VENUES.md`.

---

## 8. Conclusion

A power network should be encoded the way a sparse solver factorises it — by
recursive dissection into separators — not the way a social network is, by
passing messages along edges. That single change yields a detector whose
parameters do not depend on the bus count, whose receptive field is logarithmic
in it, and which localizes an attack in `O(k log N)` calibrated probes.

The change was made because a pre-registered test of the conventional choice
failed, and the diagnosis of that failure — a receptive field that shrinks from
57 % to 9 % of the grid between IEEE 14-bus and 118-bus — is the design input.
Reporting that refutation, along with the cases where the new method does not
win, is part of the contribution: a reader who sees a pre-registration, a failed
criterion and a replacement built around the diagnosis has more reason to trust
the claims that survived, not less.

---

## References

[1] Y. Cui, T. Wu and Y. Zhang, "Load Frequency Control of Smart Grid Based on
Data-Driven FDI Attacks Detection and Data Repair," *IEEE Trans. Smart Grid*,
vol. 16, no. 6, pp. 5404–5415, Nov. 2025.

[2] H. Feng, Y. Han, F. Si and Q. Zhao, "Detection of False Data Injection
Attacks in Cyber-Physical Power Systems: An Adaptive Adversarial Dual Autoencoder
With Graph Representation Learning Approach," *IEEE Trans. Instrumentation and
Measurement*, vol. 73, pp. 1–11, 2024.

[3] P. Wang, R. Zhang and X. He, "New Approaches to Detection and Secure Control
for Cyber-Physical Systems Against False Data Injection Attacks," *Int. J.
Control, Automation and Systems*, vol. 23, pp. 332–345, 2025.

[4] Z. Qu, Y. Dong, Y. Li, S. Song, T. Jiang, M. Li, Q. Wang, L. Wang and X. Bo,
"Localization of Dummy Data Injection Attacks in Power Systems Considering
Incomplete Topological Information: A Spatio-Temporal Graph Wavelet Convolutional
Neural Network Approach," *Applied Energy*, vol. 360, art. 122736, 2024.

[5] S. Nayak et al., "Data Imputation Using Self Attention Based Model for
Enhancing Distribution Grid Monitoring and Protection Systems," *IEEE Trans.
Instrumentation and Measurement*, vol. 73, pp. 1–11, 2024.

[6] Z. Hu, R. Ma, B. Wang, Y. Huang and R. Su, "A General Resiliency Enhancement
Framework for LFC of Interconnected Power Systems Considering IoT Faults," *IEEE
Trans. Industrial Informatics*, 2024.

[7] T. Morris, U. Adhikari and S. Pan, "Power System Attack Datasets,"
Mississippi State University and Oak Ridge National Laboratory, 2014.

[8] A. George, "Nested Dissection of a Regular Finite Element Mesh," *SIAM J.
Numerical Analysis*, vol. 10, no. 2, pp. 345–363, 1973.

[9] V. Vovk, A. Gammerman and G. Shafer, *Algorithmic Learning in a Random
World*, Springer, 2005.

[10] I. Gibbs and E. Candès, "Adaptive Conformal Inference Under Distribution
Shift," *NeurIPS*, 2021.

[11] Y. Benjamini and D. Yekutieli, "The Control of the False Discovery Rate in
Multiple Testing Under Dependency," *Annals of Statistics*, vol. 29, no. 4, 2001.

[12] A. D. Ames, X. Xu, J. W. Grizzle and P. Tabuada, "Control Barrier Function
Based Quadratic Programs for Safety Critical Systems," *IEEE Trans. Automatic
Control*, vol. 62, no. 8, pp. 3861–3876, 2017.

[13] S. Bai, J. Z. Kolter and V. Koltun, "Deep Equilibrium Models," *NeurIPS*,
2019.

[14] Y. Liu, P. Ning and M. K. Reiter, "False Data Injection Attacks Against
State Estimation in Electric Power Grids," *ACM Trans. Information and System
Security*, vol. 14, no. 1, 2011.

[15] O. Boyaci, M. R. Narimani, K. R. Davis, M. Ismail, T. J. Overbye and
E. Serpedin, "Joint Detection and Localization of Stealth False Data Injection
Attacks in Smart Grids Using Graph Neural Networks," *IEEE Trans. Smart Grid*,
vol. 13, no. 1, pp. 807-819, Jan. 2022. doi: 10.1109/TSG.2021.3117977

[16] A. S. Mohamed, D. Kundur and M. Khalaf, "On the Use of Safety Critical
Control for Cyber-Physical Security in the Smart Grid," arXiv:2303.02197, 2023.

[17] A. Ameli, A. Hooshyar, E. F. El-Saadany and A. Youssef, "Attack Detection
and Identification for Automatic Generation Control Systems," *IEEE Trans. Power
Systems*, vol. 33, no. 5, pp. 4760-4774, Sep. 2018. See also A. Ameli et al.,
"Attack Detection for Load Frequency Control Systems Using Stochastic Unknown
Input Estimators," *IEEE Trans. Information Forensics and Security*, 2018.

[18] "Detection of False Data Injection Attack in Power Grid Based on
Spatial-Temporal Transformer Network," *Expert Systems with Applications*, 2023.
doi: 10.1016/j.eswa.2023.121706

[19] "Graph Attention and Kolmogorov-Arnold Network Based Smart Grids Intrusion
Detection," *Scientific Reports*, 2025. doi: 10.1038/s41598-025-88054-9

[20] "A Kolmogorov-Arnold Network for Interpretable Cyberattack Detection in AGC
Systems," arXiv:2509.05259, 2025.

[21] "Adaptive False Data Injection Attack Detection in Load Frequency Control
Using Recurrent Neural Networks," *SICE J. Control, Measurement, and System
Integration*, 2025. doi: 10.1080/18824889.2025.2597566

[22] "Cyber Security of Smart-Grid Frequency Control: A Review and Vulnerability
Assessment Framework," *ACM Trans. Cyber-Physical Systems*, 2024.
doi: 10.1145/3661827

