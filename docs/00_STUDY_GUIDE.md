# SCEPTRE — Study Guide

**Read this before the notebook, the deck, or the manuscript.**

This document assumes you know basic linear algebra and basic neural networks. It
assumes **nothing** about power systems, conformal prediction, control barrier
functions, or nested dissection. Everything else is built up from scratch.

---

## Contents

1. [The 60-second version](#1-the-60-second-version)
2. [Power-system background you actually need](#2-power-system-background-you-actually-need)
3. [What Load Frequency Control is](#3-what-load-frequency-control-is)
4. [False Data Injection, and why the base paper's attack is too easy](#4-false-data-injection-and-why-the-base-papers-attack-is-too-easy)
5. [The base paper, in detail](#5-the-base-paper-in-detail)
6. [Why we deleted the graph neural network](#6-why-we-deleted-the-graph-neural-network)
7. [Nested dissection and the separator tree](#7-nested-dissection-and-the-separator-tree)
8. [RSTE — the encoder (C1)](#8-rste--the-encoder-c1)
9. [HALO — logarithmic attack localization (C2)](#9-halo--logarithmic-attack-localization-c2)
10. [NL-LFC — the nonlinear benchmark (C3)](#10-nl-lfc--the-nonlinear-benchmark-c3)
10b. [SSC — why one F1 number is almost meaningless (C4)](#10b-ssc--why-one-f1-number-is-almost-meaningless-c4)
10c. [The threat space — how to compare threat models honestly](#10c-the-threat-space--how-to-compare-threat-models-honestly)
11. [Conformal prediction, from zero](#11-conformal-prediction-from-zero)
12. [The blind-spot certificate (C4)](#12-the-blind-spot-certificate-c4)
13. [Control barrier functions and the shield (C5)](#13-control-barrier-functions-and-the-shield-c5)
14. [Stackelberg self-play](#14-stackelberg-self-play)
15. [How to read every figure](#15-how-to-read-every-figure)
16. [What the results actually say](#16-what-the-results-actually-say)
17. [Glossary](#17-glossary)
18. [Viva / defence questions with answers](#18-viva--defence-questions-with-answers)

---

## 1. The 60-second version

A power grid is controlled by measurements sent from substations to a control
centre. An attacker who can tamper with those measurements can make the
controller take the wrong action, destabilising system frequency.

The base paper (Cui, Wu & Zhang, IEEE TSG 2025) builds a detect → repair →
control loop for this and reports 99.78 % detection accuracy. **We measured its
attack model and found it scores up to 18,000× a bad-data threshold that has
existed since the 1970s.** It is defending against something already solved.

Attacks that actually matter are *stealthy*: they live in the null space of the
state estimator, so the classical test cannot see them at all. Against those, you
need a learned detector. Everyone builds that detector as a **graph neural
network**, because a grid is a graph.

**We tested that assumption with a pre-registered experiment and it failed.** A
2-layer GNN reaches only 2 hops, but IEEE 118-bus has a diameter of 14 hops — so
each bus sees 9 % of the grid. Message passing is an *information bottleneck*
that gets worse as the grid grows.

**Our answer:** encode the grid the way a sparse linear solver factorises it —
by recursive dissection into separators — rather than by passing messages along
edges. This gives:

* a receptive field of `O(log N)` instead of `k` hops,
* a parameter count **independent of the number of buses**, so one trained model
  runs on 14 buses and 118 buses unchanged,
* and a tree that doubles as a **segment tree**, letting us localize an attack in
  `O(k log N)` calibrated probes instead of testing all `N` buses.

---

## 2. Power-system background you actually need

### Buses, lines, and per unit

* A **bus** is a node — a substation busbar. IEEE 14-bus has 14 of them.
* A **line** (or transformer) is an edge connecting two buses.
* **Per unit (p.u.)** is a normalisation: every quantity is divided by a base
  value, so voltages sit near 1.0 and powers are small numbers. It makes systems
  of different voltage levels comparable. When you see `0.05 p.u.`, read "5 % of
  the base value".

### What is measured

At each bus the SCADA system reports roughly:

| symbol | name | meaning |
|---|---|---|
| `|V|` | voltage magnitude | how "hard" the bus is being pushed, ≈ 1.0 p.u. |
| `θ` | voltage angle | phase of the AC waveform at that bus |
| `P` | active-power injection | real power flowing in (generation − load) |
| `Q` | reactive-power injection | the out-of-phase component |

Those four channels per bus are exactly the input tensor in this project:
`x ∈ R^{W × N × 4}` for a window of `W` timesteps over `N` buses.

### Power flow

**Power flow** (or load flow) is the calculation that answers: given the loads
and generation, what are the voltages and angles everywhere? The exact equations
are nonlinear:

```
    P_i = Σ_j |V_i||V_j| ( G_ij cos(θ_i − θ_j) + B_ij sin(θ_i − θ_j) )
```

Note the products of voltages and the trigonometric functions — this is why the
problem is nonlinear. It is solved iteratively by **Newton–Raphson**.

### The DC approximation

Because the full equations are expensive, engineers often linearise:

* assume all `|V| ≈ 1`,
* assume angle differences are small so `sin(θ_i − θ_j) ≈ θ_i − θ_j`,
* ignore reactive power and losses.

You get a **linear** model `z = Hx`, where `x` is the vector of bus angles and
`H` is a constant matrix. This is the **DC power flow**, and `H` is the
**measurement Jacobian**. It is fast, and it is what makes the stealth attack
possible — see §4.

**The base paper uses the DC model. We use the AC model.** That difference is
contribution C3.

### State estimation

The control centre has more measurements than unknowns, and they are noisy. It
solves a least-squares problem:

```
    x̂ = argmin ‖z − Hx‖²      ⟹      x̂ = H⁺z
```

The **residual** is what the measurements failed to explain:

```
    r = z − H x̂ = (I − H H⁺) z
```

If `r` is large, something is wrong — a broken sensor, or an attack. The
classical **χ² bad-data test** flags `‖r‖²/σ² > threshold`. This test has been
standard since the 1970s.

---

## 3. What Load Frequency Control is

An AC grid runs at a nominal frequency (50 or 60 Hz). Frequency is a global
indicator of the **power balance**:

* generation > load → the machines speed up → frequency **rises**
* load > generation → the machines slow down → frequency **falls**

Frequency must stay within a very tight band or protection equipment trips and
you get a blackout. **Load Frequency Control (LFC)** is the automatic loop that
adjusts generator setpoints to hold frequency at nominal.

### Two areas and the tie-line

A real interconnection is split into **control areas**, each responsible for its
own balance, joined by **tie-lines**. This project uses the standard two-area
model. Each area must:

1. hold its own frequency, and
2. not steal power from its neighbour through the tie-line.

Those two goals combine into one scalar, the **Area Control Error**:

```
    ACE_i = ΔP_tie,i + B_i · Δf_i
```

where `ΔP_tie` is the deviation of tie-line power from schedule, `Δf` the
frequency deviation, and `B_i` the frequency bias. **Driving `ACE` to zero is the
entire job of the controller.** Every control result in this project is reported
in terms of `ACE`.

### The plant, block by block

```
   setpoint u ──▶ [governor] ──▶ [turbine] ──▶ ΔP_m ──┐
                    T_g            T_t                │
                                                      ▼
       load ΔP_L ─────────────────────────────▶ [rotating mass] ──▶ Δf
                                                   2H, D            │
                                                      ▲             │
       tie-line ΔP_tie ◀──────────────────────────────┴─────────────┘
```

* **Governor** (`T_g`): valve that responds to speed error, with droop `1/R`.
* **Turbine** (`T_t`): converts steam/water flow to mechanical power.
* **Rotating mass** (`H` inertia, `D` damping): the swing equation.
* **Tie-line**: couples the two areas.

The **nonlinearities** live in these blocks, and the base paper removes most of
them. See §10.

---

## 4. False Data Injection, and why the base paper's attack is too easy

### The base paper's attack (call it A0)

```
    V ← V · (1 + m_v · g),     g ~ N(0,1),   m_v = 1
```

fired with probability 0.3 per step. With `m_v = 1` this roughly **doubles or
zeroes** a measurement.

**We measured what the classical χ² test does to it.** Notebook Cell 9:

| system | χ² threshold (p = .01) | A0 statistic | ratio |
|---|---|---|---|
| case14 | 38.9 | 3.96 × 10⁴ | **1,018 ×** |
| case30 | 66.2 | 1.87 × 10⁵ | **2,823 ×** |
| case57 | 113.5 | 4.39 × 10⁵ | **3,865 ×** |
| case118 | 234.9 | 4.25 × 10⁶ | **18,087 ×** |

A detector from 1972 catches this instantly. Reporting 99.78 % accuracy against
it is not evidence of a modern contribution.

### The stealth attack (A1) — the one that matters

Here is the whole idea in three lines. The residual is

```
    r(z) = (I − H H⁺) z
```

Take **any** vector `c` and inject `a = Hc`. Then

```
    r(z + Hc) = (I − H H⁺)(z + Hc) = r(z) + (Hc − H H⁺H c) = r(z)
```

because `H⁺H = I`. **The residual does not change. At all.** The χ² test is blind
to this by construction — not by tuning, not by luck, by linear algebra.

We verify it numerically: the relative change in the χ² statistic is
`~3 × 10⁻¹⁵`, i.e. machine epsilon.

**Intuition:** `Hc` is a measurement pattern that is *perfectly consistent with
some valid grid state*. The attacker is not injecting noise; they are injecting a
plausible lie. The estimator happily believes it and returns `x̂ + c`.

**This is why you need a learned detector.** The physics test is provably
useless here, so something must learn what real telemetry looks like.

### The adaptive attack (A2)

A small Gaussian policy chooses the attack parameters `(rate, magnitude, mix)`
and is trained by REINFORCE **against the live defender**, in alternating rounds.
Most "adaptive attacker" papers train the attacker once against a *frozen*
detector; here both move. See §14.

---

## 5. The base paper, in detail

> Y. Cui, T. Wu, Y. Zhang, "Load Frequency Control of Smart Grid Based on
> Data-Driven FDI Attacks Detection and Data Repair," IEEE Trans. Smart Grid,
> 16(6):5404–5415, Nov 2025.

Three coupled stages:

1. **MSA3E detection** — a semi-supervised **adversarial autoencoder** with a
   multi-head self-attention encoder and two discriminators (one on a categorical
   latent, one on a Gaussian latent). Trained in three phases: reconstruction,
   adversarial, supervised.
2. **LSTM repair** — trained on clean data; when the detector flags a sample, the
   LSTM reconstructs it.
3. **MARL-A3C control** — one actor–critic agent per area, centralised training /
   decentralised execution, cooperative reward `r_i = −Σ κ·ACE² − ‖a‖`.

**What we keep.** The plant (Eqs. 4–10), the ACE definition, the reward, and the
CTDE structure. This is the spine and we reuse it.

**What we replace.** The threat model (A0 → A1/A2), the detector front end
(flat attention → separator tree), the plant realism (DC/linear → AC/nonlinear),
and we *add* localization, calibrated uncertainty, a safety certificate and a
blind-spot certificate.

---

## 6. Why we deleted the graph neural network

A power grid *is* a graph, so the field's instinct is a **Graph Neural Network**:
each node repeatedly aggregates from its neighbours. After `k` rounds a node has
seen everything within `k` hops. This is called **message passing**.

### The pre-registered test

This project wrote down, **before running**, a claim and a falsification
criterion (`docs/03_PREREGISTRATION.md`):

> Topology-aware attention beats flat attention. Success: ΔF1 > 0 with t > 2, or
> ΔRMSE < 0 with |t| > 2. Anything else is failure. On failure the claim is
> dropped.

Result at n = 5 seeds across `{case14, case30} × {w=12, w=24}`:

| system | window | ΔF1 | t | verdict |
|---|---|---|---|---|
| case14 | 12 | +0.0066 | +0.65 | fails |
| case14 | 24 | +0.0164 | +1.33 | fails |
| case30 | 12 | −0.0031 | −0.28 | fails |
| case30 | 24 | +0.0034 | +0.39 | fails |

**Failed in every cell. The claim was dropped.**

At n = 3 the headline cell had read `ΔF1 = +0.0182, t = +1.70` — tempting to call
"approaching significance". At n = 5 it shrank to `+0.0066, t = +0.65`. That is
textbook regression to the mean, and writing the criterion down first is the only
reason it was not mis-reported.

### The diagnosis — and this is the important part

The failure has a measurable cause. Compare the **graph diameter** (how many hops
separate the two furthest buses) with the **model depth**:

| system | diameter | 2-layer GNN reach | fraction of grid each bus sees |
|---|---|---|---|
| case14 | 5 | 2 | **57.1 %** |
| case30 | 6 | 2 | **29.8 %** |
| case57 | 12 | 2 | **15.0 %** |
| case118 | 14 | 2 | **9.1 %** |

Buses in *opposite control areas* — exactly the pairs whose correlation a tie-line
attack would create — can never exchange information at all.

**So the failure is not "topology doesn't help". It is "message passing destroys
information, and worse as N grows."** The right response is not to abandon
topology; it is to encode topology **without** a hop-limited receptive field.

---

## 7. Nested dissection and the separator tree

### The classical idea

**Nested dissection** (George, 1973) is how sparse linear solvers order variables
before factorisation. To factorise a sparse matrix cheaply:

1. Find a small set of vertices — a **separator** `S` — whose removal splits the
   graph into two roughly equal disconnected halves `L` and `R`.
2. Order `L` first, then `R`, then `S` last.
3. Recurse on `L` and `R`.

Because `L` and `R` do not touch, eliminating `L` cannot create fill-in in `R`.
The recursion produces a **separator tree**: a balanced binary tree whose internal
nodes are separators and whose leaves are individual buses.

### Why this is the right object for a power grid

A separator in the electrical graph is a set of buses whose removal **electrically
decouples** two regions. In DC power flow, conditioned on the separator's angles,
the two halves are independent. **The separator is the physics bottleneck.** That
is a far stronger statement than "these two buses share an edge".

### How we build it

We do not use METIS. We use **recursive spectral bisection**:

1. Build the susceptance-weighted adjacency `W` (`W_ij = 1/x_ij`, the electrical
   coupling strength).
2. Form the normalised Laplacian `L̃ = D^{−1/2}(D − W)D^{−1/2}`.
3. Take the **Fiedler vector** — the eigenvector of the second-smallest
   eigenvalue. Its sign structure is the classic minimum-cut relaxation, so
   splitting at its median cuts where the network is *electrically weakest*.
4. The separator is the set of vertices incident to a cut edge.
5. Recurse.

> **Why the second-smallest eigenvalue?** The smallest is always 0 with a constant
> eigenvector (no information). The second — the Fiedler vector — is the smoothest
> non-trivial function on the graph, so nodes with similar values are strongly
> connected. Splitting at its median gives a balanced, low-cut partition.

### Measured result

| system | N | tree nodes | depth | ⌈log₂N⌉ | ratio | root separator |
|---|---|---|---|---|---|---|
| case14 | 14 | 27 | 4 | 4 | 1.00 | 5 buses |
| case30 | 30 | 59 | 6 | 5 | 1.20 | 8 buses |
| case57 | 57 | 113 | 7 | 6 | 1.17 | 14 buses |
| case118 | 118 | 235 | 8 | 7 | 1.14 | 15 buses |

Depth tracks `log₂ N` within ~1.2×. The grid genuinely decomposes.

---

## 8. RSTE — the encoder (C1)

**RSTE = Recursive Separator-Tree Encoder.**

### The two sweeps

Bus features `h_b ∈ R^{N×d}` come from a temporal encoder (identical across every
model in the benchmark, so only the *spatial* operator is under test).

**Leaf init.** Each leaf takes the mean of its buses:
`h_v = LayerNorm(mean_{b∈v} h_b)`.

**Bottom-up (deepest level first).** For each internal node `p` with children
`lo`, `hi` and separator `S_p`:

```
    h_p = LayerNorm( Merge([ h_lo ; h_hi ; mean_{b∈S_p} h_b ]) )
```

Information flows **upward**: the root ends up summarising the whole grid.

**Top-down (root first), repeated K times.**

```
    g   = σ( Gate([h_parent ; h_v]) )          gate in (0,1)
    h_v = LayerNorm( h_v + g ⊙ Bcast([h_parent ; h_v]) )
```

Information flows **downward**: each node learns its own state *in the context of
the whole grid*.

**Scatter.** Leaf states are written back to their buses.

### The three properties that matter

**(1) Weight tying ⟹ N-independence.** `Merge`, `Bcast`, `Gate` and `LeafOut`
are the *same tensors* at every level and every node. So the parameter count does
not depend on `N` or on the tree shape. Measured: **116,614 parameters on case14
and on case118 — identical.**

That is what makes zero-shot transfer possible. Rebinding to a new grid is pure
bookkeeping: recompile the `TreePlan`, reuse the weights.

By contrast, a flat Transformer needs a **positional embedding table with N rows**
to tell buses apart. Change `N` and that table has the wrong shape. Those models
are not "worse" at transfer — they are **undefined** at a different `N`.

**(2) Receptive field is O(log N).** Any two buses communicate through their
lowest common ancestor: at most `2 × depth` tree hops. A `k`-layer GNN reaches
`k` graph hops, and diameters here are 5–14.

**(3) The repeated top-down sweep is a truncated fixed point.** Applying the same
weights repeatedly is a **deep-equilibrium**-style iteration — but over a spatial
decomposition rather than over time. `K` is ablated at `{0,1,2,3}` in Part 13.

### Making it fast

A naive recursion is a Python loop over `2N−1` nodes. We compile the tree **once**
into a level-order schedule (`TreePlan`), so each sweep is a handful of batched
`index_add` / `index_copy` calls. Forward cost is `O(depth) = O(log N)` **kernel
launches** regardless of `N` — which is why case118 is not slower than case14 in
the measured latency.

---

## 9. HALO — logarithmic attack localization (C2)

**Detection** answers "was there an attack?". **Localization** answers "which
buses?" — which is what an operator actually needs.

Every published localizer scores all `N` buses and thresholds: `O(N)` work, and
an `N`-fold multiplicity penalty that destroys statistical power exactly when `N`
is large.

**The separator tree is already a segment tree over the bus set.** HALO exploits
that.

### Step 1 — a node statistic

For each tree node `v`: `S_v = max_{b ∈ v} s_b`, where `s_b` is the detector's
per-bus localization logit. (Max, not mean, because attacks are sparse.)

### Step 2 — conformal p-values

On **clean** calibration windows, record the null distribution of `S_v`. For a
test window:

```
    p_v = ( 1 + #{ i : S_v^{(i)} ≥ S_v^{test} } ) / ( n_cal + 1 )
```

This is a valid finite-sample p-value under exchangeability — **no distributional
assumption on the detector's scores at all**. See §11.

### Step 3 — the descent

```
    test the root
    if not rejected: stop, declare "no attack"
    else: for each family of two siblings, apply Benjamini–Hochberg at level q
          descend only into rejected children
    flag the buses of every rejected leaf
```

An attack on `k` buses is isolated in `O(k log N)` node evaluations.

### Step 4 — FDR

Testing a node only after its parent is rejected is **Yekutieli's hierarchical
testing** structure; BH within each sibling family then controls FDR over the
tree. Crucially, the correction applies to **families of size 2**, not to `N`
simultaneous hypotheses — so power does not decay as the grid grows.

### The honest caveat — you must know this

The calibration set is *clean windows*, so the p-value is valid for

> H₀(v): the bus set of node `v` contains no tampered measurement.

It is **not** valid per-bus. A null-space attack `a = Hc` perturbs **every** bus
simultaneously — that is exactly what makes it stealthy — so an *untampered* bus
inside an *attacked* window is not exchangeable with a bus from a clean window.
Its score is elevated by its neighbours' corruption.

**Consequence: the measured per-bus FDR exceeds nominal `q`, for HALO and for the
flat test alike.** Both share the same scores and the same `q`, so the
*comparison* is sound; the absolute level is not a calibration claim and must not
be quoted as one. Fixing this is open work.

---

## 10. NL-LFC — the nonlinear benchmark (C3)

The base paper's plant is linear in everything that matters. NL-LFC restores
eight mechanisms:

| # | mechanism | form | why it matters |
|---|---|---|---|
| 1 | AC power flow | Newton–Raphson | voltage–angle coupling the DC model deletes |
| 2 | ZIP load | `P_L = P₀(a_z V² + a_i V + a_p)` | load depends on voltage → a feedback path |
| 3 | Sinusoidal tie-line | `P_tie = (V₁V₂/X)·sin(δ₁−δ₂)` | saturates; can lose synchronism past 90° |
| 4 | Governor dead band | backlash hysteresis | non-smooth AND history-dependent |
| 5 | Rate constraint (GRC) | `|dP_m/dt| ≤ 0.0017` p.u./s | hard saturation — a big disturbance cannot be answered fast |
| 6 | Reheat turbine | second-order with a lead term | slower, oscillatory response |
| 7 | Renewables | Ornstein–Uhlenbeck × diurnal envelope | coloured noise, not white |
| 8 | Regime switching | 3-state Markov chain | non-stationarity |

Plus a realistic **channel**: piecewise-constant latency (0–2 steps), 1 %
instrument noise, 12-bit quantization, 1 % dropout.

> **Ordering matters.** The channel is applied **before** the attack, because an
> adversary corrupts the telemetry that *arrives*. Getting this backwards (as an
> earlier version of this code did) decouples the attack from its own label and
> the detector learns nothing — AUC 0.43, worse than chance.

### Backlash, explained

A dead band is usually taught as `y = 0 if |x| < δ`. Real governor dead band is
**backlash**: the output only moves when the input has moved more than `δ` *since
the last move*. It has memory. Trace a sine wave through it and you get a
hysteresis loop (notebook Figure 5a), not a flat spot. It is genuinely
history-dependent, which is much harder for a model than a static nonlinearity.

### The AC response surface — and the honest limitation

Newton–Raphson cannot be called 200 times per episode inside an RL loop. So we
fit a **full second-order polynomial** from the driver vector
`u = [ΔP_L1, ΔP_L2, ΔP_res1, ΔP_res2]` to the bus measurements, using true NR
solutions over a ±40 % load / ±25 % renewable envelope, and **verify it on
held-out true solutions** (the R-squared table in Cell 9).

**State this honestly:** `R²` for the best *linear* model is already ≥ 0.997. The
AC map is dominated by its linear part over this envelope. The defensible claim is
narrower and better:

> The residual the DC model discards is **10–30× larger** than it needs to be,
> and that residual is precisely the scale at which a stealthy null-space attack
> operates.

Where the two plants separate *decisively* is the closed loop: measured
frequency-excursion standard deviation is **0.2519 Hz for NL-LFC against 0.0275 Hz
for the linear plant — a 9.2× difference** (Figure 5g). Rate limiting, backlash
and tie-line saturation let a disturbance build up that the base paper's plant
simply cannot produce.

> **A statistic we report and do NOT use.** We also computed an AR-based
> "nonlinearity index". It does **not** significantly separate the two plants.
> It is printed anyway, because it was measured. Do not cite it as evidence.

---

## 10b. SSC — why one F1 number is almost meaningless (C4)

### The problem, stated bluntly

Every FDI paper reports **one F1** on a test set built by **its own attack
generator**. That number is not comparable across papers, because F1 depends
overwhelmingly on **how hard the author made the attacks** — and nobody reports
that.

Two papers reporting 0.97 and 0.89 might be the same detector on different
difficulty distributions. There is no way to tell from what is published.

We can demonstrate this inside our own notebook: the same detector scores near
1.0 on the base paper's loud A0 attack and near chance on a low-severity A1
attack. **The number in the abstract is a statement about the generator, not the
model.**

### What SSC does instead

Every attacked sample we generate ships a **physics-derived certificate**:

| symbol | meaning |
|---|---|
| `σ` | **stealth index**: `1` = invisible to χ², `0` = fully visible |
| `Δχ²` | exact displacement of the classical test statistic |
| `‖a‖` | attack energy |
| `μ` | **attacker cost** — meters that must be compromised |
| `‖c‖` | **attacker payoff** — how far the state estimate moves |
| `k` | compromised substation neighbourhoods |
| `ε` | the attacker's network-model error |

None of these depends on any model. So we report **`F1(σ)` — a curve, not a
point** — and two groups can compare at matched difficulty even with different
generators.

### The subtlety that makes σ interesting

Here is the thing that surprised us. If the attacker uses the operator's **exact**
`H`, then `σ = 1` *identically* — the whole stratification collapses to one point.
We measured exactly that: 100 % of samples in one stratum.

**That degeneracy is not a bug. It is the omniscient-attacker assumption the whole
field makes without comment, made visible.**

A real adversary does not have the utility's network model. They infer line
parameters from public filings or stale diagrams, so *their* Jacobian `H̃` carries
an error `ε`. When they build `a = H̃c`, the mismatch **leaks out of the null
space** and the defender's residual sees it:

```
    ‖P_⊥ a‖ = ‖P_⊥ (H̃ − H) c‖ > 0    whenever H̃ ≠ H
```

So `σ` becomes a direct measure of **how good the attacker's intelligence is**:

| `ε` | who this is | `σ` |
|---|---|---|
| 0 | textbook omniscient attacker | exactly 1 |
| 0.05 | good reconnaissance | ≈ 0.99 |
| 0.8 | public-data attacker | ≈ 0.89 |
| 3.0 | topology largely wrong | ≈ 0.71 |

### The second finding, and it is the alarming one

Because `a = He_i` is just the `i`-th **column** of `H`, a perfectly stealthy
attack needs only the meters in **one substation neighbourhood**.

Measured: on IEEE 118-bus that is **8.3 of 304 meters — 2.7 % of the system.**

The cost curve is strongly sublinear, so most of an attacker's reach is bought
with the first few substations. This is directly actionable — it says *which
substations to harden first* — and we have not found it reported anywhere.

### The rule for reading a stratified table

Evaluate on all clean windows **plus** the attacked windows of that stratum, with
the **decision threshold frozen** at the full-test value.

> Why frozen? Re-optimising the threshold per stratum lets the model trade
> precision for recall differently in each slice, and reports a number no deployed
> system could achieve — a real detector has *one* threshold and does not know
> which stratum it is looking at. It inflates the hard strata a lot.

---

## 10c. The threat space — how to compare threat models honestly

This section exists because of a problem that reading fifteen papers makes
obvious and that no single paper admits: **"stealthy" is not a defined word.**

One paper says its attack is stealthy because it evades a threshold detector.
Another says stealthy because the residual does not change. A third says
stealthy because a human operator would not notice the frequency deviation.
Those are three different claims, and when a survey table puts them in the same
column labelled "stealthy: yes", the table is lying by abbreviation.

### Three numbers instead of one adjective

Part 4 defines three quantities, and every attack this project generates carries
all three as a certificate:

**Stealth, `sigma`.** Split the attack vector `a` into the part the residual
test can see and the part it cannot:

```
sigma = 1 - ||P_perp a|| / ||a||
```

`sigma = 0` means the whole attack lands in the residual — the chi-square test
sees all of it. `sigma = 1` means none of it does. This is a statement about
*one specific detector*, the chi-square bad-data test, and nothing else. That
qualification matters enormously and is the source of the result in §16 that
surprises people.

**Cost, `mu`.** The number of meters the attacker must actually compromise —
`|supp(a)|`. This is the only one of the three that a defender can act on: it
names how many field devices must be breached, and therefore how much hardening
budget buys how much protection.

**Knowledge, `eps`.** The relative error in the attacker's assumed line
susceptances. `eps = 0` is the omniscient attacker who holds the operator's own
model. Every value above zero is an attacker who has *estimated* the grid.

### Why `eps` is the axis the field is missing

Here is the mechanism, and it is worth being slow about.

A perfectly stealthy attack is `a = Hc` — a linear combination of the columns of
the measurement Jacobian. The residual cannot see it because `P_perp H = 0`
exactly. But an attacker does not have `H`. They have their own estimate,
`H_tilde`, built from whatever reconnaissance they managed. So what they can
actually inject is `a = H_tilde c`, and

```
P_perp a = P_perp (H_tilde - H) c    which is NOT zero
```

The attack leaks into the residual **in proportion to how wrong the attacker's
model is**. That is why `sigma` becomes a continuous quantity once `eps > 0`,
and why it is pinned at exactly 1.0 in every paper that assumes `eps = 0`.

This is not a minor modelling refinement. It is the difference between a threat
model with a dial and a threat model with a single point. **If every attack in
your test set has `sigma = 1`, you have not evaluated your detector across
stealth — you have evaluated it at one value of stealth and reported a single
number.**

### What Figure 19 shows

Figure 19 places all ten surveyed threat models on these axes, against the region
the SSC dataset actually occupies.

* **Panel (a), `sigma` against `eps`.** **All ten** surveyed papers sit exactly on
  the `eps = 0` line. The interior — stealthy *and* imperfectly informed — is almost
  empty. That interior is not an exotic corner case; it is the only place a real
  attacker can be, because no adversary has the operator's susceptance database.
* **Panel (b), `sigma` against cost.** The surveyed papers are scattered, but
  none of them *reports* `mu` — their positions here are inferred. The vertical
  line is this project's measurement: the median cost of a perfectly stealthy
  attack, as a fraction of the system's meters. It is small.
* **Panel (c).** The same four gaps, tabulated, with the count of papers each
  applies to.

Read the marker style carefully. **Filled** means the paper states the quantity
numerically. **Hollow** means it was placed at the value its assumptions imply.
Figure 19 is a claim about *threat models*, not a re-measurement of anyone's
reported accuracy, and you should say exactly that if asked in a viva.

### The honest caveat

Placing other people's work on axes they did not use involves interpretation. A
paper that never mentions `eps` is placed at `eps = 0` because that is what its
equations assume, not because its authors claimed omniscience. If a reviewer
disputes a specific marker, the correct response is to move it — the argument
does not rest on any single paper's position. It rests on the shape of the
cloud, and the cloud has all ten points on a single line.

---

## 11. Conformal prediction, from zero

### The problem

A neural network outputs `0.83`. Is that 83 % confidence? **No.** Softmax outputs
are not calibrated probabilities. For a safety-critical loop you need a real
guarantee.

### The idea

**Split conformal prediction** gives you one, with no assumption about the model
or the data distribution. You only need **exchangeability** (roughly: the
calibration and test points are drawn from the same distribution, order
irrelevant).

Procedure:

1. Hold out a **calibration set** the model never trained on.
2. On it, compute a **nonconformity score** `s` — anything measuring "how wrong
   was the model here", e.g. `|prediction − truth|`.
3. Take the `⌈(n+1)(1−α)⌉`-th smallest score, call it `q̂`.
4. For a new point, the interval `prediction ± q̂` contains the truth with
   probability **at least `1 − α`**.

That is it. Four lines, finite-sample valid, distribution-free.

### Why the guarantee is weaker than it looks

The guarantee is **marginal**: averaged over everything. It says nothing about
any particular subgroup. Your intervals can be 99 % right on easy cases and 50 %
right on hard ones and still average to 90 %.

### Mondrian (regime-conditioned) conformal

Fix: **calibrate separately within each group**. Here the groups are keyed on the
detector's own flag (flagged / not flagged) — deliberately, because that is an
observable the controller *has at run time*. A guarantee conditioned on something
you cannot observe online is useless.

### Adaptive conformal (Gibbs–Candès)

Exchangeability breaks when the attacker adapts — the calibration set goes stale.
Fix: update the level online,

```
    α_{t+1} = α_t + γ ( target − error_t )
```

Miss too often → `α` shrinks → intervals widen → coverage recovers. No retraining
of the underlying model at all.

### Where the width is used — this is the point

Most papers report an interval and stop. Here the width does **two jobs**:

1. it **tightens the CBF safe set** (§13), so more uncertainty ⟹ more safety
   margin reserved;
2. it enters the **MARL reward** as `−β·w_i`, so the controller is penalised for
   operating in states where the estimate is untrustworthy.

That makes the width a *control quantity*, not a diagnostic.

---

## 12. The blind-spot certificate (C4)

### The problem with accuracy

"99.78 % accuracy" is a property of **the attacks the authors generated**. An
adversary does not sample from your test set. The number is not a security claim.

### What we report instead

```
    ρ(m) = fraction of attack directions at severity m that are
           SIMULTANEOUSLY invisible to the χ² test AND below the
           learned detector's threshold
```

Estimated by Monte Carlo over directions drawn uniformly on the unit sphere of
the attack subspace, with **Clopper–Pearson** intervals (the exact binomial
interval — correct even when the count is 0, which normal approximations are not).

Then:

```
    m*(ε) = min { m : ρ(m′) < ε  for EVERY m′ ≥ m }
```

`m*` is the smallest attack severity the deployed stack cannot be fooled by.

### Why this is a stronger claim

`ρ` is a property of the **(physics, detector) pair**, not of any test set. It
cannot be inflated by choosing a favourable attack rate or a favourable test
split. It is the closest thing to a security guarantee in the whole project.

### The caveat that must travel with it

`ρ` is **not guaranteed monotone** in severity. Intuition says a bigger lie is
easier to catch; the measurement can show a blind *band* rather than a blind
floor. That is why `m*` is defined with a "for every larger `m`" quantifier
rather than as a first crossing — a first-crossing definition would be
misleading.

**And below the certified threshold, the detector is not what keeps the plant safe. The CBF is.**
That is the entire argument for having a guarantee that does not depend on
detection.

---

## 13. Control barrier functions and the shield (C5)

### The idea

You have a learned controller. You cannot prove anything about a neural network.
But you *can* wrap it in a filter that provably keeps the state inside a set you
declare safe.

Define a **barrier function** `h(x)` that is positive inside the safe set and
zero on its boundary:

```
    h_ace = ACE_max² − ACE²  ≥ 0
    h_f   = f_max²   − f²    ≥ 0
```

Enforce the first-order condition

```
    ḣ ≥ −α h          (α > 0)
```

Read it: *the closer you get to the boundary (`h → 0`), the less you are allowed
to approach it.* Satisfying this at all times guarantees the safe set is
**forward invariant** — once inside, you stay inside.

Because `ḣ` is **affine in the control** `u`, this is a linear constraint, and
"find the smallest correction to `u` satisfying it" is a tiny quadratic program —
here solved in closed form as a projection.

### The conformal coupling

```
    ACE_max ← max( ACE_max − κ·w ,  0.25·ACE_max )
```

where `w` is the calibrated interval width. Large uncertainty ⟹ the shield
reserves more room. Small uncertainty ⟹ it gets out of the way.

Because `ΔP_tie` appears in **both** areas' barriers, area `i`'s uncertainty also
shrinks area `j`'s admissible action. There is no single-agent analogue of that.

### Calibrating the safe set — a methodological point worth understanding

Our first run declared `ACE_max = 0.05` and the undefended plant violated it on
**68 % of steps**. That is not a safe set; it is a saturated constraint. The
barrier is active everywhere, the experiment measures nothing.

Fix: **measure** the undefended plant's excursion distribution first and place the
bound at its 88th percentile, so a violation is genuinely uncommon and a shield
can plausibly prevent it. The calibration is in the notebook and its output is
printed.

> **General lesson:** if your baseline violates the constraint most of the time,
> your experiment cannot detect an improvement. Check the base rate first.

### Paired evaluation — the fix for the predecessor's biggest defect

The earlier project compared shielded vs unshielded runs "with the same seed" and
got `t = 0.42`, reporting the shield's benefit as unresolvable. **That comparison
was never actually paired.** The instant the shield changed an action the
trajectories diverged, and from then on the two arms saw *different* attacks and
*different* noise. It was comparing two unrelated random worlds.

Fix: `EpisodeSchedule` pre-draws attack timing, attack direction, sensor noise and
the disturbance **before** the episode, and replays the identical schedule through
every configuration. Differences are taken **within episode**, so `n` = episodes
(120), not seeds (3).

---

## 14. Stackelberg self-play

A **Stackelberg game** has a leader who commits first and a follower who best-
responds. In security, the defender commits to a policy and the attacker responds.

Here both sides move, alternately:

1. The attacker (a 3-parameter Gaussian policy over rate, magnitude, mix) is
   trained by **REINFORCE** against the **live** defender.
2. Reward = damage done (`RMS ACE`) minus a stealth penalty (`0.55·mag²`), so a
   loud attack is discounted — the attacker is pushed toward stealth.

### The equilibrium gap

The number that makes this a claim rather than a narrative:

> Give the attacker **one more best-response budget** against the frozen
> defender. How much does its objective improve?

* Large gain ⟹ **not converged**; more rounds needed. Report it as such.
* Small gain ⟹ at or near a local Stackelberg equilibrium.

Either way it is *reported*, never assumed. And note the scope: the attacker
parameterisation is only 3-dimensional, so convergence is a statement about **this
attack family**, not about attacks in general.

---

## 15. How to read every figure

There are 20 figures. These are the ones that carry the argument.

| Figure | What it shows | The one thing to take away |
|---|---|---|
| **1** | Four IEEE systems, AC-solved | Diameter grows 5 → 14 hops from case14 to case118 |
| **2** | **The stealth geometry** | (c): with attack energy fixed, χ² only rises once energy is rotated *out* of range(H). Stealth is linear algebra, not cleverness |
| **3** | Nested dissection, unfolded | (e): separator size falls monotonically with depth — the grid genuinely decomposes |
| **4** | **The receptive-field argument** | **The load-bearing measurement.** A 2-layer GNN sees 57 % of case14, 9 % of case118. No training fixes that |
| **5** | Each NL-LFC nonlinearity, isolated | (g): under an identical load step the nonlinear and linear plants are not the same system |
| **6** | Attack realism | (a) base-paper attack is 10³–10⁴× the χ² threshold. (d) a stealth attack needs 2.7 % of case118's meters. (f) σ vs attacker model error |
| **7** | The SSC dataset | Severity, stealth strata, attacker cost — the three axes the literature conflates |
| **8** | Ten-architecture benchmark | Read (h), the paired t-tests, not the bar heights |
| **9** | **Stratified evaluation** | What a single F1 hides. F1 is a *curve* over stealth, severity and attacker cost |
| **10** | Zero-shot transfer 14 → 118 | Five of ten baselines are **n/a** — not bad, *undefined*. That is the categorical result |
| **11** | HALO cost / power / FDR | (a) log vs linear cost. (c) FDR inflated for *both* methods — read §9's caveat |
| **12** | Conformal calibration | (b) whether the two regimes actually differ; (c) whether adaptation tracks drift |
| **13** | Blind-spot certificate | (c) the χ² test catches **0 %** at *every* severity, because `a = Hc` is exactly in its null space |
| **14** | Closed loop + self-play | Paired **within episode**, n = 120. Violation rate is the highest-variance metric |
| **15** | Ablations + measured cost | The **red bar** is the decisive control: a random tree of the same depth and parameter count |
| **16** | System architecture | The manuscript's Fig. 1 |
| **17** | The RSTE mechanism | How one forward pass works: bottom-up merge, then K weight-tied top-down sweeps |
| **18** | **A traced HALO descent** | The grey subtrees were never touched — one rejected p-value prunes an entire branch |
| **19** | **The threat space, with the literature in it** | The surveyed field sits at `eps = 0` or `sigma < 1`. The stealthy-*and*-imperfectly-informed interior is nearly empty — and that is where a real attacker lives |
| **20** | Graphical abstract | Problem, mechanism, four headline numbers. Every number is computed, not typed |

---

## 16. What the results actually say

Three things to internalise before you defend this work.

### (a) SCEPTRE does not win the IEEE 14-bus table — and that is fine

Measured, 3 seeds, case14:

| model | detection F1 | localization F1 | params |
|---|---|---|---|
| GCN | **0.9937 ± 0.0000** | **0.8883 ± 0.0266** | 50,694 |
| KAN | 0.9901 ± 0.0050 | 0.8720 ± 0.0428 | 124,422 |
| BiLSTM | 0.9892 ± 0.0063 | 0.8503 ± 0.0509 | 69,126 |
| **SCEPTRE** | 0.9870 ± 0.0071 | 0.8450 ± 0.0093 | 116,614 |
| MSA3E (base paper) | 0.9563 ± 0.0149 | 0.7913 ± 0.0176 | 110,854 |

SCEPTRE beats the base paper's MSA3E significantly (**paired t = +4.19**) but a
plain GCN edges it (t = −1.33 detection, −2.58 localization).

**This is the expected result.** At `N = 14` a 2-layer GNN already reaches 57 % of
the grid, so the bottleneck that motivates a separator tree does not bite. The
claims that carry weight are the ones a baseline **cannot make at all**:
`N`-independent parameters, zero-shot transfer, `O(k log N)` localization.

**Never open a talk or a paper with the case14 detection table.** It is the
weakest evidence in the project.

### (b) Detection saturates — the severity sweep is the real comparison

At a loud attack, everything clears 0.95. The base paper's 99.78 % is a statement
about an easy attack. Below severity ≈ 0.3, *every* model collapses to the
all-positive classifier's F1 — that is the **stealth limit**, and below it the
certificate (§12), not the detector, is what bounds the damage.

### (c) Negative results are reported, not buried

* The graph claim was pre-registered and **refuted**.
* The AR nonlinearity index does **not** separate the plants.
* HALO's per-bus FDR **exceeds** nominal, with the mechanism explained.
* The Mondrian calibrator does not always beat the pooled one on this split.

A reviewer who sees a pre-registration, a failed criterion, and a replacement
built around the diagnosis will trust the surviving claims **more**, not less.

---

## 17. Glossary

| term | meaning |
|---|---|
| **ACE** | Area Control Error, `ΔP_tie + B·Δf`. The scalar the controller drives to zero |
| **AUC** | Area under the ROC curve; probability a random positive scores above a random negative |
| **Backlash** | History-dependent dead band; output moves only after the input moves enough |
| **BH / FDR** | Benjamini–Hochberg procedure / False Discovery Rate = expected fraction of flags that are wrong |
| **Bus** | A node in the grid (a substation busbar) |
| **CBF** | Control Barrier Function; a certificate of forward invariance for a safe set |
| **Clopper–Pearson** | Exact binomial confidence interval; correct even at 0 or 100 % |
| **Conformal prediction** | Distribution-free, finite-sample valid uncertainty from a calibration set |
| **CTDE** | Centralised Training, Decentralised Execution (multi-agent RL) |
| **DC power flow** | Linearised power flow: `z = Hx`. Fast, and the reason stealth attacks exist |
| **DEQ** | Deep Equilibrium model; a weight-tied layer iterated to a fixed point |
| **Exchangeability** | Joint distribution invariant to reordering. The only assumption conformal needs |
| **FDI** | False Data Injection attack |
| **Fiedler vector** | Eigenvector of the second-smallest Laplacian eigenvalue; the spectral-bisection cut |
| **GDB / GRC** | Generation Dead Band / Generation Rate Constraint |
| **Jacobian `H`** | DC measurement matrix mapping bus angles to measurements |
| **LFC** | Load Frequency Control |
| **Mondrian conformal** | Conformal calibration performed separately within each group |
| **Nested dissection** | Recursive separator-based ordering for sparse factorisation (George, 1973) |
| **Null-space attack** | `a = Hc`; provably leaves the state-estimation residual unchanged |
| **p.u.** | Per unit; normalised units |
| **REINFORCE** | Policy-gradient method using sampled returns |
| **Residual** | `r = z − Hx̂`; what the measurements failed to explain |
| **Segment tree** | Balanced tree over an index range supporting logarithmic queries |
| **Separator** | Vertex set whose removal disconnects a graph into balanced halves |
| **Stackelberg game** | Leader commits, follower best-responds |
| **ZIP load** | Load model with constant-impedance, constant-current, constant-power parts |
| **χ² bad-data test** | Classical residual test for corrupted measurements |

---

## 18. Viva / defence questions with answers

**Q. Why not just use a GNN? A power grid is a graph.**
A. We tested it with a pre-registered criterion at n = 5 seeds and it failed in
all four cells. The cause is measured: a 2-layer stack reaches 2 hops but case118
has diameter 14, so each bus sees 9 % of the grid. Message passing is an
information bottleneck that worsens with `N`. We keep the topology and drop the
hop limit.

**Q. A GCN beats you on IEEE 14-bus. Why is your method better?**
A. It is not better *on 14 buses*, and we say so. At `N = 14` a 2-layer GNN
already covers 57 % of the grid so the bottleneck doesn't bite. Our claims are
`N`-independent parameters, zero-shot transfer to 118 buses with identical
weights, and `O(k log N)` localization — none of which a GCN provides, and five of
the ten baselines cannot be *evaluated* at a different `N` at all.

**Q. Isn't 99.78 % accuracy in the base paper already solved?**
A. Against their attack, yes — and that is the problem. We measured it at up to
18,000× a χ² threshold from the 1970s. Against a null-space attack the same
statistic doesn't move at all (3 × 10⁻¹⁵ relative), and that is the regime where
a learned detector is actually necessary.

**Q. Why is the separator tree better than any balanced binary tree?**
A. Ablated directly (Figure 15): a random balanced tree of the same depth loses
accuracy, and removing the separator features loses more. The gain comes from the
*nested-dissection structure*, not from merely having a hierarchy.

**Q. Your FDR is inflated. Doesn't that invalidate HALO?**
A. It invalidates any *absolute* calibration claim, and we state that. The
p-values are valid for subtrees, not per-bus, because `a = Hc` perturbs every bus
so an untampered bus in an attacked window isn't exchangeable with one from a
clean window. The flat baseline is inflated identically, so the comparison — cost
and power at matched `q` — stands. Fixing the per-bus null is open work.

**Q. Is this real data?**
A. No. It is simulation end to end, and the abstract must say so. The MSU/ORNL
testbed dataset is feature-space with no multi-area LFC ground truth, so
transferring to it needs a feature bridge we have not built. It is listed as open,
not claimed.

**Q. What is the single strongest result?**
A. Figure 10. Trained on IEEE 14-bus, SCEPTRE runs on IEEE 118-bus with the same
weight tensors and no gradient steps. Five of the ten baselines — including
the base paper's MSA3E — cannot be evaluated there at all, because their
positional table has `N` rows. That is a categorical capability difference, not a
percentage.

**Q. Boyaci *et al.* already did GNN detection *and* localization for FDI, in
2022, in your target journal. So what is left of HALO?**

A. Correct, and we cite it as [15] and say so in §4.3 rather than waiting to be
told. What is left is the **cost class** and the **error guarantee**, which are
different claims from "graph structure helps localize".

* Boyaci *et al.* score **every bus** — `O(N)` evaluations — and apply **no
  multiplicity correction**. So does Qu *et al.* [4].
* HALO descends a tree: `O(k log N)` node evaluations, with Benjamini–Hochberg
  applied **within each sibling family** of size 2 rather than across `N`
  simultaneous hypotheses.

The second point matters more than the first. Under flat BH across `N`
hypotheses, power decays as the grid grows, because the correction gets harsher
with `N`. Under family-wise descent it does not, because the family size stays at
2 regardless of `N`. That is a structural difference, not a constant factor.

If a reviewer says "so it's just a speed-up" — no. It is a speed-up **and** a
power result, and §6.4 reports both, along with the FDR caveat.

**Q. Mohamed, Kundur & Khalaf published a CBF shield for LFC under attack in
2023. Isn't your C6 just that?**

A. The mechanism is theirs and we cite it as [16]. Our claim is narrower and is
about how the safe set and the margin are *chosen*:

1. **The safe set is calibrated from the plant**, at the 88th percentile of the
   undefended excursion distribution. This is not decoration. Our first attempt
   used a hand-chosen bound that the plant violated on roughly two-thirds of
   steps, which pinned the barrier at saturation and made the experiment measure
   nothing at all. §6.6 records that defect and the fix, because a reader
   deserves to know that the number moved for a reason.
2. **The margin is conformally tightened** — the interval width enters both the
   safe set and the MARL reward — so the shield's conservatism is tied to a
   finite-sample guarantee rather than a constant someone picked.

The one-line answer: *known mechanism, calibrated safe set, conformal margin, and
we measure whether the margin actually helps.*

**Q. You downgraded two of your six contributions. Doesn't that weaken the
paper?**

A. It strengthens it. Both downgrades came from our own search, are recorded in
`02_LITERATURE.md` Part C, and are reflected in `01_NOVELTY.md` and in the
manuscript. The alternative was to submit two overclaims and have a reviewer find
them — which costs a review cycle and credibility on the *surviving* claims.

A related point worth making if pressed: one entry in an earlier draft of our own
literature file cited a conformal-prediction-for-FDI paper that we could not
subsequently verify. Rather than quietly deleting it, the file carries the
retraction. That is the standard we want applied to us.

**Q. How do your numbers compare to the base paper's 99.78 %?**

A. They do not compare, and the paper says so explicitly. Cui *et al.* measure
99.78 % on their own attack generator; §1.2 measures that attack at roughly
`1.8 x 10^4` times a χ² bad-data threshold, i.e. several orders of magnitude
louder than ours. Putting the two numbers side by side would be meaningless.

The defensible comparison is §6.1: every architecture, including a
reimplementation of MSA3E, trained and evaluated on **one** benchmark, one attack
generator, one split, three seeds. `docs/07_COMPARISON.md` Table IV keeps other
papers' reported figures in a physically separate column from ours, with the
reason stated in the caption.
