# Literature — verified references and the benchmark set

Every reference below was checked against a live bibliographic source during this
work. Entries are marked:

* **[VERIFIED]** — the paper exists with the stated authors, venue and year.
* **[VERIFIED, CORRECTED]** — exists, but the citation in the Zeroth-Review deck
  needed a correction (shown).
* **[BENCHMARK]** — added by this work as a comparison target; not in the deck.

---

## Part A — the base paper and the Zeroth-Review reference list

### [1] Base paper — **[VERIFIED]**

> Y. Cui, T. Wu, and Y. Zhang, "Load Frequency Control of Smart Grid Based on
> Data-Driven FDI Attacks Detection and Data Repair," *IEEE Transactions on Smart
> Grid*, vol. 16, no. 6, pp. 5404–5415, Nov. 2025.
> doi: `10.1109/TSG.2025.3590035`

**Method.** MSA3E — a semi-supervised adversarial autoencoder with a multi-head
self-attention encoder and two discriminators — for detection; LSTM data repair;
MARL-A3C for two-area LFC.

**Contribution.** First unified detect → repair → control loop for LFC. Reports
99.78 % detection accuracy on IEEE 14-bus.

**Limitations this project acts on.**
1. The attack model `V ← V(1 + m_v g)` with `m_v = 1` roughly *doubles* a
   measurement when active. **Measured in Cell 9 of the notebook: it scores up to
   1.8 × 10⁴ times a χ² bad-data threshold.** A 1970s detector catches it.
2. Single topology (IEEE 14-bus only).
3. Repair compared only against random forest — no attention/Transformer arm.
4. Fully equation-generated data; DC power flow; linear tie-line.

### [2] **[VERIFIED, CORRECTED]**

> H. Feng, Y. Han, F. Si, and Q. Zhao, "Detection of False Data Injection Attacks
> in Cyber-Physical Power Systems: An Adaptive Adversarial Dual Autoencoder With
> Graph Representation Learning Approach," *IEEE Transactions on Instrumentation
> and Measurement*, vol. 73, pp. 1–11, 2024.

*Correction:* the deck's short title omits "Cyber-Physical Power Systems"; the
full title is above. Detection only — no repair, no control loop.

### [3] **[VERIFIED]**

> P. Wang, R. Zhang, and X. He, "New Approaches to Detection and Secure Control
> for Cyber-Physical Systems Against False Data Injection Attacks,"
> *International Journal of Control, Automation and Systems*, vol. 23,
> pp. 332–345, 2025. (Springer) doi: `10.1007/s12555-024-0352-z`

Not power-grid specific; no deep sequence-model repair stage.

### [4] **[VERIFIED]**

> Z. Qu, Y. Dong, Y. Li, S. Song, T. Jiang, M. Li, Q. Wang, L. Wang, X. Bo *et al.*,
> "Localization of Dummy Data Injection Attacks in Power Systems Considering
> Incomplete Topological Information: A Spatio-Temporal Graph Wavelet
> Convolutional Neural Network Approach," *Applied Energy*, vol. 360,
> art. 122736, 2024. Preprint: `arXiv:2401.15321`

**Directly relevant to C2.** This is a *localization* paper and it performs the
`O(N)` per-bus scoring that HALO replaces. It is the closest prior art to the
localization contribution and should be cited as such.

### [5] **[VERIFIED]**

> S. Nayak *et al.*, "Data Imputation Using Self Attention Based Model for
> Enhancing Distribution Grid Monitoring and Protection Systems," *IEEE
> Transactions on Instrumentation and Measurement*, vol. 73, pp. 1–11, 2024.

Repairs *missing* data, not adversarially corrupted data.

### [6] **[VERIFIED]**

> Z. Hu, R. Ma, B. Wang, Y. Huang, and R. Su, "A General Resiliency Enhancement
> Framework for LFC of Interconnected Power Systems Considering IoT Faults,"
> *IEEE Transactions on Industrial Informatics*, 2024.
> doi: `10.1109/TII.2024.3397400`

Fault-centric, not adversary-centric; model-based, not data-driven.

### [7] **[VERIFIED]**

> T. Morris, U. Adhikari, and S. Pan, "Power System Attack Datasets,"
> Mississippi State University & Oak Ridge National Laboratory, 2014.

**Not used in this notebook.** It is a *feature-space* dataset (128 relay/PMU
features, 78 377 samples) with no multi-area LFC frequency ground truth, so a
detector trained on NL-LFC bus telemetry cannot be transferred to it without a
feature-space bridge that would itself need justification. Listing it as
"external validity" in the deck overstates what it can deliver. This is recorded
as an open item, not as a completed objective.

---

## Part B — benchmark targets added by this work

These are the architectures reimplemented in Cell 13 and compared head-to-head in
Figure 8. They exist so that the comparison is against **2023–2025 practice**,
not only against the base paper.

### [B1] Spatio-temporal Transformer — **[BENCHMARK, VERIFIED]**

> "Detection of false data injection attack in power grid based on spatial-temporal
> transformer network," *Expert Systems with Applications* (Elsevier), 2023.
> doi: `10.1016/j.eswa.2023.121706`

Implemented as the `transformer` arm: full pairwise attention over buses plus a
learned positional embedding. **Its positional table is why it cannot transfer to
a different bus count** — a point Figure 10 makes concrete.

### [B2] Graph attention + Kolmogorov–Arnold network — **[BENCHMARK, VERIFIED]**

> "Graph attention and Kolmogorov–Arnold network based smart grids intrusion
> detection," *Scientific Reports* (Springer Nature), 2025.
> doi: `10.1038/s41598-025-88054-9`

Implemented as the `kan` and `gat` arms.

### [B3] KAN for AGC cyberattack detection — **[BENCHMARK, VERIFIED]**

> "A Kolmogorov-Arnold Network for Interpretable Cyberattack Detection in AGC
> Systems," `arXiv:2509.05259`, 2025.

**The closest competitor by application.** AGC is the same control layer as LFC,
and the paper explicitly considers system nonlinearities. Reports 95.97 %
detection. Must be discussed in related work.

### [B4] Mamba / selective state-space anomaly detection — **[BENCHMARK, VERIFIED]**

> "KambaAD: Enhancing State Space Models with Kolmogorov–Arnold for Time Series
> Anomaly Detection," OpenReview, 2024.

Implemented as the `s4` arm.

### [B5] Graphormer-style spatial bias — **[BENCHMARK]**

Attention with an additive bias indexed by hop distance. This is the arm the
pre-registration tested and dropped; it is retained in the benchmark as the
honest report of a negative result.

### [B6] Conformal prediction in grid security — **[UNRESOLVED — do not cite as written]**

An earlier draft of this file asserted that a specific "ST-former trained with
distributionally robust optimisation and calibrated by conformal prediction"
paper exists. **A targeted search did not confirm it, so the claim is withdrawn
rather than carried forward.** It is recorded here as a retraction so the error
cannot silently re-enter the manuscript.

What the search *does* support:

* Conformal prediction is being actively adopted across power-systems machine
  learning — electricity-price forecasting, PV market bidding, optimality-gap
  bounding — so the technique is established in this field and needs no defence.
* A specific, verifiable paper applying conformal prediction to **FDI detection**
  was not located. That is a weaker statement than "none exists", and the
  manuscript must phrase it that way.

**Action before submission.** Search once more at submission time and cite
whatever exists. Do not claim conformal-for-FDI as unoccupied on the strength of
one negative search; claim only the narrower thing we can defend, which is the
*hierarchical, separator-tree-structured* use in C2 — conformal p-values per
separator node with BH control within each sibling family.

### [B7] Deep equilibrium layers in power systems — **[BENCHMARK, VERIFIED]**

> "Acceleration of Power System Dynamic Simulations Using a Deep Equilibrium Layer
> and Neural ODE Surrogate," `arXiv:2405.06827`, 2024.

Establishes that implicit/fixed-point layers are accepted in this field. RSTE's
repeated top-down sweep is a truncated fixed-point iteration in the same family,
so this is the right precedent to cite for the mechanism — while noting that the
fixed point here is over a *separator tree*, not over time.

---

---

## Part C — prior art that **constrains** our claims

Part B lists methods we benchmark against. This part is different and more
important: these are papers that already do something we might otherwise have
claimed as new. Each one narrows a contribution, and each is cited in the
manuscript at the point where it does so. A reviewer who knows this literature
will look for exactly these papers; the paper is stronger for finding them
first.

### [C1] The origin of the null-space attack — **[VERIFIED]**

> Y. Liu, P. Ning, and M. K. Reiter, "False Data Injection Attacks Against State
> Estimation in Electric Power Grids," *ACM Transactions on Information and
> System Security (TISSEC)*, vol. 14, no. 1, art. 13, pp. 1–33, May 2011.
> (Conference version: ACM CCS 2009.)

**What it establishes.** That an attacker who knows `H` can construct
`a = Hc` and pass the χ² residual test with probability 1. Everything in this
project's threat model descends from this result.

**What it leaves open, and what we do about it.** Liu *et al.* assume exact
knowledge of `H`. That assumption is inherited almost universally by the papers
in Part A, and it is the assumption our `eps` axis relaxes (Part 4, Figure 19).
We are not the first to *notice* that an attacker may have imperfect knowledge;
we are, as far as the search in Part D can establish, the first to make it a
**continuously swept evaluation axis with a per-sample certificate**.

### [C2] GNNs for joint FDI detection **and localization** — **[VERIFIED]**

> O. Boyaci, M. R. Narimani, K. Davis, M. Ismail, T. J. Overbye, and E. Serpedin,
> "Joint Detection and Localization of Stealth False Data Injection Attacks in
> Smart Grids Using Graph Neural Networks," *IEEE Transactions on Smart Grid*,
> vol. 13, no. 1, pp. 807–819, Jan. 2022. doi: `10.1109/TSG.2021.3117977`

**This is the most important paper in this file after the base paper.** The
authors state it is the first GNN work to automatically detect *and* localize
FDIA in power systems, using ARMA-type graph filters chosen specifically because
they adapt to spectral shifts better than polynomial filters.

**How it constrains us.** We must **not** claim novelty for "using graph
structure to localize FDI attacks" — that is this paper. Our localization claim
is narrower and must be stated narrowly:

* Boyaci *et al.* produce a **per-bus** score, i.e. `O(N)` evaluations, with no
  multiplicity control across buses.
* HALO produces a **hierarchical** decision with `O(k log N)` node evaluations
  and an explicit FDR-controlled stopping rule per sibling family.

The contribution is the *cost class and the error guarantee*, not the fact of
localization. §6.4 reports both, and §4.3 states the exchangeability caveat.

**Why ARMA filters do not remove the receptive-field objection.** An ARMA graph
filter has a rational frequency response and therefore a formally infinite
impulse response over the graph, which is a genuine advantage over a `K`-hop
polynomial filter. It is nevertheless *implemented* as a fixed number of
iterations, and the effective reach is set by that iteration count. Our Figure 4
measurement is about **implemented** reach, not the idealised response, and the
manuscript should say so rather than implying that ARMA filters are hop-limited
by construction.

### [C3] Control barrier functions for LFC attack mitigation — **[VERIFIED]**

> A. S. Mohamed, D. Kundur, and M. Khalaf, "On the Use of Safety Critical Control
> for Cyber-Physical Security in the Smart Grid," `arXiv:2303.02197`, Mar. 2023.
> (University of Toronto.)

**This directly precedes contribution C5.** The paper presents a safety-critical
controller based on control barrier functions to mitigate attacks against load
frequency control — the same mechanism, on the same control loop, for the same
purpose.

**How it constrains us.** We must **not** claim "CBF shield for LFC under
attack" as novel. What remains, and what §4.5 actually claims, is narrower:

* the safe set is **calibrated from the plant's own undefended excursion
  distribution** rather than assumed, because a bound the plant violates on most
  steps makes the barrier inactive-by-saturation and measures nothing (§6.6);
* the margin is **tightened by a conformal upper bound** on the detector's
  residual error, so the shield's conservatism is tied to a finite-sample
  guarantee rather than to a hand-chosen constant.

The honest one-line statement: *we apply a known mechanism with a calibrated
safe set and a conformal margin, and we measure whether the margin helps.*

### [C4] Unknown-input observers for LFC/AGC attack detection — **[VERIFIED]**

> A. Ameli, A. Hooshyar, A. H. Yazdavar, E. F. El-Saadany, and A. Youssef,
> "Attack Detection for Load Frequency Control Systems Using Stochastic Unknown
> Input Estimators," *IEEE Transactions on Information Forensics and Security*,
> 2018.
>
> A. Ameli, A. Hooshyar, E. F. El-Saadany, and A. Youssef, "Attack Detection and
> Identification for Automatic Generation Control Systems," *IEEE Transactions on
> Power Systems*, vol. 33, no. 5, pp. 4760–4774, Sep. 2018.

The model-based counterpart to this whole line of work: an observer estimates the
unknown input and flags a discrepancy. Important because it is the **baseline a
control-theory reviewer will ask about**, and because it is *not* learned — it
needs no training data and carries no distribution-shift risk. Its weakness is
the mirror image of ours: it depends on an accurate plant model, which is exactly
what the `eps > 0` regime denies the attacker and, in practice, the defender too.

### [C5] Survey of frequency-control cyber security — **[VERIFIED]**

> "Cyber Security of Smart-Grid Frequency Control: A Review and Vulnerability
> Assessment Framework," *ACM Transactions on Cyber-Physical Systems*, 2024.
> doi: `10.1145/3661827`

The right survey to cite for the framing sentence in the introduction, and a
useful independent check that the threat taxonomy in §2.5 is not idiosyncratic.

### [C6] Adaptive retraining for LFC FDI detection — **[VERIFIED]**

> "Adaptive False Data Injection Attack Detection in Load Frequency Control Using
> Recurrent Neural Networks," *SICE Journal of Control, Measurement, and System
> Integration* (Taylor & Francis, open access), 2025.
> doi: `10.1080/18824889.2025.2597566`

Directly on topic and very recent. Its stated motivation — that a learned
detector degrades when the system enters a state absent from its pretraining
data — is the same failure mode our adaptive-conformal arm addresses in §4.5,
by a different route (recalibration rather than retraining). Worth an explicit
contrast in related work.

### [C7] Broad FDI survey — **[VERIFIED]**

> "False Data Injection Attacks in Smart Grids: State of the Art and Way
> Forward," `arXiv:2308.10268`, 2023.

General background; useful for the introduction's opening paragraph.

---

## Part D — the novelty audit

Searches run against live literature during this work, with the result:

| Query | Result |
|---|---|
| `"nested dissection" OR "elimination tree" OR "separator tree"` + neural network + FDI + power system | **No prior work found.** Decision-tree and gradient-boosting papers dominate the results; none use matrix-elimination structure as a neural inductive bias. |
| hierarchical conformal prediction + FDR + attack localization + power grid | Conformal + FDR exists for *link prediction*; hierarchical conformal testing exists generically. **No prior work applying it to a power-network separator tree for attack localization.** |
| deep equilibrium / implicit networks + power system state estimation + FDI | Implicit layers used for *dynamic simulation acceleration* and physics-informed state estimation, **not** as a spatial encoder for detection. |

**Conclusion, restated after Part C.** Part C found two papers that narrow the
claims, and the honest summary is now more conservative than the first draft:

| Contribution | Status after the Part C search | Precise claim the manuscript makes |
|---|---|---|
| **C1** RSTE separator-tree encoder | No prior work found | Nested dissection as a *neural inductive bias*, with parameters independent of `N` |
| **C2** HALO hierarchical localization | **Narrowed by [C2] Boyaci 2022** | Not "GNNs can localize FDI" — that is Boyaci. Ours is the `O(k log N)` cost class plus per-family FDR control |
| **C3** NL-LFC benchmark | Extension, not category-new | Eight nonlinearities composed in one LFC plant, with each isolated and shown |
| **C4** SSC certified dataset | No prior work found | Per-sample `(sigma, mu, eps)` certificates shipped with the data |
| **C5** Conformal-tightened CBF shield | **Narrowed by [C3] Mohamed 2023** | Not "CBF for LFC" — that is Mohamed. Ours is the calibrated safe set plus the conformal margin |
| **C6** Stackelberg self-play | Extension | Reported with a non-zero equilibrium gap, i.e. explicitly not converged |

Two of six contributions were downgraded by this search. That is a good outcome
for the paper: the two survivors are stated more sharply, and the four remaining
are presented as extensions rather than as discoveries.

> **Caveat that belongs in the manuscript.** A negative literature-search result
> is evidence, not proof. The related-work section should state the search terms
> used, as this file does, so a reviewer can challenge them directly.
