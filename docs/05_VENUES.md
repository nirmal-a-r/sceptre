# Target venues — ranked, with a submission plan

Selection criteria, in order: (i) the reviewer pool understands **both** power
systems and machine learning; (ii) the venue has published the base paper or its
direct neighbours, so the comparison lands; (iii) scope admits an
architecture + statistics contribution rather than demanding a field trial.

---

## Tier 1 — primary target

### 1. IEEE Transactions on Smart Grid  (IEEE)

* **Why first.** This is the base paper's own venue. A paper that reproduces
  Cui–Wu–Zhang, measures a structural weakness in its threat model, and replaces
  its detector is *exactly* the follow-up this readership is equipped to judge.
* **Fit.** Strong. Detection + localization + control on a two-area LFC system is
  squarely in scope.
* **What must be in the paper.** The χ² measurement (Figure 13) is the hook — it
  says, quantitatively, that the field has been defending against an easy attack.
  Lead with it.
* **Risk.** Reviewers will ask for a larger system than case14 in the *closed
  loop*, not just in detection. Pre-empt it: the transfer result (Figure 10)
  covers detection to 118 buses; state plainly that the control loop is two-area.
* Typical decision time: 3–5 months.

### 2. IEEE Transactions on Industrial Informatics  (IEEE)

* **Why.** Published ref. [6] (Hu *et al.*, LFC resilience under IoT faults), so
  the "unified fault-and-attack threat model" framing has precedent here.
* **Fit.** Strong, with a slightly more systems/deployment emphasis. Lead with
  the `O(log N)` localization cost and the CBF safety certificate — the
  operational story — rather than with the architecture.
* Typical decision time: 3–4 months.

---

## Tier 2 — strong alternatives

### 3. Applied Energy  (Elsevier)

* **Why.** Published ref. [4] (Qu *et al.*), the closest prior art to C2.
* **Fit.** Good, but Applied Energy expects an *energy-system* consequence.
  Frame the conformal width → CBF margin → lost control headroom chain as the
  energy-relevant outcome, and de-emphasise the neural architecture.
* **Caution.** High volume, and the reviewer pool is less ML-literate. The
  separator-tree contribution may be undersold here.

### 4. International Journal of Electrical Power & Energy Systems (IJEPES)  (Elsevier)

* **Why.** Fast, broad, and comfortable with method papers on IEEE test systems.
* **Fit.** Very good as a *reliable* target. Lower prestige than Tier 1 but a
  markedly higher acceptance probability, and turnaround is typically quicker.

### 5. IEEE Transactions on Instrumentation and Measurement  (IEEE)

* **Why.** Published refs. [2] and [5].
* **Fit.** Good **if** the paper is reframed around *measurement integrity* —
  the channel model, quantization, the blind-spot certificate as a measurement
  guarantee. The control loop would have to become secondary.

---

## Tier 3 — Springer options

### 6. Protection and Control of Modern Power Systems  (Springer, open access)

* **Fit.** Good, growing reputation, explicitly scoped to protection and control
  of modern grids. Open access; check APC.

### 7. International Journal of Control, Automation and Systems  (Springer)

* **Why.** Published ref. [3].
* **Fit.** Best if the paper is pitched as a **control** contribution: the CBF
  shield with a conformal-tightened safe set, with detection as the front end.

### 8. Neural Computing and Applications  (Springer)

* **Fit.** Would accept the architecture contribution readily, but the power-systems
  framing would be wasted on this readership. Use only as a fallback.

---

## What this revision adds to the pitch

Three results were added or repaired since the first submission draft, and they
change which venue is the best fit.

**1. The attacker-cost measurement is the strongest hook we have.** A perfectly
stealthy attack on IEEE 118-bus needs **8.3 of 304 meters — 2.7 % of the system**.
It is short, concrete, quantitative, and directly actionable for an operator
(it names which substations to harden). Editors at **IEEE TSG** and **IEEE TII**
respond to results with an operational consequence; lead the cover letter with
this rather than with the architecture.

**2. SSC gives the paper a reusable artifact.** The dataset ships per-sample
physics certificates and a datasheet, so referees can re-stratify our results
without rerunning anything. Several of these venues now weight artifact
availability explicitly; say so in the submission.

**3. The closed-loop result is now real.** Three defects (saturated safe set,
proportional-only control, a control-authority model wrong by ~50×) had made the
CBF experiment measure nothing. After the fixes, the shield's benefit resolves
positively on every magnitude metric at n = 120 paired episodes, and the
conformal margin beats the plain shield on all of them. That makes the control
half of the paper defensible, which it previously was not — and it opens
**IJEPES** and **Int. J. Control, Automation and Systems** as genuine options
rather than fallbacks.

---

## Recommended plan

```
Step 1   Submit to IEEE Transactions on Smart Grid.
         Lead: the chi-square measurement + the separator-tree replacement.
         Prepare for a "scale the closed loop" request.

Step 2   On rejection, split the work:
           (a) architecture + localization  -> IEEE Trans. Industrial Informatics
           (b) certificate + CBF + conformal -> IJEPES  (faster, reliable)

Step 3   Keep Protection and Control of Modern Power Systems in reserve
         for an open-access route if speed matters more than tier.
```

**Do not** submit the same manuscript to two of these simultaneously; all four
publishers treat that as misconduct.

---

## What each venue will push back on, and the prepared answer

| Objection | Answer |
|---|---|
| "Only two control areas." | **Answered in v2.** The plant is now an `N`-area interconnection, run at five areas over a ring-plus-chord tie-line graph with staggered inertia and droop. Power conservation holds exactly (`sum_i P_tie,i = 0`, by antisymmetry of `T_ij sin(d_i - d_j)`), and a load step in one area is measured reaching areas with no direct tie-line — propagation that a two-area model cannot exhibit at all. `p.slice_to(2)` still reproduces the base paper's plant exactly, so the replication is not lost. |
| "No real measured data." | Stated in the abstract. The MSU/ORNL dataset is feature-space and carries no multi-area LFC ground truth (`docs/02_LITERATURE.md`, ref. [7]). Named as future work, not claimed. |
| "Detection F1 is saturated — where is the gain?" | That is the point of Fig. 9. Every architecture clears 0.95 on a loud attack; the base paper's 99.78 % is a statement about an easy attack. The claims that discriminate are transfer (Fig. 10), localization cost (Fig. 11) and the certificate (Fig. 13). |
| **"A plain GCN beats you on IEEE 14-bus."** | **Correct, and we report it.** At `N = 14` a 2-layer GNN reaches 57 % of the grid, so the receptive-field bottleneck that motivates SCEPTRE does not bite. The paper must not claim a case14 win. It claims `N`-independent parameters, zero-shot transfer, and `O(k log N)` localization — none of which a GCN provides and none of which survive in a flat model at all. Anticipate this objection *in the abstract* rather than letting a reviewer find it. |
| **"Your FDR exceeds the nominal level."** | **Answered in v2, and the earlier concession was wrong.** The conformal p-value at node `v` tests `H_v`: "no bus in subtree(v) is tampered". We were measuring FDR over **buses** — a different hypothesis. Under `a = Hc` an untampered bus beside a tampered one is not exchangeable with a bus from a clean window, so the per-bus figure inflates for that reason alone, not from miscalibration. v2 reports FDR at **both** levels: the node-level figure is the one Proposition 4 speaks about; the bus-level figure is kept because it is what an operator dispatching crews experiences. |
| **"Your stratified result shows F1 RISING with stealth. Isn't that backwards?"** | No, and this is worth explaining carefully. `σ` measures invisibility to the **χ² residual test**, not to a learned detector. An omniscient attacker's `a = Hc` is *highly structured* — exactly a column combination of `H` — which is why the residual cannot see it and why a pattern-based detector finds it comparatively easy. The two notions of stealth come apart, and demonstrating that is a contribution rather than an anomaly. |
| **"Why should we believe the dataset difficulty wasn't tuned?"** | Because we recorded the alternative and rejected it. A calibration sweep found a harder operating point at which SCEPTRE still learns while GCN and MLP collapse to the trivial floor; adopting it would have flattered this work. Severity instead spans undetectable→trivial, and the aggregate F1 is explicitly labelled a property of that mixture. Cell 15 of the notebook states this. |
| **"Your Proposition 1 is just the standard GNN receptive-field bound."** | **Correct, and we say so in the section itself.** The machinery is textbook — message-passing networks are bounded by their `K`-hop dependency cone. What is unclaimed is the corollary in this domain: a stealthy attack is `a = Hc`, so its signature is a correlation between buses that may be far apart, and the bound then says a shallow MPNN cannot represent it *at all*. No GNN-FDI paper we surveyed states this, and several build detectors below the bound. |
| **"The theory is post-hoc rationalisation of an architecture you already built."** | Then it should not predict its own failures, and it does. On IEEE 14-bus the diameter is 5, so `K = 3` already satisfies the Prop. 1 bound — the theory predicts **no** structural advantage there, and Figure 8 shows SCEPTRE tied with three baselines on that system. A rationalisation would have predicted a win everywhere. |
| "F1 is saturated, so your benchmark cannot discriminate." | v2 additionally reports **TPR at FPR = 1e-2 and 1e-3**. F1 selects the threshold that maximises F1, which no control room operates at; a balancing authority fixes the alarm budget and accepts the recall that follows. That region of the ROC is decided by the hardest attacks, not the average one, and is not saturated. |
| "Is the safe set standard?" | No — it is calibrated to the 88th percentile of the *undefended* plant's own excursion distribution, because a bound the plant violates on 68 % of steps makes the barrier inactive-by-saturation and measures nothing. The calibration procedure is in the notebook and the resulting numbers are printed. |
| "The AC surrogate is not a real power flow." | Cell 9 reports the verification against held-out Newton–Raphson solutions, with `R²` for both the quadratic surrogate and the best linear model. |
| "Why not just use a GNN?" | Because a pre-registered test said no (`docs/03_PREREGISTRATION.md`, n = 5 seeds, failed in 4/4 cells) and the receptive-field measurement in Fig. 4(d) explains why. |
| "Is the attacker–defender pair converged?" | No, and the paper says so — the equilibrium gap is reported and is non-zero. |

---

## Conference option (for early feedback)

If a faster review cycle is wanted before the journal submission: **IEEE PES
General Meeting** or **IEEE SmartGridComm**. Both would take the architecture +
localization result as a 5–6 page paper, and neither blocks a later, extended
journal submission — but confirm each venue's current prior-publication policy
before relying on that.
