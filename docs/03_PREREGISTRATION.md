# Pre-registration — the topology test that decided SCEPTRE's architecture

> **Status: CLOSED. The registered claim FAILED at n = 5 seeds and was dropped.**
>
> This file is reproduced unchanged from the predecessor project
> (`PREREGISTRATION.md` in the archived GA2Shield predecessor, kept outside
> this repository). It is the reason SCEPTRE uses a
> recursive separator tree instead of graph attention, and it is cited from the
> README and from Part 1 of the notebook.
>
> The receptive-field diagnosis recorded below is reproduced as **Figure 4(d)**
> of `SCEPTRE.ipynb`, extended to case57 and case118 (15.0 % and 9.1 % two-hop
> reachability respectively).

---


Written **before** running the experiment, and committed unchanged whatever
the outcome. The point of writing it down is that "we changed the
architecture until it worked" and "we diagnosed a flaw and fixed it" produce
identical-looking code diffs, and only this file distinguishes them.

---

## What happened first (the result being revisited)

Three seeds, IEEE 14-bus, 12-step window: the hard-masked shared Graph-Mamba
lost to an unmasked flat-attention baseline on both tasks, paired
ΔF1 = −0.0266 (t = −4.70), ΔRMSE = +0.0057 (t = +3.67). A follow-up sweep
over {case14, case30} × {w=12, w=24} × 2 seeds reproduced the direction in
all four cells.

## The diagnosis

Measured, not guessed:

| system | graph diameter | GraphMix layers | fraction of grid each bus can see |
|---|---|---|---|
| case14 | 5 hops | 2 | **57%** |
| case30 | 6 hops | 2 | **30%** |
| flat-attention baseline | — | 1 | **100%** |

`GraphMix` masks attention to direct neighbours, so after `layers` rounds a
bus has a receptive field of `layers` hops. Both test systems have a diameter
larger than that. Buses in opposite control areas — precisely the pairs whose
correlation an FDI attack propagating through the tie-line would create —
can never exchange information at all, while the baseline sees the whole grid
in a single layer.

That comparison does not test whether topology helps. It tests whether an
information bottleneck hurts, and that was never in question. Publishing it
as a refutation of topology-aware detection would be refuting a strawman.

## The change

`SpatialBiasMix`: topology enters as a learned additive bias on the attention
logits, indexed by hop distance (Graphormer-style spatial encoding):

    att_ij = q_i · k_j / sqrt(d_h)  +  b[dist(i, j)]

Two properties, both verified in `verify_ga2shield.py`:

1. **No receptive-field confound.** Every bus reaches every other in one
   layer. Distance is expressed as a preference, not a wall.
2. **Strict superset of the baseline.** With `b = 0` the arm *is* unmasked
   flat attention, and `b` is initialised to zero — training starts exactly
   at the baseline. If topology carries signal, the bias can find it. If it
   does not, the model can learn to ignore it.

Property 2 is what makes the follow-up test meaningful in both directions: a
negative result from this arm is a statement about topology itself, not about
a masking choice.

## The pre-committed test

**Primary claim.** `spatial_bias_transformer` beats `flat_attention` on
held-out data.

- Arms differ *only* in the spatial bias. Same trunk, heads, depth, width,
  optimiser, schedule, data, and seeds.
- Paired per seed, ≥ 3 seeds, on the held-out test split.
- **Success:** ΔF1 > 0 with t > 2, or ΔRMSE < 0 with |t| > 2.
- **Failure:** anything else.

**On failure the claim is dropped**, and the paper reports that
topology-aware attention does not improve FDI detection at 14–30 bus scale,
with the receptive-field ablation as supporting evidence that this is not an
implementation artefact.

**One diagnosis, one fix, one test.** No sweeping of bias parametrisations,
bucket counts, layer counts or learning rates in search of a favourable
number. If the first run fails, that is the result.

**The SSM is a separate question** and is not part of this claim. At a
12-step window there is nothing for a state-space model to do that attention
cannot do more cheaply; the linear-vs-quadratic argument only bites at
windows in the hundreds. `spatial_bias_mamba` is run for completeness but no
claim is registered for it here.

## Reported regardless of outcome

- The original hard-mask result, as an ablation row.
- The receptive-field table above, as the explanation for it.
- Both arms' parameter counts, training time and inference cost.

## Command

```bash
python3 run_scale.py --cases case14 case30 --windows 12 24 \
    --arms flat_attention spatial_bias_transformer spatial_bias_mamba \
           shared_graph_mamba \
    --seeds 3 --epochs 8
```

Results: `results/ga2_scale/scale.json`.

---

# OUTCOME (recorded after the run, criterion unchanged)

**Result at n = 3 seeds, case14, w = 12:**

| arm | F1 | area RMSE | params |
|---|---|---|---|
| `flat_attention` (baseline) | 0.8813 ± 0.0268 | 0.1674 ± 0.0105 | 27,544 |
| `spatial_bias_transformer` | **0.8995 ± 0.0121** | **0.1624 ± 0.0095** | 27,568 |

Paired, arm minus baseline:

    dF1    = +0.0182   t = +1.70    (criterion: > 0 with t > 2)
    dRMSE  = -0.0050   t = -0.96    (criterion: < 0 with |t| > 2)

## Verdict: the pre-registered criterion is NOT met.

Both metrics move in the predicted direction, and the bias arm wins on 3 of 3
seeds on F1, but neither statistic reaches t > 2. By the rule written above,
**the claim is not established at n = 3.**

What did change, and is reportable independently of this:

1. The hard 1-hop mask goes from *losing decisively* to the flat baseline
   (ΔF1 = −0.0266, t = −4.70) to the topology arm *leading, inconclusively*
   (ΔF1 = +0.0182, t = +1.70). The original refutation was therefore
   measuring the receptive-field bottleneck, not topology.
2. The receptive-field table (diameter 5–6 vs a 2-hop reach) is a concrete,
   measured explanation for why hard-masked graph attention underperforms on
   small power networks. That stands on its own and does not depend on the
   outcome of the primary test.

## What is permitted next, and what is not

The registration specified "≥ 3 seeds", so a 5-seed run is within scope.
**But adding seeds after seeing a near-miss is exactly the practice that
inflates false positives**, so it is declared here before running:

- The 5-seed run is the *final* test. Its result is reported whatever it is.
- The n = 3 result above is reported alongside it, not replaced by it.
- No further extension. If n = 5 does not clear t > 2, the claim is dropped
  and the paper reports topology-aware attention as *not established* at
  14–30 bus scale, with the receptive-field ablation as the contribution.

Command for the final test:

```bash
python3 run_scale.py --cases case14 case30 --windows 12 24 \
    --arms flat_attention spatial_bias_transformer shared_graph_mamba \
    --seeds 5 --epochs 8 --baseline flat_attention
```

---

# FINAL OUTCOME — n = 5 seeds, case14 + case30, w = 12 and 24

Run on an Apple M4 Pro, 30 cells, `--baseline flat_attention`.

> *Provenance note (added when this file moved into `docs/`):* this line records
> the hardware the CLOSED pre-registration actually ran on and is left unchanged,
> because editing a completed pre-registration would defeat its purpose. All
> **new** work in this repository targets Windows 11 + RTX 5060; see
> `06_REPRODUCIBILITY.md`. The receptive-field diagnosis is hardware-independent
> and is recomputed from scratch in `SCEPTRE.ipynb`.

## Primary claim: FAILS in every cell. The claim is dropped.

`spatial_bias_transformer` minus `flat_attention`:

| system | window | dF1 | t | dRMSE | t | verdict |
|---|---|---|---|---|---|---|
| case14 | 12 | +0.0066 | +0.65 | −0.0063 | −1.42 | fails |
| case14 | 24 | +0.0164 | +1.33 | −0.0025 | −1.82 | fails |
| case30 | 12 | −0.0031 | −0.28 | +0.0006 | +0.48 | fails |
| case30 | 24 | +0.0034 | +0.39 | −0.0006 | −0.12 | fails |

Criterion was ΔF1 > 0 with t > 2, or ΔRMSE < 0 with |t| > 2. Nothing reaches
it. **Topology-aware attention is not established as beneficial for FDI
detection at 14–30 bus scale.** As committed, no further extension is run.

## Why writing the criterion down first mattered

At n = 3 the headline cell (case14, w = 12) read ΔF1 = +0.0182, t = +1.70 —
close enough that it would have been very tempting to call it a trend. At
n = 5 the same cell reads **+0.0066, t = +0.65**. The effect shrank by two
thirds as seeds were added.

That is textbook regression to the mean, and it is exactly the outcome an
unregistered study would have mis-reported: stop at n = 3, describe t = 1.70
as "approaching significance", and publish a claim that five seeds do not
support.

## What survives, and is worth reporting

The **hard-mask ablation** is the real finding. `shared_graph_mamba`
(1-hop mask, 2 layers) versus unmasked attention:

- case14, w = 24: ΔRMSE = **+0.0114, t = +5.55** — significantly *worse*
- case30, w = 12: ΔRMSE = **+0.0038, t = +2.46** — significantly *worse*
- case14, w = 12: ΔF1 = −0.0327, t = −1.96 — worse, borderline

Replacing the hard mask with the distance bias removes that deficit
everywhere (all |t| < 2, i.e. parity) without producing a gain.

**The publishable statement:** on power networks whose diameter exceeds the
model depth, hard k-hop attention masking measurably degrades FDI detection,
because each bus sees only a fraction of the grid (57% at case14, 30% at
case30, against 100% for unmasked attention). A distance-biased formulation
removes the penalty and recovers parity, but topology-awareness confers no
measurable advantage at this scale. Papers reporting gains from masked graph
attention on small test systems should check their receptive field against
the graph diameter first.

## Data-quality note (not a get-out; recorded for transparency)

Seed 0 of `case30, w = 24` is anomalous for both attention arms
(F1 = 0.621 flat, 0.636 bias, versus 0.93–0.96 for seeds 1–4) and trained in
20 s against 222–3313 s for the other seeds. That cell appears
under-trained. It inflates the case30/w24 standard deviation to 0.147 and is
the reason that cell resolves nothing.

It does **not** change the verdict: the three other cells are unaffected and
all fail the criterion independently. Re-running seed 0 is worthwhile before
the numbers go into a paper, but it cannot rescue the claim.

The wildly varying training times for `flat_attention` at case30/w24
(20 s → 3313 s across seeds) also warrant a look — likely thermal throttling
or resumed-checkpoint accumulation rather than anything scientific, but the
timing column should not be quoted as a cost comparison until it is
explained.
