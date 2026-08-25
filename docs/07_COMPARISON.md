# Comparison tables — ready to paste into the manuscript

Six tables. Each is written so it can be dropped into a paper with only the
caption reworded, and each states what it is *not* evidence of, because the most
common way a related-work table misleads is by putting numbers from different
datasets in the same column.

Every reference here is verified in [`02_LITERATURE.md`](02_LITERATURE.md).
Entries marked **[C*]** are prior art that **narrows** one of our claims; they
are in the tables deliberately, not reluctantly.

> **The rule that governs Tables IV and V.** A number another paper reports on
> its own data is *not* comparable to a number we measure on ours. Different
> attack generators, different severities, different splits. Tables IV and V keep
> the two kinds of number in physically separate columns and never compute a
> difference across that boundary. If you take one table from this file into a
> paper, take this warning with it.

---

## Table I — Related work at a glance

The survey table. `D` = detection, `L` = localization, `R` = repair/imputation,
`C` = closed-loop control.

| # | Reference | Year | Venue | Core method | Tasks | Test systems | Attack model |
|---|---|---|---|---|---|---|---|
| [1] | Cui, Wu & Zhang — **base paper** | 2025 | IEEE Trans. Smart Grid | MSA3E: semi-supervised adversarial AE + multi-head attention; LSTM repair; MARL-A3C | D · R · C | IEEE 14 | Multiplicative bias, `m_v = 1` |
| [C1] | Liu, Ning & Reiter | 2011 | ACM TISSEC | Analytic construction of `a = Hc` | — | IEEE 9/14/30/118/300 | Null-space, omniscient |
| [C2] | Boyaci *et al.* | 2022 | IEEE Trans. Smart Grid | ARMA graph filters, GNN | D · L | IEEE 14/118/300 | Stealth FDI |
| [C3] | Mohamed, Kundur & Khalaf | 2023 | arXiv (Univ. Toronto) | Control barrier function shield | C | LFC two-area | Signal compromise |
| [C4] | Ameli *et al.* | 2018 | IEEE TIFS / Trans. Power Syst. | Stochastic unknown-input estimator | D | LFC / AGC | Stealthy FDI on AGC |
| [4] | Qu *et al.* | 2024 | Applied Energy | Spatio-temporal graph wavelet CNN | L | IEEE 14/118/300 | Dummy-data injection, incomplete topology |
| [2] | Feng *et al.* | 2024 | IEEE Trans. Instrum. Meas. | Adversarial dual autoencoder + graph representation | D | CPPS | FDI |
| [6] | Hu *et al.* | 2024 | IEEE Trans. Ind. Informatics | Model-based resiliency framework | C | Interconnected LFC | IoT **faults** (not adversarial) |
| [B1] | ST-Transformer | 2023 | Expert Syst. Appl. | Full attention + learned positional embedding | D | IEEE bus systems | FDI |
| [B2] | GraphKAN | 2025 | Scientific Reports | Graph attention + Kolmogorov–Arnold | D | Smart-grid intrusion | Intrusion/FDI |
| [B3] | KAN-AGC | 2025 | arXiv | Interpretable spline network | D | AGC | AGC cyberattack |
| [C6] | Adaptive RNN retraining | 2025 | SICE JCMSI (T&F) | RNN + retraining on distribution shift | D | IEEE 68 | FDI on LFC |
| — | **SCEPTRE (this work)** | 2026 | — | Nested-dissection recursive encoder + hierarchical conformal + CBF | **D · L · R · C** | **IEEE 14/30/57/118** | **Null-space with swept attacker knowledge `eps`** |

**What this table is evidence of:** coverage and scope. **What it is not:**
quality. No metric appears in it, deliberately.

---

## Table II — Threat-model comparison

This is the table that carries the paper's framing argument, and it is the one
no surveyed paper provides. Axes are defined in §4 and rendered as Figure 19.

| Reference | Stealth `sigma` | Attacker knowledge `eps` | Cost `mu` reported? | Swept, or single point? |
|---|---|---|---|---|
| [1] Cui *et al.* 2025 | Well below 1 — measured here at up to `1.8 x 10^4` times a χ² threshold | `0` (exact `H`) | No | Single point |
| [C1] Liu *et al.* 2011 | Exactly `1` by construction | `0` | Partially — studies constrained-meter cases | Analytic, not swept |
| [C2] Boyaci *et al.* 2022 | Stealth assumed | `0` | No | Single point |
| [C4] Ameli *et al.* 2018 | Stealth assumed | `0` | No | Single point |
| [4] Qu *et al.* 2024 | Not characterised | Incomplete **topology**, not perturbed parameters | No | Two topology conditions |
| [B3] KAN-AGC 2025 | Not characterised | `0` | No | Single point |
| **SCEPTRE** | **Five strata spanning `0.7` to `1.0`** | **Swept `eps in [0.04, 3.0]`** | **Yes — per sample** | **Swept on all three axes** |

**The one-sentence version for the abstract.** Of the surveyed threat models,
none sweeps attacker knowledge, and none reports attacker cost; `sigma` is
therefore pinned at a single value in nearly all of them, and a detector
evaluated under such a model has been tested at one point of a three-dimensional
space.

**Caveat, and it matters.** Rows for other papers encode *their stated
assumptions*, not a re-measurement of their systems. A paper that never mentions
`eps` is placed at `eps = 0` because its equations assume it. Figure 19 marks
these as hollow points for exactly this reason.

---

## Table III — Capability matrix

Capabilities are structural: a method either has one or it does not, and the
answer follows from the architecture rather than from a training run. This is
the table that makes the transfer result legible.

| Method | Params independent of `N` | Zero-shot to new bus count | Localization | Localization cost | Multiplicity control | Closed loop | Attack certificate |
|---|---|---|---|---|---|---|---|
| MSA3E [1] | ✗ | ✗ | ✗ | — | ✗ | ✓ | ✗ |
| ST-Transformer [B1] | ✗ | ✗ | ✓ | `O(N)` | ✗ | ✗ | ✗ |
| Graphormer | ✓ | ✗ (hop-bias table) | ✓ | `O(N)` | ✗ | ✗ | ✗ |
| GCN / GAT | ✓ | ✓ | ✓ | `O(N)` | ✗ | ✗ | ✗ |
| Boyaci GNN [C2] | ✓ | Not claimed | ✓ | `O(N)` | ✗ | ✗ | ✗ |
| Qu *et al.* [4] | ✓ | Not claimed | ✓ | `O(N)` | ✗ | ✗ | ✗ |
| KAN [B2, B3] | ✓ | ✓ | ✓ | `O(N)` | ✗ | ✗ | ✗ |
| BiLSTM / S4 / MLP | ✗ | ✗ | ✓ | `O(N)` | ✗ | ✗ | ✗ |
| CBF shield [C3] | n/a | n/a | ✗ | — | ✗ | ✓ | ✗ |
| **SCEPTRE** | **✓** | **✓** | **✓** | **`O(k log N)`** | **✓ (BH per family)** | **✓** | **✓ (per sample)** |

Read the last row against the third and fourth columns together. Several
baselines are `N`-independent, and several localize. **SCEPTRE's claim is the
conjunction**, plus the cost class and the error guarantee — not any single
column.

---

## Table IV — Reported results, kept separate from ours

Numbers each paper reports **on its own data and its own attack generator**.
They are here for context and must never be differenced against our column.

| Reference | Reported metric | Value | System | Comparable to ours? |
|---|---|---|---|---|
| [1] Cui *et al.* 2025 | Detection accuracy | 99.78 % | IEEE 14 | **No** — attack is `1.8 x 10^4` χ² thresholds, i.e. far louder |
| [4] Qu *et al.* 2024 | F1, strong attacks | 0.9767 / 0.9815 / 0.9843 | IEEE 14 / 118 / 300 | **No** — different attack generator and severity distribution |
| [B3] KAN-AGC 2025 | Detection accuracy | 95.97 % | AGC | **No** — different plant and attack |
| [C2] Boyaci *et al.* 2022 | Detection + localization | Not extracted | IEEE 14/118/300 | **No** — and rather than guess, we left it blank |

**Why "not extracted" appears above.** Where a number was not confirmed from the
source, this file says so instead of supplying a plausible one. An unverified
number in a comparison table is worse than a blank.

**The defensible comparison** is Table V: every architecture reimplemented and
run on **one** benchmark, one attack generator, one split, three seeds.

---

## Table V — Head-to-head on the SSC benchmark

Generated from `outputs/sceptre_results.json` by **Part 18** of the notebook; the live
values are in [`../REPORT.md`](../REPORT.md) §6.1 and §6.3. Template:

| Model | Detection F1 (14-bus) | Localization F1 | Zero-shot F1 @ 118-bus | Params @ 14 | Params @ 118 |
|---|---|---|---|---|---|
| SCEPTRE (RSTE) | — | — | — | — | *unchanged* |
| MSA3E (base paper) | — | — | **n/a** | — | *grows with N* |
| ST-Transformer | — | — | **n/a** | — | *grows with N* |
| Graphormer | — | — | **n/a** | — | — |
| GAT / GCN | — | — | — | — | *unchanged* |
| S4 / KAN / BiLSTM / MLP | — | — | mixed | — | — |

Three things to say about this table in the manuscript, in this order:

1. **SCEPTRE is not expected to win the 14-bus aggregate**, and where it does not,
   the paper says so in the same paragraph as the number. At `N = 14` a 3-layer
   message-passing stack already reaches most of the grid, so the
   receptive-field bottleneck that motivates a separator tree does not bind.
2. **`n/a` in the transfer column is not a poor score — it is an undefined one.**
   A model whose positional table has `N` rows cannot be evaluated at a different
   `N` at all. That is a capability gap, not a percentage.
3. **The discriminating results are Tables II and III**, plus §6.4 (localization
   cost) and §6.7 (the random-tree ablation).

---

## Table VI — Which contribution each competitor already covers

The self-assessment. If a reviewer builds this table themselves, it should look
like the one we published.

| Contribution | Closest prior art | What it already does | What is left for us |
|---|---|---|---|
| C1 RSTE | none found | — | Nested dissection as a neural inductive bias; parameters independent of `N` |
| C2 HALO | **[C2] Boyaci 2022** | GNN detect **and** localize, per-bus | `O(k log N)` cost class; BH-controlled stopping per sibling family |
| C3 NL-LFC | [6] Hu 2024 | Model-based resiliency under IoT faults | Eight composed nonlinearities, each isolated and measured |
| C4 SSC dataset | [C1] Liu 2011 | Analytic null-space construction | Per-sample `(sigma, mu, eps)` certificates shipped with the data |
| C5 CBF shield | **[C3] Mohamed 2023** | CBF shield for LFC under attack | Plant-calibrated safe set; conformally tightened margin |
| C6 Self-play | — | — | Reported with a non-zero equilibrium gap, i.e. not claimed converged |

Two of six were narrowed by the search in `02_LITERATURE.md` Part C. Publishing
that fact is not a weakness; failing to notice it before a reviewer does would
be.

---

## LaTeX versions

Every table here is also emitted as IEEEtran-ready LaTeX:

```bash
# Part 18 of SCEPTRE.ipynb writes this automatically; no separate script
```

writes `outputs/tables.tex` — `booktabs` rules, `\ding{51}`/`\ding{55}` marks,
`table*` two-column floats, `threeparttable` footnotes, and commented BibTeX
stubs for every citation key used. Tables I–IV and VI are static; **Table V is
generated from `outputs/sceptre_results.json`, so re-run `make_tables.py` after
the notebook** or it will carry stale numbers.

For `elsarticle`, move each `\caption` below its `tabular`; the bodies are
unchanged. In a one-column class, change `table*` to `table`.

Required preamble:

```latex
\usepackage{booktabs}
\usepackage{pifont}
\usepackage{multirow}
\usepackage{threeparttable}
```

---

## How to use these tables

* **Journal paper.** Table I and Table II in §2; Table III in §2 or §6; Table V
  in §6. Table IV usually becomes prose, not a table. Table VI belongs in the
  response letter rather than the paper.
* **Conference paper (6 pp.).** Table II and Table III only. They carry the
  argument on their own.
* **Presentation.** Table III as one slide; Figure 19 instead of Table II.
* **Before submitting.** Re-run the searches in `02_LITERATURE.md` Part D. The
  most recent entries here are months old, and the B6 retraction in that file is
  a live example of what happens when a citation is not re-checked.
