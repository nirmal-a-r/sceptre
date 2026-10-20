# Oracle-Referenced, Audited Benchmark for Stealthy FDI in State Estimation and LFC (pilot)

One notebook, `notebooks/FDI_Oracle_Benchmark.ipynb`, implements the research proposal end to end: it builds a randomized power-grid benchmark from scratch (own Newton power flow and WLS estimator, validated against pandapower), measures every detector against a hardness-indexed Neyman-Pearson oracle, audits the benchmark and the existing SCEPTRE dataset for shortcuts, tests leave-grid-out transfer, converts detection into frequency damage with a two-area LFC model, tries three methods and an adaptive attacker, and scores 14 pre-registered hypotheses mechanically. 25 figures, 97 cells, no errors (v1); v2 below adds Part 15 and 5 more figures.

## Layout

| Path | Content |
|---|---|
| `notebooks/FDI_Oracle_Benchmark.ipynb` | The notebook, **executed at full budget** (outputs included) |
| `figures/` | The 25 figures it draws |
| `outputs/results.json`, `outputs/scorecard.csv` | Every headline number and the hypothesis verdicts |
| `docs/FDI_Detection_Research_Proposal.docx` | The proposal the notebook implements |
| `reference/sceptre/` | Copied from SCEPTRE: case14 dataset (train/test/calib), reported results, datasheet, docs, original requirements |
| `requirements.txt`, `ENVIRONMENT.md`, `setup_env.ps1` | The SCEPTRE environment contract, adapted (CPU-only; PyTorch optional) |
| `LICENSE`, `CITATION.cff`, `.gitignore` | Project housekeeping (CITATION author is a placeholder) |

## Run it

```powershell
.\setup_env.ps1
.\.venv\Scripts\Activate.ps1
jupyter lab notebooks\FDI_Oracle_Benchmark.ipynb      # Run All; ~1 h at full budget from scratch, ~5 min with FDI_BUDGET=quick
```

## Version 2: extended attack families (executed at full budget)

`notebooks/FDI_Oracle_Benchmark_v2_extended.ipynb` is the executed v2 notebook (117 cells, 30 figures, no errors, CPU only; no GPU was available or needed). It adds **Part 15** (extended attack families) and moves the scorecard to **Part 16**. The v1 notebook is kept for comparison. New in Part 15: temporal attacks (ramp, pulse, persistent, replay) on a labelled Gaussian-copula surrogate for load dynamics; imperfect-stealth attacks; limited-meter and outage-masked attacks on the real AC generator; attack directions at equal temporal hardness; a family-aware Bayes reference that splits the oracle gap; and hypotheses H15 to H18.

| New hypothesis | Verdict |
|---|---|
| H15 learned temporal detector beats best clean-only temporal test by 0.03 | **not supported** (0.542 vs 0.549) |
| H16 masking costs a standard detector >= 0.10 AUROC | **not supported** (0.018; detectors are near chance either way) |
| H17 detector rankings do not transfer across families | **not supported**: cross-family Spearman is low (0.07) but within-family reliability is only 0.69, so the detectors are not distinguishable enough to rank |
| H18 oracle direction-independent (<= 0.03) but learned detector varies (>= 0.05) | **not supported**, narrowly: learned spread 0.11 to 0.14 passes, oracle spread 0.030 to 0.040 misses its bound |

What the run does show, with caveats:

* **Oracle gap split.** For coordinate-direction attack families the known-shift oracle scores 0.91 to 0.96 and the family-aware Bayes reference 0.76 to 0.87, while learned and clean-only detectors reach 0.49 to 0.69. Not knowing the direction costs about 0.09 to 0.15 (unavoidable); the rest, about 0.14 to 0.28, is closable in principle. The Bayes reference assumes the family structure and hardness prior are known, which a deployed detector would not.
* **Leaky attacks.** From a residual leak of about 4 (non-centrality per frame) upward, a simple chi-square-based detector passes the state-only oracle (AUROC 0.86 to 1.00 vs 0.89), so the oracle framing holds only up to the stealth boundary.
* **Limited meters.** The oracle stays near 0.82 to 0.86 from 10 to 88 meters touched, as the theory says; every detector stays between 0.49 and 0.59.
* **Replay.** Near chance for all detectors, no oracle.
* **Masking.** The oracle itself drops by 0.24 when the attack is hidden in an outage, so part of the difficulty is the harder negative class, not only the detector.

**Novelty check (shallow web search, abstracts only):** every attack family already exists in the literature, so none is a contribution on its own. Not found in my search: scoring every family against its own optimal detector on one hardness scale, the unavoidable-versus-closable split, and a cross-family ranking test. These are candidates, not established novelty; a proper search is needed before claiming them. Temporal results rest on an assumed surrogate, not a simulated load trajectory.

## Version 3: structure-aware detectors (Part 16, executed at full budget)

`notebooks/FDI_Oracle_Benchmark_v3_extended.ipynb` adds **Part 16** (structure-aware detectors) and moves the scorecard to **Part 17** (125 cells, 34 figures, no errors, CPU only). New methods: a **damage-coordinate test** (the standardised estimate of the tie-line flow an attacker wants to forge), a **scan GLRT** over bus neighbourhoods, a **learner given those statistics as features**, and **conformal damage control using the damage-coordinate score**. None of the ideas is new on its own (GLRT detectors for false data injection, damage-optimal attacks on load-frequency control and threshold optimisation all exist in the literature); only the pairing is a candidate contribution, and it needs a proper literature check.

| Hypothesis | Verdict |
|---|---|
| H19 damage-coordinate test within 0.03 AUROC of the oracle on tie-line attacks | **not supported** (0.63 vs 0.80 on ieee14; 0.55 vs 0.77 on syn28) |
| H20 damage-coordinate test loses <= 0.01 AUROC to the adaptive attacker and sits at its analytic floor | **supported with a caveat**: loss 0.000, observed within 0.006 of the floor; the attacker is weak |
| H21 scan GLRT beats Hotelling by 0.05 on sparse and local attacks | **not supported** (differences within +/-0.013) |
| H22 structured features add 0.03 AUROC to the learner | **not supported** (+0.018 and +0.023; 8 to 11% of the oracle gap) |
| H23 damage-aware score pays at most 70% of Hotelling's false alarms at the same damage budget | **supported** (ratio 0.02; see caveat) |

What this shows, with caveats:

* **The metric matters.** On AUROC the damage-coordinate test is far below the oracle, because AUROC ranks attacks by how *hard* they are, and a stale-Jacobian attacker realises large hardness in harmless directions that the test rightly ignores. On **damage prevented at 1% false alarms** it does best: 0.59 (ieee14) and 0.62 (syn28), against 0.38 and 0.41 for Hotelling and 0.22 and 0.14 for the summary-view GBDT. The structured learner matches it on ieee14 (0.59) but not on syn28 (0.45).
* **It cannot be steered around.** Against the white-box adaptive attacker the damage-coordinate test loses nothing, as its analytic floor predicts; the supervised detectors lose 0.004 to 0.009.
* **Damage budgets get cheap.** With the damage-coordinate score the same damage budget costs about 1% false alarms on natural windows instead of 22 to 39%. The thresholds also stay quiet on outages (2%), but Hotelling's alarm on 88 to 98% of outages, so that comparison is partly about Hotelling being a poor score for this job.
* **Limits.** The test only sees attackers whose goal is the protected quantity; the generator's tie-line direction is exactly the test's direction, so part of the result is built in; and the scan GLRT did not help here, which I have not diagnosed.
* **Where it leaves the oracle gap.** Generic detection of hard attacks is still close to chance. The structured pieces help where the attacker's goal is known; they do not close the gap in general.

## Version 4: environment zoo, realistic profiles, nonlinear LFC, deep detectors, adversarial attacker, ensemble (Parts 17-20, executed at full budget)

Notebook: `notebooks/FDI_Oracle_Benchmark_v4_executed.ipynb` (150 cells, 43 figures, no errors, 49 min on a CPU-only cloud machine; **no GPU was used**: the notebook prints `device: cpu`, and the same code picks CUDA automatically on a GPU machine; see `ENVIRONMENT.md` and `setup_nlpenv.ps1`).

What was added, and what came out (all numbers are from `outputs/results.json` / `outputs/scorecard.csv`; the scorecard now has 34 hypotheses, H24-H34 new, H9/H10 now run):

* **Environment zoo (Part 17).** IEEE 118, CIGRE MV, Oberrhein, and SimBench HV-mixed / MV-rural / MV-semi-urban / MV-commercial. SimBench grids are driven by real yearly 15-minute load and DER profiles through our own AC power flow, validated against pandapower (worst voltage difference 2.8e-6 p.u., H24 supported). Radial grids have no N-1 outages, so a feeder surge / DER trip plays the hard-negative role. Findings: a detector trained on the 8-grid IEEE/synthetic family has almost no advantage over Hotelling on the new topologies (+0.03 vs +0.07 on unseen IEEE/synthetic; the 0.036 extra loss is below the pre-registered 0.05, H25 not supported); training on the *other zoo environments* beats it by +0.046 on the held-out one (H26 supported, 5 of 7). The best detector closes 18-72% of the oracle gap, and the larger the grid, the smaller the share (Spearman -0.79 with state dimension, Figure 42C).
* **Nonlinear LFC (Part 18).** Dead band, rate limit, valve/integrator saturation and measurement delay. The linear plant over-states the damage of the largest attacks by an order of magnitude (H27) because the plant cannot move more than its headroom; an oscillating bias at the plant's resonance (0.08 Hz) does 3.7x the damage of a constant bias of equal amplitude (H29), so the constant-bias damage ceiling of Part 12 is *not* a ceiling for a time-varying attacker. The detector ranking by damage prevented is **not** robust to the plant (Spearman 0.08 and 0.68, H28 not supported): damage-based rankings in Part 12 should be read as plant-specific.
* **Deep detectors (Part 19).** A 32-parameter physics-structured head (learned positive quadratic in the eigen-coordinates plus the damage coordinate) is far better than a 12k-parameter MLP at large hardness (0.90 vs 0.75 on IEEE 14), so H9 as pre-registered ("within 0.02") fails in the direction that favours the structured head. The normalizing-flow one-class detector is indistinguishable from Hotelling (largest advantage +0.008, H10 not supported). Both are small models trained on at most 1,800 windows; this is not a state-of-the-art deep model comparison.
* **Gradient white-box attacker (Part 19).** At fixed hardness and damage, autograd takes 0.20 AUROC from the MLP versus 0.09 for the best-of-K random search of Part 14 (H30 supported). The damage-coordinate test loses exactly 0, Hotelling 0.005, and the structured head 0.015. Adversarial training against the same attacker recovers 86% of the MLP's loss at no cost on non-adaptive attacks (H31 supported); this was judged against the same attacker family used in training, so it is not a general robustness claim.
* **Ensemble and cross-environment (Part 20).** A Cauchy-combined conformal ensemble of eight detectors beats the best single detector on the worst attack family by only +0.007 / +0.010 (H32 not supported), and its false-alarm rate at nominal 1% is 2.1% on IEEE 14 (H33 not supported, marginal) and 1.0% on syn28. Detector rankings agree moderately across environments (0.59), but the within-environment reliability is only 0.63, so the agreement number is **not interpretable** (H34): with AUROCs this close to chance, rankings are mostly noise.

**Honest summary of v4.** The strongest results are the ones about *structure and attackers* (the damage-coordinate test and the quadratic head are essentially immune to a gradient attacker that destroys a plain network; adversarial training works for the plain network), the *plant sensitivity* (damage numbers and ceilings depend heavily on plant nonlinearity and time profile), and the *environment diversity* result (new topologies are not covered by IEEE/synthetic training). The weakest are the generic learned detectors, which remain near chance at hardness below 2 in every environment, and the ensemble, which does not help. Everything is simulation; SimBench profiles are synthetic-but-validated, not utility SCADA.

**Prior art (do not claim these as new).** Adversarially resilient deep FDIA detectors and adversarial training (arXiv:2102.09057), physics-informed networks for FDIA-robust state estimation without adversarial training (arXiv:2604.22784), FDI on AGC/LFC with nonlinearities and mitigation (Khalifa University work; Tan et al.), AC-model deep detectors (PowerFDNet, arXiv:2207.10805), Cauchy p-value combination (Liu and Xie 2020), SimBench itself (Meinecke et al. 2020). The contribution claimed is the oracle-referenced, audited measurement framework and the specific findings above, not any of these building blocks.

## Version 5: baselines, a threat-library detector, heavier and more varied data (prepared; the full and heavy budgets have NOT been run)

`notebooks/FDI_Oracle_Benchmark_v5.ipynb` is the v4 notebook plus Parts 21 to 27, with the scorecard moved to **Part 28** and hypotheses **H35 to H45** added. The notebook has no stored outputs.

**What was checked, and what was not.** The whole notebook was executed once end to end at `FDI_BUDGET=quick` (500 windows per split, 600-attack library) on a 2-core CPU in the cloud: it ran to the end in 19 minutes without a crash, including the parallel data generation. That run exposed and fixed a division by zero in the damage cell and a variable that was reused between the H22 and H44 rows. The `full` and `heavy` budgets and the GPU path have **not** been run. Quick-budget numbers are noisy and must not be reported.

**What changed since the first v5 draft**

* **GPU.** All neural parts (Parts 19, 21, 22 and 23) select CUDA when it is visible, with TF32 enabled; library scoring (a 20,000-member logsumexp) runs on the GPU in float64. `FDI_REQUIRE_GPU=1` (set automatically by `run_v5.ps1`) stops with an explanatory error instead of silently falling back to the CPU. The simulators themselves (Newton power flow, WLS estimation) are NumPy code and cannot use a GPU; that is why data generation is now parallel instead (next item). The networks are small, so the GPU shortens training, not the simulation.
* **Heavy data, in parallel.** Dataset blocks are generated in independent chunks on `FDI_JOBS` CPU workers (default: all cores but one; the data do not depend on the worker count). New budget `heavy`: 3,600 train and test windows, 1,500 calibration windows, 5 seeds, and a **20,000-attack threat library per environment** (6,000 at `full`, 600 at `quick`). Caches are renamed (`datasets_v5_*`, `zoo_v3_*`, `lib_v5_*`) so older caches are not reused.
* **More varied attacks.** Four new attack families join sparse, dense and lfc: `cluster` (a graph neighbourhood), `area` (one area), `vmag` and `angle` (voltage-only and angle-only manipulation of all buses). Each is scaled to a chosen hardness like the others.
* **Equal-data fairness.** The threat library is more data than the 1,800 labelled windows the baselines had, so three supervised baselines (`gbdt_big`, `dnn_big`, `t_mlp_big`) are trained on the same simulated block and compete for "strongest baseline". The label-budget test (H39) gives library and learner exactly the same windows.
* **A better head.** The stack now also fits its head on a disjoint block of simulated windows, and `tllr_plus` / `tllr_gbdt` add one library score per attack family (H45).
* **Sceptre-environment helpers.** `run_v5.ps1` runs the notebook headless in the active environment; `papers/download_papers.py` downloads the compared papers (see below).

**What a development experiment on separate data suggested (not a result of the notebook).** On IEEE 14, a 28-bus synthetic grid and IEEE 30 (each with its own dev test set, 1,800 windows), the gradient-boosted head on library scores beat the equal-data GBDT by about +0.024, +0.025 and +0.008 AUROC in the hard region (hardness 0.5 to 2). Every method sat between 0.55 and 0.61 there, against an oracle of about 0.75; with only natural clean windows as negatives the numbers were 0.52 to 0.56. In plain words: **expect a small gain over the strongest baseline, not a large one.** The quick smoke test, at a tenth of that data, showed the method behind the best baseline in most environments. Most of the oracle gap is the price of not knowing the attack direction, and the Bayes reference in Part 22 exists to show how much.

**How to run (Windows, in your sceptre environment)**

```powershell
# 1. activate the sceptre environment (venv: ..\SCEPTRE\.venv\Scripts\Activate.ps1, or conda activate sceptre)
.\setup_nlpenv.ps1                  # installs what is missing (pandapower, simbench, joblib, ...); keeps an existing CUDA torch
python -c "import torch; print(torch.cuda.is_available())"     # must print True for the GPU path
.\run_v5.ps1 quick                  # smoke test first
.\run_v5.ps1 full                   # reported budget
.\run_v5.ps1 heavy                  # larger data and library (longest)
```

Add `-AllowCpu` to run without a GPU, `-Jobs 8` to set the CPU workers. Run times were not measured on your machine; the quick budget took 19 minutes on 2 CPU cores here, so `full` should take a few hours and `heavy` longer, mostly in simulation. Open the notebook in Jupyter instead of using the script if you prefer: set `$env:FDI_BUDGET = "full"` first.

**Papers.** The sandbox used to build this cannot reach arXiv, so no PDFs were downloaded. `papers/download_papers.py` fetches 18 of the papers cited in Part 27 (17 arXiv ids plus the Liu-Ning-Reiter report) into that folder and writes `papers/INDEX.md`; run it once on a machine with internet access: `python papers\download_papers.py`. SCEPTRE's own report (`SCEPTRE_Report.docx`) is in the SCEPTRE folder, not copied here.

**What can go wrong, said in advance.**

* TL-LR may not beat the strongest baseline (H36), or may win only because the library comes from the same simulator as the test attacks. Part 23 is designed to expose that, and a failure stays a failure in the scorecard.
* The Part 21 baselines are protocol-faithful reimplementations of families, not the authors' code.
* The nonlinear-plant ranges in Part 24 and the simulator sweep ranges are my assumptions.
* No measured utility data is used. Part 25 reaches SCEPTRE (which the Part 9 audit found easy) and anything you place in `data/external`.
* The threat-library likelihood ratio is a classical construction (a Bayesian mixture test); no FDIA paper with the same combination was found, which is not proof that none exists. Repeat a targeted search before submission.

## What the full-budget run found (version 1) (read this before writing anything up)

| Hypothesis | Verdict |
|---|---|
| H1 SCEPTRE case14 is shortcut-solvable (trivial within 0.01 AUROC of best deep) | supported (gap +0.001) |
| H2 learned detectors sit >= 0.05 below the oracle at hardness 1-4, gap shrinking with data | supported with a caveat: the oracle knows the attack shift, so part of the gap is unavoidable |
| H3 Hotelling beats clean-only ML one-class detectors | **not supported** (all near 0.5 in the hard region) |
| H4 AUROC ranking differs from damage ranking | supported (Spearman 0.32 and 0.20), on a linear LFC stand-in |
| H5 supervised detectors lose more than Hotelling to an adaptive attacker | supported (0.058 vs 0.003 AUROC), weak attacker |
| H6 hardness-stratified training | **not supported** (4% gap closure) |
| H7 oracle distillation | **not supported** (slightly negative) |
| H8 conformal risk control, per-split coverage >= 95% | **not supported** (72-83% of splits; mean risk is under budget) |
| H9, H10 | not run (need the optional deep stack) |
| H11 trivial-detector gap > 0.02 in hard bins of the rebuilt benchmark | **not supported** (passes on ieee14, fails on syn28) |
| H12 loss on unseen grids >= 0.05 | **not supported** (0.016) |
| H13 broad grid family beats a single IEEE case | supported, but small (+0.008) |
| H14 hard negatives cut outage false alarms | supported (ratio is a lower bound; the "with" rate is near zero) |

Plainly: the measurement framework works and the benchmark audit is useful, but the three methods did **not** beat their baselines, and in the hard region every detector, including the clean-only test, is close to chance at these label budgets. Whether that is a publishable result depends on framing and on a literature check at submission time. Nothing here guarantees acceptance anywhere.

## Limits

Simulation only (no real utility data). The oracle knows the realised shift. The LFC plant is linear and two-area, not the nonlinear plant of the earlier project. The adaptive attacker is weak (best of K random directions). The environment was tested on Python 3.13 with the versions in `ENVIRONMENT.md`, not on Python 3.10.
