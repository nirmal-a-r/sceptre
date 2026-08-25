# Reproducibility

## 1. The environment that produced the reported numbers

| Component | Value |
|---|---|
| OS | Windows 11 Home, build 10.0.26200 |
| GPU | **NVIDIA GeForce RTX 5060 Laptop GPU** |
| Compute capability | **`sm_120`** (Blackwell) |
| VRAM | 8.5 GB |
| Python | 3.10.11 |
| PyTorch | `2.12.0.dev+cu128` |
| CUDA (torch build) | 12.8 |
| Torch arch list | `sm_75, sm_80, sm_86, sm_90, sm_100, sm_120` |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| pandapower | ≥ 3.0 (with `numba`) |
| networkx | 3.4.2 |
| Environment path | `C:/Users/nirma/Desktop/nlp/nlpenv` |

---

## 2. The RTX 50-series trap

This is the single most likely cause of a failed run on this machine, so it is
stated in full.

The system Python at
`C:/Users/nirma/AppData/Local/Programs/Python/Python310` carries
`torch 2.11.0.dev+cu126`. That build advertises:

```
arch_list = ['sm_50','sm_60','sm_61','sm_70','sm_75','sm_80','sm_86','sm_90']
```

`sm_120` is **absent**. On this GPU that build will:

* return `torch.cuda.is_available() == True`  ← misleading
* return the correct device name and capability `(12, 0)`
* **fail at the first real kernel launch** with
  `CUDA error: no kernel image is available for execution on the device`

**Always run the notebook with `nlpenv`, not with the system Python.** Cell 1
raises a `RuntimeError` with the fix if the arch list does not contain the GPU's
compute capability, so this cannot silently produce wrong results.

Verify in one line:

```bash
C:/Users/nirma/Desktop/nlp/nlpenv/Scripts/python.exe -c "import torch;print(torch.__version__, torch.cuda.get_arch_list())"
```

---

## 3. Seeding

`GLOBAL_SEED = 20260820`. `set_seed()` seeds `random`, `numpy`, `torch` and
`torch.cuda`. Each model/seed pair is re-seeded as `GLOBAL_SEED + seed` at the
top of `_train_given_model`, so:

* every architecture sees **identical** initial data ordering for a given seed;
* comparisons are **paired by seed**, and the paired t-statistics in Figure 8(h)
  are computed seed-by-seed so shared randomness cancels.

Closed-loop comparisons use a stronger form of pairing: `EpisodeSchedule.draw()`
pre-draws attack timing, attack direction, sensor noise and the load disturbance
*before* the episode, and every configuration replays the identical schedule. So
differences are taken **within episode** and `n` = episodes (120), not seeds.

---

## 4. Known sources of non-determinism

These are real and are the reason results are reported with error bars rather
than as single numbers.

1. **cuDNN autotuning.** `torch.backends.cudnn.benchmark = True` selects
   algorithms by timing, so kernel choice can vary between runs. Set it to
   `False` for bitwise determinism at a ~15 % speed cost.
2. **TF32 matmul.** Enabled for speed. It reduces mantissa precision on
   matmul/conv, so results differ slightly from an FP32 run.
3. **`index_add_` on CUDA is non-deterministic** in accumulation order. RSTE uses
   it for the leaf/separator gathers. Setting
   `torch.use_deterministic_algorithms(True)` will raise on these ops rather than
   silently accepting them — this is expected, not a bug.
4. **pandapower Newton–Raphson** uses `init="results"` warm-starting for speed,
   so the AC surrogate fit depends weakly on sample ordering. The held-out
   verification in Cell 9 is what guards this, not bitwise determinism.

**Expected run-to-run spread** on detection F1 for a fixed seed: ~±0.005. Any
conclusion in this notebook that depends on a difference smaller than that is
flagged as unresolved in the text.

---

## 5. Wall-clock budget on the RTX 5060

| Stage | Approximate time |
|---|---|
| Grid construction + separator trees (4 systems) | 5 s |
| AC response-surface fitting + verification | 20–40 s |
| Dataset generation (4 systems) | 15–25 s |
| Main benchmark — 10 models × 3 seeds × 30 epochs | 8–12 min |
| Attack-severity sweep — 5 severities × 4 models × 2 seeds | 12–18 min |
| Zero-shot transfer evaluation | 20 s |
| HALO (4 systems) | 30 s |
| Conformal (incl. drift stream generation) | 1–2 min |
| Blind-spot Monte Carlo (3 systems × 7 severities × 220 directions) | 2–4 min |
| Closed loop — 120 paired episodes × 3 configs + 6 self-play rounds | 3–5 min |
| Ablation — 6 variants × 2 seeds | 5–8 min |
| **Total** | **≈ 35–55 min** |

To cut this for a fast check, reduce in `SCEPTRE.ipynb`:

* `SEEDS = [0]` (main benchmark) — saves ~⅔ of the largest stage;
* `SEV_GRID = [0.30, 0.70]` — saves most of the sweep;
* `N_EP = 40` in the control section;
* `n_dir=60` in `blindspot_curve`.

None of these change a conclusion; they change the width of the error bars, and
the text says which conclusions depend on them.

---

## 5b. Running on a shared GPU, and surviving an interruption

A full run is roughly two and a half GPU-hours. On a single-GPU workstation that
GPU is very often shared with whatever else is training at the time, and two
practical problems follow. Both are handled explicitly rather than left to luck,
because both actually happened while this revision was being produced.

### The interruption problem

The first attempt at the final run died two thirds of the way through the
benchmark — not with a Python exception, but killed outright while a co-tenant
job was competing for the device. Every completed architecture was lost.

Every `(architecture, variant, seed)` is therefore checkpointed to
`checkpoints/` the moment it finishes, and reloaded on a rerun:

```
checkpoints/<cfg-fingerprint>_<key>_s<seed>.pt
```

The fingerprint is an MD5 over the things that would change the answer — budget
name, `d_model`, `n_layers`, `epochs`, `batch`, `window`, `GLOBAL_SEED`, the
system, and the number of training windows. Editing any of them yields a
different fingerprint, so the cache is **invalidated rather than silently
serving a stale result**. Rows restored from cache are printed with a `(cached)`
marker, so the log always distinguishes what was recomputed from what was
reloaded.

| Variable | Default | Effect |
|---|---|---|
| `SCEPTRE_RESUME` | `1` | `0` forces a cold run and ignores `checkpoints/` |
| `SCEPTRE_GPU_FRAC` | unset | caps this process's share of VRAM, e.g. `0.22` |
| `SCEPTRE_BUDGET` | `full` | `smoke` / `quick` / `full` |

A cold run and a resumed run initialise identically: the model factory is
invoked *after* `set_seed(GLOBAL_SEED + seed)` on both paths, so a resumed run
is not a different experiment.

**Reporting rule.** Numbers restored from cache are numbers this machine
actually produced under the current configuration — the fingerprint is what
guarantees that. Numbers that would need a *different* configuration are never
served from cache; they are recomputed.

### The co-tenancy problem

Peak usage at the full budget is under 1.2 GB, so SCEPTRE fits alongside most
other jobs on an 8 GB card. It should not be able to starve them regardless, so
`SCEPTRE_GPU_FRAC` sets a hard per-process cap via
`torch.cuda.set_per_process_memory_fraction`. The preflight table prints both
the cap and how much VRAM other processes already hold, so a slow run has a
visible explanation:

```
GPU memory cap            : 22% of 8.55 GB = 1.88 GB
GPU already in use        : 1.21 GB (other processes)
```

Expect wall-clock times to inflate substantially under contention — the measured
`fp32/tf32` figure in the preflight table drops in proportion, and the
`train s` column moves with it. **`ms/batch` and `MB` are the numbers to quote
for cost comparisons**, not `train s`, because the first two are measured on a
fixed batch after training and are far less sensitive to what else is running.

---

## 6. Rebuilding the notebook

The notebook is generated from cell-marked sources so that diffs are readable:

```bash
cd nb
python build.py            # -> ../SCEPTRE.ipynb
python build.py --script   # -> _full.py   (headless run, same code)
python build.py --probe    # -> _quick.py  (short learnability check)
```

Edit `SCEPTRE.ipynb` directly. The former `nb/src_*.py` sources and their
builder have been merged into the notebook and removed: the notebook is now
the single source of truth, so there is nothing that can overwrite it.

---

## 7. What is *not* reproducible from this repository

* **The pre-registration outcome** (`docs/03_PREREGISTRATION.md`) was produced by
  the archived GA2Shield predecessor (kept outside this repository), on
  different hardware, with
  the NumPy autodiff stack. Its raw artefacts are under
  that project's own `results/ga2_scale/`. The *diagnosis* — the receptive-field
  table — is recomputed from scratch in Cell 5 and is reproducible here.
* **Any claim about real measured telemetry.** There is none. This is a
  simulation study end to end, and the manuscript must say so in the abstract.
