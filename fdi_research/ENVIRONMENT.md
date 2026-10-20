# Environment

This project reuses the SCEPTRE environment contract and drops what it does not need.

| | SCEPTRE | This project |
|---|---|---|
| Python | 3.10 | 3.10 or newer |
| numpy / scipy / pandas | >=2.0 / >=1.13 / >=2.2 | same |
| pandapower, networkx, scikit-learn, matplotlib | yes | yes |
| PyTorch (CUDA 12.8, sm_120) | required | **used in Part 19-20** (small models; CPU is enough, CUDA is used automatically if present) |
| simbench | no | yes (real yearly load/DER profiles, Part 17) |
| python-docx | yes (reports) | no |
| GPU | RTX 5060 | optional (the delivered run was CPU-only; the notebook prints the device it used) |

## Versions the reported run was executed with

The notebook records its own versions in `outputs/results.json` (key `versions`). The run delivered with this folder used numpy 2.4.6, scipy 1.18.1, pandas 2.3.3, scikit-learn 1.9.1, pandapower 3.5.6, networkx 3.6.1, matplotlib 3.11.2.
On Python 3.10 pip will choose older releases of numpy and scipy (numpy 2.4 needs Python 3.11 or newer). The code uses only APIs present in numpy 2.0 and later, but **that combination was not tested here**; if a number differs, compare against `outputs/results.json` first.

## Windows setup

```powershell
cd $HOME\Desktop\FDI_Research_Proposal
.\setup_env.ps1            # creates .venv, installs requirements.txt, registers a Jupyter kernel
.\.venv\Scripts\Activate.ps1
jupyter lab notebooks\FDI_Oracle_Benchmark.ipynb
```

SCEPTRE's `.venv` is a symlink and is **not** copied; the environment is rebuilt from `requirements.txt` instead, which is the portable way to carry it.

## Running

* `FDI_BUDGET=full` (default) is what the reported numbers use.
* `FDI_BUDGET=quick` runs in a few minutes for a smoke test; its numbers are noisy and must not be reported.

PowerShell: `$env:FDI_BUDGET = "quick"; jupyter lab`.

## Running on a GPU machine (for example the `nlpenv` environment)

Everything in Parts 19 and 20 selects `cuda` when `torch.cuda.is_available()` is true. To use an existing environment instead of the project `.venv`:

```powershell
# activate your environment first (venv:  .\nlpenv\Scripts\Activate.ps1   or conda:  conda activate nlpenv)
.\setup_nlpenv.ps1          # installs only what is missing: pandapower, simbench, networkx, scikit-learn, nbformat, ...
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
jupyter lab notebooks\FDI_Oracle_Benchmark_v4.ipynb
```

The first cell of the notebook prints the device, and Part 19 prints it again before training. **The delivered run was executed on a CPU-only cloud machine; no GPU number appears anywhere in it.** The models are small (a 12k-parameter network, a 32-parameter quadratic head, a 40k-parameter flow), so a GPU changes the wall-clock time by little; it will not change the findings except through seed-level noise.

## v5: running in the sceptre environment on the GPU

* The notebook needs, beyond SCEPTRE's packages: `pandapower`, `simbench`, `joblib` (installed with scikit-learn), `networkx`, `nbformat`/`nbconvert` for the headless script. `setup_nlpenv.ps1` installs what is missing into the active environment and keeps an existing CUDA build of torch.
* `FDI_REQUIRE_GPU=1` makes Part 19 stop with a clear message if torch cannot see the GPU. An RTX 50-series card needs a CUDA 12.8 (cu128) or newer torch build.
* Environment variables: `FDI_BUDGET` (`quick`, `full`, `heavy`), `FDI_JOBS` (CPU workers for the simulators), `FDI_REQUIRE_GPU`.
* The power-flow and state-estimation simulators are NumPy code. Only the neural parts and the library scoring use the GPU.

## Using your existing sceptre environment (no copy needed)
Python environments cannot be copied between folders (they store absolute paths), so the scripts use the environment in place:

    cd C:\Users\nirma\Desktop\FDI_Research_Proposal
    .\run_v5.ps1 quick -EnvPath C:\Users\nirma\Desktop\SCEPTRE\.venv     # or your conda env folder
    .\run_v5.ps1 heavy -EnvPath <same path>

Add `-AllowCpu` only if you want it to run without a GPU. Without it, the run stops at once if CUDA is not visible.
Install the extra packages into that env first: `pip install -r requirements.txt` (a CUDA build of torch is needed for the GPU).

## One-command GPU setup (recommended)
    cd C:\Users\nirma\Desktop\FDI_Research_Proposal
    .\gpu_prepare.ps1 -ListGpuUsers      # what is using the GPU now (changes nothing)
    .\gpu_prepare.ps1 -StopGpuUsers      # optional: asks y/N before stopping each GPU-holding program
    .\gpu_prepare.ps1                    # builds .venv_fdi from your sceptre interpreter, installs CUDA 12.8 torch + this project's packages, runs a GPU test
    .\run_v5.ps1 quick                   # smoke test on the GPU (stops at once if CUDA is not visible)
    .\run_v5.ps1 heavy
A virtual environment cannot be copied between folders (it stores absolute paths), so `gpu_prepare.ps1` rebuilds it from the same Python interpreter as sceptre and saves sceptre's package list to `outputs\sceptre_pip_freeze.txt`. This script was written without access to your Windows machine and has not been run; if a step fails, the message names the step.
