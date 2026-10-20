# Installs the packages this project needs into the CURRENTLY ACTIVE Python environment (for example your existing nlpenv).
# Usage: activate the environment first, then run  .\setup_nlpenv.ps1
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
python -c "import sys; print('python', sys.version.split()[0], '|', sys.prefix)"
python -m pip install --upgrade pip
# keep an existing CUDA build of torch if one is present; install the CPU/default build only if torch is missing
python -c "import torch" 2>$null
if ($LASTEXITCODE -ne 0) { Write-Host "torch missing: installing the CUDA 12.8 build (RTX 50-series needs cu128 or newer)"; python -m pip install torch --index-url https://download.pytorch.org/whl/cu128 }
python -m pip install -r requirements.txt
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu only')"
python -c "import numpy, scipy, pandas, sklearn, pandapower, networkx, simbench; print('environment OK: pandapower', pandapower.__version__, '| simbench', simbench.__version__)"
Write-Host "Done. Open notebooks\FDI_Oracle_Benchmark_v5.ipynb."
