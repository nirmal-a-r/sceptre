# Creates a virtual environment next to the project and installs the CPU-only stack.
# Usage:  .\setup_env.ps1            (add -Python "py -3.11" to pick an interpreter)
param([string]$Python = "py -3")
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
if (-not (Test-Path ".venv")) { Invoke-Expression "$Python -m venv .venv" }
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
& ".\.venv\Scripts\python.exe" -m ipykernel install --user --name fdi-oracle --display-name "Python (fdi-oracle)"
& ".\.venv\Scripts\python.exe" -c "import numpy, scipy, pandas, sklearn, pandapower, networkx, matplotlib; print('environment OK: numpy', numpy.__version__, '| pandapower', pandapower.__version__)"
Write-Host "Done. Open notebooks\FDI_Oracle_Benchmark.ipynb and choose the kernel 'Python (fdi-oracle)'."
