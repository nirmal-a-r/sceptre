# Runs the v5 notebook headless. Uses .venv_fdi (built by gpu_prepare.ps1) automatically; otherwise the currently active environment.
# Usage:  .\run_v5.ps1 quick     (smoke test, minutes)
#         .\run_v5.ps1 full      (reported budget)
#         .\run_v5.ps1 heavy     (larger datasets, 20,000-attack threat library per environment, more seeds)
param([ValidateSet("quick","full","heavy")][string]$Budget = "quick", [int]$Jobs = 0, [switch]$AllowCpu, [string]$EnvPath = "")
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
# Default environment: the dedicated .venv_fdi built by gpu_prepare.ps1 (a copy of the sceptre setup with CUDA torch), if it exists.
if ($EnvPath -eq "" -and (Test-Path "$PSScriptRoot\.venv_fdi\Scripts\Activate.ps1")) { $EnvPath = "$PSScriptRoot\.venv_fdi" }
# Optional: -EnvPath C:\path\to\sceptre_env  (a venv folder, or a conda env folder). If omitted, the active environment is used.
if ($EnvPath -ne "") {
  if (Test-Path "$EnvPath\Scripts\Activate.ps1") { & "$EnvPath\Scripts\Activate.ps1" }
  elseif (Test-Path "$EnvPath\python.exe") { $env:PATH = "$EnvPath;$EnvPath\Scripts;$env:PATH" }
  else { throw "No Python environment found at $EnvPath" }
}
$env:FDI_BUDGET = $Budget
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
if ($Jobs -eq 0) { $env:FDI_JOBS = "" + [Math]::Max(1, [Environment]::ProcessorCount - 2) }
if ($Jobs -gt 0) { $env:FDI_JOBS = "$Jobs" }
if (-not $AllowCpu) { $env:FDI_REQUIRE_GPU = "1" }          # stop with an error if CUDA is not visible
python -c "import torch,sys; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
$out = "notebooks\FDI_Oracle_Benchmark_v5_executed_$Budget.ipynb"
python -m jupyter nbconvert --to notebook --execute notebooks\FDI_Oracle_Benchmark_v5.ipynb --output (Split-Path $out -Leaf) --output-dir notebooks --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3
Write-Host "Done: $out ; results in outputs\results.json and outputs\scorecard.csv"
