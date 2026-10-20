# One-time GPU preparation for the FDI project (Windows, PowerShell).
#   .\gpu_prepare.ps1                      checks the GPU, builds the dedicated "fdi" environment from your sceptre one, runs a GPU test
#   .\gpu_prepare.ps1 -ListGpuUsers        only shows what is using the GPU right now, changes nothing
#   .\gpu_prepare.ps1 -StopGpuUsers        asks, one by one, whether to stop each program that holds GPU memory
#   .\gpu_prepare.ps1 -Recreate            deletes and rebuilds the fdi environment
#   .\gpu_prepare.ps1 -SceptrePath D:\x\.venv   if your sceptre environment is somewhere else (venv folder or conda env folder)
# Afterwards:  .\run_v5.ps1 quick   then   .\run_v5.ps1 full   or   .\run_v5.ps1 heavy
param(
  [string]$SceptrePath = "$HOME\Desktop\SCEPTRE\.venv",
  [string]$EnvDir = "$PSScriptRoot\.venv_fdi",
  [switch]$ListGpuUsers, [switch]$StopGpuUsers, [switch]$Recreate, [string]$TorchIndex = "https://download.pytorch.org/whl/cu128"
)
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot
function Say($m) { Write-Host "[gpu_prepare] $m" -ForegroundColor Cyan }
function Fail($m) { Write-Host "[gpu_prepare] ERROR: $m" -ForegroundColor Red; exit 1 }

# ---- 1. the GPU and what is using it ----------------------------------------------------------------------------------
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { Fail "nvidia-smi not found. Install or update the NVIDIA driver first (it must support CUDA 12.8 or newer: driver 570+)." }
Say "GPU:"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,compute_cap --format=csv
Say "Programs that currently hold GPU compute memory (Windows may list only some; Task Manager > Performance > GPU shows all):"
$apps = @(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>$null | Where-Object { $_ -and $_ -notmatch "No running" })
if ($apps.Count -eq 0) { Write-Host "  (none reported)" } else { $apps | ForEach-Object { Write-Host "  $_" } }
Say "Python / Jupyter / Ollama / game processes that commonly hold the GPU:"
Get-Process -ErrorAction SilentlyContinue | Where-Object { $_.ProcessName -match "^(python|pythonw|jupyter|ipykernel|ollama|ollama_llama_server|steam|steamwebhelper)" } |
  Select-Object Id, ProcessName, @{n="MemMB";e={[int]($_.WorkingSet64/1MB)}} | Format-Table -AutoSize | Out-String | Write-Host
if ($ListGpuUsers) { exit 0 }
if ($StopGpuUsers) {
  $protect = "System","Idle","dwm","csrss","winlogon","explorer","svchost","nvcontainer","NVDisplay.Container","claude","Claude"
  foreach ($line in $apps) {
    $parts = $line -split ",\s*"; $procId = [int]$parts[0]; $leaf = Split-Path $parts[1] -Leaf; $nm = [IO.Path]::GetFileNameWithoutExtension($leaf)
    if ($procId -eq $PID -or $protect -contains $nm) { Write-Host "  skipping $nm ($procId)"; continue }
    $a = Read-Host "Stop $nm (pid $procId, $($parts[2]))? [y/N]"
    if ($a -match "^[yY]") { Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue; Say "stopped $nm" }
  }
  Start-Sleep -Seconds 2; nvidia-smi --query-gpu=memory.used,memory.total --format=csv
}

# ---- 2. find the sceptre interpreter ----------------------------------------------------------------------------------
$basePy = $null
foreach ($c in @("$SceptrePath\Scripts\python.exe", "$SceptrePath\python.exe")) { if (Test-Path $c) { $basePy = $c; break } }
if (-not $basePy -and (Get-Command conda -ErrorAction SilentlyContinue)) {
  $envs = conda env list 2>$null | Where-Object { $_ -match "sceptre" }
  if ($envs) { $p = ($envs[0] -split "\s+")[-1]; if (Test-Path "$p\python.exe") { $basePy = "$p\python.exe" } }
}
if (-not $basePy) { Say "sceptre environment not found at $SceptrePath; using the default python on PATH"; $basePy = (Get-Command python -ErrorAction Stop).Source }
Say "base interpreter: $basePy"
& $basePy -c "import sys; print('python', sys.version.split()[0])"
$lock = "$PSScriptRoot\outputs\sceptre_pip_freeze.txt"; New-Item -ItemType Directory -Force -Path "$PSScriptRoot\outputs" | Out-Null
& $basePy -m pip freeze 2>$null | Out-File -Encoding ascii $lock
Say "package list of the sceptre environment saved to $lock (for reference)"

# ---- 3. build the dedicated fdi environment ----------------------------------------------------------------------------
if ($Recreate -and (Test-Path $EnvDir)) { Say "removing $EnvDir"; Remove-Item -Recurse -Force $EnvDir }
$envPy = "$EnvDir\Scripts\python.exe"
if (-not (Test-Path $envPy)) {
  Say "creating $EnvDir"
  & $basePy -m venv $EnvDir
  if (-not (Test-Path $envPy)) { Fail "venv creation failed" }
}
& $envPy -m pip install --upgrade pip
$haveCuda = & $envPy -c "import torch; print(int(torch.cuda.is_available()))" 2>$null
if ($haveCuda -ne "1") {
  Say "installing PyTorch (CUDA 12.8 build; it also covers RTX 50-series / sm_120)"
  & $envPy -m pip install torch --index-url $TorchIndex
  $haveCuda = & $envPy -c "import torch; print(int(torch.cuda.is_available()))" 2>$null
  if ($haveCuda -ne "1") { Say "stable cu128 wheel did not give a working GPU; trying the nightly cu128 build used by SCEPTRE"; & $envPy -m pip install --pre --force-reinstall torch --index-url https://download.pytorch.org/whl/nightly/cu128 }
}
Say "installing the project packages"
& $envPy -m pip install -r "$PSScriptRoot\requirements.txt"
& $envPy -m ipykernel install --user --name fdi-gpu --display-name "Python (fdi-gpu)" | Out-Null

# ---- 4. GPU test --------------------------------------------------------------------------------------------------------
Say "GPU test inside the new environment:"
$test = @'
import torch, time, numpy, scipy, pandas, sklearn, pandapower, simbench, networkx, joblib
print("torch", torch.__version__, "| built for CUDA", torch.version.cuda, "| cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available(): raise SystemExit("NO GPU visible to torch")
cap = torch.cuda.get_device_capability(0); arch = "sm_%d%d" % cap
print("device:", torch.cuda.get_device_name(0), arch, "| arch list:", torch.cuda.get_arch_list())
if arch not in torch.cuda.get_arch_list(): raise SystemExit("torch has no kernels for " + arch)
a = torch.randn(4096, 4096, device="cuda", dtype=torch.float64); t = time.time(); (a @ a).sum().item(); torch.cuda.synchronize()
print("float64 4096x4096 matmul: %.2fs" % (time.time() - t))
free, tot = torch.cuda.mem_get_info(); print("free GPU memory: %.1f of %.1f GB" % (free / 1e9, tot / 1e9))
print("pandapower", pandapower.__version__, "| simbench", simbench.__version__, "| cpu cores", __import__("os").cpu_count())
print("ENVIRONMENT READY")
'@
$test | & $envPy -
if ($LASTEXITCODE -ne 0) { Fail "GPU test failed (see messages above). Update the NVIDIA driver (570 or newer for RTX 50-series) and run  .\gpu_prepare.ps1 -Recreate" }
Say "Done. Next:  .\run_v5.ps1 quick   (about 10-20 min smoke test, GPU required)   then   .\run_v5.ps1 heavy"
