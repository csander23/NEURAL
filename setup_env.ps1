<#
  setup_env.ps1 - build the Python environment that runs NEURAL (+ Methods_Paper).

  Creates a virtualenv at the workspace root (..\.venv), installs NEURAL's
  dependencies from requirements.txt, and installs the correct torch build for
  this machine's GPU. Run once; afterwards use .venv\Scripts\python.exe.

  Usage (from anywhere):
      powershell -ExecutionPolicy Bypass -File "NEURAL\setup_env.ps1"
      powershell -ExecutionPolicy Bypass -File "NEURAL\setup_env.ps1" -Cuda cpu   # CPU-only

  Then, e.g.:
      & "..\.venv\Scripts\python.exe" Methods_Paper\Figures\Figure_3_Cortex_GCaMP\render.py
#>
param(
    [string]$Cuda = "cu121",                              # cu121 (RTX 4000, driver 535) | cu118 | cpu
    [string]$BasePython = "$env:LOCALAPPDATA\Microsoft\WindowsApps\python.exe"
)
$ErrorActionPreference = "Stop"
$here      = $PSScriptRoot
$workspace = Resolve-Path (Join-Path $here "..")
$venv      = Join-Path $workspace ".venv"
$py        = Join-Path $venv "Scripts\python.exe"

Write-Host "Workspace : $workspace"
Write-Host "Venv      : $venv"
Write-Host "Base py   : $BasePython"
Write-Host "Torch CUDA: $Cuda`n"

if (-not (Test-Path $venv)) {
    Write-Host "==> Creating virtualenv..."
    & $BasePython -m venv $venv
}
Write-Host "==> Upgrading pip..."
& $py -m pip install --upgrade pip wheel

Write-Host "==> Installing NEURAL requirements..."
& $py -m pip install -r (Join-Path $here "requirements.txt")

Write-Host "==> Installing torch/torchvision ($Cuda)..."
& $py -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$Cuda"

Write-Host "`n==> Verifying..."
& $py -c "import sys; sys.path.insert(0, r'$workspace'); import NEURAL, numpy, scipy, pandas, matplotlib, skimage; print('NEURAL', NEURAL.__version__, 'imports OK'); import torch; print('torch', torch.__version__, 'CUDA available:', torch.cuda.is_available())"

Write-Host "`nDone. Use: $py"
