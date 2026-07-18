param([ValidateSet("cu121", "cpu")][string]$TorchVariant = "cu121")
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
python -m venv .venv
$Python = Join-Path $Root ".venv\Scripts\python.exe"
& $Python -m pip install --upgrade pip setuptools wheel
& $Python -m pip install torch==2.5.1 --index-url "https://download.pytorch.org/whl/$TorchVariant"
& $Python -m pip install -r requirements-lock.txt
& $Python -m pip install -e .
if (-not (Test-Path ".external\circuit_sparsity")) { git clone https://github.com/openai/circuit_sparsity.git .external\circuit_sparsity }
git -C .external\circuit_sparsity checkout --detach dbf1fe0d27b76c19e10d2a715f28c2e5da535e08
& $Python -m sparse_circuit_repro audit
& $Python -m sparse_circuit_repro download-models
$SmokeArgs = @("-m", "sparse_circuit_repro", "smoke")
if ($TorchVariant -ne "cpu") { $SmokeArgs += "--cuda" }
& $Python @SmokeArgs
