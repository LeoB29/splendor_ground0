param(
    [string]$VenvDir = ".venv",
    [switch]$InstallDev
)

$ErrorActionPreference = "Stop"

Write-Host "[setup] creating virtual environment at $VenvDir"
python -m venv $VenvDir

$pythonExe = Join-Path $VenvDir "Scripts\python.exe"
$pipExe = Join-Path $VenvDir "Scripts\pip.exe"

Write-Host "[setup] upgrading pip"
& $pythonExe -m pip install --upgrade pip setuptools wheel

Write-Host "[setup] installing project package"
if ($InstallDev) {
    & $pipExe install -e ".[dev]"
} else {
    & $pipExe install -e .
}

Write-Host ""
Write-Host "[next] install a CUDA-enabled PyTorch wheel using the current command from https://pytorch.org/get-started/locally/"
Write-Host "[next] example shape:"
Write-Host "       $pipExe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"
Write-Host ""
Write-Host "[next] verify CUDA visibility:"
Write-Host "       $pythonExe scripts\\verify_training_env.py --expect-cuda"
Write-Host ""
Write-Host "[next] train:"
Write-Host "       $pythonExe run_supervised.py --replay-path data\\corpus_greedy_random_001\\replays.jsonl --output-dir outputs\\warmstart_001 --device cuda --epochs 5 --batch-size 64 --exclude-stalled-games --exclude-timeout-games"
