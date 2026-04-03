# Windows CUDA Setup

This repo is prepared to move to a second Windows machine for CUDA training.

## Recommended Transfer

From the source machine, create a clean handoff archive:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\create_handoff_archive.ps1
```

By default this includes the codebase plus:

- `data/corpus_greedy_random_001`

If you want to include different replay corpora, pass them explicitly:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\create_handoff_archive.ps1 -ArchivePath splendor_cuda_handoff.zip -IncludeDataDirs @("data/corpus_greedy_random_001", "data/corpus_search_greedy_tuned_smoke_001")
```

## Target Machine Setup

1. Extract the archive.
2. Open PowerShell in the extracted repo root.
3. Create a virtual environment and install the project:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\bootstrap_windows_cuda.ps1 -InstallDev
```

4. Install a CUDA-enabled PyTorch wheel using the current command from:

- https://pytorch.org/get-started/locally/

Example command shape:

```powershell
.venv\Scripts\pip.exe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

5. Verify the environment:

```powershell
.venv\Scripts\python.exe scripts\verify_training_env.py --expect-cuda
```

## First Warm-Start Training Run

```powershell
.venv\Scripts\python.exe run_supervised.py --replay-path data\corpus_greedy_random_001\replays.jsonl --output-dir outputs\warmstart_001 --device cuda --epochs 5 --batch-size 64 --exclude-stalled-games --exclude-timeout-games
```

## Notes

- `greedy:random` is the current recommended first warm-start corpus.
- Old checkpoints from before the recent rule and action-space changes should not be reused.
- If CUDA is not visible in the verify step, fix the PyTorch install before training.
