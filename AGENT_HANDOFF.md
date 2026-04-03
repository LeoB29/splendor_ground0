# Agent Handoff

This file is the shortest path for a new agent to get productive in this repo.

## Objective

Build a Python AI for the `Splendor` base game that can eventually reach superhuman play through exact simulation, warm-start training, and later self-play.

Current project target:

- `2-player only`
- `legal observation only`
- full legal action space
- final benchmark intent: eventually beat very strong bots and human play fairly
- main training machine: `Windows + NVIDIA RTX 5070 + CUDA`

## Locked Decisions

- Rules authority: the user supplied the base-game rulebook PDF and asked that implementation not deviate from it.
- Rule clarifications explicitly confirmed by the user:
  - distinct token take can be `3`, or `2/1` if fewer colors are available
  - if no take/reserve/buy is legal, the player must `PASS`
- Warm start from heuristics/search is allowed.
- Inference target is around `1 second per move`.
- AMD/Windows/DirectML is only a compatibility path, not the main training route.

## What Exists

Core engine:

- exact base-game data
- deterministic seeded setup
- legal action generation
- transition logic
- noble resolution and end-of-round handling
- replay-oriented diagnostics

Model/training stack:

- fixed flat action codec and legal-action masks
- 256-float public observation tensor
- replay export to JSONL
- supervised policy/value MLP
- supervised training CLI

Bots/eval:

- random legal bot
- greedy heuristic bot
- shallow search bot
- match runner
- Tkinter GUI for human-vs-bot inspection

Infra:

- Windows/CUDA bootstrap script
- environment verify script
- GitHub-friendly `.gitignore`

## Current Reality

The environment is in decent shape. The weak point is corpus quality.

Observed corpus behavior:

- `greedy:random` is acceptable for a first warm-start corpus
- `greedy:greedy` produced too many token-churn loops
- `search:greedy` was improved, but still hits too many `repetition_cutoff` terminations to be trusted as the main corpus source

Current practical recommendation:

- use `data/corpus_greedy_random_001/replays.jsonl` for the first supervised checkpoint
- exclude stalled/timeout games when training
- treat search-bot improvement as ongoing work, not a blocker to getting the first model trained

## Known Technical Debt

1. The flat action space is very large.
   - It is workable for the first supervised model.
   - It is not the preferred final architecture for serious self-play scale-up.
   - The likely future refactor is a legal-action scorer instead of a giant sparse softmax.

2. Search bot quality is still not reliable enough for primary corpus generation.
   - Current truncations are mostly `repetition_cutoff`
   - failure mode is still token cycling

3. Training has not yet been run on the target CUDA machine.
   - repo prep is done
   - first real checkpoint still needs to be trained

## Recommended Next Step

On the CUDA machine:

1. Clone the repo from GitHub.
2. Bootstrap the environment with `scripts/bootstrap_windows_cuda.ps1`.
3. Install CUDA-enabled PyTorch from the current official PyTorch command.
4. Verify with `scripts/verify_training_env.py --expect-cuda`.
5. Transfer or regenerate the `greedy:random` corpus.
6. Train the first checkpoint:

```powershell
.venv\Scripts\python.exe run_supervised.py --replay-path data\corpus_greedy_random_001\replays.jsonl --output-dir outputs\warmstart_001 --device cuda --epochs 5 --batch-size 64 --exclude-stalled-games --exclude-timeout-games
```

7. Load that checkpoint into the GUI and inspect move quality manually.

## Files Worth Reading First

- `README.md`
- `docs/environment_spec.md`
- `docs/windows_cuda_setup.md`
- `src/splendor_ai/engine/env.py`
- `src/splendor_ai/encoding/action_codec.py`
- `src/splendor_ai/training/run_supervised.py`
- `src/splendor_ai/bots/heuristic_bot.py`
- `src/splendor_ai/bots/search_bot.py`

## Test Status

Before handoff, the full suite passed:

- `69 passed`

If behavior looks suspicious on the new machine, first rerun:

```powershell
python -m pytest -q
```
