# Splendor AI

This repository is the foundation for a Python AI system that aims to reach superhuman play in the base game of Splendor through a combination of exact simulation, strong baselines, and self-play training.

## Decisions Locked In

- Game: `Splendor` base game only, no expansions.
- Player count: `2-player only`.
- Fairness target: `legal observation only`.
- Action model: full legal action space, including reserve, token taking, buying from board, buying from reserve, token returns, and noble resolution.
  - Distinct-color takes follow the clarified base-game rule: take up to three distinct colors, falling back to two or one when fewer colors remain in the bank.
  - If no take, reserve, or buy action is legal, the player must pass.
- Main training hardware: `NVIDIA RTX 5070` via `CUDA`.
- Secondary backend path: backend abstraction for `AMD/Windows/DirectML` compatibility, but not as the primary training stack.
- Training horizon: multi-day training runs are acceptable.
- Inference target: design around roughly `1 second per move`.
- Warm start: allowed. Initial training may use heuristic or search-generated data before self-play RL.

## Engineering Position

The first critical milestone is not neural training. It is an exact, deterministic, testable Splendor simulator with legal-observation encoding and a stable action codec. Every later stage depends on this being correct and fast.

The practical development order is:

1. Formal game and environment specification.
2. Deterministic simulator with strong unit tests.
3. Baseline bots and evaluation harness.
4. Supervised warm start from heuristic/search data.
5. Self-play reinforcement learning with checkpoint league evaluation.

## Training Strategy

The default long-term training plan is:

1. Implement exact environment and legal observation encoder.
2. Build baseline bots:
   - random legal bot
   - greedy heuristic bot
   - search bot
3. Generate expert-ish trajectories from heuristics and search.
4. Train a masked policy/value network on those trajectories.
5. Continue with self-play RL against a league of historical checkpoints and baseline bots.
6. Evaluate at multiple move-time controls, with `1s/move` as the main benchmark setting.

## Backend Policy

- Primary backend: `torch.cuda`
- CPU fallback: `torch.cpu`
- Compatibility path: `torch-directml` on Windows when needed

Important constraint: the codebase should not assume a single accelerator backend. Device selection must be centralized so training and evaluation code can stay backend-agnostic.

## Move To CUDA Machine

For the cleanest handoff to the Windows/CUDA training machine:

1. Create a transfer archive:
   - `powershell -ExecutionPolicy Bypass -File scripts\create_handoff_archive.ps1`
2. On the target machine, extract it and bootstrap the repo:
   - `powershell -ExecutionPolicy Bypass -File scripts\bootstrap_windows_cuda.ps1 -InstallDev`
3. Install the current CUDA-enabled PyTorch wheel from:
   - `https://pytorch.org/get-started/locally/`
4. Verify the environment:
   - `.venv\Scripts\python.exe scripts\verify_training_env.py --expect-cuda`

More detail is in `docs/windows_cuda_setup.md`.

## GitHub Workflow

This repo is set up so the codebase is suitable for GitHub, while generated corpora and outputs stay local by default.

- tracked by default:
  - `src/`
  - `tests/`
  - `docs/`
  - `configs/`
  - `scripts/`
  - root launcher files and metadata
- ignored by default:
  - `data/*`
  - `outputs/*`
  - caches and temporary diagnostics

That means the recommended workflow is:

1. Push the codebase to GitHub.
2. Clone it on the CUDA machine.
3. Either regenerate replay corpora there, or transfer selected corpora separately when needed.

## Repo Layout

- `docs/`
  - formal environment and architecture notes
- `configs/`
  - experiment and runtime configuration files
- `src/splendor_ai/engine/`
  - game rules, state transitions, legality, termination
- `src/splendor_ai/encoding/`
  - legal observation and action encoders
- `src/splendor_ai/bots/`
  - baseline bots
- `src/splendor_ai/training/`
  - model, optimization, self-play, backend selection
- `src/splendor_ai/eval/`
  - tournaments, Elo, benchmark matches
- `tests/`
  - unit and integration tests

## Current Status

This scaffold includes:

- project documentation
- initial package structure
- full base-game card and noble datasets
- deterministic setup logic from seeded shuffles
- full legal action enumeration for take / reserve / buy, including token returns and gold substitution
- full state transitions for take / reserve / buy, including market refill, noble choice on buys, and end-of-round handling
- fixed action codec and legality mask layer for model-side action heads
- baseline random and greedy heuristic bots
- shallow search bot for stronger corpus generation and evaluation
- checkpoint-backed policy bot for saved supervised models
- bot-vs-bot match runner for seeded evaluation
- first flat legal-observation tensor encoder
- replay collection and JSONL export for warm-start data
- stalled-game trace export for debugging no-legal-action edge cases
- repetition / no-progress cutoff in replay corpus generation, with explicit termination reasons
- mixed-pairing and seat-swapped replay corpus generation
- baseline supervised policy/value model, dataset loader, and training loop
- simple CLI for supervised warm-start checkpoint training
- local Tkinter GUI for human-vs-bot play and rule inspection
- engine data types and environment interface
- backend selection abstraction
- initial tests that verify setup invariants, hidden-information boundaries, and package wiring

## Immediate Next Milestones

1. Tune corpus generation to minimize timed-out loop games.
2. Collect better-balanced corpora with mixed pairings and seat swapping.
3. Start collecting benchmark results from greedy-vs-random and search-vs-greedy matches.
4. Build the first warm-start checkpoints while excluding truncated games from training when needed.
5. Replace the giant flat policy head with a legal-action scorer before serious self-play scale-up.
