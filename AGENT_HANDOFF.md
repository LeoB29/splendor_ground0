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
- supervised training CLI with train/validation metrics, progress logs, final-vs-best-validation checkpoint benchmarking, and saved metrics JSON

Bots/eval:

- random legal bot
- greedy heuristic bot
- shallow search bot with anti-token-churn loop penalties
- match runner
- checkpoint benchmark CLI with saved per-game JSON output
- Tkinter GUI for human-vs-bot inspection

Infra:

- Windows/CUDA bootstrap script
- environment verify script
- GitHub-friendly `.gitignore`

## Current Reality

The environment and warm-start pipeline are in much better shape than the original handoff. The key milestone is that search-generated data now produces a checkpoint that beats the greedy heuristic baseline in a larger benchmark.

Important local artifacts:

- `data/corpus_greedy_random_003/replays.jsonl`
  - 2,000 games, 145,570 steps, 0 stalled, 0 timed out
- `data/corpus_search_greedy_001/replays.jsonl`
  - 200 games, 12,710 steps, 0 stalled, 0 timed out
- `outputs/warmstart_search_001/supervised_policy_value.pt`
  - current champion checkpoint
- `outputs/benchmarks/warmstart_search_001_50g.json`
  - saved 50-game benchmark report

Observed results:

- Scaling clean `greedy:random` helped reach roughly random-level play, but did not reliably beat `greedy`.
- Search bot token-loop behavior was improved with game-history and take/return churn penalties.
- After the search fix, `search:greedy` corpora are clean in the tested probes and in `corpus_search_greedy_001`.
- Search-only supervised training on `corpus_search_greedy_001` produced the first checkpoint with a clear winning benchmark against `GreedyHeuristicBot`.

Current practical recommendation:

- Treat `outputs/warmstart_search_001/supervised_policy_value.pt` as the current champion.
- Use the benchmark CLI for future checkpoint comparisons instead of ad hoc inline Python.
- Generate more clean `search:greedy` data or experiment with mixed `search:greedy` + `greedy:random` training next.
- Continue excluding stalled/timeout games when training.

## Known Technical Debt

1. The flat action space is very large.
   - It is workable for the first supervised model.
   - It is not the preferred final architecture for serious self-play scale-up.
   - The likely future refactor is a legal-action scorer instead of a giant sparse softmax.

2. Replay validation loss is only loosely correlated with benchmark strength.
   - The training CLI now benchmarks both final and best-validation checkpoints.
   - Pick champions by benchmark results, not validation loss alone.

3. Search corpus generation is slow.
   - `search:greedy` is much slower than `greedy:random`.
   - Keep probing cleanliness before committing to very large search corpora.

4. Search bot quality may still have hidden loop or style issues.
   - The obvious `repetition_cutoff` token-churn pattern was improved.
   - Continue watching timeout traces for any new pathologies.

## Recommended Next Step

Near-term:

1. Run a larger robust benchmark of the current champion if needed:

```powershell
.venv\Scripts\python.exe run_benchmark.py --checkpoint outputs\warmstart_search_001\supervised_policy_value.pt --device cuda --games 100 --opponents random greedy --seed-start 2000 --output-path outputs\benchmarks\warmstart_search_001_100g_seed2000.json
```

2. Generate a larger clean search corpus if benchmarks remain strong:

```powershell
.venv\Scripts\python.exe run_replay_corpus.py --output-dir data\corpus_search_greedy_002 --games 500 --seed-start 6000 --seat0-bot search --seat1-bot greedy --swap-seats --max-turns 400 --repetition-limit 4 --no-progress-limit 60 --search-depth 2 --search-max-branching 8 --search-buy-branching 5 --search-reserve-branching 2 --search-take-branching 2 --log-every 10
```

3. Train either search-only on the larger search corpus, or add multi-corpus/mixed training support and mix `search:greedy` with `greedy:random`.

## Files Worth Reading First

- `README.md`
- `docs/environment_spec.md`
- `docs/windows_cuda_setup.md`
- `src/splendor_ai/engine/env.py`
- `src/splendor_ai/encoding/action_codec.py`
- `src/splendor_ai/training/run_supervised.py`
- `src/splendor_ai/eval/run_benchmark.py`
- `src/splendor_ai/bots/heuristic_bot.py`
- `src/splendor_ai/bots/search_bot.py`

## Test Status

Before this handoff update, the full suite passed:

- `79 passed`

If behavior looks suspicious on the new machine, first rerun:

```powershell
python -m pytest -q
```

## Recent Commit Summary

Changes prepared in this session:

- Added train/validation split metrics, progress logging, and final-vs-best checkpoint handling to supervised training.
- Added saved checkpoint benchmarking and a reusable `run_benchmark.py` CLI.
- Improved `ShallowSearchBot` loop behavior by penalizing repeated no-progress token take/return cycles.
- Added regression tests for search-loop behavior, training metrics/artifacts, benchmark summarization, and CLI helpers.
- Created local Codex skill `repo-handoff-commit` for future handoff-update-plus-commit workflows.
