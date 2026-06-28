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
- multi-corpus replay loading and repeated `--replay-path` support
- checkpoint-driven replay corpus generation with checkpoint/fallback provenance metadata
- supervised replay filtering by per-game checkpoint fallback trigger count
- optional pre-action replay state snapshots for reconstructing exact training positions
- shallow-search replay action improvement tooling for policy-target relabeling

Bots/eval:

- random legal bot
- greedy heuristic bot
- shallow search bot with anti-token-churn loop penalties
- match runner
- checkpoint benchmark CLI with saved per-game JSON output
- explicit benchmark loop diagnostics: `repetition_cutoff`, `no_progress_cutoff`, per-game termination metadata
- loop-aware checkpoint inference fallback with per-game trigger counts and CLI controls
- replay corpus CLI supports `checkpoint` bot pairings for champion-vs-baseline/champion-vs-champion probes
- Tkinter GUI for human-vs-bot inspection

Infra:

- Windows/CUDA bootstrap script
- environment verify script
- GitHub-friendly `.gitignore`

## Current Reality

The warm-start pipeline is in a stronger place than the prior handoff. Continued sharded `search:greedy` corpus scaling plus multi-corpus supervised training produced a new local champion checkpoint, and a narrow inference-only loop fallback now mitigates the rare checkpoint repetition failures seen in greedy benchmarks.

Important local artifacts:

- `data/corpus_greedy_random_003/replays.jsonl`
  - 2,000 games, 145,570 steps, 0 stalled, 0 timed out
- `data/corpus_search_greedy_002/replays.jsonl`
  - 500 games, 31,954 steps, 0 stalled, 0 timed out
- `data/corpus_search_greedy_003a/replays.jsonl`
  - 125 games, 7,992 steps, 0 stalled, 0 timed out
- `data/corpus_search_greedy_003b/replays.jsonl`
  - 125 games, 7,960 steps, 0 stalled, 0 timed out
- `data/corpus_search_greedy_003c/replays.jsonl`
  - 125 games, 7,942 steps, 0 stalled, 0 timed out
- `data/corpus_search_greedy_003d/replays.jsonl`
  - 125 games, 8,012 steps, 0 stalled, 0 timed out
- `data/corpus_search_greedy_004[1-5]/replays.jsonl`
  - 125 games total, 7,964 steps, 0 stalled, 0 timed out
- `outputs/warmstart_search_003/supervised_policy_value_best.pt`
  - prior champion checkpoint
- `outputs/warmstart_search_004/supervised_policy_value_best.pt`
  - previous champion checkpoint before gated policy improvement
- `outputs/policy_improve_margin500_001/supervised_policy_value.pt`
  - current local champion checkpoint, local/ignored artifact
- `outputs/benchmarks/warmstart_search_003_best_100g_seed5000.json`
  - 100-game benchmark: 98-2 vs random, 83-16-1 vs greedy, 2 repetition-cutoff games vs greedy
- `outputs/benchmarks/warmstart_search_003_best_100g_seed6000.json`
  - 100-game benchmark: 100-0 vs random, 86-14 vs greedy, 1 repetition-cutoff game vs greedy
- `outputs/benchmarks/warmstart_search_003_final_100g_seed5000.json`
  - final checkpoint benchmark: worse than `best_validation`
- `outputs/benchmarks/loop_fallback_enabled_100g_seed5000.json`
  - local/ignored artifact, 100-game benchmark vs greedy with fallback enabled: 85-15-0, 0 timed out, 100 completed, 8 fallback triggers
- `outputs/benchmarks/loop_fallback_enabled_100g_seed6000.json`
  - local/ignored artifact, 100-game benchmark vs greedy with fallback enabled: 85-15-0, 0 timed out, 100 completed, 5 fallback triggers
- `outputs/benchmarks/loop_fallback_disabled_seed5008_probe.json`
  - local/ignored artifact, confirms seed 5008 still hits `repetition_cutoff` when fallback is disabled
- `outputs/warmstart_mix_001/supervised_metrics.json`
  - local/ignored artifact, naive mixed search+greedy_random training; benchmark champion final checkpoint went 17-3 vs random, 5-15 vs greedy on seed block 7000
- `outputs/warmstart_mix_002_search_weighted/supervised_metrics.json`
  - local/ignored artifact, search-weighted mixed training; benchmark champion final checkpoint went 19-1 vs random, 9-11 vs greedy on seed block 7000
- `outputs/benchmarks/warmstart_search_003_best_20g_seed7000.json`
  - local/ignored artifact, then-current champion comparison on same seed block: 20-0 vs random, 16-4 vs greedy
- `outputs/warmstart_search_004/supervised_metrics.json`
  - local/ignored artifact, search-only training on `002 + 003[a-d] + 004[1-5]`; best-validation checkpoint selected as benchmark champion
- `outputs/benchmarks/warmstart_search_004_best_100g_seed5000.json`
  - local/ignored artifact, 100-game benchmark vs greedy: 85-15-0, 0 timed out, 4 fallback triggers
- `outputs/benchmarks/warmstart_search_004_best_100g_seed6000.json`
  - local/ignored artifact, 100-game benchmark vs greedy: 86-14-0, 0 timed out, 1 fallback trigger
- `outputs/benchmarks/warmstart_search_004_best_random_100g_seed5000.json`
  - local/ignored artifact, 100-game sanity benchmark vs random: 100-0-0, 0 timed out, 4 fallback triggers
- `outputs/warmstart_search_005/supervised_metrics.json`
  - local/ignored artifact, search-only training on `002 + 003[a-d] + 004[1-5] + 005[1-5]`; benchmark champion final checkpoint went 20-0 vs random and 15-5 vs greedy on seed block 7000, so it was not promoted
- `data/corpus_checkpoint_greedy_probe_001/`
  - local/ignored artifact, 2-game checkpoint-vs-greedy end-to-end replay probe: 126 steps, 0 stalled, 0 timed out, 0 fallback triggers
- `data/corpus_checkpoint_greedy_00[1-5]/`
  - local/ignored artifacts, 125 games total, 8,156 steps, 0 stalled, 0 timed out, 4 fallback triggers across 4 games
- `outputs/warmstart_champion_mix_001/supervised_metrics.json`
  - local/ignored artifact, search-only champion data plus checkpoint-vs-greedy data with fallback-triggered games excluded; benchmark champion final checkpoint went 20-0 vs random and 15-5 vs greedy on seed block 7000, so it was not promoted

Observed results:

- Scaling clean `greedy:random` helped reach roughly random-level play, but did not reliably beat `greedy`.
- Search bot token-loop behavior was improved with game-history and take/return churn penalties.
- After the search fix, `search:greedy` corpora remained clean through `corpus_search_greedy_002` and sharded `corpus_search_greedy_003[a-d]`.
- Added benchmark-side loop diagnostics so repeated-state issues surface as explicit `termination_reason` fields instead of opaque 400-turn outcomes.
- Added multi-corpus supervised training support; `run_supervised.py` now accepts repeated `--replay-path` arguments and records per-source sample counts in metrics.
- Training on `002 + 003[a-d]` produced `warmstart_search_003`, where the `best_validation` checkpoint is materially stronger than the final checkpoint.
- `warmstart_search_003_best.pt` beat `GreedyHeuristicBot` in two 100-game runs at 83% and 86%, while remaining near-perfect vs random.
- Added a conservative `CheckpointPolicyBot` loop fallback:
  - tracks repeated public state signatures and the bot's own consecutive non-progress actions
  - only intervenes when legal buy actions exist
  - chooses the model's highest-logit legal buy, and only if it is within the configured logit-gap threshold
  - can be disabled/tuned through benchmark CLI flags
  - reports per-game and aggregate fallback trigger counts in benchmark JSON
- Reused a single loaded checkpoint bot across benchmark games in both standalone and post-training benchmark paths, avoiding repeated checkpoint reload overhead while resetting per-game loop counters.
- Validated fallback on the two known greedy seed blocks:
  - seed block 5000: 85-15-0, 0 timeouts, 8 fallback triggers
  - seed block 6000: 85-15-0, 0 timeouts, 5 fallback triggers
  - combined: 170-30-0 across 200 games, 0 timeouts, 13 fallback triggers
- Ran two mixed-data warm-start experiments after accepting the fallback:
  - `warmstart_mix_001`: existing search corpora plus `corpus_greedy_random_003`, 209,430 samples, 8 epochs, final checkpoint selected by benchmark, 5-15 vs greedy on seed block 7000
  - `warmstart_mix_002_search_weighted`: search corpora repeated 3x plus `corpus_greedy_random_003`, 337,150 samples, 8 epochs, final checkpoint selected by benchmark, 9-11 vs greedy on seed block 7000
  - then-current champion `warmstart_search_003/supervised_policy_value_best.pt` went 16-4 vs greedy on the same seed block, so neither mixed checkpoint should be promoted
- Generated clean `search:greedy` shard block `004[1-5]`:
  - 125 games, 7,964 steps, 0 stalled, 0 timed out
- Trained `warmstart_search_004` on search-only data (`002 + 003[a-d] + 004[1-5]`):
  - 71,824 total samples, 8 epochs, best validation epoch 6
  - post-train 20-game seed-7000 benchmark: best-validation checkpoint went 20-0 vs random and 19-1 vs greedy
  - 100-game greedy validation: 85-15 on seed block 5000, 86-14 on seed block 6000
  - 100-game random sanity benchmark: 100-0 on seed block 5000
  - combined vs greedy: 171-29-0 across 200 games, 0 timeouts, 5 fallback triggers
  - compared to fallback-enabled `warmstart_search_003_best`: 170-30-0 across 200 games, 0 timeouts, 13 fallback triggers
  - promote `warmstart_search_004/supervised_policy_value_best.pt` as the new local champion; the win-rate gain is small, but fallback use is materially lower
- Added checkpoint-driven replay generation:
  - `run_replay_corpus.py` now accepts `checkpoint` as a bot name in `--seat0-bot`, `--seat1-bot`, and `--pairing`
  - checkpoint bots use `--checkpoint-path` and `--checkpoint-device`
  - checkpoint loop fallback can be enabled/disabled/tuned for replay generation
  - checkpoint models are cached per bot factory/seat, avoiding repeated checkpoint reloads while keeping separate per-seat state
  - replay rows now include `game_bot_metadata`, `game_loop_fallback_triggers_by_seat`, and `game_model_loop_fallback_triggers`
  - corpus `summary.json` now includes aggregate `loop_fallback_triggers`, `loop_fallback_games`, and CLI provenance metadata
- Added supervised replay filtering for checkpoint fallback games:
  - `SupervisedReplayDataset` accepts `max_model_loop_fallback_triggers`
  - `run_supervised.py` exposes `--max-game-model-loop-fallback-triggers`
  - rows without checkpoint fallback metadata count as 0, preserving compatibility with older search corpora
- Trained `warmstart_search_005` after generating clean `corpus_search_greedy_005[1-5]`, but it regressed in the quick seed-7000 benchmark:
  - benchmark-selected final checkpoint: 20-0 vs random, 15-5 vs greedy
  - `warmstart_search_004_best` on comparable post-train seed-7000 benchmark: 20-0 vs random, 19-1 vs greedy
  - do not promote `warmstart_search_005`
- Validated the new path with `data/corpus_checkpoint_greedy_probe_001`:
  - pairing `checkpoint:greedy`, then-current champion checkpoint, CUDA, swap seats
  - 2 games, 126 steps, 0 stalled, 0 timed out, 0 fallback triggers
- Generated `data/corpus_checkpoint_greedy_00[1-5]`:
  - 125 games, 8,156 steps, 0 stalled, 0 timed out
  - 4 fallback triggers across 4 games
- Trained `warmstart_champion_mix_001` on search data through `004[1-5]` plus checkpoint-vs-greedy data, excluding any game with more than 0 checkpoint fallback triggers:
  - dataset after filtering: 79,708 samples
  - benchmark champion final checkpoint: 20-0 vs random, 15-5 vs greedy on seed block 7000
  - current champion `warmstart_search_004_best` on the comparable seed block was 20-0 vs random, 19-1 vs greedy
  - do not promote `warmstart_champion_mix_001`
- Added the first policy-improvement tooling:
  - `run_replay_corpus.py --include-state-snapshots` writes compact pre-action state snapshots into replay rows without changing the default corpus format
  - `deserialize_state_snapshot` reconstructs exact `SplendorState` objects from those rows
  - `ShallowSearchBot.rank_actions(...)` exposes ranked/scored root actions
  - `run_improve_replay_actions.py` reads snapshot-enabled replay JSONL, reranks each position with shallow search, and writes relabeled policy rows plus a summary JSON
  - improved rows preserve `original_action_index`, `original_action_payload`, and `improvement_metadata`
  - metadata explicitly marks `target_semantics` as `policy_relabel_only_original_final_value_retained`
- Ran the first policy-improvement probe:
  - generated `data/corpus_checkpoint_greedy_snapshots_001`
  - 25 games, 1,650 steps, 0 stalled, 0 timed out, 0 loop fallback triggers
  - relabeled all 1,650 rows with depth-2 shallow search into `improved_search_policy_full.jsonl`
  - search changed 1,123 rows and left 527 rows unchanged
  - added `run_supervised.py --init-checkpoint` and fine-tuned from `outputs/warmstart_search_004/supervised_policy_value_best.pt`
  - trained `outputs/policy_improve_001` for 4 policy-only epochs with `--value-loss-weight 0.0`
  - quick seed-7000 benchmark: 19-1 vs random with 1 `repetition_cutoff`, 18-2 vs greedy with 0 timeouts
  - do not promote `policy_improve_001`; then-current champion remained `warmstart_search_004_best`, whose comparable quick seed-7000 result was 20-0 vs random and 19-1 vs greedy
- Added confidence-gated shallow-search relabeling:
  - `run_improve_replay_actions.py` now supports `--min-search-margin` and `--exclude-changed-action-types`
  - the gate requires the original action to be present in the pruned search candidate set when a margin is requested
  - filtered rows are counted separately for low margin, missing original score, and excluded selected action type
- Ran `policy_improve_margin500_001`, the first successful policy-improvement candidate:
  - relabel settings: `--min-search-margin 500 --exclude-changed-action-types TAKE_TOKENS --write-unchanged`
  - relabel output: `data/corpus_checkpoint_greedy_snapshots_001/improved_search_policy_margin500_no_take.jsonl`
  - 901 rows written: 374 high-confidence changed labels and 527 unchanged anchors
  - filtered 193 low-margin changes, 399 changes where the original action was pruned from search candidates, and 157 selected-token-take changes
  - fine-tuned from `outputs/warmstart_search_004/supervised_policy_value_best.pt` for 4 policy-only epochs with `--value-loss-weight 0.0`
  - quick seed-7000 benchmark: 20-0 vs random, 19-1 vs greedy, 0 timeouts
  - 100-game random sanity seed block 5000: 100-0-0, 0 timeouts, 0 fallback triggers
  - 100-game greedy seed block 5000: 94-6-0, 0 timeouts, 0 fallback triggers
  - 100-game greedy seed block 6000: 95-5-0, 0 timeouts, 1 fallback trigger
  - combined greedy validation: 189-11-0 across 200 games, 0 timeouts, 1 fallback trigger
  - this beats `warmstart_search_004_best` on the same two 100-game greedy blocks: 171-29-0 with 5 fallback triggers
  - promoted `outputs/policy_improve_margin500_001/supervised_policy_value.pt` as the new local champion after checkpointing the code/handoff
- Ran a second gated policy-improvement iteration, `policy_improve_margin500_002`:
  - generated `data/corpus_checkpoint_greedy_snapshots_002` from `policy_improve_margin500_001` vs greedy
  - 25 games, 1,634 steps, 0 stalled, 0 timed out, 0 fallback triggers
  - relabel settings unchanged: `--min-search-margin 500 --exclude-changed-action-types TAKE_TOKENS --write-unchanged`
  - relabel output: `data/corpus_checkpoint_greedy_snapshots_002/improved_search_policy_margin500_no_take.jsonl`
  - 882 rows written: 362 high-confidence changed labels and 520 unchanged anchors
  - filtered 164 low-margin changes, 426 changes where the original action was pruned from search candidates, and 162 selected-token-take changes
  - fine-tuned from `outputs/policy_improve_margin500_001/supervised_policy_value.pt` for 4 policy-only epochs with `--value-loss-weight 0.0`
  - quick seed-7000 benchmark: 20-0 vs random, 19-1 vs greedy, 0 timeouts
  - 100-game random sanity seed block 5000: 100-0-0, 0 timeouts, 0 fallback triggers
  - 100-game greedy seed block 8000: 97-3-0, 0 timeouts, 0 fallback triggers
  - 100-game greedy seed block 5000: 96-4-0, 0 timeouts, 1 fallback trigger
  - this improves over `policy_improve_margin500_001` on the checked blocks: seed 8000 improved from 87-13 to 97-3, seed 5000 improved from 94-6 to 96-4
  - recommend promoting `outputs/policy_improve_margin500_002/supervised_policy_value.pt` as the current local champion

Judgement calls made for checkpoint replay support:

- Used one shared `--checkpoint-path` for all `checkpoint` bots rather than separate per-seat checkpoint flags. This keeps the first champion-assisted workflow simple and unambiguous; per-seat checkpoint paths can be added later when we intentionally compare two different model generations.
- Kept the checkpoint loop fallback enabled by default for replay generation, but recorded fallback counts in every game and summary. This avoids known loop failures while preserving the ability to filter out or separately analyze fallback-influenced games.
- Cached checkpoint bots per factory/seat instead of reloading the model every game. This is much faster and relies on the existing per-game reset logic in `CheckpointPolicyBot`; separate factories still give each seat its own bot instance.
- Added provenance to both replay rows and `summary.json` rather than only printing it in logs. Future training filters can then exclude high-fallback games or select by checkpoint source without rerunning generation.
- Ran only a tiny 2-game checkpoint probe for validation. Checkpoint-generated data can reinforce model habits, so the next useful step is a controlled probe block and analysis, not immediate large-scale self-play.
- Trained one controlled champion-data mix after the 125-game probe, but rejected it on benchmark strength. This suggests naive imitation of the current champion against greedy is not enough; the next champion-assisted step needs a better target, weighting, or policy-improvement signal rather than simply adding model-play rows.

Judgement calls made for replay action improvement:

- Made state snapshots opt-in through `--include-state-snapshots`. This avoids bloating ordinary replay corpora while giving improvement scripts exact hidden decks, visible markets, nobles, and player zones when needed.
- Stored card/noble IDs rather than full card payloads. The base component table is canonical in code, so ID snapshots are smaller and less likely to drift.
- Added a search ranker instead of only calling `choose_action`. This lets relabeled rows record search value, heuristic score, loop penalty, and whether the original action was considered by the search candidate cap.
- Defaulted `run_improve_replay_actions.py` to writing changed rows only. This keeps improvement corpora focused and reduces duplicate imitation rows; `--write-unchanged` is available for diagnostics or full relabeled datasets.
- Kept original `final_value` on relabeled rows and marked that fact in metadata. This is the important caveat: if the action is changed, the old game outcome is no longer a clean value target. Use these rows primarily as policy-improvement targets, or train with reduced/zero value loss for this source until we add rollout-backed value labels.
- Defaulted relabeling to stateless per-row search. Because changing an action makes later original replay states off-policy, carrying a search history across all rows could inject inconsistent loop context. `--stateful-search-history` exists for sequential diagnostics.
- For `policy_improve_001`, fine-tuned from the current champion rather than training from scratch. A tiny relabeled corpus is not enough to relearn the whole policy; the right test was whether it can nudge the champion upward.
- After a power outage interrupted the first relabel attempt, verified the 25-game snapshot corpus was complete and restarted only the relabel step. The half-written relabel output was empty, so there was no useful partial data to salvage.
- Used `--write-unchanged` for the first training probe. Since 68% of actions changed, keeping unchanged rows gave the model both positive corrections and explicit "search agrees" anchors.
- Used `--value-loss-weight 0.0` for the fine-tune because changed actions retain original game outcomes and are not clean value targets.
- For the successful gated run, used `--min-search-margin 500` and excluded selected `TAKE_TOKENS` labels. This preserved the useful tactical corrections, mostly buy/reserve over original token takes, while avoiding the noisy high-volume token-take relabels that likely hurt the ungated run.
- Reused the same gate for `policy_improve_margin500_002` because the relabel distribution stayed similar to the first successful run. The model was slower in greedy benchmarks, so future iterations should track elapsed time / turns as secondary signals, not just win rate.

Current practical recommendation:

- Treat `outputs/policy_improve_margin500_002/supervised_policy_value.pt` as the current champion.
- Keep `outputs/policy_improve_margin500_001/supervised_policy_value.pt` as the previous champion baseline.
- Keep `outputs/warmstart_search_004/supervised_policy_value_best.pt` as the older pre-policy-improvement baseline.
- Use the benchmark CLI for future checkpoint comparisons instead of ad hoc inline Python.
- Keep the loop fallback enabled for GUI/benchmark play, but continue recording trigger counts so it remains measurable and removable.
- Important caveat: the fallback is a pragmatic guardrail, not a pure policy improvement. It may slightly affect marginal benchmark strength by overriding the model in rare loop-risk states, so compare future champions using both win rate and fallback trigger counts.
- Keep using sharded search corpora plus multi-path training instead of single giant replay jobs.
- Continue excluding stalled/timeout games when training.
- Watch raw checkpoint loop behavior separately from guarded benchmark behavior. The fallback mitigates observed loops, but the long-term fix should come from stronger data/training/self-play targets.

Machine transfer checklist:

- Git contains the code, tests, and handoff, but `data/` and `outputs/` are intentionally ignored. To continue from the same state on another machine, copy at minimum:
  - `outputs/policy_improve_margin500_002/supervised_policy_value.pt`
  - `outputs/policy_improve_margin500_002/supervised_metrics.json`
  - `outputs/policy_improve_margin500_001/supervised_policy_value.pt`
  - `outputs/warmstart_search_004/supervised_policy_value_best.pt`
  - `outputs/benchmarks/policy_improve_margin500_002_100g_seed5000.json`
  - `outputs/benchmarks/policy_improve_margin500_002_100g_seed8000.json`
  - `outputs/benchmarks/policy_improve_margin500_002_random_100g_seed5000.json`
- Optional but useful if continuing the exact data lineage without regenerating:
  - `data/corpus_checkpoint_greedy_snapshots_001/`
  - `data/corpus_checkpoint_greedy_snapshots_002/`
- User said they plan to copy the full suggested set above, including the optional corpus lineage directories. On the next machine, assume these artifacts may be present, but still run the `Test-Path` checks before launching training/eval.
- On the new machine, first verify:

```powershell
Test-Path outputs\policy_improve_margin500_002\supervised_policy_value.pt
.venv\Scripts\python.exe -m pytest -q --basetemp .codex_pytest_tmp_all
```

- NVIDIA CUDA remains the validated fast path. CPU can run benchmarks if needed; training and relabel loops are much slower without CUDA.

## Known Technical Debt

1. The flat action space is very large.
   - It is workable for the first supervised model.
   - It is not the preferred final architecture for serious self-play scale-up.
   - The likely future refactor is a legal-action scorer instead of a giant sparse softmax.

2. Replay validation loss is only loosely correlated with benchmark strength.
   - The training CLI now benchmarks both final and best-validation checkpoints.
   - For `warmstart_search_003`, the `best_validation` checkpoint is the true playing-strength winner.
   - Pick champions by benchmark results, not validation loss alone.

3. Search corpus generation is slow.
   - `search:greedy` is much slower than `greedy:random`.
   - Sharding is the preferred operational pattern now.

4. Search bot quality may still have hidden loop or style issues.
   - The obvious `repetition_cutoff` token-churn pattern was improved.
   - Corpus generation is currently clean, but continue watching new search styles if branching/depth change.

5. Raw checkpoint policy can still produce rare non-progress benchmark games.
   - The loop fallback is an inference guardrail, not a training fix.
   - It is intentionally narrow and has validated well so far, but it can still change move choice at the margin when it fires.
   - Track `model_loop_fallback_triggers`; a future stronger model should need it less often, and ideally not at all.
   - With fallback disabled, seed `5008` still ends at turn 39 with `termination_reason='repetition_cutoff'`, `final_scores=(0, 0)`, and `no_progress_streak=27`.
   - With fallback enabled, seed `5008` completed in 82 turns, model won 15-7, and the fallback fired once.
   - This remains a learned policy behavior / benchmark cutoff issue, distinct from the earlier `ShallowSearchBot` corpus-generation loop fix.

## Recommended Next Step

Near-term:

1. Treat the current champion as:

```powershell
outputs\policy_improve_margin500_002\supervised_policy_value.pt
```

The previous champion remains:

```powershell
outputs\policy_improve_margin500_001\supervised_policy_value.pt
```

2. Do not promote the mixed-data experiments.
   - Adding `greedy:random` data diluted the stronger search policy target and regressed benchmark strength.
   - Even a 3x search-weighted mix only reached 9-11 vs greedy on seed block 7000, while the current champion reached 16-4.
   - The lesson is that broader data needs quality control or weighting by playing strength, not just more rows.

3. Next experiment should continue the gated policy-improvement loop or run a small margin sweep.
   - `warmstart_search_005` was trained after another clean search-only block but regressed in the quick seed-7000 benchmark, so do not promote it without further study.
   - The first controlled checkpoint-vs-greedy mix also regressed in the quick seed-7000 benchmark, so do not promote it.
   - Do not add more naive imitation data. The useful recipe is:
     - generate a snapshot-enabled champion-vs-greedy corpus from `policy_improve_margin500_002`
     - run `run_improve_replay_actions.py` with the margin gate
     - train a small policy-focused candidate from `policy_improve_margin500_002` using `--value-loss-weight 0.0`
     - benchmark against `policy_improve_margin500_002` before promoting
   - First ungated version of this probe was completed as `policy_improve_001` and rejected. The useful lesson is that shallow-search relabeling changed many champion actions, but directly fine-tuning on all labels slightly hurt benchmark strength and reintroduced one random-game repetition cutoff.
   - The gated version `policy_improve_margin500_001` succeeded, and `policy_improve_margin500_002` improved again on the checked blocks.
   - Next improvement attempt can either repeat the same gated recipe for one more 25-game shard, or run a small margin sweep before scaling heavily.
   - If repeating directly, use `data/corpus_checkpoint_greedy_snapshots_003`, seed start `9500`, and output `outputs/policy_improve_margin500_003`.
   - If doing a margin sweep, reuse one snapshot corpus and compare at least `--min-search-margin 250`, `500`, and `1000` before training larger candidates.

4. Use the sharded replay pattern if continuing data scaling:

```powershell
.venv\Scripts\python.exe run_replay_corpus.py --output-dir data\corpus_search_greedy_004a --games 125 --seed-start 8000 --seat0-bot search --seat1-bot greedy --swap-seats --max-turns 400 --repetition-limit 4 --no-progress-limit 60 --search-depth 2 --search-max-branching 8 --search-buy-branching 5 --search-reserve-branching 2 --search-take-branching 2 --log-every 10
```

5. Train with repeated replay paths rather than merging corpora by hand.

6. Benchmark the resulting checkpoint with fallback metrics enabled, and choose champions by 100+ game benchmark performance rather than validation loss alone.

Checkpoint replay command that was validated:

```powershell
.venv\Scripts\python.exe run_replay_corpus.py --output-dir data\corpus_checkpoint_greedy_001 --games 25 --seed-start 9100 --seat0-bot checkpoint --seat1-bot greedy --checkpoint-path outputs\warmstart_search_004\supervised_policy_value_best.pt --checkpoint-device cuda --swap-seats --max-turns 400 --repetition-limit 4 --no-progress-limit 60 --log-every 5
```

Train on checkpoint-generated data only after inspecting `loop_fallback_triggers`, `timed_out_games`, and the per-row `game_model_loop_fallback_triggers` fields.
Use `--max-game-model-loop-fallback-triggers 0` to exclude all fallback-triggered games from supervised training.

Snapshot-enabled champion replay command template:

```powershell
.venv\Scripts\python.exe run_replay_corpus.py --output-dir data\corpus_checkpoint_greedy_snapshots_001 --games 25 --seed-start 9300 --seat0-bot checkpoint --seat1-bot greedy --checkpoint-path outputs\warmstart_search_004\supervised_policy_value_best.pt --checkpoint-device cuda --swap-seats --max-turns 400 --repetition-limit 4 --no-progress-limit 60 --include-state-snapshots --log-every 5
```

Action-improvement command template:

```powershell
.venv\Scripts\python.exe run_improve_replay_actions.py --input-path data\corpus_checkpoint_greedy_snapshots_001\replays.jsonl --output-path data\corpus_checkpoint_greedy_snapshots_001\improved_search_policy.jsonl --summary-path data\corpus_checkpoint_greedy_snapshots_001\improved_search_policy_summary.json --search-depth 2 --search-max-branching 10 --search-buy-branching 6 --search-reserve-branching 3 --search-take-branching 3 --log-every 1000
```

## Files Worth Reading First

- `README.md`
- `docs/environment_spec.md`
- `docs/windows_cuda_setup.md`
- `src/splendor_ai/engine/env.py`
- `src/splendor_ai/encoding/action_codec.py`
- `src/splendor_ai/training/run_supervised.py`
- `src/splendor_ai/training/replay.py`
- `src/splendor_ai/training/improve_replay_actions.py`
- `src/splendor_ai/eval/run_benchmark.py`
- `src/splendor_ai/eval/match.py`
- `src/splendor_ai/diagnostics.py`
- `src/splendor_ai/bots/heuristic_bot.py`
- `src/splendor_ai/bots/search_bot.py`

## Test Status

Direct verification completed in this session:

- `.venv\Scripts\python.exe -m pytest -q --basetemp .codex_pytest_tmp_all` passed: 85 passed
- `.venv\Scripts\python.exe -m py_compile` over modified source files succeeded
- Focused tests cover normal masked argmax behavior, fallback activation after loop evidence, and fallback logit-gap refusal.
- Benchmark seed 5008 before/after:
  - fallback enabled: completed, model won 15-7, 1 fallback trigger
  - fallback disabled: `repetition_cutoff` at turn 39, greedy adjudicated winner, 0 fallback triggers
- 100-game fallback validation:
  - seed block 5000: 85-15-0 vs greedy, 0 timeouts, 8 fallback triggers
  - seed block 6000: 85-15-0 vs greedy, 0 timeouts, 5 fallback triggers
- Mixed-data experiments:
  - `warmstart_mix_001`: 85 passed before run, trained successfully, benchmark champion final checkpoint 17-3 vs random and 5-15 vs greedy on seed block 7000
  - `warmstart_mix_002_search_weighted`: trained successfully, benchmark champion final checkpoint 19-1 vs random and 9-11 vs greedy on seed block 7000
  - then-current champion comparison on seed block 7000: 20-0 vs random and 16-4 vs greedy
- `warmstart_search_004`:
  - generated `corpus_search_greedy_004[1-5]`: 125 games, 7,964 steps, 0 stalled, 0 timed out
  - trained successfully on 71,824 search-only replay samples
  - post-train benchmark selected `supervised_policy_value_best.pt`
  - seed block 5000 vs greedy: 85-15-0, 0 timeouts, 4 fallback triggers
  - seed block 6000 vs greedy: 86-14-0, 0 timeouts, 1 fallback trigger
  - seed block 5000 vs random: 100-0-0, 0 timeouts, 4 fallback triggers
- Checkpoint replay support:
  - `.venv\Scripts\python.exe -m pytest -q --basetemp .codex_pytest_tmp_all` passed before filtering work: 89 passed
  - focused replay/checkpoint tests passed: 16 passed
  - `.venv\Scripts\python.exe -m py_compile` over modified replay/corpus/checkpoint files succeeded
  - `warmstart_search_005` completed but was rejected: final checkpoint 20-0 vs random and 15-5 vs greedy on seed block 7000
  - `data/corpus_checkpoint_greedy_probe_001`: 2 games, 126 steps, 0 stalled, 0 timed out, 0 fallback triggers
- Champion-data filtering and mix:
  - focused dataset/training/replay tests passed: 28 passed
  - `data/corpus_checkpoint_greedy_00[1-5]`: 125 games, 8,156 steps, 0 stalled, 0 timed out, 4 fallback triggers
  - `warmstart_champion_mix_001`: trained with `--max-game-model-loop-fallback-triggers 0`, final checkpoint 20-0 vs random and 15-5 vs greedy on seed block 7000; rejected
- Replay action improvement tooling:
  - focused tests passed: `.venv\Scripts\python.exe -m pytest -q tests\test_replay_export.py tests\test_improve_replay_actions.py --basetemp .codex_pytest_tmp_improve` -> 6 passed
  - full suite passed: `.venv\Scripts\python.exe -m pytest -q --basetemp .codex_pytest_tmp_all` -> 94 passed
  - CLI smoke probe passed:
    - generated 1 snapshot-enabled random-vs-random probe game
    - relabeled first 5 rows with `run_improve_replay_actions.py`
    - cleaned temporary probe output afterward
- Policy-improvement probe:
  - `data/corpus_checkpoint_greedy_snapshots_001`: 25 games, 1,650 steps, 0 stalled, 0 timed out, 0 fallback triggers
  - `improved_search_policy_full_summary.json`: 1,650 rows written, 1,123 changed, 527 unchanged, 0 missing snapshots
  - `outputs/policy_improve_001`: policy-only fine-tune from current champion, best validation epoch 4
  - benchmark final/best checkpoints both produced 19-1 vs random with 1 repetition cutoff and 18-2 vs greedy with 0 timeouts on seed block 7000
  - `.venv\Scripts\python.exe -m pytest -q --basetemp .codex_pytest_tmp_all` passed after the probe: 94 passed
- Gated policy-improvement probe:
  - focused gated relabel tests passed: `.venv\Scripts\python.exe -m pytest -q tests\test_improve_replay_actions.py --basetemp .codex_pytest_tmp_improve` -> 4 passed
  - `improved_search_policy_margin500_no_take_summary.json`: 901 rows written, 374 changed, 527 unchanged, 193 filtered by margin, 399 filtered by missing original score, 157 filtered by selected action type
  - `outputs/policy_improve_margin500_001`: policy-only fine-tune from `warmstart_search_004_best`, best validation epoch 4
  - quick seed-7000 benchmark: 20-0 vs random, 19-1 vs greedy, 0 timeouts
  - 100-game seed-5000 random sanity: 100-0-0, 0 timeouts, 0 fallback triggers
  - 100-game seed-5000 greedy: 94-6-0, 0 timeouts, 0 fallback triggers
  - 100-game seed-6000 greedy: 95-5-0, 0 timeouts, 1 fallback trigger
  - full suite passed after the gated probe: `.venv\Scripts\python.exe -m pytest -q --basetemp .codex_pytest_tmp_all` -> 96 passed
- Second gated policy-improvement probe:
  - `data/corpus_checkpoint_greedy_snapshots_002`: 25 games, 1,634 steps, 0 stalled, 0 timed out, 0 fallback triggers
  - `improved_search_policy_margin500_no_take_summary.json`: 882 rows written, 362 changed, 520 unchanged, 164 filtered by margin, 426 filtered by missing original score, 162 filtered by selected action type
  - `outputs/policy_improve_margin500_002`: policy-only fine-tune from `policy_improve_margin500_001`, best validation epoch 4
  - quick seed-7000 benchmark: 20-0 vs random, 19-1 vs greedy, 0 timeouts
  - 100-game seed-5000 random sanity: 100-0-0, 0 timeouts, 0 fallback triggers
  - 100-game seed-8000 greedy: 97-3-0, 0 timeouts, 0 fallback triggers
  - 100-game seed-5000 greedy: 96-4-0, 0 timeouts, 1 fallback trigger
  - full suite passed after the second gated probe: `.venv\Scripts\python.exe -m pytest -q --basetemp .codex_pytest_tmp_all` -> 96 passed

Plain pytest without `--basetemp` can still hit Windows temp permission issues around local pytest temp directories. Use a repo-local basetemp:

```powershell
.venv\Scripts\python.exe -m pytest -q --basetemp .codex_pytest_tmp_all
```

## Recent Commit Summary

Changes prepared in this session:

- Added conservative loop-aware inference fallback to `CheckpointPolicyBot`.
- Added fallback trigger counts to `GameResult` and benchmark JSON summaries.
- Added benchmark CLI flags to disable/tune the loop fallback.
- Reused loaded checkpoint bots across benchmark games in standalone and post-training benchmark paths.
- Added focused tests for fallback behavior and updated benchmark summary expectations.
- Validated fallback on greedy seed blocks 5000 and 6000, removing observed repetition cutoffs while preserving roughly the same win rate.
- Ran and rejected `warmstart_mix_001` and `warmstart_mix_002_search_weighted`; then-current champion remained `warmstart_search_003/supervised_policy_value_best.pt`.
- Generated clean `corpus_search_greedy_004[1-5]`, trained `warmstart_search_004`, and promoted `outputs/warmstart_search_004/supervised_policy_value_best.pt` based on 200-game greedy validation.
- Generated clean `corpus_search_greedy_005[1-5]` and trained `warmstart_search_005`, but rejected it because the quick greedy benchmark regressed.
- Added checkpoint bot support to replay corpus generation, including checkpoint provenance and fallback-trigger metadata in replay rows and corpus summaries.
- Added supervised filtering by checkpoint fallback trigger count, generated `corpus_checkpoint_greedy_00[1-5]`, trained `warmstart_champion_mix_001`, and rejected it because the quick greedy benchmark regressed.
- Added optional replay state snapshots, exact snapshot reconstruction, shallow-search action ranking, and `run_improve_replay_actions.py` for policy-improvement relabeling.
- Added `run_supervised.py --init-checkpoint`, ran `policy_improve_001`, and rejected it because the quick benchmark was slightly worse than `warmstart_search_004_best`.
- Added confidence/margin-gated relabeling, trained `policy_improve_margin500_001`, and promoted it because it beat the previous champion by 18 games over the same two 100-game greedy validation blocks while reducing fallback use.
