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
- 295-float public observation tensor
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

Strategic assessment:

- The current path is the right bootstrap path: exact simulator first, legal observation/action masking, baselines, benchmark tooling, heuristic/search warm start, and gated policy improvement.
- This path should not be mistaken for the final superhuman training recipe. More shallow-search relabeling against `greedy` can improve the current champion, but it risks producing a strong anti-greedy specialist rather than a broadly superhuman Splendor player.
- Splendor is not Go: it has hidden deck/refill uncertainty, structured economic actions, tactical tempo, and a much more irregular action space. AlphaGo is a useful reference for the overall loop of policy/value learning plus search, but the architecture should be adapted rather than copied.
- The likely superhuman path is: keep the exact simulator and benchmark discipline, harvest one or two more small gated warm-start improvements if they keep working, then pivot toward a structured legal-action scorer and self-play/league training against historical checkpoints and stronger search baselines.
- Greedy is no longer a sufficient long-term benchmark. Keep it as a sanity check, but champion promotion should increasingly depend on stronger tuned search opponents, previous neural champions, fallback-trigger counts, and eventually human/expert-style evaluation.

Legal-action scorer pivot status:

- Added `src/splendor_ai/encoding/action_features.py`.
  - Encodes each candidate legal action into a 79-float structured feature vector.
  - Includes action type, board/reserved location, target card, chosen noble, token flow, and derived economics.
  - Latest version also includes opponent-relative context features for score margin, threat state, reserve denial, token overflow, affordability shift, and opponent proximity to the target card.
- Public observation tensors now include opponent reserved-card identities.
  - This is legal public information in Splendor and matters for reserve denial / opponent-threat modeling.
  - Old checkpoints remain loadable because inference and dataset loading truncate/pad encoded vectors to the checkpoint's saved config size.
- Added `LegalActionReplayDataset`.
  - Requires snapshot-enabled replay rows.
  - Reconstructs the pre-action state, regenerates legal actions, encodes legal-action features, and maps the replay `action_index` to `target_action_row`.
- Added variable-length scorer batching through `collate_legal_action_replay_samples`.
- Added `LegalActionScorerModel`.
  - Encodes state observation and each legal action separately.
  - Produces one score per legal action plus a value estimate.
- Added legal-action scorer supervised training.
  - CLI entry point: `run_legal_action_supervised.py`.
  - Saves `legal_action_scorer.pt`, `legal_action_scorer_best.pt`, and `legal_action_scorer_metrics.json`.
  - Supports optional search-ranking distillation:
    - `run_improve_replay_actions.py` now stores `search_action_scores` for ranked searched candidate actions.
    - `LegalActionReplayDataset` loads those stored search values into padded per-legal-action targets.
    - `run_legal_action_supervised.py` accepts `--search-distill-loss-weight` and `--search-distill-temperature`.
    - This lets the scorer learn from the search ranking over legal actions, not only the single top-1 relabeled action.
- Added `LegalActionScorerBot` for checkpoint-backed inference.
- Added benchmark support for scorer checkpoints.
  - Use `run_benchmark.py --model-kind legal_action_scorer`.
  - The default remains `--model-kind flat_policy` for old `CheckpointPolicyBot` checkpoints.
  - Smoke benchmark passed for `outputs/legal_action_scorer_001/legal_action_scorer_best.pt` vs random, 1-0, 0 timeouts.
- Added direct checkpoint-vs-checkpoint benchmark support.
  - Use `--baseline-checkpoint` plus `--baseline-model-kind`.
  - Candidate win/loss accounting is seat-based, so it works even if both sides are model-backed.
- Added `search` as a benchmark opponent.
  - Use `run_benchmark.py --opponents search`.
  - Search settings are recorded in benchmark JSON: `--search-depth`, `--search-max-branching`, `--search-buy-branching`, `--search-reserve-branching`, and `--search-take-branching`.
- Added legal-action scorer checkpoint support to replay corpus generation.
  - Use `run_replay_corpus.py --seat0-bot checkpoint --seat1-bot checkpoint --checkpoint-model-kind legal_action_scorer`.
  - Metadata records checkpoint path, model kind, device, and loop-fallback settings.
  - Added per-seat checkpoint overrides for league corpora:
    - `--seat0-checkpoint-path`
    - `--seat0-checkpoint-model-kind`
    - `--seat1-checkpoint-path`
    - `--seat1-checkpoint-model-kind`
  - Shared `--checkpoint-path` / `--checkpoint-model-kind` remain backward-compatible defaults.
  - Smoke corpus `data/corpus_league_smoke_001` verified `legal_action_scorer_003` vs `legal_action_scorer_002` with 2 completed games, 130 steps, 0 stalls/timeouts/fallback triggers.
- Added structured replay corpus manifests for league/evaluation hygiene.
  - `run_replay_corpus.py` now writes `manifest.json` next to `replays.jsonl` and `summary.json`.
  - CLI flag: `--source-type`, default `generated_replay`; use labels like `league`, `self_play`, or `search_vs_greedy`.
  - Manifest records source type, replay/summary filenames, seed block, aggregate results, matchup-level results, bot classes, checkpoint paths, checkpoint model kinds, and CLI provenance metadata.
  - This is intended to make future league training filters explicit instead of relying on filenames and handoff notes.
- Added manifest reporting utility:
  - `run_manifest_report.py <corpus_dir_or_manifest> [...]`
  - Prints source type, seed block, games, steps, seat W-L-D, stalls/timeouts, fallback triggers, and per-matchup rows.
  - Use the per-matchup rows for actor-vs-actor interpretation when `--swap-seats` is enabled; the top-level W-L-D is seat-based.
- Generated first structured league probe with manifests:
  - All shards used `--source-type league`, `--include-state-snapshots`, `--swap-seats`, 8 games each, 40 games total.
  - `data/corpus_league_manifest_001_001_vs_003`: 8 games, 518 steps, 0 stalls/timeouts/fallback triggers.
    - `001` as seat0 vs `003`: 4-0-0.
    - `003` as seat0 vs `001`: 2-2-0, so actor-level `001` result is 6-2 over this tiny block.
  - `data/corpus_league_manifest_001_001_vs_flat`: 8 games, 484 steps, 0 stalls/timeouts/fallback triggers.
    - Actor-level `001` result vs old flat champion: 6-2.
  - `data/corpus_league_manifest_001_003_vs_flat`: 8 games, 476 steps, 0 stalls/timeouts/fallback triggers.
    - Actor-level `003` result vs old flat champion: 5-3.
  - `data/corpus_league_manifest_001_001_vs_search`: 8 games, 508 steps, 0 stalls/timeouts/fallback triggers.
    - Actor-level `001` result vs shallow search: 3-5.
  - `data/corpus_league_manifest_001_003_vs_search`: 8 games, 508 steps, 0 stalls/timeouts/fallback triggers.
    - Actor-level `003` result vs shallow search: 1-7.
  - These are data-quality probe shards, not promotion benchmarks. They are small but clean and suitable as structured league input for the next training-loop experiment.
- Generated scaled structured league corpus block `corpus_league_manifest_002_*`:
  - 5 shards x 50 games = 250 games total.
  - All shards used `--source-type league`, `--include-state-snapshots`, `--swap-seats`, fresh seed blocks 42000-42449.
  - All shards were clean: 0 stalls, 0 timeouts, 0 fallback triggers.
  - `data/corpus_league_manifest_002_001_vs_003`: 50 games, 3,064 steps.
    - Actor-level `001` vs `003`: 18-32-0. This block favors `003`.
  - `data/corpus_league_manifest_002_001_vs_flat`: 50 games, 3,082 steps.
    - Actor-level `001` vs old flat champion: 27-23-0.
  - `data/corpus_league_manifest_002_003_vs_flat`: 50 games, 3,178 steps.
    - Actor-level `003` vs old flat champion: 29-21-0.
  - `data/corpus_league_manifest_002_001_vs_search`: 50 games, 3,142 steps.
    - Actor-level `001` vs shallow search: 9-39-2.
  - `data/corpus_league_manifest_002_003_vs_search`: 50 games, 3,128 steps.
    - Actor-level `003` vs shallow search: 12-38-0.
  - This is now a usable manifest-audited league data block for the next conservative training experiment. It still should not be treated as a promotion benchmark because each matchup has only 50 games.
- First conservative manifest-league training candidate, `legal_action_scorer_011`:
  - Added legal-action scorer training support for:
    - `--policy-target-filter winner`, which trains only on actions by the eventual game winner.
    - outcome sample weights: `--winner-action-weight`, `--loser-action-weight`, `--draw-action-weight`.
    - `--validation-split-unit row|game|source`; use `game` by default for league training.
  - Trained from `outputs/legal_action_scorer_003/legal_action_scorer_best.pt`.
  - Training data: all five `corpus_league_manifest_002_*` replay files, winner-action rows only.
  - Training setup: 7,735 total rows, 6,987 train / 748 game-heldout validation, 6 epochs, CPU, `--value-loss-weight 0.0`, learning rate `0.0001`.
  - Output: `outputs/legal_action_scorer_011/`; best validation checkpoint was epoch 3.
  - Final train: loss 0.6773, accuracy 0.7736. Final validation: loss 0.7938, accuracy 0.7714.
  - Benchmarks:
    - vs `legal_action_scorer_003`, 100 games seed block 43000: 52-48-0, average margin -0.13, 0 fallback triggers.
    - vs `legal_action_scorer_001`, 100 games seed block 43100: 46-54-0, average margin -0.25, 0 fallback triggers.
    - vs old flat champion `policy_improve_margin500_002`, 100 games seed block 43200: 69-31-0, average margin +2.75, candidate fallback triggers 0, baseline fallback triggers 1.
    - standardized shallow-search block seed 20000: 24-76-0, average margin -4.46, 0 fallback triggers.
  - Conclusion: reject `011`. Winner-only league imitation from this small league block did not improve the stable neural/search gates. It still beats the old flat model, but it is worse than `001`/`003` versus search and not clearly stronger head-to-head.
- Added policy-vs-search disagreement analysis:
  - CLI: `run_policy_search_disagreement.py`
  - Supports `--model-kind legal_action_scorer` and `--model-kind flat_policy`.
  - Reconstructs snapshot replay states, ranks the checkpoint's legal actions, ranks shallow-search actions, and writes:
    - overall agreement rate
    - agreement by model action type
    - agreement by search action type
    - agreement by early/mid/late phase
    - model-action-type vs search-action-type matrix
    - top high-search-margin disagreement examples
  - Analysis outputs:
    - `outputs/analysis/003_search_disagreement.json`
    - `outputs/analysis/001_search_disagreement.json`
  - Both used 1,000 rows from the corresponding `corpus_league_manifest_002_*_vs_search/replays.jsonl` shard and the standard depth-2 search settings.
  - `legal_action_scorer_003` vs search:
    - agreement rate: 0.506
    - disagreements: 494 / 1,000
    - model action missing from searched candidate set: 256 disagreements
    - phase agreement: early 0.325, mid 0.588, late 0.600
    - by model action type:
      - `BUY_VISIBLE`: 520 rows, agreement 0.690
      - `BUY_RESERVED`: 130 rows, agreement 0.785
      - `TAKE_TOKENS`: 180 rows, agreement 0.078
      - `RESERVE_DECK`: 102 rows, agreement 0.157
      - `RESERVE_VISIBLE`: 68 rows, agreement 0.221
  - `legal_action_scorer_001` vs search:
    - agreement rate: 0.526
    - disagreements: 474 / 1,000
    - model action missing from searched candidate set: 252 disagreements
    - phase agreement: early 0.330, mid 0.629, late 0.605
    - by model action type:
      - `BUY_VISIBLE`: 495 rows, agreement 0.711
      - `BUY_RESERVED`: 145 rows, agreement 0.869
      - `TAKE_TOKENS`: 167 rows, agreement 0.084
      - `RESERVE_DECK`: 128 rows, agreement 0.164
      - `RESERVE_VISIBLE`: 65 rows, agreement 0.200
  - Main diagnosis:
    - The neural scorer is reasonably aligned with search when it chooses buy actions.
    - The large mismatch is early-game token-taking and reserve behavior. Agreement is only about one-third in early positions.
    - Many token/reserve disagreements are missing from the pruned search candidate set, which means the model often likes actions that search's heuristic ordering does not even keep.
    - The biggest high-margin examples are late-game tactical misses: buying the wrong visible card or missing a decisive reserve/buy.
  - Implication: the next work should be targeted, not another broad imitation pass. Focus on action feature/label quality for token-taking, reserve decisions, and late-game tactical buy selection.
- Added targeted search-correction corpus generation:
  - CLI: `run_targeted_search_corpus.py`
  - Compares a model's top legal action with shallow search on snapshot replay states.
  - Writes only targeted disagreement rows, with the replay target replaced by the search action.
  - Stores original action, model top action, search margin, full searched action scores, source replay path, and `target_source="targeted_search_disagreement"`.
  - Intended targets are early token/reserve disagreements and late tactical buy/reserve misses, not broad imitation.
- First targeted-corpus smoke results:
  - Completed smoke command on 100 rows from `data/corpus_league_manifest_002_001_vs_search/replays.jsonl`.
  - Output: `data/corpus_targeted_search_smoke_001/`.
  - Summary: 100 rows analyzed, 54 model/search disagreements, 29 rows written.
  - Written target types: `RESERVE_VISIBLE` 13, `TAKE_TOKENS` 11, `RESERVE_DECK` 4, `BUY_RESERVED` 1.
  - Phase split: early 20, mid 8, late 1.
  - A larger CPU attempt capped at 1,000 rows was interrupted after the command timeout; it left a usable partial file with 153 targeted rows at `data/corpus_targeted_search_001/targeted_search_disagreements.jsonl`, but no summary because the command did not finish.
- Smoke fine-tune using the partial targeted file:
  - Output: `outputs/legal_action_scorer_012_targeted_smoke/`.
  - Parent: `outputs/legal_action_scorer_001/legal_action_scorer_best.pt`.
  - Training data: the three original improved checkpoint/greedy corpora plus the 153-row partial targeted file.
  - Total rows: 2,752; game-heldout validation; 2 epochs; CPU.
  - Best checkpoint: epoch 2, validation accuracy 0.7387, validation loss 0.9203.
  - Quick 20-game head-to-head vs `legal_action_scorer_001`, seed 52000: 9-11, 0 stalls/timeouts/fallback triggers.
  - Quick 20-game benchmark vs shallow search, seed 52100: 4-16, 0 stalls/timeouts/fallback triggers.
  - Conclusion: workflow is functional, but this is not a promotion candidate. The targeted file is too small and old-checkpoint fine-tuning cannot use the new 295/79 feature widths.
- Full targeted-corpus fine-tune from `legal_action_scorer_001`, `legal_action_scorer_013_targeted_full_from001`:
  - Full targeted corpus: `data/corpus_targeted_search_full_001/targeted_search_disagreements.jsonl`.
  - Corpus summary: 6,270 rows analyzed, 3,058 disagreements, 1,842 targeted rows written, 0 missing snapshots.
  - Written target types: `TAKE_TOKENS` 803, `RESERVE_VISIBLE` 720, `RESERVE_DECK` 184, `BUY_VISIBLE` 118, `BUY_RESERVED` 17.
  - Phase split: early 939, mid 628, late 275.
  - Training output: `outputs/legal_action_scorer_013_targeted_full_from001/`.
  - Parent: `outputs/legal_action_scorer_001/legal_action_scorer_best.pt`; because of this, the model still uses the old 256/64 feature widths.
  - Training data: three original improved checkpoint/greedy corpora plus the full targeted corpus.
  - Total rows: 4,425; 4 epochs; game-heldout validation; best epoch 4; training time about 184 seconds on CPU.
  - Validation accuracy is low, around 0.366 at best epoch, but this is expected because targeted rows intentionally contradict the old policy distribution.
  - Benchmarks:
    - vs `legal_action_scorer_001`, 100 games seed 53000: 50-50-0, 1 timed-out game.
    - vs shallow search, 20 games seed 53100: 8-12-0, 0 stalls/timeouts/fallback triggers.
    - `legal_action_scorer_001` on the same search seed block 53100: 3-17-0.
    - vs `legal_action_scorer_003`, 100 games seed 53200: 55-45-0, 0 stalls/timeouts/fallback triggers.
  - Current read: promising but not yet promoted. It improves the same-seed 20-game search probe versus `001`, ties `001` directly, and beats `003` directly, but needs a 100-game standardized search benchmark before being treated as a new champion.
- Wide-feature from-scratch targeted model, `legal_action_scorer_014_wide_targeted_fromscratch`:
  - Trained without `--init-checkpoint`, so this is the first candidate that actually uses the new 295-float observation and 79-float legal-action feature vectors.
  - Training data: three original improved checkpoint/greedy corpora plus `data/corpus_targeted_search_full_001/targeted_search_disagreements.jsonl`.
  - Output: `outputs/legal_action_scorer_014_wide_targeted_fromscratch/`.
  - Total rows: 4,425; game-heldout validation; 8 epochs; CPU; best validation epoch 7.
  - Best validation accuracy: 0.5208; final train accuracy: 0.5982; training time about 320 seconds.
  - Quick neural benchmarks:
    - vs `legal_action_scorer_001`, 100 games seed 54000: 57-43-0.
    - vs `legal_action_scorer_003`, 100 games seed 54100: 60-40-0.
  - Quick search probe seed 54200:
    - `014` vs search, 20 games: 11-9-0.
    - `001` same seed block: 7-13-0.
    - `003` same seed block: 6-14-0.
  - Standardized search gate seed 20000:
    - `014`: 28-71-1, avg margin -2.79, avg model score 11.37, 0 timeouts.
    - `013`: 29-70-1, avg margin -3.06.
    - `001`: 30-70-0, avg margin -3.92.
    - `003`: 29-71-0, avg margin -3.86.
  - Current read: `014` is not a clean promotion because it did not improve standardized search win rate, but it is the best model so far by search score margin and clearly beats `001`/`003` head-to-head. It is a useful parent for the next branch, not a final champion.
- Policy-guided search branch:
  - Added `PolicyGuidedSearchBot`.
  - It reuses the existing `ShallowSearchBot` minimax/leaf evaluation but adds a legal-action scorer prior to action ordering/pruning.
  - The policy prior is rank-normalized per state, then blended into the heuristic ordering score with `--policy-prior-weight` / `policy_prior_weight`.
  - Added `CheckpointLegalActionPriorScorer` for loading legal-action scorer checkpoints as search priors.
  - Benchmark support:
    - `run_benchmark.py --model-kind policy_guided_search`
    - Uses the existing `--search-depth`, `--search-max-branching`, `--search-buy-branching`, `--search-reserve-branching`, `--search-take-branching` settings for the guided search.
    - New flag: `--policy-prior-weight`, default `2000`.
  - Replay-corpus support:
    - `run_replay_corpus.py --checkpoint-model-kind policy_guided_search`
    - Metadata records `model_kind="policy_guided_search"` and `policy_prior_weight`.
  - Verification:
    - Full suite passed after implementation: `python -m pytest -q --basetemp .codex_pytest_tmp_all` -> 143 passed.
  - Initial smoke benchmarks with `014` as the prior:
    - `outputs/benchmarks/policy_guided_search_014_vs_search_20g_seed55000.json`
      - Policy-guided search `014` vs plain shallow search: 15-5-0, avg margin +3.25, 0 timeouts.
    - `outputs/benchmarks/policy_guided_search_014_vs_greedy014_20g_seed55100.json`
      - Policy-guided search `014` vs greedy `014`: 14-6-0, avg margin +1.40, 0 timeouts.
  - 100-game confirmation:
    - `outputs/benchmarks/policy_guided_search_014_vs_search_100g_seed56000.json`
      - Policy-guided search `014` vs plain shallow search: 44-55-1, avg margin -0.58, avg model score 12.58 vs 13.16, 0 timeouts.
      - Seat split was asymmetric: model as seat 0 went 26-24-0 with avg margin +0.58; model as seat 1 went 18-31-1 with avg margin -1.74.
  - Current read: the 15-5 smoke was likely variance. Policy-guided search is close to plain shallow search but not a confirmed improvement at `policy_prior_weight=2000`; tune/ablate it before using it as the next teacher.
- Smoke-trained the first real scorer on `data/corpus_checkpoint_greedy_snapshots_003/improved_search_policy_margin500_no_take.jsonl`.
  - Output: `outputs/legal_action_scorer_smoke_001/`
  - 800 samples, 720 train / 80 validation, 1 epoch, CPU.
  - Small model config: state hidden 64, action hidden 32, scorer hidden 64.
  - Result: train loss 4.3837, train accuracy 0.2542, validation loss 4.0197, validation accuracy 0.2125.
  - This is a functionality smoke test only, not a strength candidate.
- Verification after scorer scaffolding:
  - `python -m pytest -q --basetemp .codex_pytest_tmp_all`
  - 111 tests passed.
- Verification after scorer benchmark support:
  - `python -m pytest tests/test_run_benchmark.py tests/test_legal_action_supervised.py tests/test_model_bot.py -q --basetemp .codex_pytest_tmp_benchmark`
  - 13 tests passed.
- First full legal-action scorer candidate:
  - Training output: `outputs/legal_action_scorer_001/`
  - Trained on snapshot/improved corpora `001`, `002`, and `003`, 2,583 total rows, 8 epochs, CPU.
  - Best validation checkpoint: `outputs/legal_action_scorer_001/legal_action_scorer_best.pt`.
  - Validation accuracy reached 0.6705 by epoch 7; best validation loss was epoch 4.
  - 100-game sanity benchmark, seed block 8000: 100-0 vs random, 96-4 vs greedy, 0 timeouts/stalls/fallback triggers.
  - Direct 100-game head-to-head vs old flat champion `outputs/policy_improve_margin500_002/supervised_policy_value.pt`, seed block 9000: 64-32-4 for the legal-action scorer, 0 timeouts/stalls/fallback triggers.
  - Direct-match average score margin was +2.25; candidate score average 13.91 vs baseline 11.66.
  - Seat split: candidate as seat 0 went 36-11-3, candidate as seat 1 went 28-21-1. This shows a first-player effect but still a positive result for the scorer from both seats.
  - Confirmation direct 100-game head-to-head on seed block 10000: 63-37-0 for the legal-action scorer, 0 timeouts/stalls/fallback triggers.
  - Confirmation average score margin was +3.14; candidate score average 13.93 vs baseline 10.79.
  - Confirmation seat split: candidate as seat 0 went 35-15-0, candidate as seat 1 went 28-22-0.
  - Shallow-search benchmark, depth 2 / max branching 10 / buy 6 / reserve 3 / take 3, seed block 11000, 20 games: scorer 4, search 16, draws 0, 0 timeouts/stalls/fallback triggers.
  - Search benchmark average score margin was -4.25 for the scorer; scorer average score 10.00 vs search 14.25.
  - Search benchmark seat split: scorer as seat 0 went 4-6, scorer as seat 1 went 0-10. This is the clearest current weakness and confirms search remains a stronger benchmark than old neural/greedy opponents.
  - Old flat champion `policy_improve_margin500_002` on the same search benchmark/seed block went 2-18, draws 0, 0 timeouts/stalls/fallback triggers.
  - Old flat champion average search margin was -5.85; model average score 9.20 vs search 15.05. The scorer therefore improved against search, but search remains clearly stronger than both neural checkpoints.
- First scorer self-play/search-relabel iteration, `legal_action_scorer_002`:
  - Generated `data/corpus_legal_scorer_selfplay_snapshots_001` from `legal_action_scorer_001` self-play.
  - 10 games, 626 steps, 0 stalled, 0 timed out, 0 fallback triggers.
  - Relabeled with shallow search using depth 2 / max branching 10 / buy 6 / reserve 3 / take 3, `--min-search-margin 500`, `--exclude-changed-action-types TAKE_TOKENS`, and `--write-unchanged`.
  - Relabel output: `data/corpus_legal_scorer_selfplay_snapshots_001/improved_search_policy_margin500_no_take.jsonl`.
  - 340 rows written: 25 high-confidence changed labels and 315 unchanged anchors.
  - Filtered 112 low-margin changes, 79 changed rows where the original action was pruned/missing from search candidates, and 95 selected-token-take changes.
  - Fine-tuned from `outputs/legal_action_scorer_001/legal_action_scorer_best.pt` on the three prior improved corpora plus the new scorer self-play relabel corpus.
  - Output: `outputs/legal_action_scorer_002/`; best validation checkpoint was epoch 4.
  - Training dataset: 2,923 total rows, 2,631 train / 292 validation, 4 epochs, CPU, `--value-loss-weight 0.0`.
  - Final train accuracy 0.7355; validation accuracy 0.6952; validation loss 0.9917.
  - Direct 100-game head-to-head vs `legal_action_scorer_001`, seed block 13000: `legal_action_scorer_002` went 54-45-1, 0 timeouts/stalls/fallback triggers.
  - Confirmation direct 100-game head-to-head vs `legal_action_scorer_001`, seed block 14000: `legal_action_scorer_002` went 49-51-0, 0 timeouts/stalls/fallback triggers.
  - Direct 100-game head-to-head vs old flat champion `policy_improve_margin500_002`, seed block 13000: `legal_action_scorer_002` went 62-38-0, 0 timeouts/stalls/fallback triggers.
  - Shallow-search benchmark on the same seed block/settings as `legal_action_scorer_001`, 20 games seed 11000: `legal_action_scorer_002` went 6-14, improving over `legal_action_scorer_001` at 4-16 and old flat champion at 2-18.
  - Search benchmark average score margin improved to -2.90 from `legal_action_scorer_001` at -4.25, but search remains clearly stronger.
  - Conclusion: `legal_action_scorer_002` is directionally useful because it improved the search benchmark, but it is not clearly stronger than `legal_action_scorer_001` in direct neural head-to-head yet.
- Second scorer self-play/search-relabel iteration, `legal_action_scorer_003`:
  - Generated `data/corpus_legal_scorer_selfplay_snapshots_002` from `legal_action_scorer_002` self-play.
  - 25 games, 1,528 steps, 0 stalled, 0 timed out, 0 fallback triggers.
  - Relabeled with shallow search using the same settings and gate as `002`: depth 2 / max branching 10 / buy 6 / reserve 3 / take 3, `--min-search-margin 500`, `--exclude-changed-action-types TAKE_TOKENS`, and `--write-unchanged`.
  - Relabel output: `data/corpus_legal_scorer_selfplay_snapshots_002/improved_search_policy_margin500_no_take.jsonl`.
  - 834 rows written: 68 high-confidence changed labels and 766 unchanged anchors.
  - Filtered 279 low-margin changes, 219 changed rows where the original action was pruned/missing from search candidates, and 196 selected-token-take changes.
  - Fine-tuned from `outputs/legal_action_scorer_002/legal_action_scorer_best.pt` on the three prior improved corpora plus both scorer self-play relabel corpora.
  - Output: `outputs/legal_action_scorer_003/`; best validation checkpoint was epoch 1.
  - Training dataset: 3,757 total rows, 3,382 train / 375 validation, 4 epochs, CPU, `--value-loss-weight 0.0`.
  - Epoch 1 best validation: train accuracy 0.7732, validation accuracy 0.7840, validation loss 0.9121. Final epoch validation accuracy stayed 0.7840 but validation loss rose to 0.9191.
  - Direct 100-game head-to-head vs `legal_action_scorer_002`, seed block 16000: `legal_action_scorer_003_best` went 46-54-0, 0 timeouts/stalls/fallback triggers.
  - Confirmation direct 100-game head-to-head vs `legal_action_scorer_002`, seed block 17000: `legal_action_scorer_003_best` went 60-40-0, 0 timeouts/stalls/fallback triggers.
  - Final epoch checkpoint also went 60-40-0 vs `legal_action_scorer_002` on seed block 17000.
  - Direct 100-game head-to-head vs `legal_action_scorer_001`, seed block 16000: `legal_action_scorer_003_best` went 47-53-0, 0 timeouts/stalls/fallback triggers.
  - Direct 100-game head-to-head vs old flat champion `policy_improve_margin500_002`, seed block 16000: `legal_action_scorer_003_best` went 64-35-1, 0 timeouts/stalls/fallback triggers.
  - Shallow-search benchmark on the same seed block/settings as `001` and `002`, 20 games seed 11000: `legal_action_scorer_003_best` went 10-10, improving over `002` at 6-14, `001` at 4-16, and old flat champion at 2-18.
  - Search benchmark average score margin improved to -0.35 from `002` at -2.90 and `001` at -4.25.
  - Confirmation shallow-search benchmark, 20 games seed 12000: `legal_action_scorer_003_best` went 3-17, average score margin -5.25, 0 timeouts/stalls/fallback triggers.
  - Combined shallow-search result across seed blocks 11000 and 12000: `legal_action_scorer_003_best` went 13-27, average score margin -2.80.
  - Standardized 100-game shallow-search benchmark, seed block 20000:
    - `legal_action_scorer_001`: 30-70-0, average margin -3.92.
    - `legal_action_scorer_002`: 25-75-0, average margin -3.99.
    - `legal_action_scorer_003`: 29-71-0, average margin -3.86.
  - Conclusion: the larger search benchmark says `001` and `003` are effectively tied against search, while `002` is slightly worse. The earlier `003` 10-10 result was noise. `003` is not a clean promotion because it is volatile against `002`, lost to `001` on the checked direct block, and does not clearly beat `001` on the standardized 100-game search block.
- First mixed league/search-relabel iteration, `legal_action_scorer_004`:
  - Generated four 5-game league corpora with per-seat checkpoint pairings:
    - `data/corpus_league_004_003_vs_001`
    - `data/corpus_league_004_003_vs_002`
    - `data/corpus_league_004_003_vs_flat`
    - `data/corpus_league_004_001_vs_002`
  - Combined corpus size: 20 games, 1,258 steps, 0 stalled, 0 timed out, 0 fallback triggers.
  - Relabeled all four corpora together with shallow search using depth 2 / max branching 10 / buy 6 / reserve 3 / take 3, `--min-search-margin 500`, `--exclude-changed-action-types TAKE_TOKENS`, and `--write-unchanged`.
  - Relabel output: `data/corpus_league_004_improved/improved_search_policy_margin500_no_take.jsonl`.
  - 699 rows written: 68 high-confidence changed labels and 631 unchanged anchors.
  - Filtered 211 low-margin changes, 180 changed rows where the original action was pruned/missing from search candidates, and 168 selected-token-take changes.
  - Fine-tuned from stable parent `outputs/legal_action_scorer_001/legal_action_scorer_best.pt` on the three prior improved corpora, both scorer self-play relabel corpora, and the new league relabel corpus.
  - Output: `outputs/legal_action_scorer_004/`; best validation checkpoint was epoch 3.
  - Training dataset: 4,456 total rows, 4,011 train / 445 validation, 4 epochs, CPU, `--value-loss-weight 0.0`.
  - Best validation epoch 3: train accuracy 0.7886, validation accuracy 0.7753, validation loss 0.9639.
  - Direct 100-game head-to-head vs `legal_action_scorer_001`, seed block 25000: `legal_action_scorer_004_best` went 48-52-0, average margin -0.12, 0 timeouts/stalls/fallback triggers.
  - Direct 100-game head-to-head vs `legal_action_scorer_003`, seed block 25000: `legal_action_scorer_004_best` went 51-49-0, average margin +0.07, 0 timeouts/stalls/fallback triggers.
  - Direct 100-game head-to-head vs old flat champion `policy_improve_margin500_002`, seed block 25000: `legal_action_scorer_004_best` went 63-37-0, average margin +1.69, 0 timeouts/stalls/fallback triggers.
  - Standardized 100-game shallow-search benchmark, seed block 20000: `legal_action_scorer_004_best` went 21-79-0, average margin -4.54, 0 timeouts/stalls/fallback triggers.
  - Conclusion: do not promote `legal_action_scorer_004`. The mixed league corpus did not improve strength; `004` is tied-ish with `001`/`003`, still beats the old flat champion, but regressed versus search compared with `001` at 30-70 and `003` at 29-71 on the same 100-game search block.
- Search-ranking distillation support, ready for the next candidate:
  - Implemented after rejecting `004`.
  - Future relabeled corpora include `search_action_scores` for the searched candidate list.
  - Training can add a soft cross-entropy term from those search values using `--search-distill-loss-weight`.
  - Tiny smoke run succeeded:
    - relabel: `data/distill_smoke_001/improved_search_scores.jsonl`, 12 rows
    - train: `outputs/legal_action_scorer_distill_smoke_001/`, 1 CPU epoch from `legal_action_scorer_001`
  - This smoke checkpoint is only a functionality check, not a strength candidate.
- First search-ranking distillation candidate, `legal_action_scorer_005`:
  - Attempted to regenerate one full combined depth-2 search-relabeled corpus with `search_action_scores` from the existing snapshot sources.
  - CPU relabeling was too slow for a full pass in-session. The combined relabel job was stopped after 502 valid written rows.
  - Distillation slice:
    - `data/corpus_search_distill_005/improved_search_policy_margin500_no_take.jsonl`
    - 502 rows, all with `search_action_scores`
    - 216 changed labels, 286 unchanged anchors
    - no normal summary JSON, because the long relabel was intentionally stopped before completion
  - Trained from `outputs/legal_action_scorer_001/legal_action_scorer_best.pt`.
  - Training used the prior top-1 improved corpora plus the 502-row scored distillation slice:
    - `corpus_checkpoint_greedy_snapshots_001`
    - `corpus_checkpoint_greedy_snapshots_002`
    - `corpus_checkpoint_greedy_snapshots_003`
    - `corpus_legal_scorer_selfplay_snapshots_001`
    - `corpus_legal_scorer_selfplay_snapshots_002`
    - `corpus_league_004_improved`
    - `corpus_search_distill_005`
  - Training command used `--search-distill-loss-weight 0.1`, `--search-distill-temperature 1000`, `--value-loss-weight 0.0`, 4 epochs, CPU.
  - Output: `outputs/legal_action_scorer_005/`; best validation checkpoint was epoch 4.
  - Dataset: 4,958 rows, 4,463 train / 495 validation.
  - Best validation: loss 1.2618, accuracy 0.7515.
  - Direct 100-game head-to-head vs `legal_action_scorer_001`, seed block 26000: `005` went 48-52-0, average margin -0.57, 0 fallback triggers.
  - Direct 100-game head-to-head vs `legal_action_scorer_003`, seed block 26000: `005` went 47-53-0, average margin -0.72, 0 fallback triggers.
  - Standardized 100-game shallow-search benchmark, seed block 20000: `005` went 25-75-0, average margin -4.38, 0 fallback triggers.
  - Conclusion: do not promote `legal_action_scorer_005`. The small scored distillation slice did not improve over `001`/`003`; it is below `001` at 30-70 and `003` at 29-71 on the same search block.
- Full search-ranking distillation candidate, `legal_action_scorer_006`:
  - Completed the full overnight depth-2 scored relabel pass:
    - `data/corpus_search_distill_full_001/improved_search_policy_margin500_no_take.jsonl`
    - summary: `data/corpus_search_distill_full_001/improved_search_policy_margin500_no_take_summary.json`
    - 8,326 rows read
    - 4,454 rows written
    - 1,223 high-confidence changed labels
    - 3,231 unchanged anchors
    - 1,113 low-margin changes filtered
    - 1,787 changed rows filtered because original action was pruned/missing
    - 972 selected-token-take changes filtered
    - 0 missing snapshots, 0 rows without search candidate
    - all 4,454 written rows have `search_action_scores`
  - Trained from `outputs/legal_action_scorer_001/legal_action_scorer_best.pt`.
  - Training used only the new full scored relabel corpus, with `--search-distill-loss-weight 0.1`, `--search-distill-temperature 1000`, `--value-loss-weight 0.0`, 4 epochs, CPU.
  - Output: `outputs/legal_action_scorer_006/`; best validation checkpoint was epoch 3.
  - Dataset: 4,454 rows, 4,009 train / 445 validation.
  - Best validation: loss 1.1412, accuracy 0.8022.
  - Direct 100-game head-to-head vs `legal_action_scorer_001`, seed block 27000: `006` went 49-50-1, average margin -0.03, 0 fallback triggers.
  - Direct 100-game head-to-head vs `legal_action_scorer_003`, seed block 27000: `006` went 53-47-0, average margin -0.20, 0 fallback triggers.
  - Standardized 100-game shallow-search benchmark, seed block 20000: `006` went 28-72-0, average margin -4.13, 0 fallback triggers.
  - Conclusion: do not promote `legal_action_scorer_006`. Full search-score distillation with this recipe is not clearly better than `001` or `003`; it remains slightly below `001` at 30-70 and `003` at 29-71 on the same standardized search block.
- Distillation ablation candidates, `legal_action_scorer_007` and `legal_action_scorer_008`:
  - Both used the same full scored corpus:
    - `data/corpus_search_distill_full_001/improved_search_policy_margin500_no_take.jsonl`
  - Both trained from `outputs/legal_action_scorer_001/legal_action_scorer_best.pt`, 4 epochs, CPU, `--value-loss-weight 0.0`.
  - `legal_action_scorer_007`:
    - `--search-distill-loss-weight 0.03`, `--search-distill-temperature 1000`
    - output: `outputs/legal_action_scorer_007/`; best validation checkpoint was epoch 4
    - best validation: loss 0.9200, accuracy 0.7865
    - standardized 100-game shallow-search benchmark, seed block 20000: 22-78-0, average margin -4.80, 0 fallback triggers
    - conclusion: reject
  - `legal_action_scorer_008`:
    - top-1 only on the same corpus, `--search-distill-loss-weight 0.0`
    - output: `outputs/legal_action_scorer_008/`; best validation checkpoint was epoch 4
    - best validation: loss 0.8038, accuracy 0.8022
    - standardized 100-game shallow-search benchmark, seed block 20000: 28-72-0, average margin -4.41, 0 fallback triggers
    - conclusion: reject
  - Overall conclusion from `006`/`007`/`008`: this full scored relabel corpus improves supervised validation metrics but does not improve playing strength versus the shallow-search benchmark. Do not keep tuning this exact supervised/distillation recipe as the main path.
- Weighted gated-relabel candidates, `legal_action_scorer_009` and `legal_action_scorer_010`:
  - Added `training_weight` support for legal-action scorer training.
    - `run_improve_replay_actions.py` can write `training_weight`.
    - `LegalActionReplayDataset` loads it as `sample_weight`.
    - legal-action scorer policy/value/distillation losses use weighted means.
    - Old corpora remain backward-compatible with default weight 1.0.
  - Built weighted copy of the full scored corpus:
    - `data/corpus_search_distill_weighted_001/improved_search_policy_margin500_no_take_weighted.jsonl`
    - source: `data/corpus_search_distill_full_001/improved_search_policy_margin500_no_take.jsonl`
    - 4,454 rows total
    - 1,223 changed rows weighted 2.0
    - 3,231 unchanged anchors weighted 0.5
  - `legal_action_scorer_009`:
    - parent: `outputs/legal_action_scorer_001/legal_action_scorer_best.pt`
    - top-1 only, no soft distillation, 4 epochs, CPU
    - output: `outputs/legal_action_scorer_009/`; best validation checkpoint was epoch 3
    - best validation: loss 1.0685, accuracy 0.7955
    - standardized 100-game shallow-search benchmark, seed block 20000: 27-73-0, average margin -4.04, 0 fallback triggers
    - conclusion: reject
  - `legal_action_scorer_010`:
    - parent: `outputs/legal_action_scorer_003/legal_action_scorer_best.pt`
    - same weighted corpus and training setup
    - output: `outputs/legal_action_scorer_010/`; best validation checkpoint was epoch 2
    - best validation: loss 1.0450, accuracy 0.8090
    - standardized 100-game shallow-search benchmark, seed block 20000: 20-80-0, average margin -5.26, 0 fallback triggers
    - conclusion: reject
  - Overall conclusion from `009`/`010`: weighting confident search corrections more heavily did not improve strength. It may overfit to the shallow-search correction style or overweight noisy teacher corrections.
- Value-model branch:
  - Added a dedicated state-value model path.
    - `ValueMLP` / `ValueModelConfig` in `src/splendor_ai/training/model.py`
    - `ValueReplayDataset` derives targets from final score margin from the acting player's perspective.
    - `run_value_supervised.py` trains value checkpoints.
    - `run_value_evaluate.py` evaluates value checkpoints on arbitrary replay corpora.
    - `CheckpointValueEvaluator` can load a value checkpoint and feed a heuristic-scale value into `ShallowSearchBot`.
  - `outputs/value_model_001/`:
    - trained on the stronger snapshot/league set only, 8,326 rows, margin target scaled by 15, 12 epochs.
    - random row-split validation looked strong: MSE 0.0606, MAE 0.1308, sign accuracy 0.9167, Pearson 0.9239.
    - held-out `data/corpus_search_greedy_001/replays.jsonl` exposed poor generalization: MSE 0.2171, MAE 0.3669, sign accuracy 0.6789, Pearson 0.4074.
    - conclusion: not suitable for search integration.
  - `outputs/value_model_002/`:
    - trained on broader mixed data, 66,701 rows, holding out `data/corpus_search_greedy_001/replays.jsonl` for external evaluation.
    - random validation: MSE 0.0387, MAE 0.1132, sign accuracy 0.9551, Pearson 0.9543.
    - external eval on `corpus_search_greedy_001`: MSE 0.1556, MAE 0.3070, sign accuracy 0.7071, Pearson 0.5704.
    - external eval on `corpus_search_greedy_cutoff_001`: MSE 0.0775, MAE 0.1842, sign accuracy 0.8481, Pearson 0.8350.
    - external eval on `corpus_pass_smoke_001`: MSE 0.1244, MAE 0.2673, sign accuracy 0.7472, Pearson 0.8193.
  - Added game/source-level validation splitting for value training:
    - `run_value_supervised.py` now accepts `--validation-split-unit row|game|source`.
    - `game` groups by `(source replay path, game_seed)`, so positions from the same game cannot leak across train and validation.
    - Use `--validation-split-unit game` for normal value experiments; `source` is stricter and useful when testing whole-corpus generalization.
  - `outputs/value_model_003/`:
    - same broad mixed data as `value_model_002`, but trained with `--validation-split-unit game`.
    - dataset split: 66,701 total rows, 60,132 train / 6,569 validation, 12 epochs, CPU.
    - game-heldout validation selected epoch 1 as best; final epoch overfit the training games.
    - final epoch train: MSE 0.0205, MAE 0.0839, sign accuracy 0.9719, Pearson 0.9750.
    - final epoch game-heldout validation: MSE 0.2405, MAE 0.3557, sign accuracy 0.8159, Pearson 0.6994.
    - best-checkpoint external eval on `corpus_search_greedy_001`: MSE 0.1910, MAE 0.3441, sign accuracy 0.6999, Pearson 0.5063.
    - best-checkpoint external eval on `corpus_search_greedy_cutoff_001`: MSE 0.1723, MAE 0.3146, sign accuracy 0.6994, Pearson 0.6573.
    - best-checkpoint external eval on `corpus_pass_smoke_001`: MSE 0.1437, MAE 0.3013, sign accuracy 0.7415, Pearson 0.7858.
    - conclusion: do not promote `value_model_003`; it validates the split fix and shows the previous row-level validation was too optimistic, but `value_model_002` remains the stronger external-eval value checkpoint.
  - Value-assisted search probes using `value_model_002`:
    - value blend: `leaf = 0.9 * heuristic + 0.1 * neural_value * 15000`
    - 100-game direct search-vs-value-search benchmark, seed block 30000: value-search went 48-52-0 vs normal search, average margin -0.72.
    - stronger value blend 0.25, 20-game probe seed block 30100: 10-10-0, average margin +0.35.
    - conclusion: do not promote value-assisted search yet. The learned value is helpful enough to experiment with, but it does not clearly improve depth-2 search.

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

- Treat `outputs/legal_action_scorer_001/legal_action_scorer_best.pt` as the current leading candidate.
- Keep `outputs/policy_improve_margin500_002/supervised_policy_value.pt` as the old flat-policy champion baseline.
- Keep `outputs/policy_improve_margin500_001/supervised_policy_value.pt` as the previous flat-policy champion baseline.
- Keep `outputs/warmstart_search_004/supervised_policy_value_best.pt` as the older pre-policy-improvement baseline.
- Use the benchmark CLI for future checkpoint comparisons instead of ad hoc inline Python.
- Use direct head-to-head benchmark mode for promotion decisions against neural checkpoints:
  - `--checkpoint <candidate> --model-kind <candidate_kind> --baseline-checkpoint <baseline> --baseline-model-kind <baseline_kind>`
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

0. Current immediate recommendation after the targeted-corpus smoke:
   - Do not promote `outputs/legal_action_scorer_012_targeted_smoke`.
   - Do not promote `outputs/legal_action_scorer_013_targeted_full_from001` as a champion; it did not improve the standardized search gate.
   - `outputs/legal_action_scorer_014_wide_targeted_fromscratch` is the most interesting current neural parent: it beats `001`/`003` directly and improves search score margin, but not search win rate as a greedy policy.
   - `PolicyGuidedSearchBot` using `014` as the prior looked strong in a 20-game probe, but the 100-game confirmation went 44-55-1 vs plain shallow search with avg margin -0.58.
   - Next do a small prior-weight ablation before using policy-guided search as a teacher/player: compare weights such as 0, 250, 500, 1000, and 2000 on the same seed block.

1. Treat the current local neural champion set as unresolved between:

```powershell
outputs\legal_action_scorer_003\legal_action_scorer_best.pt
```

and:

```powershell
outputs\legal_action_scorer_002\legal_action_scorer_best.pt
```

and:

```powershell
outputs\legal_action_scorer_001\legal_action_scorer_best.pt
```

`legal_action_scorer_001` and `legal_action_scorer_003` are the best current neural baselines. `legal_action_scorer_004` should not be promoted because it regressed versus search on the standardized 100-game block. Keep `001`, `002`, and `003` as active baselines rather than discarding them.

The old flat-policy champion baseline remains:

```powershell
outputs\policy_improve_margin500_002\supervised_policy_value.pt
```

2. The legal-action scorer has now confirmed as the preferred neural architecture.
   - First direct 100-game result, seed block 9000: scorer 64, old flat champion 32, draws 4.
   - Second direct 100-game result, seed block 10000: scorer 63, old flat champion 37, draws 0.
   - First scorer self-play/search-relabel iteration produced `legal_action_scorer_002`, which improved vs search but remains roughly tied with `legal_action_scorer_001` directly.
   - Second scorer self-play/search-relabel iteration produced `legal_action_scorer_003`, which improved the first 20-game search benchmark to 10-10 but fell to 3-17 on the confirmation block; on the standardized 100-game search block it was effectively tied with `001`.
   - First mixed league/search-relabel iteration produced `legal_action_scorer_004`, which did not improve over the `001`/`003` baseline set and regressed against search.

3. Do not promote the mixed-data experiments.
   - Adding `greedy:random` data diluted the stronger search policy target and regressed benchmark strength.
   - Even a 3x search-weighted mix only reached 9-11 vs greedy on seed block 7000, while the current champion reached 16-4.
   - The lesson is that broader data needs quality control or weighting by playing strength, not just more rows.

4. Do not spend the next main effort on more flat-policy gated policy-improvement unless the scorer confirmation fails.
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
   - Stop scaling this recipe aggressively if gains flatten or benchmark strength becomes opponent-specific.

Strategic next phase after the immediate gated experiment:

5. Continue the structured legal-action scorer path.
   - Score the legal actions present in a state rather than learning a huge sparse softmax over every possible encoded action.
   - Splendor actions have natural structure: buy board card, buy reserved card, reserve, take tokens, return tokens, and pass. Action features plus state features should generalize better than a flat index.
   - The first trained scorer already beat the old flat-policy champion head-to-head, so the next work should improve and validate this architecture rather than reverting to flat softmax by default.

6. Use the new search benchmark as one promotion gate, not the only one.
   - `legal_action_scorer_003` is useful as the search-improved candidate, but direct neural strength versus `legal_action_scorer_001`/`002` is mixed.
   - The next scorer candidate should be judged against both `policy_improve_margin500_002` and `--opponents search`.
   - The first scorer self-play/search-relabel recipe worked directionally but was small: `002` went 54-45-1 then 49-51-0 vs `001`, while improving vs search to 6-14 from 4-16.
   - The second scorer self-play/search-relabel recipe improved search on one block but was unstable: `003` reached 10-10 then 3-17 vs search, and on the 100-game search block scored 29-71 vs `001` at 30-70. Direct neural results were 46-54 then 60-40 vs `002`, and 47-53 vs `001`.
   - The first mixed league recipe `004` did not work: it scored 21-79 vs search on the standardized 100-game block.
   - Promotion should require a clearer head-to-head result against the neural pool plus at least matching `001`/`003` on the standardized search block.

7. Search distillation with the current recipe should not be the next main path.
   - `005` showed that a 502-row scored slice is not enough.
   - `006` showed that even the full depth-2 scored corpus, trained with `--search-distill-loss-weight 0.1`, does not clearly beat `001`/`003`.
   - `007` with lower distillation weight `0.03` got worse versus search.
   - `008` top-1-only on the same full corpus matched `006` but still failed the promotion bar.
   - `009`/`010` tested explicit example weighting, with changed rows weighted 2.0 and unchanged anchors 0.5; this also failed.
   - Keep the full scored corpus because it is reusable for ablations.
   - Do not spend the next main effort on more single-corpus supervised tuning from this data.
   - Do not use `004`, `005`, `006`, `007`, `008`, `009`, or `010` as parents unless a later benchmark promotes them. The stable parents remain `001` and maybe `003`.

8. Value learning is implemented but not yet a stronger search evaluator.
   - `value_model_002` is the best current value checkpoint, but it did not improve search in the 100-game probe.
   - Value training now supports game/source-level heldout validation. Prefer `--validation-split-unit game` for value experiments because random row validation leaks positions from the same games.
   - `value_model_003` confirmed the leak concern but did not beat `value_model_002` on external corpora.
   - The next value-model work should focus on target quality and game-outcome/value usage in the league loop, not simply plugging this value into search again.

8. Move the main learning signal from greedy/search imitation to self-play and league evaluation.
   - Train against a pool of historical checkpoints, the current champion, tuned search bots, and baseline bots.
   - Keep greedy/random as regression tests, not as the main strength target.
   - Compare candidates by benchmark win rate, fallback-trigger count, game length/tempo, and performance against previous neural champions.

9. Use search selectively rather than copying AlphaGo-style MCTS wholesale.
   - Shallow/medium search remains useful for relabeling, benchmark opponents, and possibly inference under the 1-second move budget.
   - Be careful with hidden deck/refill information and off-policy value labels when search changes replay actions.
   - The long-term fix for loops should be learned policy quality and stronger self-play targets, not reliance on the inference fallback.

Future directions worth keeping from the external expert feedback:

- Expert iteration may become useful later: periodically review high-margin model/search disagreements and encode the missing Splendor concepts explicitly, especially token-taking plans, denial reserves, and race-to-15 tactical buys.
- Consider policy-guided search ordering once the scorer is stronger. The current shallow search is a strong opponent/teacher, but a scorer-guided move ordering could make deeper or broader search fit the 1-second inference budget.
- Revisit value learning only when target quality improves. Game-level validation is now available; the next useful value work is better outcome/tempo targets from league/self-play, not just plugging the current value model into search.
- Keep league data structured by source, matchup, seed block, and checkpoint identity. This is important for avoiding leakage and for diagnosing whether a candidate is a general improvement or only an anti-teacher specialist.
- Do not over-scale one supervised relabel recipe. If targeted corrections do not improve strength after a properly sized run, move toward league/self-play policy improvement or policy-guided search rather than more variants of the same corpus.

Operational notes:

10. Use the sharded replay pattern if continuing data scaling:

```powershell
.venv\Scripts\python.exe run_replay_corpus.py --output-dir data\corpus_search_greedy_004a --games 125 --seed-start 8000 --seat0-bot search --seat1-bot greedy --swap-seats --max-turns 400 --repetition-limit 4 --no-progress-limit 60 --search-depth 2 --search-max-branching 8 --search-buy-branching 5 --search-reserve-branching 2 --search-take-branching 2 --log-every 10
```

11. Train with repeated replay paths rather than merging corpora by hand.

12. Benchmark the resulting checkpoint with fallback metrics enabled, and choose champions by 100+ game benchmark performance rather than validation loss alone.

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

- `python -m pytest -q --basetemp .codex_pytest_tmp_all` passed: 141 passed after the observation/action feature update and targeted-corpus tooling.
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
