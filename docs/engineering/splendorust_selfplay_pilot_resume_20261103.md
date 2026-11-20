# Self-play pilot continuation: forecasts are advisory

3 November 2026. This is a separate amended continuation of the frozen pilot in
`outputs/splendorust_selfplay_pilot_20261102_02`. The original attempt remains
incomplete. It stopped cleanly after game 31 because its rolling collection
forecast failed; no move or active game was interrupted.

The user's explicit instruction was:

> i dont care about the resources !! please stop these stupid stops because of a resource forecast jesus christ/ Resume the pilot now

This authorizes removing forecast-based admission and stopping decisions. Every
forecast remains recorded, including failed predicates, for information and
later interpretation. A pessimistic estimate cannot block preparation, startup,
collection, fitting, operational games, assessment or verification. The original
resource-stop records remain unchanged; their failed forecasts are recomputed
and preserved, not relabelled as passed. Available-RAM reserve checks and their
stability waits are also advisory in this continuation: the controller does not
pause or reject work merely because the host reports less than 4 or 2 GiB free.
Initial disk admission uses the actual 5 GiB reserve, rather than requiring the
entire projected evidence budget to be free in advance.

## Preserved scientific design

The target remains 256 independently seeded teacher games: 192 training, 32
validation and 32 offline assessment games. The 32 already completed pilot games
are retained in place under their exact original file hashes. Collection resumes
at game 32, block 4, with the original remaining 224 specifications. No seed,
split, initial state, teacher target, terminal outcome or sample is substituted.
Engineering qualification games and checkpoints are not training data.

The teacher is the unchanged original full network with 512 simulations, three
worlds, exploration coefficient 0.4, depth 64, float32 arithmetic, one inference
thread and CPU0. The original warm-start checkpoint and model architecture remain
unchanged. The fixed two-epoch fitting recipe, losses, weighting, order and seed
are unchanged. There is one final checkpoint and no checkpoint selection.

The final candidate and original parent both use 128 simulations in six
operational games and 400 paired assessment games. The existing schedules,
swapped seats, outcome handling, statistics, acceptance criteria and sensitivity
analysis remain unchanged. There is no automatic promotion or external testing.

## Prefix verification and corrected rules

The complete old pilot and launcher evidence trees, original frozen dependencies,
owner identities and exit records are sealed in a separate continuation manifest.
All five original source worker trees are closed and absent. A bounded saved-only
review reuses the original frozen readers to validate the 32 complete games,
1,820 decisions, 1,819 eligible roots, five complete raw-log closures, 55 graph
fixtures, 335 actual entrypoint checks and the 32 capture qualification decisions.
The initial OFF/ON capture fixtures are reused through their exact preserved
paths; they are not rerun. A final reader may reuse the passed raw-prefix review
only while the entire sealed inventory and its source dependencies remain exact.
The final complete corpus loader still validates all 256 trajectories and targets.

All new collection, fitting and evaluation retain the corrected ruleset,
catalogue and turn contract from the qualified pilot. Every model worker calls
`verify_runtime_sources()` once at startup and runs its actual graph and corrected
entrypoint qualification. New source lifecycles are `collect_004` through
`collect_031`, followed by `replay_on`: 29 new trees, combined with five old trees
for the unchanged total of 34 source lifetimes. At most two inference trees are
resident at once. All 48 source and up to 96 assessment reproductions remain.

## Actual limits and clocks

Forecasts are advisory; actual hard limits remain enforceable. This continuation
inherits the original pilot monotonic origin 904900.656 and Unix origin
1793649048.764625. The 72-hour deadline remains 5 November 2026 at 19:50:48.764625
UTC; the 60-hour model cutoff remains 5 November at 07:50:48.764625 UTC. The
original collection start and its 34-hour deadline are retained, including the
elapsed stopped interval. Original 76-hour outer workflow origins are also
retained. No original clock is reset or extended.

The 20-hour assessment ceiling, stage limits, 48 GiB cumulative evidence ceiling,
8 GiB final reserve, 2 GiB inference process-tree limit and 4 GiB fitter limit
remain. The available-RAM reserve and startup stability requirement are zero;
host memory readings are reported without preemptive waiting. The original
three-hour cumulative RAM-wait budget and its already consumed time remain in
the accounting, but no new available-RAM reserve waits are scheduled. The exact
original evidence byte count is charged to the continuation. The owner still
checks actual process memory, the 5 GiB disk reserve, process ownership and elapsed
time. Allocation failures remain genuine execution failures and are preserved.

Every move retains its nine-second internal and ten-second response guard. One
second and 700 ms are descriptive timing thresholds. No active decision is retried,
no failed clock is reset, no failed game is replaced, and no sample is shortened.
If a real hard limit or correctness check fails, preserve the attempt and report
it as incomplete. Forecasts alone never trigger this stop.

## Publication and acceptance

New source and receipts belong only to the separate
`tools/splendorust_selfplay_pilot_resume_20261103` package and
`outputs/splendorust_selfplay_pilot_resume_20261103_01` output. The original source,
freeze, stop, user hold and all historical failed attempts remain intact.

Launch requires the exact sealed continuation manifest, passed saved-prefix
review, model-free regression results and independent source/admission reviews.
Completion requires the full combined source corpus, unchanged fitting and paired
assessment, raw transport and replay checks, independent saved verification,
actual successful process exits, cleanup and owner absence. A successful result
is an explicitly amended continuation; it never retroactively completes the
original stopped attempt. No additional user confirmation is required to launch
the authorized continuation after these checks.
