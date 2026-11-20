# Saved reliability failure diagnosis

25 October 2026. The saved failure followed a slowdown affecting both players,
while completed moves retained their normal search and neural-evaluation counts.
The next proposed fix is to keep supervision off the player's physical CPU core.
This is a testable execution hypothesis, not an established cause or cure.

The original failure, players, one-second limit, internal 900 ms guard and all
verification are unchanged. No model, saved-position replay, game or training
was run for this diagnosis. No next experiment has been launched.

## Evidence and scope

The original live attempt remains
`outputs/splendorust_live_boundary_20261025_01`, freeze
`50ca8363a364b06dbf452778b90aa27c5ea763d8f2d76d2c4efab8bca582ec6b`.
It stopped in main game index 72, the 73rd main game, at ply 28. The complete
recorded prefix comprises six operational games, 72 naturally completed main
games and one forfeited main game. There are 4,371 attempted decisions and
4,370 successful decisions. The other 127 main games remain unplayed.

The original final verification passed, including 48 sampled completed-decision
reproductions, raw receipts, science, transitions, graph checks, logs and owning
cleanup. That verifies the stopped prefix; it does not turn this into a complete
200-game workload or an independent strength estimate.

The new offline extractor is
`tools/splendorust_saved_failure_diagnosis_20261025_v2.py`; derived evidence is
`outputs/splendorust_saved_failure_diagnosis_20261025_02`. It recomputed all
4,371 live-decision metrics from saved move records. It checked 4,480 consumed
members of the original artifact seal and eight frozen source files, bound
unsealed final summary/lifecycle receipts separately, and reread all 4,496
consumed inputs unchanged. This is a scoped preservation check, not another
full historical seal audit. Execution took 4.44 seconds and read 177.43 MB,
within its 300-second and 1 GiB read limits.

The first offline-reader attempt is preserved in the `_01` directory and its
original source. It stopped because it expected a derived performance summary
to belong to the artifact seal. Version 2 explicitly distinguishes unsealed
summary receipts from sealed primary records and recomputes the claims from
the latter. Neither analysis attempt wrote original evidence or executed a
player.

## Where the failed decision spent its time

| Recorded interval | Elapsed time |
|---|---:|
| Trusted parent start to child input read | 51.534 ms |
| Input read to decision-handler start | 0.061 ms |
| Inside decision handler, ending in failure | 900.477 ms |
| Trusted error-packet arrival from parent start | 980.018 ms |

The exception was detected at the deadline check immediately after a neural
evaluation returned: `tools/splendorust_search_scale_20261024/adapters.py:134`.
It occurred while the native search was still awaiting that evaluation's
response. The error arrived before the outer one-second deadline, but there
was no valid action. The preserved loss is therefore correct under the existing
policy. The supervisor's later 1.064-second return includes subsequent checking
and is not the packet-arrival clock.

Both timing sidecars exist. The failed request has no completed result, partial
callback count or per-evaluation timings. Its raw stream has one error packet
and no successful response. Thus the records do not identify which evaluation
failed, how much search remained, or how long that one evaluation took. The
900.477 ms measures the entire handler, including all preceding work and waits.

## The new finding is a shared slowdown

The following intervals were selected after inspecting the failure. They are
descriptive comparisons within one game, not an independently selected test.

| Player | Early handler mean, plies 0–9 | Later handler mean, plies 14–27 | Early sampled request CPU | Later sampled request CPU |
|---|---:|---:|---:|---:|
| optimized64 | 402.8 ms | 672.9 ms | 331.3 ms | 359.4 ms |
| unchanged32 | 210.2 ms | 378.2 ms | 196.9 ms | 212.1 ms |

Every completed optimized64 move in that game recorded 64 simulations and 65
neural callbacks. Every unchanged32 move recorded 32 simulations and 33 callbacks.
All ten successful moves above the descriptive 700 ms mark across the entire
attempt likewise had 64 simulations and 65 callbacks. Four occurred in the
final game, at plies 16, 18, 20 and 22.

The substantially larger rise in elapsed time than in sampled CPU time supports
shared execution, waiting or scheduling variability as the next hypothesis.
These CPU counters are coarse cumulative request differences, include some work
outside the handler and exclude the supervising process. Subtracting them from
handler time would not yield an exact waiting-time measurement. Fixed callback
counts also do not prove identical native traversal cost or per-callback cost.
The failed move's callback count remains unknown.

Python-process resident memory stayed nearly flat in the final game: optimized64
232.043–232.355 MiB and unchanged32 226.500–226.758 MiB. Both retained six observed
Python-process threads and affinity `[0]`; both stderr logs are empty. This
does not support a growing worker-memory leak. It does not reconstruct global
RAM availability or rule out operating-system paging.

## Why supervisor placement is the next candidate

The frozen hardware topology places logical processors 0 and 1 on one physical
performance core. Player Python processes and their native helpers are pinned
to logical processor 0. The supervisor's saved allowed mask includes logical
processors 0–19; the runner and watchdog never exclude the player's core.

During moves, supervision performs process snapshots, ownership and executable
checks, memory/CPU/affinity sampling, response reading and aggregate resource
checks. This work is permitted to share the player's physical core. The saved
records do not show where its threads actually ran during the failure. Other
applications, operating-system activity, shared caches and power/frequency
effects remain possible explanations. No specific competing process or thermal
cause can be named from these records.

Source anchors:

- `tools/local_player_qualification_128/clock.py:80`: topology uses the caller's allowed affinity.
- `tools/local_player_qualification_128/clock.py:222`: player affinity is set explicitly.
- `tools/local_player_qualification_128/clock.py:315`: receive loop and resource sampling.
- `tools/local_player_qualification_128/clock.py:380`: sampled request CPU accounting.
- `tools/splendorust_timing_opt_20261024/sampler.py:176`: authenticated per-poll process-tree sampling.
- `tools/splendorust_live_boundary_20261025/runner.py:75`: aggregate resident-worker memory checks.

## Recommended next fix and its acceptance conditions

Prepare one separately versioned supervisor-placement change: retain every
player process on logical processor 0, while excluding both 0 and its physical
sibling 1 from owned supervision, readers, writers, logging and the watchdog.
Keep normal priorities, one numerical thread, the original players, weights,
precision, search quotas, observer, deadlines, sampling cadence and all checks.
This excludes our own supervisory work; it does not reserve the core against
unrelated applications.

There is an implementation trap: worker construction derives permissible CPUs
from the supervisor's current affinity, and runtime verification compares that
topology against the freeze. Simply narrowing the supervisor mask first would
reject player processor 0 or select a different processor. A new version must
explicitly bind host topology and the distinct supervisor/player masks, preserve
player processor 0, and verify masks after every relevant process/thread creation.
Do not weaken existing player-affinity verification to make the change pass.

Qualification should compare original and isolated supervision on identical
saved positions with reversed execution order, both players, exact outputs,
actions, traces and transitions, and the unchanged 1.000/0.900-second clocks.
Keep all failed observations. Any future saved-root test is a separate diagnosis,
never a replacement move or erasure of the original forfeit. Fresh bounded
operational games follow only if correctness and timing admission pass; a longer
live test is still needed to assess rare failures. A successful replay alone
would not establish a cure.

This recommendation targets a specific avoidable source of competition before
changing neural calculations again. If isolation produces no repeatable benefit,
the next evidence needed is bounded scheduling/work-wait attribution, not a
larger clock or removal of verification. No change is implemented here.

## Research direction

The earlier optimized64 strength result remains intact, including its historical
losses. The current obstacle is delivering that strength reliably within one
second, rather than demonstrating a faster average move. Learning and larger-data
work remain deferred. The new finding narrows the next engineering experiment
without changing the player or claiming that the underlying cause is proved.

## Derived records

- [Diagnosis](../../outputs/splendorust_saved_failure_diagnosis_20261025_02/diagnosis.json)
- [Final-game timeline](../../outputs/splendorust_saved_failure_diagnosis_20261025_02/final_game_timeline.json)
- [All decision metrics](../../outputs/splendorust_saved_failure_diagnosis_20261025_02/all_decision_metrics.json)
- [Input preservation manifest](../../outputs/splendorust_saved_failure_diagnosis_20261025_02/input_manifest.json)
- [Offline completion](../../outputs/splendorust_saved_failure_diagnosis_20261025_02/completion.json)
- [Independent scoped review](../../outputs/splendorust_saved_failure_diagnosis_20261025_02/independent_review.json)
