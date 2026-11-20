# Remaining operational games and replays with bounded memory waiting

20 October 2026. The owner authorized checking RAM stability and resuming the
remaining games and replay checks under a new bounded plan, with the same
players, 120-second allowance and every recorded failure retained. The owner
also asked whether execution can continue through a RAM shortage. This plan
adds waiting before workers start; an active request's clock never pauses.

This is an operational correctness and timing continuation. It is not a new
playing-strength sample. Keep matchup outcomes closed and the larger-data
study deferred.

## Preserve both prior attempts and the exact continuation

The original attempt is `outputs/live_game_operational_20261019_01/`, freeze
SHA256 `21ca9cb617ca65ad9745eeed76a2f5f91153dd638981c7fbfd76f4e348a27c51`.
The immediate source is `outputs/live_game_operational_recovery_20261020_01/`,
freeze SHA256 `3af87db0e61a9d014b4ec60e8b031f9a36a2b6f90214b559f2485f60b75ae716`.
Both processes and their completion helpers have closed in preserved holds.
The first stopped for low physical RAM after five games; the second stopped
for the same guard after sixteen games. Their elapsed times, deadlines,
failed preparation, failed requests, incomplete requests and diagnostics are
historical records and must not be overwritten or reset.

New output is `outputs/live_game_operational_recovery_20261020_02/`.
The unchanged schedule is
`6,8,10,7,9,11,12,14,16,13,15,17,18,20,22,19,21,23`.
Its first sixteen game records are final. The four local timeout forfeits
at `g006/t0000`, `g007/t0007`, `g017/t0006` and `g019/t0007` remain final.
Do not retry these requests or replace these games.

Game 21 has completed plies 0 through 10. Reconstruct the environment from
its original seed and every accepted action; verify lawful observers, receipt
provenance, state hashes and independent dictionary transitions. Continue
from ply 11, whose prior incomplete opponent attempt has no worker directory,
initialization, request receipt or accepted move. Preserve that incomplete
attempt separately. Game 23 is unstarted. No other game, seed or request is
added. Carry the reconstructed environment forward, including its random state.

Authenticate the immediate source's complete file inventory, its original
snapshot/import mappings, and all original frozen inputs. Copy only required
game/validation metadata byte for byte, retaining original receipt and trace
references. Read original continuation traces through authenticated paths;
never write to either previous evidence directory. Treat the three execution
segments separately in receipts, process inventories and costs.

## Unchanged players and scientific checks

Keep the four-candidate, 128-shared-world local planner with fits
161001/161002/161003, raw035 continuations, serial arithmetic and original ties.
Keep profile `four_candidate128_corrected_live_operational_20261019_v1`.
Keep SplendoRust profile
`entity_onehot_puct6400_canonical_atomic_prereveal_cpu_120s_v1`,
SHA256 `6f74c3b6d77707c42619801d1b1bef3fbf203d1ee6ff548afbe935169c257599`.
Preserve corrected rules, card catalogue, observations, public search streams,
legal action handling, seat assignments and all original success criteria.

The local numerical implementation and independent continuation replay are
the unchanged 20261019 modules. Only orchestration, output paths, authenticated
audit input paths and memory admission change. Workers remain fresh per request,
normal priority, one numerical thread and original performance CPU 0.
The supervisor must retain the original full CPU eligibility, since the
hardware helper observes the calling process's allowed CPUs.

Keep a real 120-second complete-move deadline, separate 60-second initialization
limit, 600-second replay limit, and SplendoRust's 108-second internal limit.
No partial-search fallback, active clock suspension, extra thinking time,
automatic worker retry, new training or compilation is permitted.

Reuse the twelve passed validation decisions and their exhaustive replays only
after authenticating their original lineage. There are no new validation
decisions. Once all eighteen game records exist, select the original combined
audit manifest once across all completed local moves from all three segments:
eight sampled worlds per root and the original full stress-cell replays,
with the unchanged seed, priorities and deduplication over 81 possible cells.
All these game-continuation audits remain outstanding. Missing cells and
unreplayed trajectories must remain explicitly untested.

## Memory waiting and the new bound

The new execution gets one prospective 12-hour maximum, 43,200 seconds, from
runner launch before RAM admission and state reconstruction. Both wall-clock
and monotonic limits apply; use whichever has less time left. Count initial
admission, memory waits, reconstruction, games, replay audits, cleanup, final
source checks, independent analysis and finalization. Preserve a six-hour
post-game reserve and 600-second final reserve. Preparation is separate and
does not run players.

Before the first new request, require at least 2 GiB available physical RAM
and 4 GiB available commit for thirteen consecutive readings, ten seconds
apart, spanning at least two minutes. A separate two-minute preparatory
observation is informative; the runner must repeat admission within its clock.

Before every subsequent move, replay worker and final analyzer, read memory
again. A good reading permits immediate admission. A low reading begins a
recorded wait with no worker running. Resume only after the same thirteen
consecutive readings over two minutes. Any bad reading resets the stability
streak. Keep monitoring disk, evidence size and both clocks while waiting.

Limit each admission episode to two hours and the sum of all admission
durations to four hours. Charge the larger wall-clock or monotonic elapsed
time for each episode, including healthy fast checks and initial admission;
clock adjustments cannot refund time. These are ceilings within the 12-hour
budget, not extensions to it. The six-hour game reserve can stop a wait sooner.
Record per-admission JSON and append-only reading logs, exact thresholds,
streaks, clocks, cumulative charges, and the gate bound to each new request.

There is no worker to time out while waiting between requests. Once a worker
starts, its original initialization/request clock and process-memory limits
remain unchanged. Host memory pressure during a move can still slow it down
or cause failure; waiting between moves cannot guarantee freedom from timeouts.
Do not suspend that clock, retry the move, or reclassify its failure.
The source runs already disabled physical-headroom enforcement inside active
requests; retain that behavior and the independent 2 GiB worker-tree RSS limit.

If a waiting, global, disk, evidence, initialization or scientific check fails,
preserve a terminal hold. No automatic new recovery follows. Recognized player
failures remain recorded forfeits and the prescribed schedule continues.
Never close unrelated applications, change the page file, or weaken resource
limits automatically.

Cap new evidence at 4 GiB with a 32 MiB in-flight/final reserve, keep 5 GiB
free disk, and require 9 GiB free before preparation. Original evidence is
retained by reference outside the new allocation. The unchanged 2 GiB process
tree, message and log limits continue to apply.

## Review and acceptance

Freeze this protocol, source, tests, review, preparation receipts, both preserved
lineages, source inventory and exact import mappings before launch. Tests must
cover partial-state reconstruction, provenance, no completed-decision retries,
trace path boundaries, unchanged numerical code, memory recovery/streak resets,
waiting limits, both clocks, protected reserves and admission before workers.

The independent analyzer must check all three receipt/clock/process contexts,
all actual game transitions and endings internally, exact combined audit
selection and prescribed proofs, full source preservation and worker closure.
Independently validate memory gate readings/logs, cumulative charges, ordering,
thresholds and worker creation after its gate. Include the analyzer's own
admission, created by the completion helper before it starts.

An authenticated completion helper may run the frozen analyzer once after
successful collection and replay, within the same 12-hour cap. A successful
finalization validation and absence of a coordinator hold are required.
Report admission time separately while retaining it within total segment time.
Report each old segment and the new segment separately; do not claim that the
combined work met either old uninterrupted budget.

All four prior timeout forfeits and any new failures remain visible in the
operational report. Successful verification means the evidence was checked;
it does not establish failure-free timing or playing strength. With roughly
one game plus most of another still to play, a planning scenario is about
one hour of games and one to two hours of combined replay checks, plus final
checks. Allow roughly three to four hours without RAM waits; this is an
estimate, not a promised completion time.
