# Interrupted matched benchmark: inspection and bounded recovery options

22 October 2026. The owner authorized inspection and resumption if possible,
and asked whether the earlier timeout should count as a loss. This document
records the discovered limitation and a concrete decision required before a
new playing request. It does not amend or launch the frozen benchmark.

## What the evidence establishes

The original run completed 59 games and plies 0–42 of game 59. During the next
local request, `g059/t0043`, the runner, administrative launcher and worker
disappeared. Initialization succeeded; only the ready message survives. There
is no complete response, request receipt, persisted trusted request-start
timestamp, decision record or resumable numerical checkpoint. The compressed
trace is truncated. Last worker progress was at 13:34:44 Paris time. The cause
is unconfirmed; the reviewed Windows events did not identify it.

This is an unresolved active-request interruption, not evidence of a timeout
or a completed action. The frozen clock stores request timing durably only
after the receive operation returns. File timestamps cannot reconstruct a
trusted remaining thinking allowance. Neither restarting with 120 seconds nor
salvaging partial search is an unchanged continuation.

The prior recovery precedent does not solve this case: it resumed a request
that had stopped before any worker was started. The frozen deadline policy
explicitly says that an unresolved active-request fault blocks acceptance.

The earlier request `g039/t0007` is different. Its trusted complete-response
timestamp is **121.6558109 seconds**, beyond the **120-second** deadline. The
late response is retained, with a completed 512-continuation trace; the clock
correctly did not accept it. Supervisor elapsed time was 122.2571484 seconds,
and sampled request CPU time was 96.0625 seconds. There is no recorded resource
guard or worker exception explaining the difference. These numbers support a
late-delivery forfeit under the frozen contract, but do not prove that search
computation alone needed more than 120 seconds or establish a causal RAM fault.

Counting that timeout as a loss measures the delivered player's reliability
within a wall-clock allowance. Both players face the same external rule.
Initialization is separate; required local trace delivery is inside the move
clock by prior agreement. This is not an unlimited-compute strategic comparison.
Granting grace after seeing this failure would change the primary assessment.
The loss remains final under either option below.

## Option A: one disclosed infrastructure retry, with amended analysis

Recommended if completing a practical comparison is the priority. This needs
explicit authorization for the exception, because the current instruction to
resume if possible does not establish how to change the frozen clock rule.

- Preserve the entire original attempt, its partial search and genuine timeout.
  Reconstruct the final game's environment from the original seed and accepted
  actions, including referee random state, without exposing matchup totals.
- Retry **only `g059/t0043`, once**, in a new process with a new 120-second
  allowance. Do not reuse its partial search. This new allowance is the explicit
  deviation; do not claim the two attempts met one uninterrupted move deadline.
- Keep the same players, fits, seats, seeds, rules, numerical code and subsequent
  120-second move deadlines. Genuine player failures remain losses. Another
  unclassified interruption stops the recovery; no automatic retry follows.
- Verify all retained and new decisions, actual game transitions and prescribed
  continuation replays. Freeze the separate recovery implementation, tests,
  source inventory, amendment and independent acceptance checks before launch.
- Keep the original study labelled **incomplete under its frozen protocol**.
  Report any successful continuation separately as an **amended recovered
  benchmark**, with the exception prominent. Also show sensitivity to the
  unresolved original final-game score: its full possible effect on the pooled
  point estimate is 1/60, or 1.667 percentage points. Confidence intervals and
  conclusions still require separate validation; no original success criterion
  is retrospectively relaxed.

## Option B: verification only, leaving the original study incomplete

Authenticate the 59 completed games and final-game prefix, then replay the
recorded completed local decisions using the frozen sampling/stress rules.
Do not send another playing request, invent a last-game outcome or replace a
seed. Any later partial report must explicitly state the missing game and
show its possible effect; the original complete-sample primary estimate remains
unaccepted. This option also requires selecting the partial-report scope before
opening outcomes.

## Bounds shared by either selected recovery

Use a separate versioned output directory and immutable references to original
evidence. No old file is overwritten, no missing historical exit receipt is
fabricated, and no frozen source/player is edited. Authenticate process closure,
source hashes, original decisions and the twelve passed adapter/full-replay
checks before reuse. Fresh code requires independent review and relevant tests.

Execution is capped at **12 hours from its own launch and the original absolute
deadline of 24 October 2026 at 01:01:13.594567 Paris time**, whichever comes first.
No reset of the original 72-hour accounting occurs. Protect six hours of the
recovery for replay/final checks and retain the original eighteen-hour reserve
against the original deadline before any new playing move.

Retain CPU0 for numerical workers, normal priority, one numerical thread,
fresh processes, no GPU, a 2 GiB process-tree ceiling, 5 GiB free disk and the
original **8 GiB combined evidence ceiling**, including retained source output
and new recovery evidence. The final analyzer gets at most 3,600 seconds plus
600 seconds for finalization, inside both deadlines.

Require fresh two-minute stable RAM admission (2 GiB physical/4 GiB commit),
and retain all previously charged RAM waiting against the original six-hour
total; each new episode remains capped at two hours. Active new move clocks are
never paused or refunded. Stop on any cap, unexpected failure or verification
mismatch. The larger-data study remains deferred.

Inspection evidence is stored separately in
`outputs/matched_strength_interruption_review_20261022_01/`. Its hash inventory
does not open matchup outcomes or launch any model/game request. Neither option
has been executed or presented as an accepted result.
