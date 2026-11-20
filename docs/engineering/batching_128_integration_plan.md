# Conditional batch16 integration check at 128 worlds

2026-09-21. Preparation only, while the existing version 3 batching benchmark
is running. Its performance adoption screen remains unmeasured at the time this
document is written. This document changes no implementation, experiment
protocol, configuration, existing check receipt or frozen evidence. It is not
part of the existing `outputs/learned_proposal_128/checks_02` identity. Any later
implementation must receive fresh checks and be bound before the study freezes.

## Entry condition and bounded decision

Consider this integration only after the exact version 3 benchmark completes
equivalence successfully and reports both prespecified ratios: summed collection
speed at least 1.25 and full collection-plus-audit speed at least 1.10. Bind the
actual benchmark artifacts and decision; do not infer a pass from partial work.
These are engineering adoption screens, not gameplay or universal-speed claims.

The owner-authorized integration allowance is four elapsed hours including
implementation, review, checks and validation, beginning when conditional
integration work starts after the screen passes. Record that start and deadline.
Failure of the screen, any unresolved behavioral mismatch, or inability to
complete the bounded work means retain the checked serial backend. Preserve
partial/failed evidence, stop queued integration work and do not extend the
allowance or select a different batch size. An already-running atomic validation
job may finish and retain its evidence; this does not extend adoption eligibility.

## Fixed source and selection provenance

Use the already sealed metadata and source references in:

- `outputs/batching_benchmark_v3/preparation_01/plan.json`, SHA256
  `97b0f5f72c54d9bfed340bd43a947448ef333ebf702062a4d1386f998aa17b3c`.
- `outputs/batching_benchmark_v3/preparation_01/prepared.json`, SHA256
  `cc00f9a4effaf900ad20aa71f773f315b75ac69f6453b862529592eb4c23f349`.
- Preparation binding
  `5e468e2f3e4722cd6b3a96b979a7dcdee4df3720b516aa404f823485e097f66e`.

These nine states were selected using only root identity, phase, acting seat,
assigned fit, offered support and private-reserve metadata already present in
that plan. No outcomes, choices, continuation lengths, historical runtimes or
current benchmark pass/timing observations influenced the selection. This is a
fixed engineering coverage sample from consumed development evidence, not a
random strength sample or a claim of exhaustive coverage.

Before any 128-world integration execution, create a write-once validation plan
binding this document, the checked adapter/verifier code, all nine source-root
hashes below, their original game/audit references and inherited model identities
from the sealed plan. Reverify the plan/seal hashes and original provenance links.
Copy each state's original `search_stream` and assigned fit from its bound
`spec`; do not generate a new stream or choose a model by its behavior. No source
state may be edited, replaced or added after validation outcomes are observed.

All source roots are under
`outputs/learned_proposal_arena/main/roots/<root identity>.json`.
Privacy categories are relative to the acting player.

| Root identity | Phase | Acting seat | Assigned fit | Privacy | Offered support |
| --- | --- | ---: | ---: | --- | ---: |
| `game_0238/turn_000` | Early | 0 | 161002 | Neither | 4 |
| `game_0373/turn_011` | Early | 1 | 161001 | Both | 4 |
| `game_0398/turn_032` | Middle | 0 | 161001 | Neither | 4 |
| `game_0183/turn_025` | Middle | 1 | 161002 | Neither | 4 |
| `game_0204/turn_064` | Late | 0 | 161003 | Opponent only | 4 |
| `game_0153/turn_059` | Late | 1 | 161003 | Own only | 4 |
| `game_0379/turn_042` | Late | 0 | 161003 | Own only | 1 |
| `game_0266/turn_016` | Early | 0 | 161001 | Both | 2 |
| `game_0307/turn_022` | Middle | 0 | 161001 | Both | 3 |

The first six states cover every phase/seat combination once, each fixed fit
from both seats once, and all four privacy categories. The final three cover
natural short supports. Phase boundaries remain 0-19, 20-39 and 40+.

| Root identity | SHA256 |
| --- | --- |
| `game_0238/turn_000` | `0e65bba64697592ace8b8dc6baab814112e60e9294eaeb25daa25b2030e89df9` |
| `game_0373/turn_011` | `08603962b2e40e3b559335534c4b4ebd7c446a9b1585b2e8af9eb3981023d132` |
| `game_0398/turn_032` | `d367feba716dad7ff8a6dfaa423f3eb21510a527e288ca5bc963821cff3cc7c4` |
| `game_0183/turn_025` | `de9c343f5da0af1fe978c7cac6b84546aaec40c4b9948338380baa758448c00f` |
| `game_0204/turn_064` | `c252a416502ba02661f6414808182a199fe9315172882aacf8061b6af30a9293` |
| `game_0153/turn_059` | `09ee501df143a68a5319c419576f2b096afb2ec42e3667ead76265b9fb3b7621` |
| `game_0379/turn_042` | `018ea9ce7a6389dc68a77e882d1cd61d40662a2f8dcc108db5eba3758c317454` |
| `game_0266/turn_016` | `5e6d53e9a5301e5298999d8587f96adca69cdb95c21594f7750e4ca9f09d6580` |
| `game_0307/turn_022` | `6f928667792962cb990525794ec37f0d17c14e0c1de1e0b7731cadaca925cdec` |

## Roles and implementation boundary

At each state test BOTH root roles, regardless of its original source role:

1. Learned candidate: the assigned frozen 161 fit chooses the fourth proposal
   outside the original first three through the unchanged successful rule.
2. Original reference: no learned fit is used; retain exact 160's original 035
   top two plus two random alternatives, including the second random proposal.

Keep the acting seat and source state fixed. Label the role explicitly in the
isolated fixture; do not mutate the saved source spec or game. For auditor role
dispatch, the isolated role-specific candidate-seat field equals the actor for
the learned role and the other seat for the reference role. Preserve the original
source spec separately. Do not relax production schedule/spec checks to enable
these development fixtures. Assigned fit remains provenance for the reference,
not an operative model.

Compare the serial implementation with batch16 only. Reference serial execution
must retain the unchanged `world_budget_160.choose_root(..., 128)` behavior;
the candidate retains the checked serial `learned_proposal_128` behavior. Replace
only continuation scheduling in the proposed backend. Preserve lawful inputs,
concrete representatives, canonical order, proposal scores, original-order ties,
and the exact 128-world schedule and its original 32-world prefix. Flatten jobs
in world-major/action-minor order and restore outcomes to those exact positions.
Every simulated actor continues with greedy 035. Do not change old global
configuration, checkpoints, evaluator policy, outcome utility or verifier.

Short supports retain the entire original offered list in both roles. Each root
still has 128 times its supplied support continuation jobs. Do not combine
unrelated roots into one inference batch. Do not rerun sizes 1 or 4 or choose a
different batch size after results.

## Exact validation and accounting

Run all nine states x two roles x two backends: 36 root executions. The support
sum is 30, giving 30 x 128 x 2 x 2 = **15,360 collected continuations**. Fully
independently replay every branch of every execution: **15,360 additional
continuations**. Synthetic tests and worker fixtures are separate work and must
be reported separately, not hidden in these counts.

For every paired root-role execution require:

- Exact complete action/state-hash traces, final state, action-trace digest,
  final-state digest, ending reason, winner and turn count for every continuation.
  Trace capture must observe the existing serial trajectory, not introduce a
  second scalar policy. If capture requires extra execution, disclose and budget
  that additional work before launch rather than silently changing these counts.
- Exact persisted scientific-record equality between serial and batch16:
  snapshot, privacy, proposal fields apart from timing, raw root scores,
  schedules, candidates, all ordered worlds/outcomes, means, choice and returned
  concrete action. Enumerate the excluded timing/run metadata fields explicitly
  before execution. Compare persisted JSON to persisted JSON to avoid the known
  in-memory integer-key versus JSON string-key discrepancy. No numerical
  tolerance may conceal a changed decision, trace, outcome or world assignment.
- The unchanged independent root auditor reconstructs lawful inputs, proposals,
  hidden-information invariance, all 128 worlds and all root values/choices,
  and replays every offered action in all 128 worlds for BOTH backends. Use the
  isolated full-fixture path; retain serial independent continuation replay.
  Do not substitute ordinary eight-world sampling for this integration check.
- Validate the exact 32-world schedule prefix. Compare source scalar evidence
  for that prefix where its role and proposal contract match. A 128-world mean
  or choice need not equal the original 32-world mean or choice.
- Account for every job exactly once, including world indices 32-127. Record
  actual batch widths and require real full batches of 16, rather than merely
  recording a requested width. Keep short supports and mixed continuation
  lengths. Reverify source state hashes after execution.

Retain the existing bounded synthetic checks for exact ties, forced actions,
private reserves, refills and terminal boundaries under the selected backend,
with serial comparison and full replay. Do not demand that a favorable natural
state happen to exhibit every boundary case. A finite pass supports this
integration; it does not prove universal batching equivalence.

## Hidden eight-worker integration fixture

The existing tiny pipeline uses two workers; it does not establish eight-worker
operation. Add a separate isolated fixture with eight explicitly assigned tiny
root tasks, at most four actions and exactly 128 worlds each. Use six learned
tasks covering all three fixed fits in both seats, plus two original-reference
tasks covering both seats. Bind the synthetic snapshots and task mapping before
execution. They must be terminal-near fixtures, not new natural games.

Use the actual hidden detached launcher/spawn/worker initialization path, with
one deterministic Torch thread per worker. Require eight distinct initialized
worker PIDs and evidence that each performs its assigned real batch16 workload;
a pool configured for eight with fewer active workers is insufficient. A bounded
initialization barrier may ensure coverage; a timeout fails the fixture rather
than waiting indefinitely. Fully replay each fixture root independently, retain
per-worker health/output/audit receipts and verify no console attachment,
exceptions, nonfinite output, changed source state or missing task.

This is at most 4,096 collected and 4,096 replayed fixture continuations, reported
separately, and does not require eight full games. Keep write-once isolated
outputs, production-path guards, launch acknowledgement and process cleanup.
Do not alter the scientific six-test-game, 12-probe-game or 402-main-game
schedules to occupy eight workers. Do not overlap validation with the ongoing
benchmark or another heavy study.

## Cost and completion

At the historical approximately 18.55 seconds per 32-world/full-support root
collection, comparable serial replay cost and linear scaling, the natural panel
has an aggregate anchor around 75 minutes. This is a planning calculation, not
a measured 128-world forecast. Allow roughly 1-2 hours on one worker or about
15-40 minutes with eight independent workers, plus implementation/check time.
Trace I/O, startup, uneven root lengths and memory contention can change these
figures. The existing benchmark supplies a speed screen; the later study probe
supplies the actual production runtime forecast.

Save a single integration report with source/implementation/check identities,
complete root/branch coverage, exact-comparison results, independent audits,
worker receipts, costs, failures and the serial-or-batch16 decision. A pass plus
the prior speed screen permits the already-authorized backend decision before
the study freezes. Bind the chosen backend and new verification evidence through
fresh checks and the experiment identity. No backend may change mid-study.
Preserve serial if the four-hour allowance cannot accommodate all required work.

Preparation of this document ran no inference, games, fitting, validation or
launch, and changed no existing file. It does not itself adopt batching.
