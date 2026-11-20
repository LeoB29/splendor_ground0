# BreckEmert adapter qualification and 60-game screen

Completed 27 October 2026. Study: `breck_screen_20261026_01`.

## Outcome

The separate BreckEmert adapter passed correctness and timing qualification,
then completed all six operational games and the frozen 60-game screen against
unchanged32. Breck lost all 60 scored games, with no draws or player faults.
It scored 0/30 in each seat. The six operational games are excluded from that
score.

The frozen admission rule required at least 15 natural points from 60 games,
zero player faults and all verification checks. This exact exported policy
therefore **fails the competitiveness gate**. It is not admitted to the proposed
optimized64 comparison. The screen was completed successfully; its negative
finding is not an implementation or timeout failure.

Runner status is `completed_verified`. Independent receipt-only reconciliation
accepted the completed attempt and rejected Breck as a competitive opponent.
It directly checked 132 consumed files and the 66 small game-result records,
reconciled the existing passed verification receipts, and confirmed all three
owning processes were absent. It did not rerun models, games or exhaustive
scientific/seal verification. The review took 13.882 seconds; its SHA256 is
`f43c7a92e1dbb53b1835615870a4508d3c7313f2840f5d1f5134b5193b33aa08`.

## What was implemented

New versioned implementation: `tools/breck_screen_20261026/`. Existing players,
weights, search settings and frozen historical evidence were preserved.

The external player is the exported NumPy policy from
`BreckEmert/Splendor-Webapp` at commit
`6ec66efb02406e982510d2279ae1f29b84f80ace`, using the pinned inference weight
SHA256 `5044383cadd98aa6c753b9d0a2bd8557b1ad017c21b6d8ba762027f63277438a`.
Its single output head uses a 251-512-512-256-141 float32 network. No search,
training, alternative checkpoint, precision change or fallback was added.
This export has not been established as the author's strongest historical
checkpoint.

The adapter retains the source feature and network arithmetic while translating
its 141 output indices into corrected legal actions. Payment, token-return and
noble choices are fixed before new information is revealed. Unknown opponent
reservations use one deterministic completion from lawful public information;
actual hidden identities, deck order and referee seeds are not decision inputs.
Both players use the corrected catalogue and the existing atomic turn contract.

The control remains the previously qualified unchanged32 player. Both entrants
retain the one-second complete-response deadline; unchanged32 also retains its
existing internal 0.900-second allowance. Supervision stays on separate CPUs.

## Qualification and verification

- 95 model-free implementation tests passed. Initial fixture/harness failures
  remain preserved in `prelaunch/`; they were corrected before the freeze.
- The actual control startup suite passed 52 decisions. The external startup
  suite passed 64 handler decisions, 64 independent source-reference forwards,
  13 rejection checks and the 141-index mapping checks.
- All 256 ordinary qualification requests passed, over 64 structural positions
  with two hidden variants and two repetitions. Sixty positions had genuinely
  different hidden variants; four had no hidden alternative. These are not 256
  independent positions, and not every structural fixture is proven naturally
  reachable. Maximum response time was 0.1172972 seconds, below the frozen
  0.700-second qualification margin.
- Six fresh operational games passed. Their outcome-blind timing and storage
  forecast admitted the fixed 60-game, 30-pair screen.
- All 66 games and 3,400 live decisions passed the independent action,
  transition and terminal replay checks. The scored screen contains 3,074 moves.
- All 48 selected decisions reproduced successfully, 24 per player; no sampling
  cells were missing. All 182 worker lifecycles passed log, raw exchange and
  placement checks.
- The final evidence seal contains 24,685 artifacts and 178,602,924 bytes.
  The runner checked its readback and the 1,351 frozen input identities, along
  with the pinned external runtime and numerical libraries.
- Runner, watchdog and guardian each have a separately observed OS exit code 0.
  Cleanup receipts report no errors. There were no retries or replaced games.

The execution freeze is
`8fd1343e418242cfd0dab6af61cac8bd5ff46721fe5be4982fc2acc01773990d`.

## Timing and resources

The following live timings cover all 66 games, including the six excluded
operational games. They measure the full supervised response, not just neural
inference.

| Player | Decisions | Mean | 95th percentile | Maximum | Timeouts |
|---|---:|---:|---:|---:|---:|
| Breck policy | 1,700 | 0.0650 s | 0.0970 s | 0.1385 s | 0 |
| unchanged32 | 1,700 | 0.3027 s | 0.4440 s | 0.6781 s | 0 |

No live move reached the descriptive 0.700-second margin. The enforced limit
remained 1.000 second. Zero observed faults in this finite run is not a claim
of zero future timeout risk.

The run began 26 October at 23:51:16 Paris time and its guardian exited on
27 October at 00:38:41: approximately 47 minutes 25 seconds, within the four-hour
cap. The 60-game collection took 33 minutes 17 seconds; final verification took
6 minutes 52 seconds. Peak sampled combined worker memory was 340,713,472 bytes,
below the 2 GiB worker limit. Larger-data work and training remain deferred.

## Interpretation and next recommendation

This adapted checkpoint is fast and passed the prescribed correctness checks,
but it is too weak against unchanged32 to provide the intended competitive
external benchmark. The result concerns this precise checkpoint and adapter;
it does not establish the strength of every player in the upstream project.

Optimized64 did not participate. Its earlier 64.25% score against unchanged32
remains separate evidence; this screen neither confirms nor refutes whether
that advantage extends to independent opposition.

Recommend qualifying the predeclared reserve, `domdomegg/splendor` PPO512, under
its own bounded plan and running its 60-game screen against unchanged32 only
if qualification passes. Do not extend Breck's sample, lower its threshold or
tune its adapter after seeing the outcome. Do not launch the proposed 400-game
confirmation against this rejected opponent. No reserve-opponent execution or
confirmation study has been launched by this task.

The wider objective remains a strong, reliable Splendor player under one second
per move. Earlier work established a gain from 32 to 64 search simulations and
qualified its reliability. Independent opposition is the next test of how
broadly that gain carries. A usable opponent is still missing; additional
training and larger-data investment remain deferred.

## Evidence

- Protocol: `docs/engineering/breck_screen_20261026.md`.
- Frozen attempt and all receipts: `outputs/breck_screen_20261026_01/`.
- Core results: `qualification_summary.json`, `screen_assessment.json`,
  `performance.json`, `coverage.json`, `verification_summary.json`.
- Lifecycle acceptance: `completion.json`, `execution.json`,
  `watchdog_execution.json`, `guardian_os_exit.json`.
- Independent reconciliation: `independent_results_review.json`.

Historical incomplete attempts and their separate accepted recovered
verification retain their original statuses and evidence.
