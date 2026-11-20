# Live-game adapter and six operational games: accepted

19 October 2026. The corrected four-candidate planner completed one swapped-seat
pair per fixed proposal model against the qualified SplendoRust profile at
120 seconds per complete move. All six games, 306 actual moves, twelve adapter
validation requests and 153 prescribed game continuation audits passed. There
were no player timeouts, forfeits, initialization failures, illegal delivered
actions, privacy mismatches or resource holds. Independent analysis v2 passed.

The original final verifier failed on a starting-state envelope-key bug. Its
failure is preserved, and the separately frozen repair is disclosed below.
Matchup outcomes remain closed; these six games contribute **zero strength
observations**. The larger independent-data study remains deferred.

**Recommend preparing the remaining nine operational pairs under a new bounded
plan, retaining both players, all three local fits and the 120-second allowance.**
The observed local timing margin is thin. This pilot supports continuing
operational qualification; it does not establish playing strength or guarantee
future deadline compliance. No further games have been launched.

## Exact players and transfer validation

The local profile is
`four_candidate128_corrected_live_operational_20261018_v1`: four candidates,
128 shared worlds, fixed fits 161001/161002/161003, unchanged raw035 continuations,
serial computation and original tie rules. The versioned adapter changes the
live context, random-stream identity and evidence destination while retaining
the corrected proposal, world, continuation and reduction primitives.

The opponent remains
`entity_onehot_puct6400_canonical_atomic_prereveal_cpu_120s_v1`, profile SHA256
`6f74c3b6d77707c42619801d1b1bef3fbf203d1ee6ff548afbe935169c257599`.
Its qualified model, native executable, observer-only interface, search settings
and declared heuristic branches are unchanged. Both use
`splendor_base_qualified_rules_20261018_v1`, catalogue SHA256
`db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312`.

All 239 preflight tests passed. The three predetermined saved roots each supplied
legacy/new-context and original/hidden-variant requests, giving twelve timed
decisions. All six privacy pairs matched exactly. Every legacy scientific
decision matched its accepted original after removing only the declared profile
label, and compressed traces matched exactly. All twelve received full replay:
1,536 worlds, 6,144 continuations and 365,044 dictionary transitions, including
358,900 reexecuted raw035 choices. No case was retried or replaced.

The three paired starting worlds were independently reconstructed, with exactly
the same initial world in each pair and entrant seats reversed. The frozen order
was 0,2,4,1,3,5. All 306 actual observer/action/successor transitions and sealed
endings passed independent checks. Referee seeds stayed outside player inputs.
The collision inventory covered 230 declared files and excluded 22 files over
8 MiB; this is a documented inventory check, not a universal noncollision claim.

## Timing and execution cost

Both players had the same externally enforced 120-second complete-response
clock, one assigned performance-core logical processor and one numerical
thread. Fresh initialization was separately capped at 60 seconds for every
decision. Local trace writing and flushing were included in its move clock.

| Live-game metric | Local planner | SplendoRust |
|---|---:|---:|
| Completed / attempted decisions | 153 / 153 | 153 / 153 |
| Mean complete-move time | 46.750s | 26.822s |
| Median | 48.216s | 30.563s |
| 95th percentile | 87.315s | 34.911s |
| Maximum | 113.498s | 41.340s |
| Recorded player failures | 0 | 0 |

The local maximum leaves **6.502 seconds, or 5.42%**, below the limit. These are
correlated decisions from only three independent starting-seed pairs. They do
not establish a reliable tail-failure rate. Equal allowances also do not mean
equal consumed thinking time. The separate one-second deployment target remains
unresolved by this exercise.

Game execution took about 3h24m, including fresh initialization. Adapter
validation, exhaustive validation replay and admission took 41m27s. Post-game
continuation audits took 42m45s: 29 full audits averaged 58.288s and 124 sampled
audits averaged 7.055s, including their initialization and cleanup. Validation
audits are already included in the validation stage and are not added again.
The runner finished in 4h49m43s; the accepted independent analysis finished about
4h59m06s after the original launch, including the verifier repair.

All 483 worker initializations completed; mean receipt time was 2.607s and maximum
5.111s. Peak sampled combined worker RSS was about 275.2 MiB. Evidence occupied
about 396 MiB at accepted analysis, below the 2 GiB cap. The original twelve-hour
deadline and final reserve were retained throughout. The process receipt checks
483 unique worker identities and closure of 1,272 recorded worker/helper
identities plus the supervisor. Filesystem timing envelopes corroborate serial
execution; they are not an independent absolute monotonic clock proof. Analyzer
closure is recorded separately in its execution receipt.

## Continuation verification and limits

Every completed local decision received independent proposal/selection checks,
reconstruction of all 128 world identities, and checks of all recorded outcomes,
aggregates, choices and trace structure. The frozen outcome-blind manifest then
selected eight complete worlds per root and full 128-world replay at stress
roots. Overlapping selections were deduplicated.

| Game continuation coverage | Count |
|---|---:|
| Completed local roots | 153 |
| Full stress roots / sample-only roots | 29 / 124 |
| Recorded worlds / uniquely replayed worlds | 19,584 / 4,704 |
| Recorded continuations / uniquely replayed continuations | 73,088 / 16,568 |
| Replayed dictionary transitions | 590,870 |
| Reexecuted raw035 choices | 574,302 |
| Observed stress cells / predefined cells | 33 / 81 |

All prescribed audits passed. Forty-eight stress cells were absent and remain
untested: 46 phase/seat/privacy combinations and model 161002's candidate-support
sizes one and three. They are neither failures nor grounds for adding targeted
games to this completed sample. The exact missing cells are listed in the
accepted analysis. Unsampled continuations received structural and arithmetic
checks but were not replayed. Full-root replay includes its eight sample worlds;
the table does not count them twice.

SplendoRust's reported work and legal outputs were checked: 153 searched Main
decisions, 979,200 simulations, 812,909 host inference calls and 22 auxiliary
heuristic phases. No forced-pass, single-choice Main or late-turn heuristic
branch was observed. The audit retains the qualified native/model parity trust
boundary; it does not reexecute every internal Rust search edge, backup or neural
output. Local replay shares frozen model/feature kernels and corrected support
dependencies; dictionary-ledger independence is not an entirely separate engine.

## Preserved verifier failure and accepted repair

Execution 01 of the original final verifier stopped at the independent initial
world comparison. It used `ruleset` instead of the corrected snapshot's
`ruleset_id` envelope key. Model-free diagnosis found all six saved starting
worlds exactly matched with the correct key. No game, model or continuation
search was rerun to address this defect.

The original verifier and failure remain unchanged. A separate v2 corrected
that key, retained the exact initial-state comparison and every other scientific
gate, and added separate repair authentication/output names. All 74 original
and new regression tests passed, including corruption rejection and checks
that other gates remain unchanged. Another reviewer cleared the complete diff.
The repair was frozen prospectively before its one full execution, which passed
in 30.73s. No allowance restarted and no criterion was relaxed.

All 17,704 original inputs (603,229,652 bytes) and all nine repair inputs retained
their hashes and sizes. The original completion receipt deliberately still says
`completed_pending_independent_analysis`; acceptance is established by the
separate v2 result and finalization receipt, without rewriting that history.

| Provenance | SHA256 |
|---|---|
| Original experiment freeze | `e5c635c1cd5f0e4e9e499a421ab25f8a34b82076707ee946e10d3f0644f904ba` |
| Separate verifier repair freeze | `fa27d136c589f54f0e2e993beb3b706e150520e1e744375e81daf9ee705611e6` |
| Accepted independent analysis v2 | `d3259938221b8e5836bd79d1a7bdbcf30762f3bd5cfc57413ac04c914986a0db` |

Evidence is under `outputs/live_game_operational_20261018_01/`. Key records are
`independent_analysis_v2.json`, both `independent_analysis_execution_*.json`
receipts and logs, `sealed/independent_analysis_failure_01.json`,
`analyzer_repair_02/freeze.json`, `audit_manifest.json`,
`process_metadata_closure.json` and `finalization.json`. The prospective protocols
are [the six-game plan](live_game_operational_20261018.md) and
[the verifier repair plan](live_game_operational_20261018_analysis_repair.md).

## Recommended next step

Prepare a separately frozen continuation for the remaining nine operational
pairs, three per fit, retaining the same players, clock, work and closed-outcome
rule. The current live context only admits game IDs 0 through 5; extending it
requires an explicit versioned schedule/context boundary and transfer checks,
while preserving this accepted execution.

Price the continuation using measured game and audit costs before launch.
Simple three-times scaling of this six-game game/replay cost is about 12.3 hours
for eighteen more games, before new transfer validation and finalization. This
is a planning scenario: game lengths, stress coverage and replay costs can change.
Choose an adequate bounded execution plan prospectively rather than extending
this run, weakening checks or replacing failures. Current zero failures do not
justify changing candidates or moving straight to a large strength benchmark.

Keep all twelve roadmap operational pairs outside any later fresh strength
sample. Freeze that sample using precision requirements and measured cost before
opening matchup totals. The larger independent-data study remains deferred.
