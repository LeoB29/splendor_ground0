# Corrected-rules four-candidate qualification: completed

18 October 2026. The four-candidate, 128-world planner passed the prescribed
correctness and 120-second timing panel across all three frozen proposal models.
All 26 research decisions completed within their actual deadlines and passed
exhaustive replay. All 23 one-second attempts timed out and remain recorded
failures of the deployment target.

**Recommendation: 120 seconds per complete move for each player in a future
comparison against SplendoRust, conditional on the observed thin timing margin.**
The slowest accepted decision took 101.952 seconds, leaving 18.048 seconds
(15.04% of the allowance). SplendoRust still needs a separately frozen and
qualified profile at this allowance before a match. This task ran no strength
games. The larger independent-data study remains deferred.

## Exact player and panel

The new profile is `four_candidate128_corrected_rules_20261018_v1`, using
`splendor_base_qualified_rules_20261018_v1`. It retains proposal fits 161001,
161002 and 161003, the raw035 continuation checkpoint, up to four candidates,
all 128 worlds, the original public randomness schedule, ordered ties and serial
numerical backend. No model was selected by its result, retrained or optimized.

The isolated adapter applies the adjudicated six card-cost corrections and
compulsory nobles throughout proposals and continuations. A sole PASS that
awards a noble counts as progress; this necessary rules consequence is explicit
in the frozen protocol. Historical implementations and evidence were preserved.
The catalogue SHA256 is
`db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312`.

The exact earlier selection was retained: 18 core positions covering three fits
by two acting seats by three phases, plus five metadata-selected edge positions.
Each root kept its original fit assignment; this is not 23 positions crossed
with every model. The five edges cover own, opponent and both players' private
reserves, and the original two short-support strata. Three prescribed hidden-state
repeats, one per fit, complete the 120-second panel. All 49 attempts ran once,
without replacement or a partial-search fallback.

These are **corrected-rules counterfactual saved positions**. The historical
outcomes are not ground truth for acceptance under changed rules.

## Timing and acceptance

| Proposal model | One-second verified / attempted | 120-second verified / attempted | Mean at 120s | Median | Maximum |
|---|---:|---:|---:|---:|---:|
| 161001 | 0 / 8 | 9 / 9 | 42.601s | 42.898s | 68.939s |
| 161002 | 0 / 7 | 8 / 8 | 51.838s | 47.964s | 101.952s |
| 161003 | 0 / 8 | 9 / 9 | 59.889s | 63.348s | 95.295s |
| All three | 0 / 23 | 26 / 26 | 51.428s | 46.868s | 101.952s |

The pooled 95th percentile was 98.310 seconds, using linear interpolation.
These descriptive timings include the three repeats; they are not an estimate
of a universal tail-latency guarantee. Timed-out cases have no imputed completion
time. Differences between models also reflect their different assigned roots.

| Stratum | One-second result | 120-second result |
|---|---:|---:|
| Core: six positions per model | 0 / 18; all timed out | 18 / 18 verified |
| Five original edges | 0 / 5; all timed out | 5 / 5 verified |
| Three prescribed repeats | Not scheduled | 3 / 3 verified |

The trusted supervisor enforced the complete move deadline on one assigned
logical thread of a performance core, with one numerical thread. The clock
includes input serialization and delivery, search, complete compressed trace
writing and flushing, and receipt of the complete response. Fresh initialization
was outside the thinking clock under its own 60-second cap: all 75 player/audit
initializations succeeded, with median 2.788 seconds and maximum 3.096 seconds.

Each completed decision was followed by a separate exhaustive audit worker,
outside its thinking time and within the shared four-hour study cap. Audit
requests averaged 70.957 seconds and peaked at 141.657 seconds, below their
600-second allowance. The production schedule and these audits took 59m17.4s.
The successful final analysis, including the intervening checker repair, finished
66m43.8s after the original launch. These figures exclude preparation and coding.
Peak sampled combined worker/helper RSS was 242.9 MiB; evidence occupied about
69.6 MiB, well below the 256 MiB cap. Sampling does not measure continuous peaks.

## Correctness evidence and preserved failures

All 23 original/permuted lawful observer inputs matched. Actual hidden allocations
and order were changed, while private opponent card faces and actual deck order
remained outside the player input. All three completed repeat pairs had exactly
matching decisions and compressed continuation traces.

The audits independently reconstructed 3,328 lawful worlds and replayed all
13,184 candidate continuations: 453,604 transitions, including reexecution of
440,420 raw035 policy choices and all 26 root proposal-model computations.
The separate dictionary referee checked every transition, score, terminal outcome
and aggregate choice. Exact scores, hashes and ordered choices agreed; no
post-launch numerical tolerance was introduced. The audits still trust the frozen
model/feature kernels, corrected legal enumeration and serialization, as disclosed
in the protocol; they are not an independent implementation of every dependency.

The preflight suite passed 137 tests. A synthetic noble-PASS integration fixture
initially violated the score/end-round ledger. It was replaced before freezing
with a ledger-valid noble-finishing fixture; the ledger was not weakened. The
failure and repair receipts remain in the preparation folder. The archived source
snapshots were captured while repair was underway, so they are explicitly not
claimed to be the exact source version of that failed test.

The first final analyzer also failed, on a receipt-schema defect: the clock's
initialization file contains five supervisor metadata fields absent from the row's
raw response copy. The original checker and failed execution remain unchanged.
A separate version 2 validates those five fields independently and compares all
shared clock fields exactly. Its 30 regression checks and a separate review
passed; timing, replay, preservation and qualification gates were unchanged.
Version 2 then passed on its only full execution. No player decision or full
continuation audit was rerun for this repair.

The successful analysis rechecked all 16,695 frozen inputs (300,137,803 bytes),
including both previous completed qualification/screen evidence trees. Every
size and SHA256 matched. The original freeze SHA256 is
`75afcb4f4412cb2b8f106db93481c9d318734b3f55a1bec07cc3df01b363b180`;
the accepted version-2 analysis SHA256 is
`4140061c31572602ea32227c3240a598ba33ff347c15574ae6495f9e3647dec2`.

For context only, corrected legal support differs from historical support on
2 of the 23 unique roots; candidate sets differ on 6 and final actions on 9
(10 action changes among all 26 including repeats). These changes are neither
failures nor evidence of stronger play. New full trace recording also differs
from the historical timing instrumentation, so the change from the historical
43.61-second mean cannot be attributed solely to the corrected rules.

## Matched-budget recommendation and next step

The prospective rule qualifies 120 seconds only when all 26 research cases pass.
It labels the margin thin if any takes more than 96 seconds. Both conditions
apply here. Retain the tested **120-second ceiling**, without inferring a lower
untested allowance or increasing the ceiling automatically. The player qualifies
for this finite research panel and fails the tested one-second deployment target.

The next recommended work is a timing-only calibration and correctness/privacy
qualification of a new SplendoRust profile at the same 120-second whole-turn
ceiling. Freeze its workload using a prospective rule, before match outcomes.
Keep the same core/thread allocation, initialization policy, observer boundary
and fail-closed deadline handling for both players. Qualify the exact future
match interface if its trace/instrumentation costs differ from this one.

The previously qualified SplendoRust PUCT16 profile has a one-second external
ceiling and a 0.90-second internal limit. Its existing loader explicitly limits
that profile to 0.90 seconds internally: a new profile and qualification are
needed, not an edit to the frozen screen. Its 58-2 screen against raw035 does not
establish its strength against this planner. Equal wall allowances also do not
imply equal simulations, inference work or actual CPU use.

After both exact interfaces qualify, a separately authorized paired, seat-swapped
strength study can address the roadmap's real question: stronger play against an
independent opponent. Retain all three local fits and predeclare the matchup and
sample size. This fixture study does not promote four candidates above the
historically stronger eight-candidate player or establish corrected-rules strength.
No new match, larger dataset, training, global engine migration or cleanup was run.

## Evidence locations

- Prospective design: `docs/engineering/corrected_planner_qualification_20261018.md`.
- Main evidence: `outputs/corrected_planner_qualification_20261018_01/`, including
  `freeze.json`, `schedule.json`, `rows/`, `attempts/`, `audits/`, `completion.json`,
  `independent_analysis_v2.json` and both numbered analyzer execution receipts.
- Checker repair: `analyzer_repair_v2.json` and `.diff` in the same output folder;
  `tools/analyze_corrected_planner_qualification_20261018_v2.py` and
  `tests/test_corrected_planner_analyzer_v2_20261018.py` are separate new files.
- Preparation failure: `outputs/corrected_planner_preparation_20261018_01/`.
- Prior opponent result: `docs/engineering/external_strength_screen_20261018_02_result.md`.
- Final report/handoff identities and retained tail hashes: the main output's
  `finalization.json` and `handoff_completed_update.json`.
