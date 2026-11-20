# Experiment 144 result: search improvement not established

Completed and independently verified on 2026-09-08. The frozen 143 public
score/resource evaluator, combined with exact 035 canonical priors and real
PUCT at .4 seconds per move, scored **52.75% against raw 035**, paired 95% CI
**[48.5%, 57.0%]**. All 400 games finished normally. The result is valid and
inconclusive for superiority; neither inferiority nor the practical 55% target
is excluded. Frozen decision: `improvement_not_established`.

## Results

| Measure | Result |
|---|---:|
| Fresh starting seeds / seat-swapped games | 200 / 400 |
| Candidate wins / draws / losses | 210 / 2 / 188 |
| Candidate score share | 52.75% |
| Paired whole-seed bootstrap CI95 | 48.5% to 57.0% |
| Difference from 50% | +2.75 percentage points, CI95 -1.5 to +7.0 |
| Candidate score share in seat 0 / seat 1 | 60.0% / 45.5% |
| Mean candidate score margin | +.275 points |
| Mean turns per game | 63.185 |
| Normal completions / cutoffs / deadlocks | 400 / 0 / 0 |
| Mean candidate / raw decision time | .404755 / .004358 seconds |
| Ratio of mean decision times | 92.87x |
| Candidate decision time p95 / maximum | .409692 / .479336 seconds |
| Mean / median completed simulations | 60.501 / 54 |
| Simulation count min / p95 / max | 9 / 93 / 650 |
| Mean maximum depth reached per search | 2.289 |
| Maximum-depth statistic median / p95 / max | 2 / 4 / 6 |

Score share counts draws as half a win. The CI resamples complete seed pairs,
50,000 replicates with the frozen RNG seed, rather than treating seat-swapped
games or individual moves as independent. Success required a CI lower bound
above 50% and a point estimate of at least 55%; neither condition was met.
The CI includes 55%, so this run does not rule out a practically useful gain.
There was no outcome-driven extension, exclusion, replacement or tuning.

Seat figures are descriptive. They do not isolate a treatment-by-seat effect
from first-player advantage and do not justify a seat-specific policy change.
Score margin was not a success criterion. Depth is each search's maximum depth,
not its average path length; the observed searches were generally shallow.

## Interpretation

Experiment 143 established that this simple public evaluator predicts fresh
035-policy outcomes better than its neural value heads and a neutral value.
Experiment 144 does **not** establish that this predictive advantage yields
stronger gameplay at the tested search allowance. The small positive point
estimate remains compatible with no gain, a modest loss, or a useful improvement.
This is an inconclusive strength estimate, not an invalid experiment, proof of
equivalence, or rejection of public value learning or planning in general.

The candidate spends substantially more compute than raw 035. The .4-second
allowance passed its prespecified overhead gates (mean <= .44, p95 <= .50),
but the measured 92.87x latency ratio buys no established strength improvement
in this sample. Timings reflect this host, eight concurrent workers and one
Torch thread per worker. They are not an equal-consumed-compute comparison.

This tests a combined system: canonical-prior search, a frozen public evaluator,
and extra computation. It does not isolate the evaluator's contribution. The
small number of completed simulations and shallow depths identify the tested
operating regime; they do not prove that more search would repair it. Value
calibration under decisions selected by search, prior allocation, search depth
and opponent continuation assumptions remain possible limitations.

Keep 084 as research champion and 035 as fast reference. This result does not
justify a policy update, replacement, promotion or another full learning cycle.
If continuing this line, the next clean attribution question is the frozen
public evaluator versus a neutral leaf evaluator under the same search and
decision allowance, on new seat-paired games. That would test whether the value
signal itself adds strength. It is a proposal, not a launch authorization;
choose its precision and practical threshold before collecting any outcomes.
Do not add games to 144 or use its outcomes to tune the frozen evaluator.

## Execution and verification

The owner authorized the 143 next proposal with "ok, please test that too".
Protocol: `docs/engineering/public_value_search_144.md`. Preparation:
`docs/engineering/public_value_search_144_preparation.md`.

Fourteen focused tests and the complete 1,002-test repository suite passed.
Two separate-seed practice games passed the trajectory/root checks before
launch. All code, protocol, configuration and input hashes were frozen before
the first protected game. Earlier intentionally dirty work was preserved.

Arena parent PID 69048, execution session 13884, used eight CPU worker processes
and one Torch thread each. All game collection finished after 655.536 seconds;
the complete arena stage including aggregation took 657.569 seconds. Session
13884 exited successfully. The saved game JSONs occupy about 105.08 MiB.
No training ran and no model weights changed.

Independent audit session 36564 exited successfully after 64.741 seconds:

- All 400 complete trajectories, 25,274 legal moves and every state hash verified.
- All 12,637 raw-035 choices reproduced with the original scorer.
- All 12,637 search roots checked for exact public value, canonical action set,
  unchanged 035 priors, visit accounting, selection and timing records.
- First and last search in every game reproduced exactly: 800 full searches
  using recorded RNG seeds and completed simulation counts. Their root edges,
  values, chosen actions and expansion counts matched. Middle moves received
  structural/prior/value audits; their full trees were not regenerated.
- All final outcomes, game hashes, summary statistics and independent paired
  bootstrap calculations matched. All 273 current and 272 earlier 143 frozen
  dependencies matched before and after execution.

The audit uses the same engine and search implementation. It is a separate
reconstruction and arithmetic check, not an independent game-rules implementation
or a reproduction of operating-system wall-clock scheduling.

## Artifacts

- Frozen manifest digest:
  `8cccbc2e525e40b89870e3746a5a1f181dd415b8c4481f02cbc6aca5ec9a620c`.
- Raw result: `outputs/public_value_search_144/result.json`, SHA256:
  `fd277b031f3aeffab76deba8ae3975055a5c620c154577b3b8dc9bd963dcf1ea`.
- Independent audit: `outputs/public_value_search_144/verification.json`, SHA256:
  `63feb6296b0747d15d6b08c18a3ac0976e34b792d478811794f0a22377494427`.
- Completion record: `outputs/public_value_search_144/final_status.json`.
- Logs: `outputs/public_value_search_144/run.log` and `verification.log`.
- Games: `outputs/public_value_search_144/games/game_0000.json` through `game_0399.json`.

All experiment processes have exited. Do not restart or extend the arena.
