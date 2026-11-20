# Experiment 145 result: public-value benefit not established

Completed and independently verified on 2026-09-08. Under the same .4-second
search allowance and frozen 035 move priors, the 143 public score/resource
evaluator scored **52.375% against neutral unfinished-position values**,
paired 95% CI **[49.25%,55.5%]**. All 800 games completed normally and both
time gates passed. Frozen decision: `value_benefit_not_established`.

This is a valid but inconclusive attribution experiment. It establishes neither
a positive value benefit nor inferiority, and does not exclude the practical
55% target. The small positive point estimate is not evidence of equivalence
or proof that the evaluator is useless.

## Results

| Measure | Result |
|---|---:|
| Fresh starting seeds / seat-swapped games | 400 / 800 |
| Public wins / draws / losses | 417 / 4 / 379 |
| Public score share | 52.375% |
| Whole-seed paired bootstrap CI95 | 49.25% to 55.5% |
| Difference from 50% | +2.375 percentage points, CI95 -.75 to +5.5 |
| Public score share in seat 0 / seat 1 | 58.75% / 46.0% |
| Mean public score margin | +.4025 points |
| Mean / minimum / maximum game turns | 63.6975 / 50 / 82 |
| Normal completions / cutoffs / deadlocks | 800 / 0 / 0 |

| Search measure | Public value | Neutral value |
|---|---:|---:|
| Decisions | 25,479 | 25,479 |
| Mean decision seconds | .404434 | .404546 |
| Decision seconds p95 | .409154 | .409376 |
| Maximum decision seconds | .480075 | .490988 |
| Mean CPU seconds per decision | .369679 | .369742 |
| Mean completed simulations | 65.179 | 64.486 |
| Median / p95 simulations | 58 / 104 | 57 / 101 |
| Minimum / maximum simulations | 7 / 849 | 9 / 854 |
| Mean maximum depth reached per search | 2.316 | 2.491 |
| Maximum-depth statistic median / p95 / max | 2 / 4 / 8 | 2 / 4 / 8 |

The ratio of mean public to neutral decision time was **.999724**, a difference
of about .028%, inside the frozen [.95,1.05] range. Both arms passed mean <= .44
seconds, p95 <= .50 seconds, and full .4-second allowance checks. Matched wall
time does not imply identical simulations, depths or visited positions; these
can change as a consequence of the value treatment. Depth statistics describe
each search's maximum, not the average simulation path length.

Score share counts a draw as half a win. Bootstrap resampling used 50,000 whole
seed-pair replicates with the frozen RNG seed. Individual moves and games within
a seat pair were not treated as independent units. Primary success required
CI lower >50% and point share >=55%; neither condition was met. The upper bound
55.5% still includes the practical target. No games were added, replaced or
excluded, and no settings were selected after protected outcomes existed.
Seat shares and score margin are descriptive, not alternative success tests
or evidence for a seat-specific treatment.

## What this changes in the programme

Experiment 143 showed substantially better prediction of 035-policy outcomes
from basic public features than from a neutral value. Experiment 144 did not
establish that combining that evaluator with .4-second PUCT beats raw 035.
Experiment 145 now isolates the unfinished-position scalar while holding policy,
search implementation, terminal recognition and decision allowance constant.
It still does not establish a playing-strength benefit from that scalar.

The control computes the same frozen inference, including the public evaluator,
before replacing its unfinished-position value with zero. It retains exact
terminal win/loss/draw and forced-pass deadlock values. Thus the comparison does
not give neutral a cheaper inference path or remove its terminal understanding.
Every audited root retained the original 035 canonical priors in both arms.

The evidence does not justify scaling this unchanged public-value/.4-second
search recipe or using it as a validated improvement teacher. It also does not
rule out a small useful effect, better value representations, different training,
deeper planning or an eventual strong-policy learning cycle. The generally
shallow searches describe the tested regime; they do not prove that increasing
depth would solve the problem. No bottleneck is uniquely identified here.

Do not pool 144 and 145 as one common endpoint: they have different opponents
and estimands. Neither a similar point estimate across them nor a head-to-head
result versus neutral establishes a new win rate against raw 035 or champion
084. Preserve 084 as research champion and 035 as fast reference.

The protected 145 outcomes are consumed. Do not extend this experiment to make
its interval pass, tune its evaluator against these outcomes, or automatically
start another arena or learning run. Any continuation needs a new scientific
question and a design fixed before new outcomes, with a clear reason that the
effect being resolved matters to eventual strong play.

## Execution and independent verification

The owner authorized the proposed 800-game comparison with "ok do i t".
Protocol: `docs/engineering/public_value_ablation_145.md`.
Preparation: `docs/engineering/public_value_ablation_145_preparation.md`.

Twenty-one focused tests passed, and the complete repository suite passed
**1,023/1,023** in 229.95 seconds before launch. Two full separate-seed practice
games passed, including all 130 roots and eight complete search reproductions.
All code, protocol, config and input hashes were frozen before the first
protected game. Earlier intentionally dirty work and frozen sources were preserved.

Arena parent PID 25744, execution session 92976, launched about 13:34 Europe/Paris
on 2026-09-08 with eight CPU workers and one Torch thread each. Collection took
2602.444 seconds (43.37 minutes); the full arena stage including aggregation
took 2607.842 seconds. Saved game JSONs occupy about 386.21 MiB. All 800 protected
outcomes were revealed only after collection ended. No training ran.

Independent audit session 29911 completed successfully in **224.349 seconds**:

- All 800 legal trajectories, 50,958 moves, state hashes and final states checked.
- All 50,958 roots checked against original 035 canonical priors and the correct
  public/neutral scalar, plus visit totals, selection, bounded values and timing.
- First and last search of each arm in every game reproduced exactly with saved
  RNG seeds and completed simulation counts: 1,600 public and 1,600 neutral,
  **3,200 total**. Root edges, values, selected actions and expansion counts matched.
- All game hashes, outcomes, summary statistics and an independent paired
  bootstrap calculation matched. All **274 current** and **273 prior** frozen
  dependencies were unchanged before and after execution.

Middle search trees were checked structurally and against original priors/values;
they were not fully regenerated. The verifier shares the repository engine and
PUCT code, so is not an independent rules implementation or a reproduction of
OS wall-clock scheduling. No repairs, exclusions or reporting recovery were needed.

## Artifacts

- Manifest digest:
  `f110daefbfe714954639c7294f24ba2dfdb5de062cab2f3746e56bea48916672`.
- Result: `outputs/public_value_ablation_145/result.json`, SHA256:
  `e8b3fcc7b74cadb31512266a97ed8ddacc458cc8e57a03d6b7538215df700305`.
- Verification: `outputs/public_value_ablation_145/verification.json`, SHA256:
  `06be73e6b35461d8a09aeae417be0ee9c3002602643414f6ac65b4bfa0e523ea`.
- Completion record: `outputs/public_value_ablation_145/final_status.json`.
- Logs: `outputs/public_value_ablation_145/run.log` and `verification.log`.
- Game records: `outputs/public_value_ablation_145/games/game_0000.json`
  through `game_0799.json`.

Arena and audit sessions have exited. No active experiment processes remain.
No training, promotion or further experiment is authorized by this completion.
