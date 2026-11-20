# Experiment 148: additional-compute strength gain not established

Completed and independently verified on 2026-09-08. All 400 protected compute
games finished naturally. The first-stage strength gate failed, so the fixed
pipeline stopped without running the conditional raw-035 panel. No training,
promotion or successor ran. Champions remain 084 research / 035 fast.

## Primary result

The unchanged 035/public-value PUCT player at 1.6 seconds per move scored
**53.75% against its .4-second counterpart**, paired CI95 **[49.5%,58.0%]**.
Success required score share >=55% and CI lower >50%; neither condition passed.
The result is inconclusive for superiority, not evidence of equivalence or
inferiority. The interval still includes the 55% practical target.

| Measure | Result |
|---|---:|
| Starting seeds / seat-swapped games | 200 / 400 |
| Longer-search wins / draws / losses | 214 / 2 / 184 |
| Longer-search score share | 53.75% |
| Whole-pair bootstrap CI95 | 49.5% to 58.0% |
| Difference from 50% | +3.75 percentage points, CI95 -.5 to +8.0 |
| Longer-search share in seat 0 / seat 1 | 58.5% / 49.0% |
| Mean final score margin | +.3875 points |
| Mean turns per game | 63.2 |
| Natural completions / cutoffs / deadlocks | 400 / 0 / 0 |

Draws count half a win. Statistical resampling uses 50,000 complete starting-seed
pairs, not individual moves or independent treatment of the two paired games.
Seat shares and margin are descriptive; neither replaces the failed primary gate.
No protected games were added, excluded, replaced or used to select settings.

## The additional search was actually delivered

| Search measurement | 1.6-second arm | .4-second arm |
|---|---:|---:|
| Decisions | 12,640 | 12,640 |
| Mean decision seconds | 1.605280 | .404973 |
| Decision seconds p95 | 1.610718 | .409801 |
| Maximum decision seconds | 1.703360 | .490598 |
| Mean CPU seconds per decision | 1.388148 | .350423 |
| Mean completed simulations | 221.283 | 56.357 |
| Median completed simulations | 198 | 51 |
| Simulation count p95 / maximum | 290 / 2,200 | 74 / 660 |
| Mean maximum depth reached per search | 3.225791 | 2.403560 |
| Maximum-depth statistic median / p95 / maximum | 3 / 5 / 8 | 2 / 4 / 7 |

The measured mean decision-time ratio was **3.963922**, with roughly 3.93 times
as many completed simulations on average. Both budget gates passed: every search
used its allowance, mean time stayed within 1.10 times allowance, and p95 within
1.25 times allowance. Timings reflect this host, eight concurrent CPU workers
and one Torch thread each. They include root inference and exclude caller legal
enumeration and artifact serialization, consistently with 144/145.

Depth is each search's maximum reached depth, not its average path length.
These arm means describe different trajectories. They show the operating regime
and delivered computation, but do not separately identify depth, exploration or
chance sampling as a causal explanation for the strength result. The treatment
is the complete change in search allowance under otherwise fixed settings.

## Programme consequence

148 does not establish the stronger shared-tree teacher sought by this proposal.
The point estimate is positive, but compatible with no gain and with the useful
target. Do not claim that more thinking never helps, that the players are equal,
or that 55% strength is excluded. The fourfold allowance was exercised; a missing
compute intervention does not explain the inconclusive estimate.

The competence panel against raw 035 must remain unrun under the frozen rule.
This duel does not establish a new strength result versus raw 035 or 084, and
must not be pooled with 144 or 145. It does not isolate the public evaluator's
benefit at 1.6 seconds, nor test a learned successor. The positive 113 headroom
result remains valid for its different generation-11 system.

Together with 147's overfit, this gives no new basis for another unchanged value
fit or an automatic search-budget sweep. Preserve the consumed 148 panel; do not
extend it, retune budgets against it or bypass its gate. Further work needs a
substantive proposal about obtaining and learning useful decision improvements.
No specific successor is authorized by this result. Champions remain 084 research
and 035 fast; no neural checkpoint was trained, modified or promoted.

## Execution and verification

Protocol: `docs/engineering/search_teacher_headroom_148.md`. Preparation:
`docs/engineering/search_teacher_headroom_148_preparation.md`.
The owner approved the proposal with "ok, please do it".

Twenty-six focused tests and the full 1,096-test repository suite passed.
Four separate-seed preflight games finished naturally; all 260 moves, 195 timed
roots, 65 raw-035 actions and 12 full searches passed their checks before launch.
Both protected stage schedules, configurations and gates were frozen together.

Parent PID 8688 / execution session 55760 started at 17:49:47 Europe/Paris.
Game collection took 3,221.885 seconds; aggregation completed at 18:43:32,
3,224.866 seconds after the arena stage started. No running win rates were
inspected. Independent audit PID 28968 completed successfully in 226.732 seconds.
The entire frozen pipeline completed at 18:47:20 in **3,453.642 seconds
(57 minutes 34 seconds)**, excluding implementation, testing and preflight.
Parent and audit processes exited successfully; no active experiment jobs remain.

The independent audit passed:

- All 400 complete trajectories and 25,280 legal moves/state hashes/outcomes.
- All 25,280 timed roots, including original 035 canonical priors, independently
  calculated public values, visit accounting, selected actions and budgets.
- Exactly 800 long and 800 short first/last searches reproduced in full, with
  matching edges, values, actions, expansions and inference counts.
- All game hashes, independently recomputed paired share/CI and decision gates.
- All 280 current and 280 explicitly listed prior frozen dependencies, plus
  prior inputs and anchored manifest hashes, unchanged.

The verifier shares the repository engine and search code; it does not
independently implement the rules or reproduce OS wall-clock scheduling.
Middle search trees received structural and original-prior/value checks rather
than full regeneration. No repairs, exclusions or reporting recovery were needed.
Final status: `complete`, `compute_gate_failed_stop`, conditional competence
false, training false and promotion false. No protected competence directory
or failure artifact exists. Earlier dirty work and frozen experiments are preserved.

## Artifacts

- Root: `outputs/search_teacher_headroom_148/`.
- Manifest digest:
  `1abb1e379a589bda16017ae1b3e5c840e05ae6e79ed94d9ba3b5b6805499cd34`.
- Manifest file SHA256:
  `2f7a300fb8d058e05e64ab0f3e5c28d53004524820b13cd01995814215f8f257`.
- Compute result: `compute/result.json`, SHA256:
  `152a9aff0c0f673e3f2fe9098e7eaf74dae2814a9b2d45a92640370aaadf0f4d`.
- Verification: `compute/verification.json`, SHA256:
  `7d22285ca0cc8d36cfe84043d4afea8c9d91b66f1099477542f6aed67c0fe7a7`.
- Logs: `run.log`, `compute/verification.log`.
- Complete game records: `compute/games/game_0000.json` through `game_0399.json`.
- Final pipeline completion: `final_status.json`.
