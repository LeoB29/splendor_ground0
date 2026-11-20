# Experiment 154 result: fresh rollout selection passes, fully audited

2026-09-10 Paris. **Complete; all statistical and integrity gates passed.**
The fixed selector improved mean terminal WDL over exact greedy 035 by
**+0.048868815**, with a paired 95% confidence interval of
**[+0.036814372, +0.061350759]**. All three prespecified statistical gates pass.
Independent reconstruction of the statistics passed. Every source game and all
247,136 main continuations were independently replayed. Final closure and
post-run identity, artifact-hash and test-evidence verification passed.

This is prospective evidence about choosing one move on fresh baseline
positions, followed by greedy 035 for both players. It is not a complete-game
strength result. No model was fitted, and no arena, promotion or successor ran.
Champions remain 084 research / 035 fast. All dirty and frozen predecessor work
is preserved.

[Final status](C:/Users/leoba/Desktop/splendor_ground0/outputs/rollout_selector_154/final_status.json),
[frozen protocol](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/rollout_selector_154.md),
[machine-readable result](C:/Users/leoba/Desktop/splendor_ground0/outputs/rollout_selector_154/result.json)
and [per-position observations](C:/Users/leoba/Desktop/splendor_ground0/outputs/rollout_selector_154/state_results.json).

## Fixed comparison and primary result

The panel contains 768 positions from 768 distinct fresh greedy-035 source
games: 128 in each early/middle/late by acting-seat cell. The deterministic
matching rule first filled the quotas after 1,612 source games. Four extra
games from the final eight-game batch were preserved and audited, for 1,616
attempted games and 102,860 source moves. There was no outcome-based source
replacement or panel extension.

Each position retained the exact greedy-035 move, its highest-ranked distinct
canonical alternative, and two uniformly sampled remaining canonical groups.
767 positions had four real candidates; one had three. No position was removed.
There were no baseline-only positions in this fresh panel.

The selector used 32 sampled legal hidden worlds per candidate and exact
greedy-035 terminal continuations for both players. It chose the largest mean
WDL, preferring the original candidate order on exact ties, with baseline first.
All 768 choices were sealed before any evaluation began. Evaluation then used
128 independent new world draws per position, shared between the locked choice
and baseline. A coincident choice was evaluated once and retained with zero gain.

| Prespecified gate | Observed | Decision |
|---|---:|---|
| Mean paired WDL gain >= +0.02 | **+0.048868815** | Pass |
| Paired 95% interval lower bound > 0 | **+0.036814372** | Pass |
| Interval half-width <= 0.02 | **0.012268194** | Pass |
| All integrity audits pass | Full source/discovery/evaluation replay and statistics passed | Pass |

The interval uses 50,000 source-game bootstrap resamples within the six cells,
with equal state/cell weights and the frozen seed. Individual continuations
are nested observations, not extra independent source games. The independent
verifier reproduced the bootstrap through occurrence-count arithmetic and
separately interpolated quantiles.

WDL is +1 for a win, 0 for a draw and -1 for a loss from the root actor's
perspective. The gain corresponds to **+2.443 percentage points of expected
terminal score** (win=1, draw=0.5), with interval [+1.841,+3.068], under these
conditional continuations. It does not predict a 2.443-point arena win-rate gain.

## Selection optimism and descriptive breakdown

| Quantity | Observed |
|---|---:|
| Apparent gain on discovery worlds | +0.142049154 |
| Gain on independent evaluation worlds | +0.048868815 |
| Discovery minus independent gain | +0.093180339 |
| Positions where the selector changed the move | 395 / 768 (51.4323%) |
| Coincident baseline choices, all retained | 373 / 768 (48.5677%) |

The independent gain is about 34.4% of the apparent discovery gain. A useful
positive gain survived fresh evaluation, alongside substantial selection
optimism. The discovery maximum should therefore still not be treated as an
unbiased best-action value or a stable exact training label.

| Phase | Positions | Discovery gain | Independent gain | Move changes |
|---|---:|---:|---:|---:|
| Early, turn <20 | 256 | +0.174682617 | +0.014587402 | 64.8438% |
| Middle, turns 20-39 | 256 | +0.168579102 | +0.053741455 | 62.5000% |
| Late, turn >=40 | 256 | +0.082885742 | +0.078277588 | 26.9531% |

| Cell | Positions | Independent gain | Move changes |
|---|---:|---:|---:|
| Early, actor 0 | 128 | -0.006958008 | 59.3750% |
| Early, actor 1 | 128 | +0.036132813 | 70.3125% |
| Middle, actor 0 | 128 | +0.047424316 | 67.1875% |
| Middle, actor 1 | 128 | +0.060058594 | 57.8125% |
| Late, actor 0 | 128 | +0.074340820 | 25.7813% |
| Late, actor 1 | 128 | +0.082214355 | 28.1250% |

Actor-0 mean gain was +0.038269043 and actor-1 gain +0.059468587, each across
384 positions. Late positions show the largest observed gain despite fewer
move changes; the early actor-0 cell has a slightly negative mean. These are
the prespecified descriptive summaries, without separate significance claims.
They do not authorize a phase filter, new budget or retrospectively selected
variant. The source restriction and phase balancing remain part of the target
population.

## Integrity and work accounting

| Stage | Source games or states | Continuations | Moves |
|---|---:|---:|---:|
| Main source generation, all audited | 1,616 games | Not in continuation cap | 102,860 |
| Main discovery | 768 states | 98,272 | 3,381,799 |
| Main independent evaluation | 768 states | 148,864 | 5,491,334 |
| **Main continuation total** | **768 distinct source games** | **247,136** | **8,873,133** |
| Off-panel source timing probes | 2 games | Additional | 134 |
| Off-panel discovery/evaluation probes | 6 fixed probe states | 132 | 4,422 |

The main continuation count is below the fixed 294,912 ceiling. Discovery had
98,268 ordinary completions and four valid forced-two-PASS deadlocks; evaluation
had 148,863 ordinary completions and one valid forced-two-PASS deadlock.
No invalid cutoff was reassigned, excluded or replaced. The existing score and
card-count adjudication applies to the valid deadlocks.

All source games, snapshots, matching, candidate support/order and root score
privacy checks passed independent audit. Every discovery continuation was
replayed, including independent legal-world reconstruction and greedy action
selection. The complete selection lock preceded every evaluation file. The
completed discovery certificate is bound at final closure instead of replaying
that unchanged stage again. All 148,864 evaluation continuations were then
independently replayed, matching every world, action trace and terminal result.
The final audit, all 20 linked-hash checks, every source/target file hash and
the five frozen files' agreement with pre-run test evidence passed. Both
preparation and main processes exited 0; no failure artifact was recorded.

The auditor shares the frozen repository engine, canonical grouping and 149's
independent greedy replay routine. It is not an independent implementation of
Splendor's rules. World reconstruction, matching, candidate ordering, choices
and statistical arithmetic are separately implemented for this experiment.

| Stage | Measured time |
|---|---:|
| Full repository suite | 208.64 s, 1,214 passed |
| Preparation process, including independent probe audit | 21.90 s |
| Main source-panel audit | 56.16 s |
| Discovery collection, launch to last saved completion | 32m41s |
| Complete independent discovery replay audit | 28m51s |
| Independent evaluation collection, launch to last completion | 52m00s |
| Complete evaluation replay and independent final audit | 47m46s |
| Main process, including all source/discovery/evaluation work and audits | 2h43m50s |

The small outcome-blind probe projected about 90 minutes of maximum collection
and replay work. Main continuations were slower, so that estimate was optimistic.
The fixed population and budgets were preserved. Main run launched at 12:38:39
Paris; selection sealed at 13:13:41; evaluation launched at 13:42:35 and finished
at 14:34:36. Final audit and closure completed at 15:22:26; the process exited
at 15:22:27. Preparation plus main execution took 2h44m12s. Implementation,
focused tests, the full test suite and reporting are additional.

## Scientific conclusion and limits

The statistical result supports the specific 32-world selector as a source of
conditional improvements on fresh baseline positions. It weakens the explanation
that this operator's entire apparent benefit is merely reused-development or
discovery-world selection noise. It does not establish that a learned model can
recover the improvements: 149-153's failed learners retain their negative results.

This population used exact greedy-035 source games. The motivating post-150
split-world diagnostic used exploratory development positions and reused data,
so the two means are not an exact replication or a pooled estimate. Source
snapshots contain no already-held private blind reservations for either player;
blind reserve remains legal at the root and throughout continuations.

The current result changes one move and follows baseline play thereafter.
Repeated selection could encounter different positions, compounded errors and
different opponent responses. Complete-game benefit, strength versus 084, runtime
competitiveness, stable labels and a successful learning loop remain open.
Experiment 084 already established that search can improve on 035; this is not
the first proof of search headroom or evidence of superhuman play. A later
complete-game proposal is the next question supported by this audited result.
It is not launched or authorized by this report.

## Provenance

- Frozen 154 binding: `8ee50225d1e73b5ea778192ce5b32cb5fdad36d29ae2ab139c62042e6dd4350f`.
- Parent 153 binding: `0627752a8399ac74484584025135282dcbff41e376122a362a8cf32218554d3f`.
- Selection-lock SHA: `d207c402a78befe836d47140536a7131bc7803adfce5e6dbe9b3afd8114df449`.
- Discovery-audit SHA: `8c787fa178c8ecf72e92481318f51d275cfbec3ad2fa2f2e8a0b168ea8ca52e5`.
- Result SHA: `96c39b8562f8de96159b19ae42c55e9aaf09187172a54665b42a1768cc614e0c`.
- State-results SHA: `e16e588809c72be4a7e855e59b2502d8268bb7228f2dbe27cd41d810c111f6a5`.
- Evaluation-targets SHA: `c5d4235eaed7bfc384438de0a4b6a02868c41f36b630ead25ed230158bc4b3d0`.
- Final-audit SHA: `7e12fbe45201e41cb35930bb0e5a9553686d80572a8ad95b73a4b907e75d8ddf`.
- Final-status SHA: `42fcf7ee89319befb4c45ff5b9a5c8deddd1ef8d26ebb5e023b566a4dd83d9f5`.

[Complete final audit](C:/Users/leoba/Desktop/splendor_ground0/outputs/rollout_selector_154/final_audit.json)
and [post-run verification](C:/Users/leoba/Desktop/splendor_ground0/outputs/rollout_selector_154/post_run_verification.json).
Experiment 154 is closed. No automatic successor is running.
