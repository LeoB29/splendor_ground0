# Proposed strength comparison of 128 and 64 simulations

28 October 2026. **Prepared recommendation only. No implementation, freeze or
experiment is launched.** The [roadmap update](../strategy/splendorust_roadmap_update_20261028.md)
retains optimized64 as the baseline and makes 700 ms descriptive for future work.

## Question and fixed players

Does doubling total search from 64 to 128 simulations improve whole-game score
enough to justify its additional time, while retaining three root worlds?
This is an increased-compute strength comparison, not a claim of equal cost.

| Setting | Baseline policy | Proposed candidate |
| --- | --- | --- |
| Total main-search simulations | 64 | 128 |
| Root worlds | 3 | 3 |
| Network and weights | Existing Entity checkpoint | Identical |
| Numerical execution | Qualified optimized graph, float32, batch 1, one numerical thread | Identical |
| Other policy settings | Existing depth 64, PUCT settings, lawful seed A derivation, auxiliary choices and turn 124 heuristic | Identical |

Bind checkpoint SHA256
`ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb` and its explicitly
historical training origin. No fitting, tree reuse, cache change, new stopping
heuristic, world-count change or quota selection belongs in this comparison.
128 is chosen prospectively; existing 64/256 diagnostics do not qualify its entry
point. There is no fallback to another quota if its timing or result disappoints.

Keep the frozen optimized64 implementation and deployment profile unchanged.
Create separate research execution profiles for both policies with the same
validated response/evidence path and CPU allocation. Verify that the research
64 route exactly reproduces the frozen baseline's scientific outputs. The wider
research guard makes it a different execution identity, not a newly trained or
scientifically retuned baseline. Normalize only explicitly declared execution
metadata when checking equivalence; never normalize or rewrite raw clocks.

## Strength sample and analysis

Use **400 fresh main games on 200 independent standard initial setups**, each
played twice with swapped seats. Counterbalance candidate seat and game order.
Use ordinary complete games from initial deals, not selected midgame positions.
Six operational games on three other paired setups are excluded from the score.

Before the first model call, freeze one candidate, the full schedule, deterministic
seed namespace, collision checks against prior registered schedules and panels,
analysis, resource limits and failure rules. Record the exact scope of the seed
inventory check. Previous 400-game results and the 24 development positions are
motivation only, never fresh assessment evidence. Operational scores cannot select
the candidate or sample. No interim score stopping, discarded setup, replacement
game, failed-move retry or sample extension is permitted.

Give the candidate one point per win, half per draw and zero per loss. The pair
is the analysis unit. Report the score, paired 95% interval, each-seat score,
natural outcomes and all administrative forfeits. Use the established 50,000
pair-bootstrap resamples, PCG64 and linear percentiles, with a new fixed analysis
seed bound before execution and the existing degenerate-interval safeguard.
Independently reconstruct score arithmetic and verify the statistical result.

Retain the useful-gain criteria: **score at least 55%, paired 95% lower bound
above 50%, each-seat score at least 50%, and complete verification**. The sample
does not guarantee resolution of a small difference. A smaller supported effect
and an inconclusive result must be reported distinctly; neither changes the
criteria or automatically enlarges the study.

## Research clocks and failure handling

Use the existing trusted complete-response arrival clock, starting before parent
request preparation and dispatch and ending when the complete response arrives.
Apply symmetric **10.000-second external and 9.000-second internal guards**.
Keep the prescribed simulation counts even when time remains.

Correct completed responses above 700 ms, 900 ms or one second are played normally.
Those thresholds are descriptive: they do not control admission, stopping, losses,
selection or retries. Preserve every timing receipt and report distributions and
exceedance counts by arm, stage and seat, including unfinished requests separately.
Do not subtract instrumentation, transport or host delays, or reset an active clock.

A genuine attributable player 9/10-second deadline failure during the main sample
is an administrative research-guard loss; preserve it and continue the fixed
schedule. No replacement of that game or its paired counterpart is allowed.
Show full-sample sensitivity bounds assigning all administrative games to candidate
wins and losses. If such failures determine the conclusion, do not interpret the
score as a clean move-quality advantage. Source, identity, rules, evidence,
infrastructure and resource failures stop incomplete; they are not invented player
losses. Preserve the 2,000-ply safety ceiling and unresolved-deadlock handling
without inventing a natural result.

One-second deployment remains a separate objective. Research timings do not
establish a counterfactual one-second win rate: earlier aborted moves would change
the subsequent game. A later deployment comparison must use the same actual
one-second policy and verification standards for candidate and reference, with
700 ms still descriptive.

## Short qualification followed by the operational rehearsal

Future implementation must use additive versioned files and reuse qualified
correctness and evidence machinery where compatible. Do not introduce a new
timing observer or host sampler as a prerequisite for this question.

Before the six operational games, require model-free regression checks and
actual-entry-point qualification of both 64 and 128. Cover the corrected card
costs, all native card faces, noble-after-nonpurchase and final-round behavior,
gold/payment/return handling, private reserves, legal support, lawful information,
legacy-identity rejection, phase boundaries and the turn 123/124 behavior. Check
graph parity for every worker, exact baseline reproductions, direct same-quota
native parity for the new route, and exact deterministic repeats. Exercise both
arms on 18 fresh predetermined saved positions spanning phase and seat, plus
the required edge cases. Bind the finite check manifest before execution.

Use `tools/future_training_rules_20261021.py` and call `verify_runtime_sources()`
once at startup in every new evaluator and collector. Bind ruleset
`splendor_base_qualified_rules_20261018_v1`, catalogue SHA256
`db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312`, and turn contract
`atomic_prereveal_completed_turn_v1`. Follow the
[corrected-rules policy](future_training_rules_20261021.md) through the actual
encoder, hidden-world reconstruction, simulated transitions and referee. Outer
environment correctness or a manifest tag alone is insufficient. Retain the
documented atomic auxiliary-choice restriction and internal forced-pass convention.

The six fresh operational games test correctness, natural completion, evidence
integrity, cleanup and the study's resource forecast. **Neither 700 ms nor one
second is an admission gate.** A genuine research-guard failure prevents main
collection and is preserved without retry. If the correct run fits the resource
envelope, ordinary late responses do not postpone the strength question for a
timing-optimization campaign.

## Proposed resource envelope

These are prospective ceilings, not new execution authority or measured 128 costs.

| Resource | Proposed bound |
| --- | --- |
| Shared non-resettable wall clock | 24 hours from qualification through independent acceptance and cleanup |
| Qualification / six-game operational maxima | 1 hour each, within the shared clock |
| Absolute end of all collection | Hour 20, preserving up to 4 hours for final verification |
| New evidence total / final reserve | 16 GiB / 4 GiB; at most 12 GiB before finalization |
| Initial free disk / maintained reserve | 24 GiB / 5 GiB |
| Worker/helper memory | 2 GiB combined; at most two resident workers, one active decision |
| Available RAM before requests | 4 GiB, with 60 seconds initial and recovery stability |
| Cumulative RAM waiting | 600 seconds, charged to the same clock; pause only between decisions |
| Execution placement | Established CPU0 player/helper placement and CPUs2–19 supervisors, normal priority, actual topology checked |

Retain the existing startup memory guard and active supervision; the pre-request
RAM requirement must not create a new unmeasured in-move sampler. Startup graph
and bundled qualification requests retain their explicitly declared administrative
limits; ordinary handler subcases remain under the research guards.

Current planning is **roughly 8–12 hours overall**. The recent 64-versus-64
six-world study took about 6 hours 9 minutes including its original final checks.
The saved development panel measured median responses of 0.389 seconds at 64
and 1.423 seconds at 256. A simple interpolation suggests roughly 0.7–0.8 seconds
for 128 on comparable positions, but this is unmeasured and does not predict
tail latency. Different live positions, game lengths and evidence costs can
change all these estimates.

Before main collection, use all six operational games, without reference to
their outcomes, to forecast the 400-game time and bytes with a 1.5 contingency
factor. Include worker startup/closure, callback evidence, all ledger replays,
up to 48 selected model reproductions, source/runtime checks, compression,
sealing, independent statistical verification, file readback, acceptance receipts
and cleanup. Forecast the doubled callback workload, not just game logs.
Require the projections to fit the remaining absolute collection, final and
storage allowances. If they do not fit, stop preparation for main collection
and report the cost; do not quietly shrink the sample, remove checks or extend
the cap. A materially different plan would need a separate prospective decision.

## Verification and interpretation

Independently replay all 406 game ledgers and audit the complete compressed raw
transport, lawful observations, scientific responses, clocks, source/runtime
identities, placement and ownership evidence. Select up to 48 completed main
decisions by fixed hash and predeclared arm/seat/phase/hidden-reserve strata for
exact reproduction, never by timing, score or disagreement. Preserve failures
and do not replay them into successes. Reuse passed historical receipts only
with exact byte authentication and an explicit unchanged scope. Final acceptance
requires complete coverage, preservation and actual owned process exit checks.

A useful supported gain would justify considering practical work on 128; it
would not automatically replace optimized64. If the gain is absent, unresolved
or too small for the extra cost, retain optimized64 and close this fixed sample.
There is no automatic 256 experiment or training job. Any later reliability or
learning study should follow from the strength and cost evidence, with its own
concrete hypothesis and bounds.

External testing remains paused and larger-data work remains deferred. This
document creates no player, schedule, seed registry entry, execution freeze,
background process, game or model call.
