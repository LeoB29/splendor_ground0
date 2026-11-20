# Saved-world planner diagnostic results

18 October 2026. Completed at 13:01 Paris after explicit owner authorization.
The guarded execution took 9 minutes 37 seconds. All 56 synthetic checks passed;
independent arithmetic and the final source-integrity check passed.

Both frozen rankers change the planner's decisions, but their estimated gains
are small and uncertain in this diagnostic. This gives limited additional
support for funding a costly arena of these unchanged models. It does not
establish that a 128-world planner or whole-game benefit is absent.

## What was measured

At all 720 original positions, reconstruct the common first three proposals
plus each ranker's sealed fourth proposal, compared with the anchor's fourth.
Select a final action on saved selection worlds 0–63 and assess it on 64–127;
reverse the halves and average both paired differences within the position.
Retain all 720 distinct source games and all 18 balanced fit/seat/phase cells,
including the 13 empty and 15 singleton outsider sets. Unknown outcomes: zero.

These are expected conditional score differences, with win 1, draw 0.5 and
loss 0. They are not whole-game win-rate changes. This is a **post hoc development
analysis of a 64-world selector on an already-opened panel**. Its intervals do
not account for the adaptive decision to perform this analysis. No confirmatory
success criterion, model-selection rule or new +0.5-point gate was introduced.

| Model versus anchor | Mean gain, percentage points | Descriptive 97.5% interval |
| --- | ---: | ---: |
| Ridge | +0.0939 | [-0.2843, +0.4763] |
| Histogram boosting | +0.1481 | [-0.2045, +0.4812] |

The two intervals use the specified joint stratified root bootstrap: 50,000
replicates, seed 2813000000, fixed 1,000-replicate chunks and linear percentile
interpolation. Both directions stay together within each resampled root. Exact
mean numerators are 173 and 273, each divided by 184,320 before conversion to
percentage points. Neither the difference between model means nor the current
intervals establish boosting's superiority over ridge.

## How proposals affect final choices

| Model | Fourth proposal differs from anchor | Final action differs from anchor | Own fourth proposal selected |
| --- | ---: | ---: | ---: |
| Ridge | 449/720 = 62.36% | 22.85% | 23.26% |
| Histogram boosting | 300/720 = 41.67% | 14.38% | 24.31% |
| Anchor | — | — | 21.11% |

Final-choice and fourth-selection rates average the two direction indicators
within each root, then average all roots. The 1,440 directional choices are not
1,440 independent observations. Many changed proposals do not alter the final
action; frequent final changes still do not establish better actions.

The earlier restricted-ranking estimates (+0.4340 and +0.5143 points) and this
diagnostic have different endpoints. Do not treat their difference as a formal
measured loss of benefit or compare these gains directly with the structured
study's 128-world planner contrasts. Original independent assessment records
were not used because they lack complete coverage for newly selected actions.

## Verification and preserved evidence

The execution lock bound the specification, implementation, tests, runtime and
authenticated source references before computing the new metrics. The original
model and prediction seals, completion chain, root/audit links and source
dependencies were validated. All **26,292 input files** were rehashed unchanged
on completion.

A separate implementation reproduced every directional choice and integer
score difference at all 720 roots. Its win/loss tallies and bootstrap
multiplicity matrices reproduced the means, usage counts, intervals and full
integer-bootstrap digest exactly. This reused the completed studies' original
trajectory audits; no new trajectory replay, model inference, fitting or
simulation occurred. Both original studies retain their frozen classifications.

Execution used one compute thread at below-normal priority. Sampled worker
memory peaked at approximately 76 MiB, within the 1 GiB ceiling. The supervisor
exited successfully and released its exclusive lease. No arena or successor
experiment was launched.

- [Frozen execution specification](ranker_planner_bridge_128.md)
- [Result](../../outputs/ranker_planner_bridge_128/execution_20261018_01/result.json)
- [Independent arithmetic check](../../outputs/ranker_planner_bridge_128/execution_20261018_01/independent_check.json)
- [Completion and source recheck receipt](../../outputs/ranker_planner_bridge_128/execution_20261018_01/completion.json)
- [Execution lock](../../outputs/ranker_planner_bridge_128/execution_20261018_01/execution_lock.json)
- [Synthetic preparation checks](../../outputs/ranker_planner_bridge_128/preparation_20261018_01/checks.json)

Result SHA256:
`d3cbb6b3c192990f85d837c42f193ce99bb7a4af7aa7bec9001792b2db5d4d18`.

## Roadmap consequence

My recommendation is to lower the priority of an expensive arena for these
unchanged rankers. This is an investment judgment using effect size, uncertainty
and cost, not a retrospective significance gate or a proof of no useful gain.

The next substantial learning proposal should test a materially larger number
of independent positions through a prespecified learning curve, with collection
cost priced before commitment and fresh planner/gameplay confirmation retained.
Data scarcity remains a hypothesis, not an identified cause. The accepted faster
backend can inform a compatible future design; no new speed campaign is needed.
If an affordable increase is only modest, describe that limitation honestly.

Alongside that preparation, specify a thinking-time/hardware budget and prepare
external calibration of the strongest verified player against independent bots
or strong humans. The programme has one demonstrated search–learning–search
cycle; reliable repeated gains and broad playing strength remain unfinished.
This bounded diagnostic has answered its intended development question and
does not authorize another nearby reanalysis automatically.
