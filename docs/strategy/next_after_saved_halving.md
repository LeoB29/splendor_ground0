# Next after the sampling screen: learned root proposals

2026-09-17. Recommendation only. No implementation, inference, new analysis,
simulation, training or launch. All frozen files and dirty work remain intact.

## Recommendation

Run one bounded, prospective **learned fourth-proposal replacement screen**.
Use the three frozen161128-world students to suggest a candidate move outside
three common baseline candidates. Ask whether the unchanged32-world planner
then chooses a better move. No new model training is needed.

Confidence is moderate. An independent read-only challenge agrees that this is
the highest-value next scientific experiment after the small halving result.
It tests a missing route from learning into stronger planning. The successful
restricted students do not yet establish that their all-legal rankings can
serve as good proposers.

## Why this now

The completed halving screen improved conditional score by +0.108805 percentage
points,95%[+0.065072,+0.153532], using the same continuation count. Preserve that
positive result as a candidate efficiency improvement. Its small size and reused
source evidence do not make an expensive full-game confirmation the automatic
next priority. Small root gains can compound, but their whole-game effect is
unknown; do not translate this interval into an arena power calculation.

157/161 showed useful learned action ranking.158 tested a different role, learned
continuations, and did not establish stronger planning.162 and the broader panel
did not establish a productive additional refresh. Proposal support remains
unisolated: current restricted student choices already belong to the planner's
offered four, so adding that same choice back does nothing.

The all-legal scorer already exists. `fast_selector_157.choose` computes logits
for the complete lawful legal-action set, then takes a group maximum only for
offered groups. Extending ranking to unoffered groups requires a new adapter and
verification, not a new fit. Do not modify its frozen implementation.

## Fixed intervention

At each root construct the exact original155 offered list in its original order.
For four-candidate roots keep its first three entries common:035's top two
distinct canonical groups and the first uniformly sampled alternative. Compare:

| Arm | Fourth candidate |
|---|---|
| Reference | Original second uniformly sampled alternative |
| Raw-ranking control | Highest035-ranked canonical group outside the common three |
| Learned fit1 | Highest161-fit1-scored canonical group outside the common three |
| Learned fit2 | Highest161-fit2-scored canonical group outside the common three |
| Learned fit3 | Highest161-fit3-scored canonical group outside the common three |

Use the established maximum-over-concrete-actions group score for the learned
models. Preserve the existing035 canonical group representatives for every arm,
including newly offered groups. Otherwise representative changes would be an
extra intervention. Freeze original-order tie rules and lawful input invariance.
All arms keep four distinct groups when available. For support below four, keep
the unchanged original support in every arm. Retain all proposal and final-choice
coincidences; never enrich the sample by discarding unchanged roots.

Every arm selects with exactly32 common sampled worlds per offered action,
terminal WDL and unchanged greedy035 continuations for both players. Do not add
halving, batching, a new continuation policy or a larger candidate count here.
The extra learned inference cost must be reported; equal continuation counts
are not automatically equal wall time.

## Proposed bounded design

- 288 main positions from288 distinct fresh greedy035 source games,48 in each
  phase-by-acting-seat cell. Use154's outcome-blind position/matching method;
  whole selected source games remain the independent units. Freeze fresh seed
  namespaces and a1,024-source-game ceiling; incomplete quotas are an operational
  failure, not permission for outcome-selected replacement or an extension.
- Evaluate all three frozen161 fits at every selected root. This gives a common
  state population and paired controls; average all three as the learned arm.
  Never select a favorable fit. This population tests conditional proposal value
  on fresh035 positions, not every planner/student-induced state distribution.
- Use32 selection worlds per offered candidate, then seal all choices before
  generating any evaluation evidence. Use128 independent evaluation worlds per
  root, shared across the unique locked choices. No selection/evaluation reuse.
- Primary: average learned-arm minus reference held-out terminal score. Required
  attribution contrast: average learned-arm minus raw-ranking control. Compute
  50,000 paired source-game bootstrap samples within the six balanced cells,
  keeping all fits/arms together. Report score as win1/draw0.5/loss0.
- Descriptive outputs: each fit, phase, seat, legal/canonical support, proposal
  overlap, choice overlap, proposed-group035 rank, actual proposal latency,
  continuation counts and total resource use. None selects a subgroup or retry.
- An independent verifier should reconstruct every source game, legal/group
  input, proposal ranking and final choice, and fully replay every collected
  selection/evaluation continuation as in154. Independently rebuild statistics.
  This is a new prospective study, not an extension of160 or the halving screen.

These details are a recommendation, not a frozen execution protocol. New seed
values, adapters, artifact inventories and checked commands must be fixed during
authorized implementation, before any probe or main strength evidence.

## Cost and interpretation

Share computation across identical candidate groups. There are at most eight
groups in the union: three common plus five fourth-slot proposals. Evaluate only
unique locked winners, at most five. Thus the maximum is
32 x8 +128 x5 =896 new continuations/root, or258,048 for288 main roots, before
independent replay. This sharing reduces research cost; each deployed conceptual
arm still receives only its own four-by32 selection budget.

Use an18-position timing/integrity probe, balanced across the six cells and
separate from main, with all arms. Inspect timing and integrity only; probe
strength cannot tune rules or sample size. Eight hidden CPU workers, one Torch
thread each, preserve existing windowless launch/replay practice. No GPU benefit
is assumed.

For scale,154 collected and fully replayed247,136 continuations in2h44m including
its source work. The proposed main worst-case count is similar; different states,
overlap and trajectory lengths prevent a firm forecast. A provisional3--5hours
for execution is reasonable pending the probe; implementation/checks are extra.
No production batching dependency is introduced merely to make this estimate.

If both the primary and raw-ranking contrasts have intervals above zero, the
study supports a learned contribution to conditional search improvement and
justifies considering fresh full-game confirmation. Both effect sizes matter.
Beating the random slot alone cannot distinguish learned knowledge from simply
offering another035 favorite. A primary gain with an inconclusive attribution
contrast leaves that mechanism unestablished.

A negative/inconclusive result sets aside this exact unrestricted-max proposer;
it does not rule out learned proposals or learning generally. The strongest
risk is that taking the maximum over all legal groups exploits poorly trained
scores, because training compared only restricted offered groups. Do not rescue
the result with a score threshold, different fit, phase filter or extra rounds.

A positive one-move test still does not establish stronger full-game play or a
reliable multi-generation learning loop. The eventual claim needs fresh games,
broader independent opposition and declared computation limits.

Teacher batching remains a useful separate engineering task; its speedup is
unmeasured. Retained old teacher-corpus replay remains a plausible learning
alternative, with forgetting unproved. Neither should be bundled into this
proposal test. No experiment is launched by this recommendation.
