# Roadmap for strength testing before extended timing work

28 October 2026. **Keep three-world optimized64 as the baseline. Recommend one
128-versus-64 simulation strength experiment. Nothing is launched by this update.**
The next study is described in the [prepared comparison plan](../engineering/splendorust_128_vs_64_strength_plan_20261028.md).

This additive revision supersedes the prospective research order and timing
admission recommendations in the [27 October roadmap](splendorust_roadmap_update_20261027.md)
and the [timing protocol review](../engineering/splendorust_timing_protocol_review_20261028.md).
It preserves their historical evidence and every frozen attempt. The wider
[24 October programme](splendorust_revised_roadmap_2026-10-24.md) remains the basis
for the long-term objective and corrected-rules requirements.

## What the evidence now supports

Optimized64 remains our practical development baseline. Its earlier comparison
against unchanged32 scored **64.25%**, with a paired 95% interval of **58.25–70.25%**,
including both deadline-related losses. That establishes a useful gain in that
matchup, not a human ranking or a universal advantage. Subsequent recovered
reliability verification covered 200 games with no observed timeout; its maximum
optimized64 response was 854 ms, and 700 ms was descriptive in those games.
[Strength result](../engineering/splendorust_graph64_strength_20261025_results.md)
and [reliability result](../engineering/splendorust_reliability_finalization_20261026_results.md).

The completed six-world comparison scored **50.5%** over 400 games: 201 wins,
two draws and 197 losses, with a paired 95% interval of **46–55%**. All games
ended naturally. Six worlds exceeded one second once and three worlds three
times, out of 11,123 decisions each. These counts do not establish a reliability
ordering. Six worlds has no demonstrated strength gain; modest differences remain
unresolved. The first independent audit timed out, while the separately preserved
recovery passed. [Accepted result and recovery record](../../outputs/splendorust_pool6_strengthfirst_20261028_continuity/assessment_accepted_separate_review_recovery.json).

The earlier sampling-stability improvement was useful development evidence, but
it did not establish that the different decisions won more games. Repeated work
on a strict 700 ms margin delayed that essential test. The new order below
corrects that prioritization rather than rewriting any failed admission.

## Research order and timing policy

1. **Establish correctness and basic feasibility.** Use the actual entry points,
   corrected rules, lawful information, exact-output checks and a small operational
   rehearsal. Check that the intended study fits its time, memory and storage
   budget. Do not start an extended timing-optimization campaign here.
2. **Test strength and record timing together.** Choose one candidate before new
   results, use fresh paired whole games, and measure complete response time for
   both players. Saved-position stability or internal search values cannot
   substitute for a strength result.
3. **Invest in practical execution when the benefit warrants it.** A promising
   strength result can justify targeted speed or reliability work, followed by
   separate confirmation under the intended deployment clock. If no useful gain
   is demonstrated, retain the baseline without automatically enlarging the sample.

**700 ms is descriptive for future work.** It must not determine admission,
stopping, losses, candidate selection, discarded positions or retries. This rule
applies to both candidate and reference. The practical objective remains one
second per complete move; it is assessed separately from research strength.

For the next proposed strength study, accept correct completed moves over one
second under symmetric 9-second internal and 10-second response-arrival guards.
Report timing honestly, including all overruns. The guards protect against stuck
execution; they do not permit extra simulations. This is a proposed research
execution policy, not a change to the frozen optimized64 deployment identity.

## Recommended next strength experiment

Compare **128 versus 64 total simulations, with three root worlds in both arms**.
Keep the exact network, weights, precision, encoder, search depth, PUCT settings,
seed derivation and auxiliary decisions fixed. This deliberately increases search
work; it is not a fixed-compute allocation experiment.

The previous 32-to-64 gain supplies a concrete reason to test one further doubling.
128 is the nearest such step to the one-second objective. There is no measured
128-simulation timing or strength result yet. Saved 256-simulation searches had
a median response of 1.423 seconds and changed four of 24 development decisions;
neither those changes nor same-network values establish stronger play.
[Saved-decision evidence](../engineering/splendorust_decision_quality_20261027_attempt02_results.md).

Choose 128 now, before fresh timing or outcomes. Do not select a quota by whether
it passes 700 ms, switch to 256 after an inconvenient result, or conduct a parameter
sweep. A direct 256 comparison would provide a larger research contrast but is
less directly aligned with one-second play. Retuning PUCT lacks a specific supported
setting at present, and another world-count sweep has weaker motivation after the
completed six-world result. The previous general suggestion to investigate search
allocation is therefore narrowed to this simpler first strength question.

The proposed assessment is **400 fresh games on 200 independent starting setups,
with swapped seats**, preceded by six disjoint operational games. Retain the
existing useful-gain criteria: at least 55% score, paired 95% lower bound above
50%, at least 50% from each seat, and complete verification. Four hundred games
can still leave a small gain unresolved. There is no extension or automatic
promotion. The proposed envelope is 24 hours and 16 GiB of new evidence; the
current 8–12-hour planning estimate must be checked using the operational costs.

## Decisions after the experiment

A useful supported strength gain would justify considering a 128-simulation
successor and, where necessary, targeted work toward one-second execution.
Research success alone does not replace optimized64 or qualify deployment.
A smaller, uncertain or adverse result leaves optimized64 as the reference;
report the interval and close the fixed sample before choosing another question.
An infrastructure or correctness failure is incomplete evidence, not a playing
strength verdict. Preserve every failure and distinguish research-guard forfeits
from natural outcomes.

Six-world promotion and further timing work on that branch are parked. Preserve
unchanged32 and the older research players as historical controls. External
opponent testing remains paused. Larger-data collection, fitting, self-play
training and network enlargement remain deferred until storage and expected
benefit justify a separately approved study.

## Wider objective and current status

The objective remains exceptional two-player base Splendor with lawful information
and practical response times. We moved to a strong pretrained policy/value search
foundation, established a useful 64-simulation reference, and have now tested one
hidden-world hypothesis in full games. The next experiment asks whether another
small search-budget increase produces a benefit worth engineering for.

Parent comparisons establish local progress. Diverse strong opponents and eventual
expert-human comparisons remain necessary for broader strength claims. A successful
corrected-rules learning cycle may later improve the network at the same inference
cost, but neither extra training nor more search is assumed to guarantee improvement.

This update changes documentation and future recommendations only. No player,
model, frozen protocol, historical success criterion or evidence is modified.
No new seeds, implementation, freeze, model calls, games or training job are created.
