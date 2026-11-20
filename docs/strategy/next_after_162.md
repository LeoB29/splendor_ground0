# Recommendation after162: measure whether the gains generalize

2026-09-17. Recommendation and saved-artifact review only. No inference,
simulation, training, profiling, preparation/freeze or launch performed.
Preserve all frozen experiments and intentionally dirty work. Confidence:
moderate. Independent read-only reviews challenged the value of the prepared
panel against cheaper direct matches and a change to the learning mechanism.

## One recommended study

Run the prepared, bounded local-opponent benchmark. Compare all three161
anchors, all three162 refresh students and all three162 replay controls against
the same035, information-legal shallow-search and loop-safe023 opponents. This
tests whether the recent learning improves play beyond matches among closely
related students, and whether the cheaper old-data update carries similar value.

Use the existing54-game timing/integrity probe first, then the fixed5,400-game
panel if its operational cost is acceptable. It requires no new training or
teacher labeling. Freeze identities, scheduling and analysis before any probe
outcomes. Probe strength must not select models, opponents or sample size.
The implementation and commands are in
`../engineering/parallel_tools_after_162.md`; it has not been frozen or launched.

Before freezing, add **replay-minus-anchor** as a prospectively declared secondary
panel-score contrast, with a joint cluster interval, from the same games. Keep
refresh-minus-anchor primary and refresh-minus-replay secondary as already
specified. The extra contrast requires an analysis/test/documentation change and
new checks, not additional games. Existing checks_03 would no longer bind the
updated implementation. None of these contrasts is a direct replay-vs161 match.

Keep all three fits, both seats,100 shared initial-seed clusters and per-opponent
reports. Do not select the strongest fit.5,400 games are not5,400 independent
statistical units; uncertainty resamples the100 complete clusters and conditions
on these particular fits and opponents. This is calibration of direction and
effect size, not a promotion gate or a guarantee of detecting small differences.

## Why162 points here

The completed audited result is
`../../outputs/student_refresh_162/result.json`, SHA256
`93578415fa5eb3dcf32dc1d15fef4de27f3a36fde8c2b7bb2e5486b4d0bda11c`.

| Comparison | Score | Paired95% interval |
|---|---:|---:|
| Refresh versus equal-dose replay |50.4167%|48.6458--52.1875%|
| Refresh versus unchanged161 |53.0208%|50.9167--55.1250%|

The second comparison favors modest improvement, consistently in direction
across all three fits, but misses the predeclared55% practical threshold. The
first provides no clear fresh-feedback advantage and weakens the case for a
large benefit from this exact refresh. Neither equivalence nor uselessness of
all fresh-data strategies follows. No direct replay-versus161 game was played.

Saved fit curves supply no obvious case for simply continuing longer:

| Fit | Refresh selected update/loss | Replay selected update/loss |
|---|---:|---:|
|162001|448 /1.232285|448 /1.241604|
|162002|582 /1.230666|448 /1.235271|
|162003|358 /1.231899|314 /1.237317|

Every final checkpoint at update716 has worse validation loss than its selected
checkpoint. Refresh achieved lower common validation loss in every fit, yet did
not establish a gameplay advantage over replay. These reused validation results
are descriptive diagnostics, not a new causal test or a proof that the loss
function is the bottleneck. They discourage another arbitrary training extension
or a loss sweep based only on that metric.

Programme precedent makes broader measurement relevant:106's AlphaZero progress
within its own lineage did not establish improvement against035. This does not
prove the current students have the same problem; it shows why ancestor matches
alone are insufficient evidence for scaling a learning recipe.

## How the result would change decisions

- If replay improves over161 across this panel and fresh feedback shows little
  additional benefit with reasonably narrow intervals, prioritize the cheaper
  route for subsequent development. Similar point estimates alone do not prove
  equivalence or authorize promotion.
- If refresh has a credible broader advantage over replay, fresh feedback may
  have opponent-dependent value that the162 direct matchup did not establish.
  This would be new evidence, not a retrospective pass of162's gates.
- If neither update improves broadly, or an opponent-specific regression appears,
  investigate that pattern before spending on another refresh. Ancestor gains
  would then provide insufficient guidance for scaling.
- If intervals remain wide, record the uncertainty. Do not extend this panel to
  obtain a preferred result or let a diagnostic become an indefinite tournament.

## Alternatives and limits

A direct replay162-versus161 match is the closest cheap alternative. It closes a
real missing comparison, but remains within the same policy family and does not
show whether the gains generalize. The proposed panel tests both updates against
shared opponents without claiming transitivity.

Teacher batching remains worthwhile engineering. Useful teacher-budget scaling
in160 means lowering future query costs could help the programme. However, the
prototype has no representative speed or broad-equivalence evidence yet, and
making queries cheaper does not resolve which learning change to fund next.
It should not be mistaken for the next strength experiment.

Learning better proposals outside035's offered set is a major unresolved route
to stronger planning. The current student's chosen move is already in that set;
adding it back expands nothing. Testing an unrestricted learned proposer needs
its own support and equal-budget design.162 does not isolate proposal support,
architecture, target noise or value estimation as the dominant bottleneck.

The strongest objection to this recommendation is that the local panel is small
and partly related to the training programme. It cannot certify external/human
strength, low exploitability, a successful learning-to-stronger-search cycle or
equal-time superiority over the expensive planners. Easy or saturated opponents
may be insensitive to incremental gains. Report that limitation rather than
replacing opponents after observing results. Use one bounded panel to guide the
next mechanism-changing study, not to substitute evaluation for progress.

Historical results do not establish saturation: the051 resolver scored74.5%
against shallow and79% against023, while fast157 scored57.5833% against035.
Current161/162 scores against the broader panel remain unmeasured. In particular,
155's95.5% against035 cannot be assigned to its much cheaper students. The
shared strategic ancestry of the panel is a clearer limitation than an assumed
absence of remaining headroom.

No timing promise is based on the number of games alone.162's fast arena and
replays took tens of minutes, whereas teacher labeling dominated its7.5-hour
run; the shallow opponent and new adapters still need the fixed timing probe.
Monitoring remains off, and no future study is automatically authorized by this
recommendation.
