# Next after the completed ranking pilot

2026-09-27. Scientific recommendation only. Read AGENT_CURRENT first, the
completed ranking/iteration/diagnostic results, original134/137 evidence and
subsequent strategy/results. Three independent read-only reviews challenged
learning explanations, search allocation and experimental design. Some additional
arithmetic used saved artifacts; no model inference, fitting, simulation, replay,
tests, executable implementation, experiment freeze, launch or monitor ran.
Preserve all dirty work and original/recovery bindings.

## Decision

Run one bounded **breadth-versus-precision experiment** in the validated terminal
rollout planner. Retain the current useful four candidates; compare adding four
challengers with spending exactly the same additional continuation count on the
original candidates. Do not train another model in this experiment.

The three arms, when at least eight canonical actions are legal, are:

| Arm | Candidate actions | Selection worlds per action | Continuations per decision |
|---|---:|---:|---:|
| Current anchor | 4 | 128 | 512 |
| Broader candidate set | 8 | 128 | 1,024 |
| More precise evaluation | 4 | 256 | 1,024 |

The recommended additions are **two next-highest anchor-ranked remaining
canonical groups, then two uniformly sampled remaining groups**, without
replacement. Keep the original four, including the successful learned fourth.
This is one fixed expansion recipe, not a grid over counts or proposal mixtures.
Confidence that this is the best next allocation is moderate; its success is
uncertain.

## What the latest result changes

The ranking pilot completed360 fresh positions, all three fits and every
prescribed audit. Its primary conditional score gain was +0.0434 percentage
points, paired95% interval[-0.3993,+0.5306]. Against matched CE the gain was
+0.0239pp, interval[-0.2463,+0.2962]. All practical/superiority gates failed.
The first interval still includes the +0.5pp practical threshold: this is a
failed investment gate, not proof that every useful small gain is absent.

The failure was not simply an unchanged model. Ranking changed173/354 eligible
fourth proposals and66/360 final moves. Its fourth was selected89/354 times,
versus88/354 for the anchor. More changed decisions and similar use of the fourth
proposal did not produce established improvement.

Saved development validation also weakens a gross optimization/forgetting story:

| Fit | Initial pair-ranking loss | Selected ranking update | Matched CE selection |
|---|---:|---:|---:|
|161001|0.635420|0.612541|0.624007|
|161002|0.635512|0.612353|0.623979|
|161003|0.635941|0.613661|0.624623|

Ranking improves this criterion by about3.5–3.6% on the common validation panel,
including both old and newer source partitions. Two fits select the last update;
the third selects1118/1192 with slight subsequent deterioration. These are
reused development measurements, not evidence that further fitting improves
play. They neither diagnose broad overfitting nor rule out capacity limits.

A bounded post-result read of the saved roots adds a useful qualification. The
fixed ranking fourth, chosen BEFORE selection-world outcomes, has a mean score
about+1.1608pp above the fixed anchor fourth across354 eligible states; the CE
fourth is+0.7161pp above anchor. Ranking's point estimate is positive in all three
fits. These are descriptive fixed-action means from the128 selection worlds,
not selected maxima, independent chosen-move effects, confidence intervals or
new confirmation.724 linked hashes were checked; no new predictions or rollouts
were generated. This pattern makes complementarity and selection noise credible
rivals: a suggestion can look better on average without improving the choice
made alongside the retained first three. Do not claim the network learned no
better proposals, or infer that sampling noise uniquely caused the flat endpoint.

Close this exact loss/data/dose recipe as its prospective plan requires. Do not
extend its360 positions, rescue one fit/phase, run its unearned full-game arena,
or sweep nearby losses to seek a passing result.

## How the programme evidence ranks the remaining hypotheses

**Candidate information is the most actionable unresolved hypothesis.** The
current training data compare only a few offered actions at each state. Training
can rearrange those observed comparisons while still extrapolating poorly to
other legal actions or failing to propose a complementary fourth. Improving
the loss on the same comparisons did not solve that problem. This raises the
priority of measuring useful action coverage; it does not establish coverage
as the cause of the null result.

There is positive precedent: learned128 beat original128 with65.796% score,
paired95% interval[60.945%,70.522%], while retaining four candidates and128 worlds.
One learned proposal can improve search, and one search-learning-search cycle
has been demonstrated. The subsequent CE iteration scored51.493%
[47.264%,55.721%] against its anchor; neither that result nor this ranking pilot
established another improvement. Unlimited repetition of those updates is
therefore poorly supported.

**Sampling precision is the strongest competing practical investment.** The
original128-world planner beat its32-world version with65.5%
[60.875%,70.125%]. That does not establish the128-to256 slope, but makes extra
precision a serious comparator. The saved halving screen's positive+0.1088pp
conditional development gain also remains useful efficiency evidence; it is
not a fresh128-world or gameplay result.

**Continuation/evaluator bias remains the strongest limitation of this screen.**
Fresh worlds eliminate reuse of selection outcomes as assessment, not the bias
of imagining greedy035 continuations. Earlier134 was a decisive31.875% gameplay
loss despite favorable conditional diagnostics; its world law, continuation
controller and expanded operator differed from the current planner. Later139
also showed that fresh-world conditional gains alone do not certify gameplay.
These findings argue for a bounded screen and subsequent full-game confirmation,
not a universal ban on expanding candidate sets.137 constrained its particular
root-refill stratification recipe, not all chance search.

Stronger continuations remain conceivable, but140/158/159 do not supply an
established replacement for the present evaluation policy;158 remains only
provisionally audited after the owner stopped its remaining replay. The value,
representation and target variants142–153 constrain their individual recipes.
They do not rule out stronger value learning, larger models or another search
architecture, but presently give less localized interventions than this test.

**Distribution and independent opposition remain open.** The fast161 source
population is not the full distribution reached by stronger replanning agents.
The earlier refreshed fast students did not establish broader progress in the
related local panel. Neither the latest null nor a future positive conditional
screen establishes broad strength, optimal play or human superiority. A broader
benchmark is important before those claims; it does not by itself identify the
next useful learning change.

## Why this exact expansion

Keep the original035 top two, the original first random alternative and the
unchanged161 learned fourth, with their original representatives and tie order.
Use the corresponding frozen161 fit, not either newly trained ranking or CE
model. Obtain the next two canonical groups from the same full-legal anchor
ranking already needed for its fourth proposal. Then select up to two additional
distinct groups uniformly from the remaining lawful groups using a separately
seeded, identity-based random order. Append additions after the incumbents.

The mixture gives the successful learned ranking additional opportunities while
retaining exploration outside its preferences. A four-uniform-addition recipe
would test a cleaner pure-exploration intervention, but could spend this bounded
opportunity on poor challengers. The immediate question is whether a useful
fixed broader teacher is worth funding, not which addition type caused a gain.
A positive result must be attributed to the entire frozen expansion package.

The diagnostic's random fourth replacement lost about0.997pp to the anchor;
removing its fourth altogether lost about1.564pp. Neither comparison tested
adding candidates while retaining that fourth. Its larger-union estimate was
+0.309pp[-0.0175,+0.6860], uncertain and confounded with extra compute. That is a
reason to test the present contrast with a precision control, not a prior proof.

Prefer this design to4×128 versus8×64. Halving per-candidate sampling while
doubling candidates creates a plausible selection-noise cost. A negative result
would leave useful additional actions confounded with diluted precision. The
proposed three arms preserve128 worlds per added candidate and compare width
against a real equal-budget alternative. No fourth arm is necessary.

## Fresh fixed decision test

Use360 entirely new source-game-disjoint positions:120 per fixed161 source fit,
20 per fit×acting-seat×phase cell. Retain phases0–19,20–39,40+, private reserves,
short supports and unchanged decisions. Generate a fixed1,080-game source pool,
prospectively assigning60 games per cell; identity-select20 eligible games and
one eligible position per game. Use a separate54-game probe pool to select one
root per cell,18 total. Freeze all seeds, identities, eligibility, sampling and
shortage rules before new outcomes. A shortage is a hold, not permission to
extend the pool. Do not reuse the inspected ranking or diagnostic positions as
fresh confirmation, or select only disagreements.

Let G be the lawful canonical action count, m=min(G,4), n=min(G,8). The anchor
evaluates m candidates on128 worlds. Breadth evaluates n on128 worlds. Precision
evaluates the original m on **128*n/m** worlds. This is always integral: when
G<4, n=m; otherwise m=4. For G=5/6/7/8+, precision uses160/192/224/256 worlds.
At G≤4 all arms coincide and contribute zero differences while remaining in
the panel. Thus breadth and precision consume exactly128*n continuations per
root. Equal branch counts do not imply equal wall time; report both.

Use the same128 selection worlds for all breadth candidates and the anchor.
Precision also sees its required additional worlds for the original candidates.
Preserve the mature observer-legal world law, greedy035 continuations, terminal
WDL score, canonical representatives and first-maximum ties. New independent
world namespaces must cover the largest256 selection block without overlapping
the assessment block. Do not change hidden-information handling or game rules.

Persist all three choices before assessing only their distinct selected actions
(at most three) on128 disjoint fresh worlds. Shared assessment worlds provide
paired contrasts. Do not assess only changed roots or choose a new winner from
the assessment outcomes. Save full selection outcomes for provenance; they do
not become an automatically authorized training corpus or label oracle.

Primary scientific comparison: breadth minus equal-budget precision. Required
practical comparison: breadth minus the current anchor. Report precision minus
anchor as a prospectively declared secondary comparison. Average all360 source
games equally, retain all fits, and use50,000 paired bootstrap resamples within
the18 fixed cells. Intervals condition on these three fits and source population;
worlds and model fits are not additional independent games.

Recommend the following investment gate, fixed before collection:

- Breadth-minus-anchor point gain at least+0.5pp AND95% lower endpoint above0.
- Breadth-minus-precision95% lower endpoint above0.
- All integrity and operational conditions pass.

This is a conservative compute-investment gate, not a power guarantee or a
conversion from conditional score to55% arena strength. If breadth beats the
anchor but not precision, useful extra computation may be indicated while
breadth's advantage over that alternative remains unestablished. If precision
looks stronger, record that evidence for the next decision without automatically
promoting it or converting it into the primary winner after the fact.

A breadth pass supports a separately specified fresh full-game comparison. It
does not itself prove useful distillation or authorize training, promotion,
another iteration or a multi-day arena. A null closes this specific expansion
recipe; do not expand the panel, change the mixture, filter phases or sweep
candidate counts. It does not prove that no useful legal actions remain.

## Cost, verification and practical ceiling

Shared selection computation costs at most4×256+4×128=1,536 continuations per
root. At most384 assessment continuations give1,920/root: **691,200 main** and
**34,560 probe** continuations before verification. No training simulations or
fits are required.

The completed ranking screen recorded265,856 main continuations in4,710 seconds
with eight workers, followed by698 seconds of sampled verification. Scaling that
collection average to the new ceiling gives about3.4 hours for main collection
alone. Different added actions can have different trajectory lengths. Budget
sources, probe, sampled audit and finalization separately. A provisional4–6 hour
execution expectation is reasonable; it is not a measured ETA for this design.
Propose an8h execution ceiling,5GiB disk reserve and an outcome-blind18-root
runtime/storage gate before main collection. Preparation/preflight is additional.

Follow the accepted verification policy. Check every source move, model input,
proposal, realized per-arm budget, world law, selected move and outcome aggregate.
Independently reconstruct statistics and all artifact links. For mature rollout
paths, use prospectively selected whole-world replay across every root, covering
both the common128 block and the precision-only extension, plus fixed full stress
roots. Fully replay the bounded probe. Freeze exact coverage and cost accounting
before launch, with hidden workers and durable best-effort status reporting;
scientific artifact and integrity failures remain fatal. No routine full replay
of all mature continuations or historical training is proposed.

## Relation to the eventual superhuman goal

This experiment asks whether we can improve the planner's information before
asking another learner to compress it. A useful teacher with broader comparisons
could provide a more promising next learning dataset. That second transfer is
still an empirical question; greater breadth alone is not a self-improvement
guarantee. The programme ultimately needs repeated useful learning, affordable
planning, independent strong opponents and declared-resource human evaluation.

External research provides context, not a diagnosis of this code: Sampled
MuZero explicitly studies policy improvement over sampled action subsets
([Hubert et al., ICML2021](https://proceedings.mlr.press/v139/hubert21a.html)).
Gumbel planning also addresses incomplete root exploration, with improvement
arguments requiring correctly evaluated action values
([Danihelka et al., ICLR2022](https://openreview.net/forum?id=bERaNdoegnO)).
Neither transfers a guarantee to this noisy, imperfect-information rollout
planner, and neither justifies replacing its architecture before this test.

## Main local evidence

- outputs/proposal_ranking_128/{report.md,result.json,final_audit.json,final_status.json,
  fits/,selection.json,main_collection_lock.json,main_audits_lock.json}
- outputs/proposal_ranking_128/recovery_01/closure.json
- outputs/proposal_iteration_128/report.md; outputs/proposal_diagnostic_128/report.md
- outputs/learned_proposal_128/report.md
- docs/engineering/saved_halving_screen_result.md; verification_policy.md
- docs/strategy/next_after_proposal_diagnostic_128.md; next_after_learned128.md
- docs/strategy/next_after_161.md; next_after_162.md; next_after_generalization_panel.md
- docs/strategy/diagnosis_after_158.md; next_after_159.md
- docs/strategy/independent_superhuman_review_after_134_137.md
- outputs/expanded_root_resolver_fresh_champion_comparison_134/result.json and its
  owner runtime amendment; outputs/alphazero_stratified_root_refill_search_137/result.json

Recommendation only. No experiment has been prepared or launched by this review.
