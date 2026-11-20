# Recommendation after161: refresh the teacher feedback where the student plays

2026-09-17. Recommendation and saved-artifact review only, following the owner's
question about the next step. Read AGENT_CURRENT first and reviewed161, its
learning curves,160,157,158/159 and the earlier learning failures. Two independent
read-only reviews informed the decision; they differed on the value of a cheaper
learning-objective test. No successor implementation, inference, fitting,
simulation, gameplay, promotion or monitoring was launched. Preserve all frozen
predecessors and intentionally dirty work. Confidence is moderate.

## Decision

Recommend one bounded student-state teacher refresh, provisionally162:

**Let the improved161 fast players produce fresh games. Ask the unchanged strong
128-world planner to judge a fixed sample of positions they reach. Compare
learning from old plus fresh examples with an equal training dose on old examples
only, and measure whether the updated player also improves on unchanged161.**

This is a test of continuing useful learning with refreshed experience. Keep the
teacher, restricted proposals, architecture and successful classification recipe
fixed. It is not another change to the players inside simulated futures.

## What161 changes in our assessment

All prescribed checks passed:400 source exports and audits, six complete fits
and bitwise fit/checkpoint-selection replays,48 pilot games/replays and4,800
fresh main games/full replays. The original bookkeeping failure and successful
recovery remain preserved. The result is in
`outputs/teacher_budget_161/result.json`, SHA256
`813596b6228a2e3530a8b73cd577b557bf211cfd76ed433411ae2313074b2113`.

| Comparison | Score | Paired95% interval |
| --- | ---: | ---: |
|128-label update against matched32-label update|52.5417%|50.1875-54.9167%|
|128-label update against unchanged157|54.8958%|52.3333-57.4167%|

The formal classification remains `clear_success_not_established`: progress
missed the frozen55% point threshold by0.1042 percentage points. Nevertheless,
the evidence favors improvement, with positive point estimates for all three
fixed fits in both comparisons. Do not round the gate into a pass, call this
absence of learning, multiply successive win rates, or repeat the arena merely
to cross55%. The intervals condition on the three fixed fits and resample400
whole starting-seed clusters, retaining both seats and comparisons.

Together,160 and161 establish useful teacher-budget scaling and some transfer
into fast play. They do not establish repeated self-improvement, a stronger
learned value function, general superiority to other opponents, or superhuman
strength.158/159 still do not establish that the learned player improves search
when used in its simulated continuations. The old035 proposer also still limits
which moves the student can select.

### The saved curves weaken the case for just adding passes

The128 fits selected updates563/488/563 out of600. Pair-mean validation CE was:

| Fit | Initial | Selected | Final |
| --- | ---: | ---: | ---: |
|161001|1.218028|1.205828|1.210310|
|161002|1.219095|1.205440|1.205666|
|161003|1.218083|1.206423|1.207436|

Averaged across the three fits, successive complete-pass CE changes were
-0.006636,-0.002409,-0.002157,+0.000603. This is a noisy plateau, not an obvious
improving endpoint. No training-loss curves were saved, so it does not establish
overfitting or distinguish noise, capacity and optimization limits. More training
remains testable; late checkpoint selection alone is not strong evidence for it.

On the reused validation rows, the32-trained models actually matched128 teacher
labels more often than their128-trained counterparts in every fit:45.02/44.35/
44.68% versus44.44/44.23/44.56%. Yet they lost the fresh direct matchup. This warns
against treating agreement as playing strength. It does not identify hard labels
as the cause of the remaining weakness or prove that a different loss will help.
These are read-only development diagnostics from saved fit result and validation
JSON files under `outputs/teacher_budget_161/fits`, not new confirmation tests.

## Proposed bounded experiment

1. Freeze all three selected161128-label students as source players and paired
   training initializers. Generate a prospectively fixed set of fresh student
   self-play games, representing every student and both acting seats. Target
   approximately2,400 teacher queries in total, with phase coverage and a fixed
   source-game cap. Choose positions by identities before teacher outcomes;
   retain unchanged choices, small supports, wins and losses. Freeze the exact
   sampling/count/shortage rules before collection. Never keep generating until
   favorable labels or outcomes appear.
2. At each sampled position, query exact160's128-world teacher over the actual
   recorded035 top-two-plus-two-random canonical proposals. Preserve original
   representatives, first-maximum ties, lawful world sampling, terminal WDL and
   greedy035 continuations for both simulated actors. The source student's move
   is not itself a teacher label. Save the complete paired outcomes and provenance.
3. Keep entire new source games or coupled pairs together in an identity-only
   train/validation split. Pool source data prospectively across the three fixed
   students. Preserve161's old source split; do not train on old or new validation
   rows. The old validation panel has already been used for development.
4. Make two updates from each frozen161 student: old128-label rows plus new
   teacher rows, versus old128-label rows only. Use uniform rows in the combined
   treatment corpus; no confidence weighting or tuned replay ratio. Match total
   optimizer updates, batch sizes and row exposures between arms by supplying
   old rows to the control in place of the additional new rows. Match all other
   settings, including fresh optimizer state and architecture. Freeze the dose
   before outcomes rather than comparing four passes over unequal-size datasets.
5. Retain offered-group hard-label cross-entropy, frozen035 proposals at
   deployment, and no value-head training. Use one common prespecified validation
   panel and pair/game-mean CE selection rule for both arms, with the unchanged
   initial checkpoint eligible and earliest ties. Fresh validation teacher labels
   count within the query budget. No checkpoint selection by arena results.
6. Test treatment versus its matched old-only update, and treatment versus its
   unchanged161 initializer, in fresh complete games. A proposed panel is400
   starting-seed clusters across three fixed fits, both seats and both matchups:
   4,800 fast games. Couple lawful proposal streams at matching observations;
   keep all games from each initial seed together in joint bootstrap resampling.
   Report all fits, seats and latency without picking a favorite model.

Use the same distinction between attribution and useful progress as161: require
a paired95% lower score bound above50% against the matched update, and a score
at least55% with lower bound above50% against unchanged161, plus all integrity
checks. Keep the full estimates if only some criteria pass. No outcome-dependent
sample extension, automatic promotion, continuation substitution or next round.
Exact schedules, hashes, validation aggregation and namespaces require a frozen
engineering protocol if execution is authorized.

The attribution is deliberately limited: this comparison tests whether the
specified fresh student-state feedback adds value beyond more old-data practice.
It does NOT isolate student-state distribution from the value of more independent
examples. A fresh teacher-state control would be needed for that stronger causal
claim. Failure would narrow this particular refresh size, mixture and dose; it
would not reject iterative learning or all student-state collection.

Learning changes the positions a player encounters, which motivates querying an
expert on learner-visited positions in
[dataset aggregation research](https://proceedings.mlr.press/v15/ross11a.html).
That is a rationale for this test, not proof that distribution shift caused our
remaining errors or that its theoretical guarantees apply to our noisy,
restricted-support Splendor teacher.

## Cost and verification

At160's measured78.101 seconds per128-world root and eight workers,2,400 queries
give a6.51-hour collection anchor. State lengths, contention, launch overhead and
audit scope can change this. Allow an overnight run, roughly8-12hours plus
preparation as a provisional estimate, and replace it with a fixed outcome-blind
timing probe before promising an ETA. Fast fits and fresh arenas are much cheaper
than the teacher queries. Do not silently budget6.5hours for the entire workflow.

Independently replay every source game, rebuild every lawful feature/proposal/
label, reproduce all six fits and checkpoint selections, and replay every fresh
arena game. For the new teacher outcomes, use160-style independently chosen
inner-rollout checks with full replay of those sampled branches; disclose this
as sampled inner verification, not full verification of every simulated move.
Freeze that scope and sample before collection. Full inner replay would add
substantially to the runtime and must be budgeted explicitly if selected instead.
Use hidden Windows workers and the tested durable-launch pattern. Monitoring
stays off. No launch is authorized by this recommendation.

## Why this precedes the alternatives

- A gap-weighted pairwise objective on the saved128 outcomes is the closest
  cheaper competitor. One independent reviewer preferred it on cost grounds.
  It would preserve the network while using more return information, but noisy
  estimated gaps are not reliable regret weights by construction.052/053 and
  149-153 already caution against assuming that softened targets, ranking losses
  or advantage prediction improve play. Those different failures do not rule
  out this new loss; they lower the priority of changing the ingredient that
  currently has positive fresh-game evidence. No loss sweep is recommended.
- More passes are cheaper still, but the present curves supply weak evidence
  that optimization time is the main obstacle. The old-only arm measures whether
  extra practice nevertheless helps within the next experiment.
- Changing simulated continuations again lacks a new validated mechanism after
  158/159. Adding a learned move already inside035's offered set does not expand
  search support; genuinely learned proposals require a separate design.
- Adaptive world allocation is a strong search-efficiency hypothesis after160,
  but needs an expensive direct equal-budget test. Scaling uniform worlds again
  is costlier and does not test retaining knowledge through learning.
- A broader opponent panel matters before a general strength or promotion claim.
  It would assess generalization but would not itself test the next learning
  step. Success here still would not close the learning-to-stronger-search loop.

Choose one refresh, evaluate it honestly, and then reconsider. Do not turn this
recommendation into an indefinite sequence of automatically scheduled rounds.
