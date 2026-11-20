# Recommendation after156: transfer some search strength into a fast selector

2026-09-13. Recommendation only. The owner asked for a considered next step.
Read AGENT_CURRENT.md first, reviewed the closed156 result and earlier strategy,
learning, teacher and gameplay evidence, and obtained independent read-only
reviews. No implementation, fitting, inference, simulations, tests, game,
training, promotion or successor was launched. Preserve all dirty/frozen work.

## Decision

Recommend one bounded experiment, provisionally157:

**Can a network learn to choose usefully among155's offered moves, producing a
fast controller that improves on035 without running the32-world simulations?**

Keep the verified proposal mechanism: frozen035's greedy action, highest-ranked
distinct canonical alternative, and two uniformly sampled remaining canonical
groups. Replace only the expensive evaluation of those proposals with a learned
ranking. Train from the already audited155/156 candidate decisions. Compare with
a matched035-label cloning control and unchanged035 in fresh complete games.

This is a test of partial transfer. It does not require the fast controller to
retain155's entire95.5% versus035 or65.125% versus084. A pass would not be full
compression of the teacher, a standalone improved035 network, or an iterative
learning loop. The deployed product still contains the frozen035 proposer and
a learned scorer. Establishing useful fast decisions is the next missing bridge.

## Why this question now

156 confirms a direct improvement over the prior research champion:259 wins,
3 draws,138 losses versus084;65.125%, paired95% CI60.375%-69.75%, all gates and
complete replay passed.155 had already established95.5% versus035. Repeated
terminal-rollout selection now has both local independent-world evidence and
full-game evidence against two relevant opponents. Another learner should be
judged by whether it captures useful decisions from that demonstrated system.

The paid data are a material opportunity:24,342 candidate decisions from800
games, representing400 initial-seed pairs across155 and156. They are about five
times the4,940-row052 training corpus before splitting, with approximately
19,500 training decisions under the proposed80/20 split. The3.62million156
continuations are not millions of separately saved training positions; the
usable decision examples here are the candidate roots.

The proposed role of the learner is deliberately limited. At a root where the
teacher examined only four moves, an omitted legal move was not judged inferior.
Training only the choice among the actual offered moves respects that support.
It also retains random exploration during deployment rather than asking a
state-only network to predict which random proposals happened to be available.
The32-world sampling noise remains; this design does not make the labels exact.

## Earlier failures remain relevant

| Evidence | Consequence for this proposal |
| --- | --- |
|052's warm full-distribution update cut KL substantially but rarely recovered teacher rankings;052a found its blended target suppressed36.1% of resolver switches | Teach the actual selected move among its actual candidates; no arbitrary softmax of sampled returns or blend with the old policy. |
|053's anchored residual ranking and054's representation variant failed | This is not another frozen residual/anchor adjustment. All policy-affecting035 parameters may learn under one fixed classification recipe. |
|075's from-scratch hard-label model had27.99% agreement versus035's32.87% | Hard labels alone are not a remedy. Preserve the useful035 initialization and evaluate actual gameplay. |
|149 afterstate advantage models were worse than035 on fresh conditional choices;150-153 did not rescue the tested objectives/representations/allocations | Do not restart sampled-value regression or select a network by MSE. Learning rankings may be a different task, but success is uncertain. |
|114 improved held-out target fit without establishing gameplay gain;115 remained inconclusive | Agreement and validation loss cannot establish playing strength. Fresh complete games are the primary endpoint. |
|154's apparent discovery gain0.14205 fell to0.04887 on independent worlds | Do not use the chosen maximum as an exact value label or weight examples by its apparent gain. |

This is not the first distillation attempt, and156 alone does not make imitation
likely to succeed. The case for this bounded pilot is the combination of the
new terminal-WDL teacher, already paid broader data, preserved proposal support,
warm classification and a cheap direct-gameplay endpoint. These changes are
tested together; a pass would not isolate which one caused learning to work.

One independent review preferred a random-versus-ranked candidate ablation,
citing the old learning failures. That is the strongest alternative. My judgment
favors this pilot because a positive fast-play result would change the programme
more, while a failed fixed recipe costs much less than another full planner match.

## Concrete proposed design

### Data, inputs and targets

Use only the155-selector decision roots from the main155 and156 games. Exclude
preflight data and084 decision roots. Do not combine084 score-margin targets
with155's terminal-WDL targets, or use game wins/losses as example filters.

Create a deterministic80/20 training/validation split separately within each
experiment:160 training pairs and40 validation pairs from each. Every turn and
both seat-reversed games from one initial seed remain in the same split. This
gives320 training pairs/640 games and80 validation pairs/160 games. All155/156
data are reused development evidence; the later fresh arena is confirmatory.
The reviewed156 pair0 belongs to training because its root action means were
opened during schema inspection. Freeze the remaining membership by identifier
hash before fitting; do not balance on outcomes, agreement or regret.

Retain all decisions, including unchanged-baseline choices, supports1-4, private
reserves, both actors and all phases. Record split coverage rather than deleting
unusual states. Source snapshots are omniscient replay records; derive model
inputs solely through the existing observer-relative035 encoders. Own private
cards stay known. Opposing blind identities, hidden deck order, actual future
refills, final outcome, sampled worlds, search scores and RNG identifiers are
not model inputs. Never expose a hidden draw by stepping a real snapshot before
constructing a pre-action input.

Initialize both arms from exact035. The attention model sees the complete
concrete legal-action list, as035 does. Each proposed canonical group receives
the maximum student logit among its concrete legal members. The loss is ordinary
cross-entropy normalized over only the1-4 offered groups:

- Treatment target: the actual155-selected group.
- Control target: the group containing the exact greedy035 action (candidate0).

The recorded selection is one sample from the finite-budget stochastic teacher,
not a guaranteed optimal action. Do not regress its discovery maximum, invent
visit counts, add a value loss, apply gap-dependent weights, train on wins only,
or turn unproposed actions into negative labels. Supports of one have zero
classification loss but remain in all accounting.

Candidate membership restricts the loss and deployed choice set. It is not an
extra input to the scorer; the hypothesis is that a useful consistent ranking
can generalize across proposed subsets. World-sampling noise and limitations of
this ranking model remain possible failure modes.

### Bounded fitting and checkpoint choice

Use the existing035 architecture; train its policy-affecting parameters, leaving
the unused value head unchanged. Three prespecified batch-order seeds, paired
between treatment and control, yield six fits; initial weights are always035.
No best-seed selection, ensemble, architecture grid, extra teacher corpus or
adaptive data collection.

Proposed optimization: four complete passes through the training rows, AdamW
learning rate1e-5, weight decay1e-4, batch64 and gradient-norm cap1. These are a
single starting recipe, not an inferred optimum. Freeze batching/padding and
all numerical details before fitting. Keep the initial checkpoint eligible;
evaluate early updates1,2,4,8,16,32,64 and quarter-pass checkpoints so the early
learning window missed in149 is not omitted. Complete all four passes.

Select each fit's checkpoint by its own target's validation classification
loss, averaged within starting-seed pair and then equally across pairs; ties
favor the earliest update. Report the full curves, agreement, action movement,
phase/seat/private-card summaries and every seed. The validation data select
checkpoints only. No teacher-agreement, MSE or movement threshold decides whether
a numerically valid trained model receives its fresh-game test.

### Deployed players and fresh gameplay

The treatment and control each retain the unchanged035 proposal builder,
representatives, small supports and ordering. Give them new independent legal
information/random-stream namespaces. Their learned scores choose among the
offered groups; exact ties favor original candidate order, baseline first.
Return the original proposed concrete action, not a newly selected alias.

At initial weights this complete controller must exactly reproduce raw035,
including pathological score ties and duplicate concrete representations. This
requires maximum-group scores plus the preserved proposal order; summing alias
probabilities would alter the policy. Attention over only representatives would
also alter035 and is not allowed.

Freeze all six selected checkpoints before any arena outcome. Evaluate every
one against exact greedy035 on the same400 fresh initial seeds, both controller
seats per seed:800 games per model,4,800 total. Share the initial-seed panel
across arms/seeds and prospectively couple proposal streams where the same legal
information is encountered. No search or teacher invocation during these games;
the fast controller uses035 for proposals and its trained scorer for ranking.

The matched clone shares states, model, optimizer, batches and selection rule;
only its teaching action differs. Compare treatment and clone scores against
the same unchanged opponent. This isolates the usefulness of the teaching signal
under the complete recipe without mistaking degradation of the clone for progress.
It is not a direct head-to-head claim between treatment and clone.

### Success, uncertainty and stopping

Primary practical criterion: the mean treatment score against035 is at least
55%, with complete-pair95% CI lower strictly above50%. Also require positive
mean final margin and at least50% from each seat.

Attribution criterion: treatment score against035 exceeds the matched clone's
score against035 by at least5 percentage points, with the paired difference's
95% CI lower strictly above zero. Both criteria are required. Beating a damaged
clone while failing to beat unchanged035 cannot pass.

For each initial seed, first average the two seat outcomes and all three fixed
training-seed outcomes within an arm. Bootstrap the400 complete initial-seed
clusters50,000 times, retaining paired arm outcomes, with a new frozen seed.
Moves, model seeds and continuations are not extra independent initial games.
Report each fit separately; the aggregate interval is conditional on the three
fixed fitted models and is not a precise characterization of all possible
training runs.800 games per model gives a bounded screen, not guaranteed power
for every five-point effect. Do not extend the panel after seeing an interval.

Require all privacy/legal-input, provenance, numerical and replay checks, and
confirm that the controller actually avoids rollout work. Report complete
decision wall/CPU and two-network overhead. A proposed practical latency gate
is mean decision time no more than10 times raw035 and below0.1 seconds on the
same host, including proposals. This is a generous fixed fast-controller target;
freeze measurement and the treatment-seed aggregation before launch.

Independently reconstruct all exported source features/candidates/labels and
split memberships; reproduce all six complete fits and selected checkpoints;
replay every actual arena trajectory and its proposal/score/choice evidence;
independently recompute statistics. Share the engine/scorer primitives explicitly.
Use the established natural/equal-turn and valid forced-two-PASS endings, with
all cutoff/illegal/exception cases preserved as failures, never replacements.

Full repository tests precede freeze. A small off-panel timing/privacy/reload/
gameplay preflight precedes the protected arena. No silent change to budgets
or recipe if preflight reveals a blocker. Audit the entire valid collected
arena on either statistical outcome. Immutable records, exclusive locks and
identical explicitly authorized crash resume follow the existing pattern.

Both outcomes stop. A pass establishes partial useful transfer into this cheap
controller and supports a separately proposed test inside stronger planning.
A failure closes this exact pilot; it does not justify an automatic optimizer,
capacity or threshold sweep or prove that useful policy learning is impossible.

## Cost and alternatives

No new expensive teacher searches are needed.155 measured raw035 at about
0.00427s per decision, versus19.485s for its planner. This controller needs two
network evaluations and proposal construction; exact speed must be measured.
4,800 fast games are therefore a different cost class from another400-game duel
between planners. Source export, training and independent fit replay may dominate.

The actual052 policy fit took100.80s for four epochs over4,940 training rows,
with batch256. Its trainer thread count was not recorded/enforced like the game
workers, so multiplying that figure is only a crude order-of-magnitude check.
The new candidate loss, batch size, early validation, private positions and full
verification also differ. Expect an hours-scale pilot rather than assume another
41h planner run, but obtain a real preflight estimate before promising duration.

| Alternative | Why it is not my first choice now |
| --- | --- |
| Same-budget random-versus-top-ranked candidate arena | Clean causal test of diversity and a credible alternative. It answers which proposal mechanism to keep, while useful transfer would open a cheaper improvement path. It requires fresh expensive games; frequency of random choices is not causal evidence. |
| More worlds, more candidates or recursive stronger continuations | Could improve strength;154 leaves substantial selection noise. It adds planning cost without yet answering whether improved decisions can be retained cheaply. |
| Learn a proposer while keeping the32-world evaluator | Strong fallback learning direction with evaluator protection. A first match also needs a frozen035-proposer control to separate learning from guided enumeration, adding cost. |
| Full-policy behaviour cloning over every legal action | It penalizes actions the teacher never considered and marginalizes over random proposal availability. The proposed restricted choice objective respects the recorded support. |
| Further broad opponents or humans immediately | Necessary for eventual superhuman claims.156 already answered the strongest immediate in-programme comparator; breadth is less directly informative about the learning bottleneck. |

## Research context and limits

Classification-based policy-iteration research distinguishes approximating a
useful policy from accurately fitting every action value. Its guarantees depend
on assumptions and losses not established here; it motivates testing the
distinction, not assuming that this unweighted classification pilot will work.
[Lazaric, Ghavamzadeh and Munos, JMLR2016](https://www.jmlr.org/beta/papers/v17/10-364.html).

A student can reach different positions once its own choices affect play.
That is why fresh whole games decide success rather than held-out imitation
accuracy. The general issue is documented by
[Ross, Gordon and Bagnell, AISTATS2011](https://proceedings.mlr.press/v15/ross11a.html).
This proposal does not silently add their iterative data-aggregation algorithm.

Even a stronger cheap controller would not automatically be a better rollout
policy: useful simulation estimates and strong direct play are distinct roles.
The distinction has primary experimental precedent in
[Silver and Tesauro, ICML2009](https://research.ibm.com/publications/monte-carlo-simulation-balancing--1).
Any replacement of035 inside155's simulations requires its own test.

## Evidence and status

- `docs/engineering/rollout_selector_champion_156_result.md`
- `docs/engineering/rollout_selector_gameplay_155_result.md`
- `docs/engineering/rollout_selector_154_result.md`
- `docs/experiments/resolver_selfplay_distillation_052.md`
- `docs/experiments/resolver_distillation_disagreement_audit_052a.md`
- `docs/experiments/resolver_residual_ranking_distillation_053.md`
- `docs/experiments/from_scratch_resolver_bc_pilot_075.md`
- `docs/engineering/afterstate_advantage_149_result.md` and150-153 results
- `docs/strategy/independent_superhuman_review_after_134_137.md`
- `outputs/resolver_selfplay_distillation_052/training.json`

156 remains closed, binding
`dab0e5502a4eca5005c04c02d2095acc601a7cadee196c2173725d85bce69774`;
result SHA `c468037bd8459957c455d8869c02502a4c2fee4aeb6b74f174839aeedb73fb8f`;
audit SHA `20e079af9fcb851f36c6c2fa460912325a2fd0b75a4564d111f0c709dc4dd2ab`.
This recommendation authorizes nothing. No157 code, model fit or game has run.
