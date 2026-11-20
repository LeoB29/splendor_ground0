# Experiment159: independent-world replication of fixed planner preferences

Authorized2026-09-14 after the owner accepted the narrowed diagnostic. This is
an explanatory experiment, with no training, arena, promotion, automatic
extension, successor, recovery or scheduled monitoring. Preserve all dirty work
and all158 artifacts. The owner-stopped158 result remains provisional:402 games
were collected, but only19 main games received full independent replay.

## Question and estimand

Do the moves selected by the existing32-world greedy-continuation planner (GG)
and learned-continuation planner (LL) have material, repeatable differences on
fresh futures under those same two continuation objectives? Both simulated
actors use035 in GG, or the same assigned exact157 controller in LL. The root
proposal set is preserved from158: greedy035, second distinct canonical group,
two uniformly proposed remaining groups, or the smaller available support.

For each root, independently select one action under each objective using32
shared fresh lawful hidden-world draws, with the original first-maximum tie
rule. Freeze every selected action for the entire panel before dispatching any
evaluation job. Evaluate only those two fixed actions on128 additional draws,
shared across actions and policies. Selection and evaluation have separate RNG
namespaces; this means independent draws, not rejection of naturally repeated
hidden states. LL proposal streams are also separate between stages and share
corresponding action/world/step/actor slots within a stage. Exact157 adds only
the current simulated actor's lawful belief; no root-private information seeds
the opposing simulated actor. Use the frozen158 trajectory and natural-ending
rules with no truncation, invalid-outcome exclusion or retries.

The primary effects D_GG and D_LL are root-averaged fresh-world score differences
between the LL-selected action and GG-selected action, under each objective.
Score is win1/draw0.5/loss0; a difference of.025 is2.5 percentage points. If both
planners select the same action, retain the root with exact zero difference and
skip its redundant evaluation. This is an algebraic saving, not a root filter.

Neither continuation objective is ground truth or a planner win probability.
The diagnostic does not establish why158's arena result was null, whether a
different planner wins more games, or whether instability is harmful. Choices
are conditioned on preserved proposals, with the original32-world budget.

## Panel and precision fixed before new simulations

180 roots from180 distinct158 source seed pairs. Exactly five roots in each of
36 model(3) x phase(3) x acting-seat(2) x source-planner(2) cells. Phases use turn
index<20,20..39,>=40. GG/LL source labels mean which planner actually visited
the source position. Private reserves, forced decisions and small proposal sets
are retained. This is an equal-weight phase/seat/source-planner mixture
conditional on the three fixed157 fits, not the natural gameplay visitation
distribution and not a sample of newly trained models.

Use all402 hash-verified saved game traces only to enumerate pair/turn/actor
eligibility. Hash-rank the source pair order, cell order within a pair, and root
position within each eligible cell. Capacity-constrained augmenting-path
matching assigns each pair to at most one cell; stop at the first fully filled
36-cell panel. If it cannot fill, stop without relaxing criteria. No terminal
outcomes, saved choices, means, gaps, confidence measures or disagreements
participate in matching or ranking. Copy the selected root's snapshot and
original proposal evidence after matching and seal all source paths/hashes.

Prospective planning assumption: total root-level SD.14 after128 evaluation
worlds. With z(.9875)=2.2414, a2.5-point interval halfwidth requires
ceil((2.2414*.14/.025)^2)=158 roots; round up to180 for balanced36-cell coverage.
This is a planning assumption, not a guaranteed power/precision result. A saved
opening-only16/16 split diagnostic suggested128-world-adjusted SDs about.112
and.088; those selected moves and phases differ from this experiment. Later
states and full32-world choices may violate the assumption. The fixed cap is
180 roots regardless of interim or final outcomes. Missing the final precision
target yields an explicitly inconclusive precision result, with no extension.

Maximum collection:180*(2*4*32+2*2*128)=138,240 terminal continuations, reduced
only by smaller support or identical frozen choices. Eight workers, one Torch
thread each; deterministic inference, fixed checkpoints1220/1220/1144.

## Analysis and decisions

The independent unit is the source pair, with one root per pair. Resample five
roots with replacement independently within each of36 fixed cells,50,000
replicates, NumPy seed1593000000 for exploratory intervals. Preserve the full
paired vector of objectives and evaluation halves within every root. Because
each cell has only five roots, primary inference uses unbiased within-cell
sample variances: v_cell=s_cell^2/5, SE=sqrt(sum(v_cell))/36, and
Welch-Satterthwaite df=(sum(v_cell))^2/sum(v_cell^2/4). The two marginal97.5%
t intervals form an approximate simultaneous95% family via Bonferroni. Zero
variance gives a point interval and null degrees of freedom. The two primary
interval halfwidths must each be<=.025. The exploratory bootstrap cell-mean
deviations are multiplied by sqrt(5/4) before aggregation to correct the usual
small-cell bootstrap variance shrinkage. No extra within-root bootstrap: the
root-level sampling already includes selection and evaluation variability.
Models remain fixed; these are approximate, not exact finite-sample intervals.

Materiality band: +/- .025 score units. Prespecified labels:

- LL effect interval above+.025 and GG below-.025: material conditional
  preference reversal, giving a concrete repeatable disagreement to dissect.
  The opposite direction receives an explicit reverse-direction label.
- Both intervals above+.025, or both below-.025: a common conditional advantage
  for one planner's selected moves, without a gameplay-strength claim.
- Both intervals inside[-.025,+.025]: small average fixed-choice effects only.
- Otherwise: inconclusive for material classification. A missed precision
  target overrides the classification with an inconclusive precision result.

Also report the signed interaction J=D_LL-D_GG and its exploratory95% interval.
Split the128 evaluation draws into disjoint64-world halves A/B and report the
root average J_A*J_B with an exploratory95% interval. Conditional on the frozen
choices this estimates mean squared root interaction, exposing variation that
may cancel in the signed average. Preserve negative sample estimates and
intervals; do not clip them into claimed variance estimates or take an invalid
square root. Subgroups by phase/model/actor/source planner and choice-change
rates are descriptive, with no selection of a winning subgroup. Small primary
means alone never exclude large canceling differences at individual positions.

## Integrity and operational contract

New files only:159 module, entry point, verifier, tests and this protocol. Bind
their hashes, numerical runtime versions,158 frozen identity,157 selections,
158 game lock and provisional result. Run focused159 and frozen154/158 tests
before freezing. Seal the matched panel and test evidence before launch.

159 deliberately uses a proportionate, fixed sampled replay. Independently
reconstruct every selected source prefix from the actual saved actions, verify
every selected snapshot/proposal/privacy binding, reconstruct every sampled
hidden world, validate every terminal record and recompute every selection and
effect. Independently replay every action/policy at selection world indices0
and16 and evaluation indices0 and64. Identical-choice evaluations have no
branches to replay. This is at most4,320 replayed continuations, not a full
replay of every branch. Report that scope explicitly; do not certify158 or call
159 a fully trajectory-replayed experiment. Statistics receive separate
implementation/test checks.

All selection records and their immutable hash lock must exist before
evaluation; selection audits must pass before evaluation starts. Immutable
records and locks are never replaced. Save append-only stdout and separate
failure artifacts. Use a hidden detached process, a new process group and
Windows job breakaway so audit shell windows are unnecessary. No visible shell
windows or periodic monitoring. A failure stops this one run without retries.
Finish by sealing results, report and final status after the prespecified checks.
