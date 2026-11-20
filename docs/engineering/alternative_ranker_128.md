# Fixed alternative learners on an untouched restricted-ranking panel

Owner authorized on 2026-10-16: prepare and freeze the alternative-model test
before the new results, then investigate simulation speed; leave the running
structured_proposer_128 experiment unchanged. This is an independent add-on,
not an amendment to that experiment. No current source, protocol, process,
resource setting, gate, or output is changed.

## Question and scope

Can either of two deliberately different, strongly regularized learning recipes
rank recorded teacher alternatives better than the original anchor on fresh
positions? This tests a combined change of features, model and objective. It
does not uniquely identify representation, regularization or data quantity as
the cause. A null rejects these recipes, not learnability in general.

The confirmatory panel is the existing study's planned 720 main positions from
distinct fast-161 source games, balanced across 18 fit/seat/phase cells. It exists
only if that study independently passes its acquisition and operational gates.
No new source games or simulations are authorized for this add-on. If the panel
does not materialize, this study remains unevaluated; no substitute panel or
relaxation of the original gates is automatic.

The endpoint concerns ranking within the teacher's recorded alternatives outside
the common first three, at most five actions. It is not full-legal fourth-move
proposal quality, production planner improvement, a game win rate or promotion.

## Freeze and separation from outcomes

Before accessing any new probe/main outcomes, freeze this document, exact code,
feature schema, training files, all hyperparameters, seeds, software versions and
statistical rules. The freeze binds an unambiguous deterministic training and
selection algorithm; final weights are subsequently sealed after independent
reproduction. No tuning, best-model selection or checkpoint selection uses old
evaluation panels or the new panel. Both recipes are always reported.

The initial freeze records an existence-only check that the current study has
not created its fresh probe/main output directories. Freeze and verification
read only that study's already-frozen manifest, TRAIN data lock and original
model references. They do not read current fitting metrics or fresh outcomes.

Before assessment, create a separate prediction seal from source snapshots and
original anchor models, without reading fresh simulation results. Fix every
canonical candidate, model score and chosen candidate. Only then may the
evaluation command read authenticated completed outcomes. There is no automatic
monitor, evaluation on arrival, arena, successor or deployment.

## Data and features

Reuse exactly 9,461 old TRAIN roots and 11,520 added TRAIN roots, with their
original 128-world teaching labels. Authenticate the two immutable tensor files
through the current study's sealed data lock. Validation and all historical
evaluation panels are excluded from model fitting and selection.

Each concrete action has 465 float32 features, derived only from the established
observer-legal observation295 and action79 encodings:

- Original observation295 and action79.
- Thirteen after-action quantities: six token counts/10, five bonuses/7,
  score/20 and reservation count/3.
- For each of 18 known-card slots: gold-adjusted remaining shortfall before and
  after/10, availability after and own-reserved ownership after. Unknown slots
  are zero and before-action card identity is retained.
- For each of three noble slots: total remaining bonus deficit before and
  after/20. Eligibility, including a claimed noble, has zero remaining deficit;
  the original slot identity is retained.

The exact channel layout and arithmetic are additionally bound in FEATURE_CONFIG
and code. No hidden reserve identity, hidden deck order, unknown refill or new
information is introduced. Canonical action features are the arithmetic mean of
the concrete members' feature vectors, with fixed member order. This differs
deliberately from the neural models' maximum-logit aggregation.

Only recorded teacher outsiders enter fitting. Target each action's original
expected terminal score minus its root's mean outsider score. Empty supports
remain accounted for with no training samples; singleton supports contribute a
zero target. Eligible roots have equal total weight within a pool. Each pool
has half the total weight, and weights sum to the number of candidate samples.
The raw input features are not centered by root.

## Exactly two fixed recipes

One pooled model per recipe uses all eligible TRAIN roots; there is no
hyperparameter search or validation-based early stopping.

1. Ridge regression, alpha=100, with weighted TRAIN-only centering/scaling and
   an unpenalized intercept. Accumulate sufficient statistics in bounded chunks
   and solve the regularized normal equations in float64.
2. HistGradientBoostingRegressor, squared-error loss, 250 iterations, learning
   rate .05, at most eight leaves, minimum 64 samples/leaf, L2 regularization10,
   no early stopping, random seed2801000000, raw float32 features. Exact remaining
   library options and version are bound by the code/configuration freeze.

Use one CPU thread and one additional process at a time. Prepare compact
outsider-only memory-mapped arrays. Independently repeat each deterministic fit
and compare fitted parameters and every TRAIN prediction before sealing models.
Training diagnostics are descriptive and cannot choose or discard a recipe.

## Fresh comparison and statistics

For each source root, reconstruct the same ordered teacher outsiders using only
source state and frozen original models. Each alternative chooses its highest
score; ties use the first member of this fixed order. The comparator is the
assigned original anchor's highest-scoring member of the same support.

The fixed choices are assessed on all 128 recorded selection worlds. These
outcomes were not used to train, select, or choose actions for either alternative
or the comparator, so no additional simulation block is needed for this ranking
contrast. Do not reuse these outcomes to optimize the choices. This is separate
from the original planner, which selected moves using this block and therefore
required disjoint assessment worlds.

All 720 roots remain in the mean. Empty/singleton supports contribute zero paired
gain, rather than being dropped. Independently reconstruct identity, fit, seat,
phase, support, choices, world count, outcome arithmetic and completed source
study audit references. Do not drop roots whose proposed full-legal move would
lack recorded outcomes; full-legal proposals are outside this endpoint entirely.

Primary endpoint for each recipe: paired mean action-value improvement over
anchor, in expected-score percentage points, win1/draw.5/loss0. Resample whole
source roots jointly across both recipes, within each of the 18 balanced cells,
50,000 times with seed2713000000. Report two-sided97.5% bootstrap intervals for
each recipe (quantiles .0125/.9875), a Bonferroni adjustment across the two primary
contrasts. No extra confirmatory subgroups or best-fit selection.

Unknown outcomes retain conservative bounds; never impute draws. Identical chosen
actions cancel exactly. A lower confidence bound above zero supports a positive
restricted-ranking effect. An interval crossing zero is inconclusive; it does
not establish no effect. A wholly negative interval supports deterioration.
There is no new +.5pp or ten-capture-point gate and no automatic arena decision.

Report absolute improvements and intervals first. Empirical regret capture is
supplementary: its maximum over noisy candidate means is optimistic, so capture
is not an unbiased fraction of true remaining skill. Retain the denominator,
support counts, coverage, unknowns and short-root counts alongside it.

## Verification and operational boundaries

Follow docs/engineering/verification_policy.md. This add-on reuses authenticated
completed simulation evidence, without claiming new trajectory replays. Require
the source study's complete720-root result and all prescribed audits to pass
before accepting the fresh ranking result. Independently check every new feature
fixture, changed prediction, arithmetic result and model reproduction. Tests
cover outcome blindness, privacy, candidate ties/short supports, wrong hashes,
missing/duplicate roots, unknown outcomes and multiple-comparison accounting.

No production shared files may be edited. Own outputs are confined to
outputs/alternative_ranker_128 and named test/preflight directories. One extra
CPU thread, below-normal process priority, at most two hours for preparation and
both fits/reproductions, at most2GiB own data/artifacts, and a5GiB machine disk
reserve. Check memory before work; defer safely if insufficient, without touching
the current experiment. Start only with at least768MiB available physical RAM
and2GiB commitment headroom. A two-second operational guard stops only the new
worker if available RAM falls below128MiB, commitment headroom below1GiB, its own
resident memory above1.5GiB, its artifact or time limit is reached, or the current
study enters its source/simulation stage. A fixed-algorithm fit deferred until
the current study finishes remains permissible without opening its results.
No outcome-based retry, dose extension or model change.
An interruption preserves partial files; any recovery must retain the frozen
algorithm and cannot use fresh results to change it.

The subsequent speed investigation uses separate code and completed historical
positions. No optimization is deployed into the running experiment.
