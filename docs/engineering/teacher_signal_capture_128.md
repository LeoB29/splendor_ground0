# Teacher signal and frozen learner capture at 128 worlds

Owner authorized on 2026-10-08. This is one bounded mechanism study with fresh
assessment, not a training run or player promotion. Preserve the intentionally
dirty worktree, every frozen predecessor and all failed or partial evidence.
Implementation is tools/teacher_signal_capture_128, entry point
run_teacher_signal_capture_128.py, output outputs/teacher_signal_capture_128.

## Question and inherited evidence

The stronger eight-candidate planner has demonstrated useful search and gameplay.
Subsequent winner imitation, ranking, broader supervision, outsider specialization,
planner-state transfer and additive retention did not establish another useful
learned proposal. This study asks where independently useful information is lost:
in the recorded teaching preferences, their capture on training versus validation
examples, or their use as supported proposals. It does not ask whether search can
help in general or assume a uniquely identified learning bottleneck.

Use original breadth-arena candidate-side decisions belonging to the actual
11,950-row addition in the closed broader-learning corpus. Authenticate its source
closure, original train/validation assignments, selected snapshots, recorded action
representatives, old128-world outcomes, tensor-export membership and original
simulation audits. Inherit historical trajectory and fitting audits; do not rerun
old games, old rollouts or old fits. Selected games/roots remain development data.

Use all nine original selected models from the broader-learning registry: three
unchanged161 initializers, three BROAD learners, three matched NARROW learners.
Keep the original selected checkpoints and assign the corresponding source fit.
The specialist is excluded to retain the original clean matched comparison and
avoid silently substituting its differently reselected control. No model selection,
training, new source games, continuation-policy change or architecture change.

## Fixed source selection

Main has 336 roots from84 original source pairs. Retain all42 original new-corpus
validation pairs (14 per fit). Identity-hash-select42 training pairs,14 per fit.
Select two different phases per pair and one candidate-side root in each selected
phase of each reversed-seat game: four roots per pair, two per game.

Every source game has an eligible candidate root in all three phases, verified
before any new assessment. Within each fit, separately identity-rank the14 TRAIN
and14 VALIDATION pairs and match ranks j. Omit phase (j+fit_index) modulo3 in both
members of each rank match. Choose roots by identities within the remaining phases.
This gives112 roots in each phase globally and identical train/validation phase
counts. Phases are turns0-19,20-39,40+. No outcome, margin, support size, model
disagreement or prior simulated winner selects a pair or root.

Probe uses nine unused TRAIN pairs, three per fit, with one fixed phase per pair
and one root from each game:18 roots, disjoint from all main pairs. The probe is
for integrity/time/storage only; none of its scores selects a scientific rule or
enters the main endpoint. All source shortages hold; never extend a pool to obtain
favorable observations. Exact hash tags and identities are sealed before launch.

## Choices fixed before fresh assessment

Preserve the original ordered up-to-eight offered candidates and their concrete
representatives. The original up-to-four prefix is the anchor planner. The common
first three define the unchanged proposal context. Score full lawful legal inputs
with the frozen models and retain canonical maximum logits and original tie rules.
Record both full-legal rankings and the restricted choices within recorded support.

At each root lock these choices before any fresh simulation:

- Anchor: original128-world best action in the original four-candidate prefix.
- Teacher: original128-world best action among all recorded candidates.
- Original fourth: recorded index3, or absent when support is at most3.
- Teacher outsider: original128-world best among recorded indices3 onward.
- For each anchor/BROAD/NARROW model, direct choice: its highest-scored recorded
  candidate; outsider proposal: its highest-scored recorded candidate outside
  the common first three; supported decision: old128-world selection among the
  common first three plus that outsider.

Original-order ties are preserved. Confirm that the original anchor's supported
choice reproduces the old four-candidate decision and that its outsider is the
original fourth. Preserve every small support. When no outsider exists, outsider
actions and absolute scores are null and outsider-only paired effects are exactly
zero as structural no-ops; do not drop these roots or invent a fourth action.

Seal source identity, model predictions, old outcomes and all scored choices
before the fresh assessment starts. New worlds cannot modify those choices.
Old unresolved outcomes require conservative exact original-order certification;
remaining ambiguity holds the study. No draw imputation or new old-label rollouts.

## New independent outcome block

Assess every recorded candidate on128 new common observer-legal worlds, with
unchanged greedy035 terminal continuations. New seed base2210000000, stage offset
0 for probe,1000000 for main,2000000 for isolated fixtures; index stride512 and
world offsets0-127. Check new namespaces against source environment and old world
seeds. Same hidden-world draw is shared by all candidates in that world.

Fresh assessment can provide descriptive rank/choice repeatability against the old
teacher. Never grade its own new argmax on those same outcomes as an improvement,
use it to replace the old teacher choice, or reverse-score learners on old targets
used in fitting/checkpoint selection. Main maximum new continuations336*8*128
=344,064; probe maximum18*8*128=18,432, before short supports.

Natural terminal WDL maps to score (WDL+1)/2. Repetition outcomes remain unknown
WDL in[-1,1], with conservative score/contrast bounds. Identical chosen actions
cancel exactly in contrasts. Invalid termination reasons hold/fail; do not remove
roots, impute draws or change the source population after outcomes.

## Inference and interpretation

Primary diagnostic effects are teacher minus original anchor, supported BROAD
minus original anchor, supported BROAD minus supported NARROW, and teacher outsider
minus original fourth (equivalently the independently checked anchor outsider).
Other direct-choice, outsider-choice and teacher-minus-learner contrasts are
descriptive measures of remaining capture gaps. No best-fit/phase/model rescue.

Report TRAIN and VALIDATION separately and the validation-minus-training effect
differences. Weight all four roots in each pair equally; every split has42 pairs.
Bootstrap50,000 times using whole pairs within fit and split (14 pairs per cell),
seed2213000000, chunks1000. Keep every arm paired within a draw and resample TRAIN
and VALIDATION independently. Retain conservative lower/upper envelopes for
unknowns. Independently reproduce the bootstrap with a separate implementation.
Intervals condition on these fixed fits, sources and split assignments.

The +0.005 score (+0.5pp) material threshold is an investment guide. An individual
diagnostic has a material signal if its lower mean bound reaches that threshold
and its95% interval lower endpoint is positive. An interval upper endpoint below
the threshold argues against that target on this panel; other outcomes remain
uncertain. Pointwise diagnostic intervals are not familywise confirmatory claims.
There is no automatic promotion or training gate, and no forced unique mechanism
classification. Positive training capture with weaker validation suggests a
generalization problem; poor capture even on training suggests examining fitting,
objective or representation; useful supported capture suggests revisiting proposal
use. Each remains a hypothesis requiring its own controlled test.

Raw winner agreement cannot diagnose consequential label noise: near-equal actions
may be interchangeable. Conversely, new assessment on reused development positions
does not establish fresh game strength, broad generalization, a learning ceiling,
or superiority against stronger opponents. Original validation selected checkpoints;
it was not used for gradient updates but is not a protected final test set.

## Verification, execution and stopping

Freeze code, source/model inventory, choices rule, audit sample, thresholds and
resource rules after passing focused preflight. Verify all new lawful inputs,
model predictions, old-choice arithmetic, new world identities, outcome structure,
sparse/absent action handling and statistics. Independently replay all probe worlds,
eight hash-selected whole worlds at every main root, and all unresolved new outcome
cells. Stress sampling takes one minimum-priority root in each fit/seat/phase cell
(18), each fit/private-reserve-category cell (12), and each fit/short-support1/2/3
cell (9), deduplicating overlaps and reporting absent cells, at most39 full roots.
Inherit sealed old simulation audits.
Full main inner replay is not prescribed; sampling coverage is reported honestly.

Run eight hidden CPU workers, one Torch thread each, with durable logs and no
recurring monitor. No GPU is used. Keep system sleep inhibited while executing;
display sleep is allowed. Use an outcome-blind fully verified probe to forecast
main collection, sampled/stress verification and finalization, with1.35 allowance
and a300-second closing allowance. Forecast outside eight hours from launch or
below5GiB free-space reserve holds. Cooperative boundaries enforce the same limits
throughout execution and preserve in-flight evidence after failure.

Preflight includes private input/hidden-state invariance, exact ties and nulls,
short supports, held-out separation, model/source tampering, pair bootstrap,
budget/one-shot launch/failure handling, a bounded real simulation/replay fixture,
and an actual hidden supervisor/worker loading all nine models. No late code
edits after freeze. No automatic retry, extension, refit, new arena, successor,
promotion or monitoring follows this study. Expected execution2.5-4hours after
preparation is provisional until the probe, not a promise.
