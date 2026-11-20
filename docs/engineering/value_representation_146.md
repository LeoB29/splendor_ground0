# Experiment 146: separate trainable value representation over fixed 035

Owner authorized this conditional pilot with "ok do it," on 2026-09-08.
This document, implementation, tests, verifier, model inputs and explicit prior
dependencies are frozen before training and any protected 146 outcomes.
Preserve the intentionally dirty worktree. No automatic successor or promotion.

## Scientific question

Does allowing a separate copy of 035's state encoder to learn outcome prediction
produce practically better early/middle estimates than learning only its head,
when both start at the frozen 143 public evaluator? 143's frozen head improved
overall MSE only 1.73% against neutral; the simple evaluator improved 15.59%,
mostly late. 145's equal-time public/neutral comparison was inconclusive.
Neither result establishes that representation is the bottleneck. 116 tested
another independent residual tower on a different branch and dose; it does not
settle this strong-035, controlled value-only comparison.

## Matched treatments

Original 035 policy checkpoint and all policy inference remain separate and
unchanged. Every canonical prior uses the intact complete legal action set.
Both value models copy exactly the same 035 state encoder and its 256 -> 128
GELU -> 1 head. Remove the final Tanh and zero the final linear weight and bias.
Control freezes the copied encoder; treatment trains that encoder plus the head.
No action features enter the value tower. Inputs are the existing 295 legal
observation values, which already include board cards, costs, nobles and player
resources; this is not an addition of missing board information.

Prediction is clip(public + residual, -1, 1). Public is the exact frozen 143
15-feature ridge: features rounded to float32, dot product in float64, clipped
before adding the float32 residual converted to float64. Step zero therefore
equals the public baseline exactly. This intervention changes trainability and
capacity within a recipe; it cannot cleanly distinguish information loss from
optimization or sample efficiency. Residual clipping is part of this recipe.

Pre-launch precision correction: deployment reuses PublicValueInference's exact
scalar and adds the separately evaluated residual. Development/test tensor
evaluation uses batched float64 matrix-vector arithmetic as 143 did; its last
rounding bit can differ from a scalar dot. The first off-schedule check caught
that deployment discrepancy before any fit/test. Its records, source and
identity are retained under `preflight` and `preflight_initial_*`; the corrected
probe uses `preflight_v2` with the same predetermined off-schedule seeds.

## Development and selection

Reuse only sealed 143 training IDs [0,4096) and validation [4096,4608).
No old protected IDs [4608,5120) enter fitting, selection or a new test.
All 4608 games must be natural and retain their sealed JSON/tensor hashes.
The earlier independent corpus audit remains valid through input hash binding.

Both arms use AdamW, lr .0003, weight decay .0001, gradient clip norm 1,
batch 128, seed 146010, maximum 4096 updates, one Torch thread. The paired
batch stream samples games uniformly then a position uniformly within each
game, from all training turns. Objective is actor-relative final W/D/L MSE.
Validation every 128 updates uses game-balanced MSE restricted to turns <40.
Select step zero or the first checkpoint improving the previous selected loss
by more than .001. Stop after four consecutive unselected validations or at
4096 updates. Each arm follows this same rule independently; selected update
counts may differ. No grid, retries, post-test selection or alternate candidate.
The trainable arm is the sole candidate for conditional gameplay.

Before protected collection, seal both selected models, complete training
histories and cost measurements. Reproduce both training runs deterministically
in audit; independently check the selection rule and every selected tensor.

## Fresh protected prediction test and gate

512 games, seeds 1460000000 through 1460000511. Exactly the 143 raw-035 behavior
recipe: first 20 turns, epsilon .2, top-three softmax temperature .5, behavior
RNG offset 143003; greedy afterwards. No self-play search. Eight CPU workers,
one Torch thread each. Max turns 400, repetition 4, no-progress 60; completed
and forced-pass deadlock are natural. Persist failures without replacement.

References are neutral zero, frozen public, and the matched frozen-encoder
residual learner. Primary endpoint: each game's mean squared error on turns
<40, averaged equally across games. Overall and turns >=40 are guardrails.
Turns <20 and 20..39 are separately reported descriptive phases. Resample whole
games, keeping model differences paired, 50,000 bootstrap draws, seed 146999,
percentile two-sided 95% intervals. Do not treat positions as independent.

All of these are required to proceed:

- All 512 games natural, all 512 contributing to early/middle, overall and late.
- Early/middle trainable MSE at least 3% lower than public, with difference
  interval upper bound <0; and at least 1% lower than frozen with upper <0.
- For overall and late, versus each comparator, difference interval upper
  bound <.01 absolute MSE. This is a specified noninferiority tolerance, not
  proof of exact equality or nonregression at zero margin.
- Combined policy/value inference median <=1.5x public and p95 <=2x public.
  Measure all states from two fixed off-schedule behavior games three times,
  alternating model order and discarding two warmup calls per model. Record all
  timings, verify exact unchanged policy logits. Cost measurement occurs before
  test collection. No latency retries based on a failed gate.

The 3% primary target asks for a useful early/middle gain beyond the weak 143
result. The 1% control target requires evidence specifically for encoder
trainability. These are pragmatic pilot thresholds, not calibrated guarantees
of gameplay gain. Requiring every condition is an intersection gate, not a
choice among whichever contrasts happen to succeed.

## Conditional fixed-time gameplay

Only after the independently verified prediction gate passes: 800 games, 400
fresh seat pairs, seeds 1461000000 through 1461000399, against frozen public
search. Treatment is trainable residual, opponent public only; original 035
priors and real PUCT are identical. Each move has .4 seconds including inference
and root expansion, at least one simulation and no new simulation after the
deadline. No equal-simulation cap and no artificial padding for the slower arm.
c_puct 1.5, max depth 96, zero exploration/noise, original canonical selection.
Search RNG is seed*10000 + player*1000 + turn +146013. Eight CPU workers.

Both arms must have mean decision time <=.44 seconds, p95 <=.50, every search
>=.4, and mean time ratio within [.95,1.05]. All games natural, no replacements.
Material gameplay success: score share >=55% and paired-seed bootstrap CI95
lower >50%; 50,000 resamples, seed 146998. Otherwise report uncertainty, regression
or invalidity as appropriate. A win against public search does not establish
superiority to raw 035 or champion 084; no promotion is authorized here.

Independent arena audit reconstructs every move, final state, outcome, 035
prior, root value, visit accounting, timing and selected edge. Reproduce the
first and last search of each arm in every game at recorded simulation counts.
Independently recompute paired gameplay statistics.

## Preflight, audit and stopping

Two off-schedule behavior games use 1469000000/1469000001; reconstruct all
behavior, observations, public features, latents and labels with the existing
143 independent auditor. Check both zero-residual predictions exactly equal
public at every state. Two zero-residual arena probes use seed 1469100000 with
swapped seats and all-root/four-search-per-game verification. Probe outcomes
are engineering checks only and cannot select a model. Focused tests cover
gradient isolation, exact zero identity, precision, game weighting, protected
data exclusion, baseline selection and every gate condition. Run the full suite
before freezing/launching the protected pipeline.

Freeze source and input hashes once. Lock selected candidates before generating
fresh protected data. Reconstruct all 512 new trajectories independently using
the exact raw scorer; verify labels, model predictions and paired statistics.
Audit must pass before any conditional arena. Keep failed-gate diagnostics.
If any prediction condition fails, stop without protected gameplay. No sample
extension, alternate threshold, champion promotion, retraining or successor.
