# AlphaZero Value-Uncertainty Audit 130

Date frozen: 2026-08-29

## Purpose

Experiments 110--129 repeatedly improved ordinary offline policy/value losses
without establishing a reliable gameplay successor. Experiment 113 separately
proved that 1,024-simulation search is stronger than 256-simulation search.
Experiment 129 then broadened the starting-state population and still stopped
at 53.125% against the experiment-114 control.

Experiment 130 is a no-training audit of the remaining value-interface
hypothesis. It asks whether one terminal WDL sample per visited state is mostly
noise around a broad outcome distribution, and whether that uncertainty comes
primarily from the hidden/future possible world or from search-policy
variation. A scalar expected value is theoretically sufficient for risk-neutral
search when it is known accurately; this audit does not assume otherwise. It
tests whether the present one-sample training interface makes that expectation
unnecessarily difficult to learn.

## Frozen inputs

- experiment-129 result SHA-256:
  `75cf5857ae833ba1c5609002b6d93f76660fadd3be20d2ecc98c30146c20f4fa`;
- exact generation-11 checkpoint SHA-256:
  `c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0`;
- experiment-114 terminal-control checkpoint SHA-256:
  `93e548a35f2a42194c70602928fa1ae9398fe042dbad2b2c876694fa97a9c4bd`;
- experiment-114 32-game ordinary holdout SHA-256:
  `33bb2d8f92d418f62063864f1fea7473ea5b323f1005e887aee01ed61c78be88`;
- retained installed generation-2 checkpoint SHA-256:
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`.

Verify every hash, the completed 129 decision, unchanged installation, exact
generation/checkpoint identity, normal source games, contiguous turns, legal
selected actions, and exact canonical tensors before evidence.

## State panel

Use the first 24 games of the sealed experiment-114 ordinary holdout in replay
order. Assign phases cyclically `early`, `middle`, `late`, producing eight
states per phase. Select the nearest non-`pending_round_end` replay state to
25%, 50%, or 75% trajectory progress, with ties toward the earlier state.

These games were not used to train generation 11 or the experiment-114
control. Their ordinary offline losses have been opened by prior experiments,
but no outcome in this audit and no repeated continuation from these selected
states has been used for training or model choice.

## Crossed repeated continuations

For each of the 24 observable states, construct four deterministic possible
worlds by information-legally resampling the root player's unknown opponent
blind reserves and hidden deck order. Use world seed namespace `113000000`.
All four worlds must have the exact same canonical public-belief key, legal
semantic root actions, canonical entity tensor, and raw generation-11 and
control predictions.

Cross the four worlds with four continuation-policy seed slots, yielding 16
continuations per state and 384 total. Exact generation 11 controls both seats
with 256 simulations, `c_puct=2.25`, real information-legal mode, zero root
noise, deterministic root choice, and maximum turn 200. The same policy seed
is coupled across the four worlds. Use policy seed namespace `113100000` and
the established repetition, no-progress, forced-pass, and adjudication
contracts. Every continuation must complete normally; an abnormal continuation
is an integrity failure, not an outcome.

For each policy seed, run the public root once at 256 simulations and reuse its
semantic first action across the four possible worlds. Also run the same four
public-root seeds at 1,024 simulations without playing them. Store raw network
value, search-root value, visit distribution, and chosen semantic action for
both budgets. This measures whether deeper search stabilizes the estimated
expectation and action while the continuation matrix measures actual outcome
spread.

## Frozen metrics

From the root player's perspective store terminal WDL, final score margin,
world slot, policy slot, phase, blind-reserve status, and normal-termination
telemetry.

For every state and for the full panel report:

- win/draw/loss frequencies, WDL mean, variance, entropy, and whether both a
  win and a loss occurred;
- score-margin mean and standard deviation;
- absolute error and MSE of generation-11 raw value, experiment-114 control
  raw value, mean 256-search value, and mean 1,024-search value against the
  16-continuation empirical WDL mean;
- root-value standard deviation and semantic-action agreement at 256 and
  1,024 simulations;
- the one-way state/within-state WDL sum-of-squares decomposition, reported as
  the fraction of individual terminal-target variance lying within a state;
- a balanced two-way decomposition within states into possible-world main
  effect, policy-seed main effect, and residual interaction;
- phase and opponent-blind-reserve descriptive breakdowns.

Bootstrap state clusters with 50,000 samples for the panel mean within-state
WDL variance and within-state target-noise fraction. All selection and gates
use only the frozen aggregate panel, never a phase or reserve subgroup.

## Decisions

Terminal-value distribution is considered materially under-observed only if
all of the following hold:

- mean within-state WDL variance is at least 0.50;
- at least 50% of states contain both a win and a loss;
- at least 65% of individual terminal-target sum of squares lies within state;
- the bootstrap CI95 lower bounds are at least 0.40 for mean within-state WDL
  variance and 0.55 for the within-state target-noise fraction.

Possible-world uncertainty is considered the primary component only if its
aggregate two-way sum-of-squares fraction is at least 0.40 and at least 0.10
larger than the policy-seed fraction.

- Integrity failure:
  `value_uncertainty_integrity_failed_no_model_conclusion`.
- Material terminal-target noise with primary possible-world uncertainty:
  `possible_world_value_distribution_material_freeze_distributional_pilot`.
- Material terminal-target noise without primary possible-world uncertainty:
  `trajectory_policy_value_uncertainty_material_review_joint_target`.
- Terminal-target noise gate fails:
  `scalar_terminal_value_uncertainty_not_primary_close_compact_alphazero`.

A distributional decision authorizes only a separately frozen minimal
identity-preserving value-head pilot. It does not authorize training here,
belief features, gameplay, installation, or promotion. Belief-aware features
may be considered later only if the prespecified blind-reserve breakdown is
large enough to motivate its own fresh confirmation; that subgroup is not a
gate in this audit.

## Launch guard

Default invocation verifies inputs, reconstructs all 24 states, checks the four
possible-world invariants for every state, and runs one temporary 8-simulation
root search that is not persisted as evidence. It instantiates no optimizer,
trains no checkpoint, and plays no continuation. The audit requires explicit
`--launch`. The owner's 2026-08-29 instruction to do experiment 130 authorizes
launch only after focused tests and preflight pass.
