# AlphaZero Repeated-Return Value Target A/B 131

Date frozen: 2026-08-29

## Purpose

Experiment 130 established that a single terminal WDL label materially
under-observes the return distribution of an observable state: 83.33% of its
states produced both wins and losses and 73.34% of terminal-target variation
lay within state. Possible-world variation alone was not primary; the largest
component was its interaction with the subsequent search-policy trajectory.

Experiment 131 is the minimal causal test of whether repeated-return averaging
actually improves the scalar value model used by AlphaZero search. Two
identity-matched children use the same states, architecture, initialization,
optimizer, and value-only trainable scope. The control learns from one fixed
continuation outcome per state. The treatment learns from the arithmetic mean
of all repeated outcomes for that state. The policy and shared representation
remain exact generation 11.

This is not a distributional head, belief model, action-Q model, search-value
blend, or general joint policy/value update.

## Frozen inputs

- experiment-130 result SHA-256:
  `cc884522a98a6b45105d28a8d878bd31267ad6504f444b14cb9d1dfd0df4d417`;
- exact generation-11 checkpoint SHA-256:
  `c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0`;
- retained installed generation-2 checkpoint SHA-256:
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`;
- generation-2 source replay SHA-256:
  `855b9ae3cb4e05da7b26b759ce8fa4e62c8c44aea4a0dcee2da0e01a1790de56`;
- generation-5 source replay SHA-256:
  `0f9393fb020cf01c4d06f829c03e7426913dcc29d6afde49091cb9def57772e1`;
- generation-8 source replay SHA-256:
  `96f0d346743f30bd29e329f4bf3e2a7a361e778391c6d6e5833ca2a06af9be10`;
- generation-11 source replay SHA-256:
  `59f1dfeb36fafc4af8b6fcf8b90b78f91a751be86ef4328ba0067c9514b84c7a`.

Verify every hash, the completed 130 decision, unchanged installation, exact
source/checkpoint identity, normally completed source games, contiguous turns,
legal selected actions, and exact canonical tensors before evidence.

## State schedule

Use the same four source populations as experiment 129, but only source-game
ordinals untouched by its training attempts and gameplay panel.

- Repeated-target training/development block: source ordinals 112--171,
  inclusive, 60 states per source and 240 total. Within every source, phases
  cycle `early`, `middle`, `late`, producing 20 states per source/phase cell.
  The first 16 in each cell are fitting states and the last four are
  development states: 192 fitting and 48 development states total.
- Protected repeated-return block: source ordinals 172--186, inclusive, 15
  states per source and 60 total. Phases cycle independently, producing five
  states per source/phase cell.

For each game select the nearest non-`pending_round_end` replay state to 25%,
50%, or 75% trajectory progress, ties toward the earlier state. Every source
game is used once. Selection is outcome-blind.

## Repeated continuations

Exact generation 11 controls both seats with 256 simulations, `c_puct=2.25`,
real information-legal search, zero root noise, deterministic root choice, and
maximum turn 200. Use the established repetition, no-progress, forced-pass,
and normal-completion contracts.

For every training/development state cross four deterministic possible worlds
with two coupled policy-search seed slots, yielding eight continuations. For
every protected state cross four worlds with four policy slots, yielding 16.
Use possible-world seed namespace `113300000` and policy seed namespace
`113400000`. Cache each public-root policy-slot action once and rebind the same
semantic action across its four worlds. Every continuation must end normally;
an abnormal continuation is an integrity failure, not a label.

The total is exactly 2,880 continuations: 1,920 in the training/development
block and 960 protected. Collection is atomic by state, exactly resumable, and
hashed. The protected block may be collected concurrently but its targets and
metrics must not be loaded before both arm checkpoints are selected and sealed.

## Matched value-only training

Both arms start from exact generation 11. Freeze exactly 351,745 of 356,450
parameters. Only the existing 4,705-parameter `value_head` is trainable; every
policy logit and shared representation remains bit-exact.

- single-outcome control target: WDL from world slot 0, policy slot 0;
- repeated-mean treatment target: arithmetic mean WDL across all eight
  continuations;
- fitting states: 192;
- development states: 48;
- development metric for both arms: MSE against the eight-continuation mean;
- epoch-zero parent is eligible;
- up to 50 epochs, batch 64, AdamW learning rate `1e-4`, weight decay `1e-4`,
  patience 8, minimum development-MSE improvement `1e-4`;
- common initialization/training seed `1131000` and deterministic ordering.

Select each arm independently by development repeated-mean MSE. Seal both
checkpoints before the protected target file is opened. Verify exact frozen
parameters, exact policy logits on all 60 protected states, finite values, and
checkpoint round-trip.

## Protected repeated-return gate

Evaluate generation 11, the single-outcome control, and the repeated-mean
treatment against each protected state's 16-continuation empirical WDL mean.
Report MSE, MAE, Pearson correlation, phase/source breakdowns, and paired
state-cluster bootstrap intervals with 50,000 samples.

Gameplay opens only if all checks pass:

- the treatment selects a nonzero epoch;
- all integrity, identity, frozen-parameter, policy-logit, round-trip, and
  finite-metric checks pass;
- treatment MSE is at most 90% of control MSE and strictly below generation-11
  MSE;
- treatment MAE is no worse than control MAE;
- the paired bootstrap CI95 upper bounds for treatment-minus-control and
  treatment-minus-generation-11 squared error are both strictly below zero;
- in every phase, treatment MSE is at most control MSE plus 0.02.

## Conditional gameplay

Only a protected offline pass opens gameplay. All agents use 256 simulations,
`c_puct=2.25`, real mode, zero root noise, and maximum 200 turns.

First run repeated-mean treatment versus the single-outcome control on 200
paired fresh initial seeds `2730000..2730199`, both seats, for 400 games. Pass
requires every game to complete normally, treatment score share at least 55%,
positive margin, at least 50% from each seat, and paired-bootstrap CI95 lower
bound strictly above 50%.

Only that pass opens treatment versus exact generation 11 on 200 paired seeds
`2731000..2731199`, both seats, with the same gates. Four exact workers and
50,000 paired bootstrap samples are used per panel.

## Decisions

- Integrity or protected-access violation:
  `repeated_return_value_integrity_failed_no_strength_conclusion`.
- Protected repeated-return gate failure:
  `repeated_return_value_target_not_causally_better_no_gameplay`.
- Treatment/control gameplay failure:
  `repeated_return_value_offline_gain_not_search_causal`.
- Treatment/control pass but generation-11 failure:
  `repeated_return_value_gain_not_parent_robust`.
- Both gameplay panels pass:
  `repeated_return_value_causal_authorize_joint_reanalysis_design_no_install`.

No result installs or promotes a checkpoint, consults `035`, changes the
policy, adds belief features, or automatically starts a joint training loop.

## Launch guard

Default invocation verifies all inputs, reconstructs all 300 states and their
exact split, validates possible-world/public-prediction invariants, checks the
4,705-parameter trainable scope and exact epoch-zero identity, and runs one
temporary 8-simulation root search. It instantiates no optimizer, writes no
durable continuation, trains no checkpoint, and plays no strength game. The
long run requires explicit `--launch`. The owner's 2026-08-29 approval of
experiment 131 authorizes launch only after focused tests and preflight pass.
