# AlphaZero Start-Role Interaction Audit 119

Status: frozen before 119 evidence

Date frozen: 2026-08-28

## Question

Experiment 118 confirmed a large paired search-teacher start-role signal but its
zero-initialized two-row global-entity embedding improved validation objective by
only 0.0480%. Determine, without training, whether a minimal direct
identity-preserving readout interaction has materially better and more stable
first-order access to the paired role signal.

## Frozen inputs

- exact generation-11 checkpoint SHA-256
  `c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0`;
- finalized 118 result SHA-256
  `4846b79e1fbcd6344e759155befc469463a9100423f8785ff0e352fa9127fbc1`;
- 118 decision
  `minimal_role_bridge_failed_direct_uptake_review_richer_interaction_no_gameplay`;
- sealed 118 training-target manifest SHA-256
  `278d22ab7497c0b4921c86a700c5fc88c3ba1d73982ff645ccbb169d499df79f`;
- exact 114 224-game training replay SHA-256
  `59f1dfeb36fafc4af8b6fcf8b90b78f91a751be86ef4328ba0067c9514b84c7a`.

The audit reuses all 448 paired states / 896 oriented teacher rows from 118 and
the exact frozen 179/45 whole-game split. It must not read or search the
protected 117 confirmation states or the 114 holdout replay.

## Architectures at exact identity

The exact gen11 parent is frozen in every arm. Start role is represented as
`s=-1` for a non-start-player observation and `s=+1` for a start-player
observation. Every new parameter is initialized to zero.

1. `global_antisymmetric_118_channel` is the diagnostic baseline. A single
   96-vector `g` is injected as `s*g` into the global entity before the frozen
   entity transformer, matching the role-contrast channel available to 118
   while excluding its common-mode degree of freedom. It has 96 trainable
   parameters and still feeds both heads through the shared pooled state.
2. `split_readout_additive` is the minimal candidate. Independent 96-vectors
   `p` and `v` form policy and value states `h+s*p` and `h+s*v` after the frozen
   entity encoder and pooling. The policy state feeds both the frozen
   state-to-action context and score head; the value state feeds only the frozen
   value head. It has 192 trainable parameters, is antisymmetric by construction,
   and has structurally disjoint endpoint parameters.
3. `split_readout_film` is a fallback only. Per endpoint, zero-initialized
   96-vector gain and bias parameters form
   `h*(1+s*gain)+s*bias`. It has 384 trainable parameters and the same strict
   endpoint separation.

No architecture may alter v1 encodings, parent parameters, head parameters, or
checkpoint files.

## No-training measurements

For every arm:

- verify exact epoch-zero logits and values against gen11 on all 896 rows with
  maximum absolute error at most `1e-7`;
- verify the declared trainable count, all-zero new parameters, frozen parent,
  finite outputs, and deterministic repeatability;
- compute paired role-contrast losses at identity. Policy contrast loss is the
  mean squared error between candidate and teacher start-minus-nonstart action
  probability differences, averaged over semantic actions and pairs. Value
  contrast loss is the mean squared error between candidate and teacher
  start-minus-nonstart root-value differences;
- compute, without applying an update, the gradient of each contrast loss on
  the 358-pair training split, 90-pair validation split, and all 448 pairs;
- report gradient L2 norm, per-parameter RMS, finite/nonzero status, and
  train-versus-validation cosine for each endpoint;
- verify for split designs that policy contrast loss has exactly zero gradient
  in value-only parameters and value contrast loss has exactly zero gradient in
  policy-only parameters.

The runner must not instantiate an optimizer, call backward followed by a
parameter mutation, save a candidate checkpoint, generate a search target, run
self-play, or run an arena.

## Frozen decision rule

First test `split_readout_additive`. It qualifies only if all integrity checks
pass and, for both policy and value:

- the all-pair contrast gradient is finite and nonzero;
- train-versus-validation gradient cosine is strictly positive;
- all-pair per-parameter gradient RMS is strictly greater than the corresponding
  `global_antisymmetric_118_channel` value;
- cross-endpoint gradient leakage is exactly zero.

If additive qualifies, recommend it and do not prefer the larger FiLM arm even
if FiLM has larger gradients. If additive fails, apply the same gates to
`split_readout_film`; recommend FiLM only if it passes. If neither split design
passes, close start-role-only repair and return to broader representation work.

A recommendation authorizes only a separately reviewed and frozen pilot. It
does not authorize fitting, new targets, self-play, confirmation evaluation,
gameplay, external `035`, installation, promotion, or automatic continuation.
