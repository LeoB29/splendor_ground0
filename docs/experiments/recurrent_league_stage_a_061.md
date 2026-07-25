# Recurrent Population Self-Play Stage A (`061`)

Frozen after the post-`060` strategic review and before implementation or new
recurrent-policy evidence. This is a CPU-only engineering feasibility smoke.
It performs no fitting, policy update, corpus collection, gameplay strength
gate, promotion, or GPU rental.

## Question

Can the current legal champion be extended with an information-legal recurrent
public-history residual while preserving its epoch-zero policy exactly and
remaining deterministic, checkpointable, batchable, and practical on this
CPU?

This stage validates only the machinery needed by a later, separately
preregistered multi-generation population self-play pilot. Passing does not
claim that recurrence improves play.

## Frozen inputs and architecture

- Parent checkpoint:
  `outputs/legal_action_scorer_035_legal_observation/legal_action_attention_scorer_best.pt`
- Parent SHA-256:
  `4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`
- Public event schema: `splendor-public-action-v1`
- Public-history encoder: the existing `encode_public_history` representation,
  including observer-relative actor sign and no blind-reserve card identity.
- Recurrent encoder: one GRU layer with hidden width `64`, fed by learned action
  type, card, noble, and numeric event projections.
- Policy residual input: frozen `035` state features, contextual legal-action
  features, frozen parent score, and recurrent history context.
- Residual head: hidden width `128`, with the final scalar layer initialized to
  exact zero weights and bias.
- Parent parameters remain frozen. The parent value output is unchanged in
  Stage A.

The model must accept padded batches with variable public-history and
legal-action lengths. Empty history must map to an exact zero context.

## Legal lifecycle contract

The generic match loop may call optional bot lifecycle hooks:

1. `start_game(seed)` exactly once before the first decision;
2. `observe_action(before, action, after)` exactly once on both bots after every
   legal transition, including the transition that causes a forced-pass
   deadlock.

Bots without these hooks are unchanged. A recurrent bot resets its transcript
on `start_game` and appends only `public_action_event(before, action)` on
`observe_action`. Its decision is based only on the acting player's public
observation, legal-action features, and observer-relative public transcript.

## Fixed smoke

Use seeds `610000--610007`, with the recurrent zero-residual model controlling
both seats. Run the same seeds a second time from clean lifecycle state.

For every decision:

- compare recurrent scores and value against the frozen parent on the same
  legal tensors;
- require the same greedy semantic action;
- store a public-action trace digest;
- retain a batchable trajectory input row.

Save the recurrent checkpoint, reload it on CPU, and compare its outputs on the
retained rows. Compare full-history GRU contexts with event-by-event recurrent
contexts. Time frozen-parent and recurrent inference on the same retained
batches after warmup. Timing is descriptive except for the broad feasibility
limit below.

No optimizer step, return target, advantage estimate, or outcome-conditioned
selection is allowed.

## Integrity requirements

Require all:

- exact frozen parent path and SHA-256;
- parent checkpoint kind is `legal_action_attention_scorer`;
- every parent parameter is frozen and every residual/history parameter
  intended for future learning is trainable;
- residual final layer starts at exact zero;
- recurrent and parent scores and values are exactly equal on every smoke
  decision;
- no nonfinite tensor, illegal action, ordinary stall, or hard turn cutoff;
- both runs complete all eight seeds;
- both runs have identical action traces, endings, turns, winners, and scores;
- blind reserve events contain a public tier and no card identity;
- observer-relative encodings of the same event differ only in the legal
  actor-relative feature;
- full and stepwise history contexts agree within `1e-6`;
- padded-batch and single-row greedy choices agree on every retained row;
- checkpoint reload scores and values are exactly equal;
- checkpoint provenance contains the parent path and SHA-256;
- measured recurrent batch inference is no more than `3.0x` frozen-parent
  inference on the same rows.

Any integrity failure yields
`recurrent_stage_a_invalid_no_scale_decision`.

## Decision

Passing decision:
`recurrent_stage_a_feasible_authorize_stage_b_preregistration`.

Failure decision:
`recurrent_stage_a_infeasible_repair_engineering_before_scale`.

A pass authorizes only a separate Stage B protocol for a small
multi-generation learning pilot. It does not authorize the approximately
10,000-game run, promotion, resolver training collection, or GPU rental.
