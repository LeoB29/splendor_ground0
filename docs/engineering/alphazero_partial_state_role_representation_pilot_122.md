# AlphaZero Partial State-Role Representation Pilot 122

Date frozen: 2026-08-28

## Purpose

Experiment 117 proved that exact generation-11 deep search changes when only the
public start role changes. Experiments 118, 120, and 121 then tested increasingly
expressive role adapters around a completely frozen role-blind network. Their
development-objective improvements were 0.0480%, 0.2335%, and 0.2960%, all below
the prospectively fixed 0.5% gate. Those results close small frozen-parent
adapters.

Experiment 122 is the smallest credible representation pilot beyond that closed
family. It gives the state encoder an explicit signed start-role input and lets
only its final entity-transformer block adapt jointly. It is one bounded pilot,
not a sweep or an automatic continuation.

## Frozen inputs

- completed 121 result SHA-256
  `f871427980697f4036c84b7617ba671db801563af6d6a0434f698a7d473d4e66`;
- exact generation-11 checkpoint SHA-256
  `c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0`;
- role-blind candidates 114 and 116 SHA-256
  `93e548a35f2a42194c70602928fa1ae9398fe042dbad2b2c876694fa97a9c4bd`
  and
  `bb3bf65b102e5ec4c46a351c3958dba250c0f4c94f240847b7c3bc5871a57565`;
- exact 114 partition manifest, 224-game replay, and 32-game holdout SHA-256
  `61ef0dd0a4b0f83e2a9aba7cd3cc83d9e62f7770798287b7d2f672bacdb1c4dc`,
  `59f1dfeb36fafc4af8b6fcf8b90b78f91a751be86ef4328ba0067c9514b84c7a`,
  and
  `33bb2d8f92d418f62063864f1fea7473ea5b323f1005e887aee01ed61c78be88`;
- sealed 118 fitting-target manifest SHA-256
  `278d22ab7497c0b4921c86a700c5fc88c3ba1d73982ff645ccbb169d499df79f`;
- retained installed generation-2 checkpoint SHA-256
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`.

Before evidence, verify those hashes, the finalized 121 decision, the inherited
117--121 lineage checks, exact target/action schedules, disjoint partitions, and
the retained installation.

## Architecture and trainable scope

Start from exact generation 11. Let signed public role `s` be `+1` when the
observer is the start player and `-1` otherwise. Add one zero-initialized
96-wide vector `r` only to the canonical global entity immediately before the
two-layer entity transformer: `global_input = global_input + s*r`.

Keep the first entity-transformer block and every other parent component frozen:
entity input projection and ID embeddings, action encoder and state-to-action
context, policy scorer, and value head. Jointly train only `r` and the complete
final entity-transformer block. The final block has 74,784 parameters; therefore
74,880 parameters are trainable, total capacity is 356,546, and 281,666 parent
parameters remain frozen.

The zero role vector and copied final block must reproduce exact generation-11
legal logits and values within `1e-7`, live and after checkpoint reload. Preflight
must prove that the exact trainable-name set is the role vector plus final block,
that policy and value losses both produce nonzero gradients in both trainable
parts, that frozen parameters receive no gradients, and that this audit does not
mutate any parameter. Shared policy/value gradients are intentional because this
experiment trains a joint state representation.

## Existing targets and development fitting

Reuse and fully validate the sealed 448-pair / 896-row target panel. Generate no
new fitting target and no self-play. Reuse the exact 179/45 whole-game split:
716 fitting rows and 180 development rows. This design-exposed development split
is only a gate to protected confirmation.

Use policy cross-entropy plus `0.5` search-root-value MSE, 12 epochs, batch size
128, AdamW learning rate `1e-4`, weight decay `1e-4`, gradient clipping 5, and
shuffle seed `1118118`. Select the earliest minimum-development-objective epoch;
epoch zero is retained as the safe fallback. This schedule is fixed from the
proven parent-safe full-network/capacity recipe, with no sweep.

Require a nonzero selected epoch, at least 0.5% relative development-objective
improvement, nonzero role-vector norm, nonzero final-block movement, exact
preservation of every frozen parent parameter, finite metrics, and the exact
trainable scope. Failure stops before protected confirmation.

## Protected confirmation and safety

Only a development pass may regenerate the disjoint 64-state 117 original/flipped
teacher panel: exact generation 11, 1,024 simulations, `c_puct=2.25`, real mode,
zero root noise, maximum depth 96, and coupled seed
`1117000 + 100 * panel_index`. Reproduce 117's teacher summary within `1e-9`.

Against the lowest-loss role-blind baseline among generation 11, candidate 114,
and candidate 116, require candidate policy CE and value MSE improvements overall
and in both role classes. Require 50,000-sample 32-game bootstrap CI95 upper
bounds below zero for policy, value, and weighted objective using seeds
`1122120..1122122`; raw paired policy TV at least 0.025 with CI95 lower bound
above 0.01; raw absolute value difference at least 0.05 with CI95 lower bound
above 0.01; nonzero role input; and finite metrics.

Only confirmation pass opens the 1,956-row reachable-state safety panel.
Candidate-minus-generation-11 weighted objective may be no greater than `+0.005`
overall, for start role, and for non-start role.

## Conditional gameplay

Only full development, confirmation, and safety passes open candidate 122 versus
exact generation 11 at 256 simulations, `c_puct=2.25`, real mode, zero root
noise, and operative seed `2 * game_seed + seat`. Use fresh seeds
`2220000..2220199`, both seats, 400 games in four one-thread shards.

Require score share at least 55%, positive mean margin, both seats at least 50%,
all normal terminations, and a 50,000-sample paired-bootstrap CI95 lower bound
strictly above 50% using seed `1122200`.

## Decisions and exclusions

- Development failure:
  `partial_state_representation_failed_development_no_protected_evaluation`.
- Protected confirmation failure:
  `partial_state_representation_failed_protected_confirmation`.
- Safety failure:
  `partial_state_representation_confirmed_but_reachable_regression_no_gameplay`.
- Gameplay failure:
  `partial_state_representation_offline_gain_not_converted_no_install`.
- Gameplay pass:
  `partial_state_representation_confirmed_review_role_aware_continuation_no_install`.

Do not change the epochs, rate, threshold, trainable scope, split, or search
settings after evidence. Do not tune a failed checkpoint. No result authorizes
new self-play, external `035`, installation, promotion, or automatic
continuation. Exact generation 2 remains installed.

The runner is preflight-only without `--launch`. Preflight may validate existing
fitting targets and identity/gradient/round-trip contracts, but may not optimize,
generate protected targets, search, or play. The user's 2026-08-28 instruction
to do the next suggested step authorizes launch after successful preflight.
