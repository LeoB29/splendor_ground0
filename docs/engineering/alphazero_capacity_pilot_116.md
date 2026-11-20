# AlphaZero Identity-Preserving Medium-Capacity Pilot 116

Date frozen: 2026-08-27

## Purpose

Experiment 113 established that exact generation 11 at 1,024 simulations is a
stronger teacher than the same network at 256. Experiments 114--115 showed that
the compact 356,450-parameter network can improve offline fit to those targets
but does not convert them into reproducible gameplay. Experiment 116 tests the
specific capacity hypothesis without discarding generation 11 or generating
another expensive corpus.

## Frozen inputs

- completed 114 result SHA-256:
  `dd8bbaa7876b881db69d44fb4242624e1f227981b4628da392b698491c231ad5`;
- completed 115 result SHA-256:
  `02d4d1d2557c57c6778b92950fb4c06d02e4112dc83d45d20791f638df0db993`;
- exact generation-11 checkpoint SHA-256:
  `c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0`;
- retained installed generation-2 checkpoint SHA-256:
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`;
- 114 train/holdout manifest SHA-256:
  `61ef0dd0a4b0f83e2a9aba7cd3cc83d9e62f7770798287b7d2f672bacdb1c4dc`;
- 224-game training replay SHA-256:
  `59f1dfeb36fafc4af8b6fcf8b90b78f91a751be86ef4328ba0067c9514b84c7a`;
- untouched 32-game holdout SHA-256:
  `33bb2d8f92d418f62063864f1fea7473ea5b323f1005e887aee01ed61c78be88`.

Verify the exact completed decisions, failed pooled 115 gate, gen11 lineage,
retained installation, corpus hashes, game counts, and disjoint seed sets.

## Architecture bridge

Keep exact generation 11 as a permanently frozen 356,450-parameter parent.
Add one independent canonical entity/action policy-value correction tower with
128-wide entity and action streams, 192-wide scorer, two entity layers, two
action layers, four heads, 256-wide feed-forward layers, 24-wide ID embeddings,
and zero dropout. It has 635,378 trainable parameters, for 991,828 total.

Zero-initialize the correction tower's final policy and value linear layers.
Before training, audit the internal validation split and require maximum legal
logit and value absolute differences from exact gen11 no greater than `1e-7`.
The frozen parent must remain non-trainable. The composite policy adds the
correction logits; the composite value adds and clamps the correction to
`[-1,1]`.

## Training and untouched offline gate

Reuse only the exact 224-game / 13,960-position training partition. Split whole
games 80/20 with seed `1116000`. Train only the residual tower for at most 12
epochs with batch size 256, AdamW learning rate `1e-4`, weight decay `1e-4`,
policy/value weights 1/0.5, gradient clipping 5, no policy-divergence or argmax
auxiliary loss, patience 3, and minimum validation improvement 0.001. Exact
epoch zero remains eligible for parent-safe selection.

Evaluate the selected checkpoint once on the untouched 32-game / 1,956-position
holdout. Qualify only if identity and frozen-parent checks pass, total capacity
is at least 2.5 times the parent, a nonzero epoch is selected, policy loss and
weighted objective both improve, value loss regresses by at most 0.025, and all
reported metrics are finite. If this gate fails, stop without gameplay.

## Conditional ordinary-budget primary

Only a qualified candidate faces exact generation 11. Both agents use 256
simulations, `c_puct=2.25`, real mode, zero root noise, and operative search
seed `2 * game_seed + seat`. Use seeds `2160000..2160199`, both candidate seats,
for 400 games in four concurrent one-thread shards of 50 paired seeds.

Every shard writes a protocol/checkpoint/search/schedule sidecar before its
first game and resumes only by an exact completed-game prefix. Combine only the
exact 400-game schedule and report W-D-L, share, margin, both seats, termination
reasons, and a 50,000-sample paired bootstrap using seed `1116100`.

Gameplay confirms capacity compression only at score share at least 55%,
positive mean margin, both seats at least 50%, all normal terminations, and a
paired CI95 lower bound strictly above 50%.

## Decisions and exclusions

An offline failure triggers objective/bridge review. Offline success but
gameplay failure triggers representation-design review. A full gameplay pass
authorizes only review of a short continuation using the capacity architecture.

No branch authorizes automatic self-play, `035`, installation, promotion, or
continuation. Exact generation 2 remains installed. Execution is crash-resumable
at sealed training/audit stages and exact arena-game prefixes.
