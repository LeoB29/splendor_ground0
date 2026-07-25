# Recurrent Population Self-Play Stage B (`062`)

Frozen after the successful `061` engineering smoke and before implementation
or any recurrent-policy learning evidence. This is a bounded CPU-only learning
feasibility pilot. It is not the approximately 10,000-game scale experiment and
cannot directly promote a policy.

## Question

Can a history-aware residual initialized at exact champion `035` complete
three synchronous on-policy generations while:

- learning from complete final outcomes rather than a search teacher;
- retaining each immediate parent;
- accumulating useful movement rather than remaining pinned to `035`; and
- avoiding collapse against the fixed broad league?

The experiment tests whether population self-play can cross the roadmap's
warm-start-to-iteration boundary. It does not test final playing strength.

## Frozen prerequisites

- `061` result:
  `outputs/recurrent_league_stage_a_061/result.json`
- `061` result SHA-256:
  `1b1469fca9b7abcc22525e6f57a2fcdf9091863efe80bbb5e862f6deb1665f08`
- Epoch-zero recurrent checkpoint:
  `outputs/recurrent_league_stage_a_061/recurrent_policy_epoch0.pt`
- Epoch-zero SHA-256:
  `2d807231666420bb221049aa7e2a4a28358ef62618cf1c652ede67dd7249acef`
- Frozen `035` SHA-256:
  `4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`
- Frozen `016` SHA-256:
  `876ac23a005d367ac48b3619bb7cdcb6e0230a3e9e97943d7eecdc6d07d1b056`
- Frozen clean `023` SHA-256:
  `29ef7c13c5553ec88c8bcb98f7378c925f0e600e2fcf3d944df5926efed7a070`
- Frozen clean `023` checkpoint:
  `outputs/canonical_entity_coherent_023/`
  `canonical_entity_action_scorer_coherent_width64_best.pt`

The first pre-evidence invocation exposed that the original protocol listed
the SHA-256 of the unrelated legacy attention experiment numbered `023` while
calling it the clean canonical `023`. It stopped at checkpoint construction,
before completing collection, training, or writing any evidence. This
prospective identity repair binds the already intended and historically frozen
clean canonical checkpoint above; no result or learning statistic was
available when repaired.

All acting-policy inputs use only the legal public observation, legal-action
features, and `splendor-public-action-v1` transcript established by `061`.
The embedded `035` parent remains frozen in every generation.

## Generations and league

Run at most three conditional generations, each with exactly 240 eligible
games. The actor controls one seat per game. Every opponent/actor-seat cell is
exactly balanced.

Static league members:

1. raw legal `035`;
2. `016`;
3. clean canonical `023`;
4. information-legal shallow depth-one search with the frozen operational
   contract: branching `6`, buys `4`, reserves `2`, takes `2`,
   determinizations `1`, search seed `23`, determinization seed `29`.

Generation 1 uses the four static members. Generation 2 adds accepted
generation 1. Generation 3 adds accepted generations 1 and 2. Thus historical
snapshots remain explicit league opponents rather than being overwritten.

Collection seed starts are `620000`, `621000`, and `622000`. Schedule games
round-robin by opponent and actor seat. A game is eligible only when it ends
`completed` or `forced_pass_deadlock`. Eligibility is outcome-blind. If a
scheduled seed is ineligible, continue within the generation's reserved
1,000-seed band for the same cell until exactly 240 eligible games exist;
attempts may not exceed 300. Ineligible games never enter training.

Use maximum turn `400`, repetition cutoff `4`, and no-progress cutoff `60`.
Forced-pass deadlocks are valid natural endings.

## Behavior and trajectory target

On actor turns sample from:

`0.95 * softmax(scores / 1.0) + 0.05 * uniform(legal actions)`.

Sampling seeds are deterministic functions of generation, game, and actor
decision. Store the complete public-history decision input, legal-action set,
chosen row, behavior scores/probabilities, and final result.

For actor perspective:

- `WDL = +1 / 0 / -1`;
- `margin = clip((actor score - opponent score) / 20, -1, +1)`;
- game return = `0.5 * WDL + 0.5 * margin`.

Within each opponent/seat cell, normalize game returns to zero mean and unit
standard deviation. Every decision from a game receives that game advantage,
but decision weights sum to one per game. This prevents longer and later games
from dominating. No privileged critic, handcrafted value, search value, or
temporal reward redistribution is used.

Within every cell, reserve the last 20% of eligible scheduled games as
game-disjoint validation; use the rest for training. Each cell must contribute
at least four validation games. Normalize training and validation advantages
within their own split and cell.

## Update

Start each generation from its immediate accepted parent. Only recurrent
history and residual parameters are trainable; embedded `035` parameters
remain frozen.

Use clipped on-policy regression:

- four epochs maximum;
- AdamW learning rate `3e-4`, weight decay `1e-4`;
- batch size `128`;
- clip ratio `0.20`;
- entropy coefficient `0.002`;
- gradient norm cap `1.0`;
- deterministic shuffle seed `620062 + generation * 100 + epoch`.

After each epoch measure raw policy KL on untouched validation decisions.
Stop when it reaches `0.008`. If the final proposal exceeds `0.008`,
deterministically interpolate every trainable parameter toward the immediate
parent by bisection and choose the largest interpolation fraction whose
validation raw KL is at most `0.008`. Never alter embedded `035`.

The validation surrogate is the game-balanced mean of
`advantage * log(candidate_probability / parent_probability)`.

## Per-generation gates

Require all before accepting a snapshot:

- exact league/opponent/seat and train/validation balance;
- exactly 240 eligible games and no more than 300 attempts;
- zero illegal actions and no ineligible game in either split;
- behavior log-probability reconstruction maximum error at most `1e-5`;
- finite losses, gradients, probabilities, returns, and diagnostics;
- nonzero first gradient norm;
- every non-parent parameter remains trainable and every embedded parent
  parameter remains frozen;
- validation raw KL in `[0.0001, 0.0100]`;
- validation argmax movement in `[0.005, 0.12]`;
- strictly positive validation surrogate;
- validation normalized entropy at least `0.30`;
- checkpoint reload exact on validation tensors.

Then run an untouched immediate-parent gate on 20 paired seeds (40 games),
using seed starts `623100`, `623200`, and `623300`. Require:

- all endings valid;
- candidate paired score share at least `0.45`;
- mean candidate-minus-parent paired score margin at least `-2.0`.

If any generation fails, stop without constructing the next generation.

## Final population gate

Only after all three generations pass, compare generation 3 and epoch zero
against every static league member on the same ten fresh paired seeds per
opponent. Seed starts are `624000`, `624100`, `624200`, and `624300`.

Require:

- all 160 games end validly;
- aggregate generation-3 minus epoch-zero paired score-share delta is strictly
  positive;
- at least two of four opponent deltas are nonnegative;
- no opponent delta is below `-0.10`;
- on generation-3 validation states, final-versus-epoch-zero raw KL lies in
  `[0.0005, 0.025]`;
- final-versus-epoch-zero argmax movement lies in `[0.01, 0.25]`.

No confidence-interval claim is made at this scale.

## Decisions

Integrity failure:
`recurrent_stage_b_invalid_no_learning_decision`.

Generation failure:
`recurrent_stage_b_generation_failed_diagnose_before_scale`.

All gates pass:
`recurrent_stage_b_iterates_authorize_stage_c_scale_preregistration`.

A pass authorizes only a separately preregistered scaled population-self-play
run. It does not promote generation 3, authorize resolver collection, or
justify GPU rental by itself.
