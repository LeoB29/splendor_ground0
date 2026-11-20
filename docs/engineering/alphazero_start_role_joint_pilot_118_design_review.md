# AlphaZero Start-Role Joint Pilot 118 Design Review

Date reviewed: 2026-08-27

Status: superseded by the frozen protocol
`docs/engineering/alphazero_start_role_joint_pilot_118.md`. The review itself
authorizes no target generation, training, self-play, arena, installation, or
promotion.

## Recommendation

The next experiment should be one bounded, no-new-self-play role-only
distillation pilot. Keep exact generation 11 frozen, add a zero-initialized
two-entry start-role embedding to its shared canonical entity trunk, and train
only that embedding against directly paired original-role and flipped-role
1,024-search targets reconstructed from the existing 114 training games.

This is preferable to another wide residual tower or another ordinary replay
fit. It tests the smallest model change that can remove the exact alias found by
117, preserves the complete gen11 function at epoch zero, affects policy and
value through their shared trunk, and uses paired targets that identify start
role independently of turn parity.

## Evidence controlling the choice

- Experiment 117 found a large causal teacher response when only public start
  role changed: mean visit TV 0.234634, 18.75% visit-argmax changes, and mean
  absolute root-value change 0.312712. Both policy and value gates passed.
- Experiments 114 and 116 already improved the untouched teacher-fit objective
  without converting it into ordinary-budget gameplay. Their arena seat shares
  were 57%/47.25% and 58.75%/44.5%, respectively. More role-blind capacity is
  therefore not the next controlled intervention.
- On 117's original-role holdout, both candidates improved policy loss in both
  role classes. The missing variable is not diagnosed by an aggregate
  start-versus-nonstart loss gap; it is diagnosed by the within-state role-flip
  contrast. The next gate must measure that contrast directly.
- The historical value-only experiment 037 used a zero-initialized start-role
  embedding but failed its terminal-outcome validation gates. That result still
  closes the old value/PUCT leaf line. It does not answer the newly established
  question here: whether a frozen joint gen11 model can absorb paired deep-search
  policy and root-value role contrasts. This justifies one bounded revisit, not
  an embedding or learning-rate sweep.

## Minimal architecture contract

Preserve every v1 encoder and checkpoint unchanged. The new checkpoint kind
should wrap exact gen11 and receive one explicit batch tensor,
`observer_is_start_player`, derived at deployment as
`state.current_player == state.start_player`.

Add `nn.Embedding(2, 96)` to the canonical global entity after `entity_input`
and before the existing entity transformer, matching the already tested v2
value-model placement. Initialize both embedding rows to exact zero. Freeze all
356,450 gen11 parameters and train only the 192 embedding parameters. Because
the shared transformed state feeds both the action context/policy head and the
value head, this is a genuinely joint policy/value intervention rather than a
value-side patch.

Before any optimization, require exact equality with gen11 on ordinary states
and on both orientations of every 117 pair: maximum legal-logit and value
absolute differences no greater than `1e-7`. Also require the frozen parent to
remain non-trainable and checkpoint round trips to preserve this equality.

## Direct paired target block

Do not train from natural replay rows alone. In the 114 corpus, start role is
perfectly confounded with player identity and turn parity, so ordinary rows do
not isolate the public variable whose effect 117 measured.

Instead, reconstruct the exact 224 existing 114 training trajectories and use
the same outcome-blind rule as 117: the latest two states per game before
`pending_round_end`. Subject to exact reconstruction preflight, this yields 448
source states. For each state, compare its original and flipped start role with
exact gen11 at 1,024 simulations, `c_puct=2.25`, real mode, zero root noise, and
one coupled seed. This produces 896 oriented supervised rows without generating
a new game. The policy target is the visit distribution and the value target is
the corresponding search-root value.

Use a frozen whole-game 179/45 train/validation split inside those 224 games.
The exact seed namespace, split seed, optimizer, stopping rule, bootstrap seed,
and shard schedule must be chosen and frozen before generating any target. Do
not select them from candidate results. The existing 64 pairs from the 32-game
114 holdout remain completely excluded from fitting and epoch selection.

## Required offline confirmation

The 117 pair panel is the decisive confirmation set. A future frozen protocol
should require all of the following before gameplay:

1. Exact epoch-zero identity, frozen-parent, input-role, semantic-action, and
   checkpoint-integrity checks pass.
2. A nonzero embedding checkpoint is selected on the 45-game internal
   validation split with parent-safe epoch zero eligible.
3. On all 64 held-out 117 pairs, role-aware raw policy/value predictions improve
   paired teacher-target loss relative to exact gen11 and the strongest frozen
   role-blind 114/116 baseline. Policy and value should be reported separately,
   by role, and with 32-game cluster-bootstrap intervals.
4. The model expresses material within-state role separation in the teacher's
   direction, rather than gaining only through a common shift of both embedding
   rows. Freeze a contrast-retention threshold from baseline-only arithmetic
   before training; do not choose it after seeing the candidate.
5. On the original, reachable 1,956 holdout rows, weighted objective does not
   regress versus gen11 overall or in either role class. Report all 117 phase by
   role cells. Treat this repeatedly inspected holdout as a safety/nonregression
   panel, not as a new unbiased efficacy claim.

The protocol should define paired policy cross-entropy, value MSE, contrast
retention, and their bootstrap gates exactly before evidence. The candidate
must pass both the policy and value uptake conditions because 117 established
material sensitivity in both paths.

## Conditional gameplay and stopping rule

Only a complete offline pass should open a fresh 400-game, 200-paired-seed arena
against exact gen11 at the ordinary 256-simulation budget, with both candidate
seats, zero root noise, exact-prefix resume, and the established practical gate:
at least 55% score share, positive margin, both seats at least 50%, all normal
terminations, and paired CI95 strictly above 50%.

- Offline failure: the 192-parameter role-only bridge did not absorb the causal
  teacher contrast. Stop; do not tune its embedding width, learning rate, or
  pair selection after the result. Review whether a richer role-conditioned
  interaction is warranted.
- Offline pass but gameplay failure: start-role omission is real but not
  sufficient to explain the compression failure. Close the simple role repair
  and return to training-objective/multi-generation design review.
- Full gameplay pass: authorize only review of a separately preregistered short
  continuation. Do not install or compare with `035` automatically.

## Rejected immediate alternatives

- Do not widen the 116 residual tower again; capacity already passed offline
  and failed gameplay.
- Do not mutate the shared v1 numeric tensor in place; that weakens checkpoint
  and encoder compatibility. Pass role explicitly in the new model kind.
- Do not train the whole gen11 network in this first test; that would confound a
  192-parameter representation repair with another ordinary distillation run.
- Do not fit on the 117 pairs; they are the only direct held-out role-flip
  confirmation block.
- Do not launch an arena before direct paired uptake is demonstrated.

## Immediate next action

Draft and freeze the exact Experiment 118 protocol, including power/threshold
arithmetic, hashes, seed namespaces, and terminal decisions. Then implement the
new checkpoint/inference path, paired-target builder, sealed resume, and focused
identity/role/holdout tests. Stop after verified preflight and request explicit
authorization before generating targets or starting training.
