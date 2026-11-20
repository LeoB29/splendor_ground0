# From-Scratch Resolver Behaviour-Cloning Pilot (`075`)

Frozen on 2026-08-03 before implementing a `075` dataset, trainer, evaluator, or
opening any `075` model outcome. This is an offline CPU-only learnability test on
existing `052` resolver rows. It generates no games, touches no human replay,
promotes no checkpoint, and authorizes no gameplay by itself.

## Question and relation to failed `052`--`054`

The resolver is the only demonstrated high-dose stronger operator, but every
prior student was warm-started from and/or anchored to `035`. `052a` showed that
the warm student mostly sharpened distributions without changing the inherited
ranking. `075` asks a genuinely different question: can a randomly initialized,
canonical entity/action model learn the resolver's selected semantic action from
hard one-hot labels, without a `035` anchor or soft target?

This does not reinterpret any failed gate. `052`, `053`, and `054` remain failed.
It changes initialization, representation, label semantics, loss, and primary
endpoint, and uses no new teacher outcome to choose them.

## Frozen data and privacy

Use only completed-search (`target_available=true`) rows from:

- training: `outputs/resolver_selfplay_distillation_052/train_targets.jsonl`,
  SHA-256 `6827848ca5773799a698c62712ee04f3c40b2e2a33fd1ec3c9b5e3828d3af7e5`;
- evaluation A: `validation_targets.jsonl`, SHA-256
  `865926c3e3221d8840bcbb1cdac43913bbd6efcd48b65637efca68e315a2d052`;
- evaluation B: `confirmation_targets.jsonl`, SHA-256
  `8625d9049957ce266fa544896f0262cb37ccea6d168ed0c60a847ef5325cce8f`.

Require `052/result.json` SHA-256
`60458c1bea4328b64f33ad8238529d074a25d1034fa2845e0b339deee3827153`
and its collection-integrity gates to pass. The fixed counts are 4,397 training
rows from 80 games and 2,228 evaluation rows from 40 games (1,135 plus 1,093).
The two old evaluation splits are deliberately combined: their labels and prior
warm-student outcomes are already known, so neither is honestly pristine. No
epoch, model, threshold, or hyperparameter is selected on either. They form one
fixed evaluation panel for the new from-scratch mechanism.

Reconstruct public states and legal actions from snapshots. Convert the stored
resolver action and raw-parent action to canonical public-successor keys; train
and score semantic-action agreement, not equality of arbitrary concrete return
variants. Hidden deck order, opponent blind reserve identity, outcome, future
state, `resolver_action_values`, and the stored soft distribution are forbidden
model inputs and forbidden loss terms. Code must not read
`outputs/human_games/replays.jsonl` or `games_summary.jsonl`.

## Frozen learning curve and model

Shuffle the sorted 80 training game seeds once with `random.Random(75075)`.
Train nested prefixes of 20, 40, and 80 games (25%, 50%, 100%); retain all
eligible rows from each included game. These are three prespecified learning-
curve models, not a hyperparameter sweep. Only the 100% model is the candidate.

Each model is a newly initialized `CanonicalEntityActionScorerModel` with the
established width-64 configuration: entity/action/scorer widths `64/64/96`, one
entity layer, one action layer, four heads, feed-forward width 128, ID embedding
16, dropout zero. Reset `torch.manual_seed(75075)` before each initialization.

Train exactly eight epochs, batch size 128, AdamW learning rate `3e-4`, weight
decay `1e-4`, deterministic shuffle seed 75075, CPU, value-loss weight zero.
Policy target is the resolver-selected canonical action with ordinary one-hot
cross-entropy. Add exactly

`-0.01 * mean(raw_masked_softmax_entropy / log(legal_semantic_action_count))`

to the batch loss, defining one-action rows as normalized entropy zero. This is
the same unmixed, width-normalized functional used by the entropy diagnostic;
there is no behaviour temperature or uniform mixture. Use the final epoch with
no validation selection, early stopping, retries, altered seed, or second arm.

## Primary endpoint, power, and dose

The sole primary endpoint is the 100%-model minus frozen-`035` game-weighted
canonical top-one agreement with the resolver action over the fixed 40-game
evaluation panel. Compute agreement within each game, average the 40 game means,
and form a paired 20,000-resample game-cluster bootstrap CI95 using seed 75075.
Success requires the CI95 lower bound to exceed **+3.0 percentage points**.

Expected gain is **+8.0 points**. The gate intentionally does not also demand a
point estimate of +8: doing that would cap power near 50% at the expected effect.
Design simulation on the actual 2,228-row/40-game panel starts from parent
game-weighted agreement about 32.9% and independently corrects 11.92% of parent
errors to induce +8 points. Across 5,000 trials, the lower-bound-above-+3 rule has
>99.9% power; a beta-binomial sensitivity analysis with within-game correction
ICC 0.10 has 99.88% power. This sizing claim is conditional on gains not being
more extremely game-concentrated than that declared sensitivity model.

Intervention dose is mandatory but diagnostic: report candidate-versus-`035`
argmax movement over all evaluation decisions and by phase, alongside agreement
gain. Also report agreement, NLL, normalized entropy, and movement for every data
fraction, overall and by phase. The learning curve must be shown even if it is
non-monotone; it is not a hidden selection rule.

## Integrity and collapse guardrails

Before interpreting the primary endpoint require exact source hashes/counts and
game membership; legal canonical targets on every row; finite loss/logits;
deterministic reload; byte-identical repeated evaluation; no human-data access;
and the exact architecture/optimizer/seed recipe above.

The 100% model must have mean normalized entropy at least 0.30 overall and 0.20
in each phase, measured by the exact regularized functional. No phase may have a
game-weighted agreement delta below -5 points. These are collapse guardrails,
not additional claims of improvement. Seat and winner/loser results are reported
diagnostically with no sign gates.

## Stopping and falsification

Stop after the three final-epoch models and one evaluation pass. Do not train
more epochs, change entropy weight, choose a fraction, or inspect human data.

- If integrity fails, diagnose only; there is no learnability result.
- If integrity passes but the primary lower bound is at most +3 points, or an
  entropy/phase-collapse guardrail fails, reject this candidate and close the
  from-scratch width-64 hard-label resolver-BC pilot. Do not add `052` games or
  tune the recipe to rescue it.
- A complete pass establishes offline learnability only. It authorizes a
  separately preregistered fresh-data/gameplay experiment, not immediate corpus
  generation, deployment, or promotion.

This is the explicit falsification condition: failure makes us abandon the claim
that this fixed stationary canonical student can recover a material amount of
resolver action knowledge at existing-data scale.

## Cost and value relative to alternatives

The established width-64 recipe trained eight epochs on 24,862 rows in about
6.5 CPU-minutes. `075` processes only 20+40+80 training-game prefixes and 2,228
evaluation rows, so expected measured compute is under 15 CPU-minutes after
implementation; allow up to two engineering hours. It buys a direct answer to
whether the untried no-anchor/hard-label mechanism is learnable and can close the
branch before any new simulation.

By comparison, fresh resolver collection costs 170--260 CPU-hours / 20--30 wall
hours, a powered human extension requires 68 additional interactive games, and a
new strategy-search teacher needs a new strategic instrument. `075` is therefore
the cheapest discriminating next measurement. It cannot answer gameplay strength
or whether the resolver ceiling reaches expert human play.

## Result (2026-08-04)

The complete pre-evidence suite passed 592/592. The single frozen CPU run then
completed all three final-epoch models in 101.5 seconds wall time, including
source reconstruction, evaluation, and the 20,000-resample bootstrap. All source,
privacy, reconstruction, finiteness, nested-prefix, reload, and repeated-evaluation
integrity checks passed.

The 80-game model reached 27.99% game-weighted resolver agreement versus 32.87%
for frozen `035`: candidate-minus-parent `-4.875` percentage points, bootstrap
CI95 `[-7.180, -2.452]`. The primary lower-bound-above-`+3` gate therefore failed
in the opposite direction. Agreement rose with data from 23.39% to 24.85% to
27.99%, but never reached the parent. Full-model phase deltas were `-3.82` early,
`-1.92` mid, and `-11.55` late points, so the no-phase-below-`-5` guardrail also
failed.

This was not entropy collapse. Full-model normalized entropy was 0.753 overall
and 0.791/0.761/0.671 early/mid/late; all entropy gates passed. The model moved
from `035` on 53.01% of evaluation decisions, so it changed behavior at high dose
without recovering the resolver ranking.

Formal decision:
`from_scratch_width64_hard_label_resolver_bc_pilot_rejected`. Per the frozen
stopping rule, close this exact stationary width-64, eight-epoch, hard-label
resolver-BC candidate. Do not add `052` games, train longer, adjust entropy,
change width, or run gameplay to rescue it. Fresh-data collection, gameplay, and
promotion are not authorized.

Evidence: `outputs/from_scratch_resolver_bc_pilot_075/result.json`, SHA-256
`329ec7edf93ead3ada87eb2c9e69847838c3a0b712c2068c7ec321c5ffda3894`.
