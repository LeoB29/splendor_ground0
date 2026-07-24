# Frozen-Parent Resolver Residual Ranking Distillation (`053`)

Frozen after the completed `052a` diagnostic and before implementing or
training the residual model, inspecting any `053` metric, generating any fresh
confirmation game, or running gameplay. This is one CPU-only proposal. It
performs no parameter sweep.

## Question and prerequisites

`051` proved that the exact public-belief resolver is a broadly stronger
full-game teacher. `052` produced a clean resolver-every-move corpus but its
global distribution cross-entropy update mostly flattened `035`: on `052a`
confirmation target disagreements, 87.79% of pairwise gaps moved correctly,
yet only 8.57% crossed zero and fallback KL rose to `0.186057`.

`053` asks whether a small action-specific residual can transfer the
resolver's decisive rankings while preserving frozen `035` everywhere the
teacher does not request a switch.

Require:

- `052a` result SHA-256
  `226d28a4cd9667ef27eacfc1afccf8cac628d07b30006e9dc179ca282b2eba57`;
- raw `035` checkpoint SHA-256
  `4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`;
- `052` train-target SHA-256
  `6827848ca5773799a698c62712ee04f3c40b2e2a33fd1ec3c9b5e3828d3af7e5`;
- `052` validation-target SHA-256
  `865926c3e3221d8840bcbb1cdac43913bbd6efcd48b65637efca68e315a2d052`.

The previously inspected `052` confirmation corpus is forbidden for training,
selection, gating, or threshold choice. The runner must not open it. A
validation pass is required before generating a completely fresh confirmation
corpus.

Earlier experiments `002`/`003` failed to learn noisy matched top-two rollout
preferences. This is not a retry of that closed primary direction: the present
teacher uses exact `052` resolver decisions, four adversarial values, complete
legal supports, and the resolver whose full-game strength passed `050` and
`051`. The earlier failure nevertheless forbids full-model pairwise tuning,
capacity sweeps, and learning-rate sweeps here.

## Frozen residual architecture

Load the exact `035` legal-action attention scorer. Freeze every parent
parameter and keep it permanently in evaluation mode. Recompute its
information-legal state hidden vector, contextual legal-action vectors, raw
action scores, and value.

For each legal action concatenate:

- frozen 256-dimensional state hidden vector;
- frozen 128-dimensional contextual action vector;
- its frozen raw `035` action score.

Pass the 385-vector through exactly `Linear(385,64)`, GELU,
`Linear(64,1)`. Initialize the final linear weight and bias to exactly zero, so
the initial model is bit-for-bit action-score equivalent to `035`. Initialize
the first linear layer with the PyTorch default under seed `53`. Subtract the
masked per-state mean residual and add the centered residual to the frozen
parent score. Only the 24,769 residual parameters are trainable. No parent
parameter, normalization statistic, temperature, value head, or observation
feature may change.

The self-contained checkpoint stores the frozen parent state, residual state,
model configuration, parent hash, data hashes, and optimizer recipe. Deployment
uses the same acting-player public encoders as `035`.

## Fixed supervision and loss

Use all 4,940 `052` training rows exactly once per epoch. A completed resolver
row is a **ranking row** iff its recorded resolver action differs from raw
`035`; the resolver action is preferred and raw `035` is the rejected action.
All other completed rows and every structural fallback are **anchor rows**.
This deliberately uses the resolver's actual canonically best action rather
than the `052` blended-distribution argmax that suppressed 36.1% of switches.

For a ranking row, optimize pairwise softplus loss

`softplus(0.25 - (student_score(resolver) - student_score(parent)))`.

For an anchor row, optimize
`KL(parent_policy || student_policy)` at temperature 1. For all legal rows,
also penalize the mean squared centered residual. The batch loss is:

`rank_loss + 5.0 * anchor_KL + 0.01 * residual_MSE`,

where an absent group contributes zero and each present group is averaged
before weighting. Do not apply hard-action CE, blended-target CE, value,
outcome, trajectory, search-score, or any other loss.

Train one proposal with AdamW, learning rate `3e-4`, weight decay `1e-4`,
batch size 128, 12 epochs, deterministic shuffle seed `53`, CPU, and one
PyTorch thread. Save and evaluate every epoch. Do not resume, interpolate,
ensemble, change a coefficient, or train a second proposal.

## Validation selection and gate

Evaluate from zero and after every epoch on the fixed 1,242-row `052`
validation split. An epoch is selection-eligible only if:

- fallback `KL(p035 || student) <= 0.02`;
- fallback parent-argmax retention is at least 98%;
- parent argmax is retained on at least 98% of completed rows where resolver
  and parent agree.

Among eligible trained epochs, select maximum validation resolver-agreement
gain over parent, then lower validation ranking loss, then earlier epoch. Epoch
zero is a baseline and cannot be selected. If no trained epoch is eligible,
stop.

The selected epoch must additionally satisfy all of:

1. resolver-action agreement over all completed rows improves by at least five
   percentage points over raw `035`;
2. at least 15% of resolver-versus-parent disagreement rows select the resolver
   action;
3. at least 80% of disagreement pairwise gaps move in the correct direction;
4. the two 98% anchor-retention conditions and fallback KL condition above;
5. overall student-versus-parent argmax movement is between 5% and 25%;
6. KL to the stored complete blended target is no worse than raw `035`;
7. resolver-agreement gain is non-negative independently in early/mid/late,
   seat 0/1, and eventual winner/loser strata with at least 50 completed rows;
8. train disagreement capture exceeds validation capture by no more than ten
   percentage points;
9. every score, loss, residual, and probability is finite and every selected
   action is legal.

A validation failure rejects the proposal and stops before any fresh game.

## Conditional fresh confirmation

Only a complete validation pass authorizes 20 new resolver-versus-resolver
teacher games at seeds `620300--620319`, collected with the exact `052`
teacher, target semantics, ten CPU workers, clean-terminal rules, privacy
checks, 192-branch accounting, and row/stratum gates. Store them under the
separate `053` output directory. Require at least 1,000 rows and 50 completed
search rows in every phase, seat, and winner/loser role.

Open and evaluate this corpus only after the checkpoint is fixed. Require all
validation gates on confirmation, plus a deterministic 20,000-sample
game-cluster bootstrap with seed `530053` whose 95% lower bound for
resolver-agreement gain is strictly above zero. No confirmation result may
cause retraining or reselection.

## Conditional gameplay

Only a complete confirmation pass authorizes a fresh 100-pair / 200-game
candidate-versus-raw-`035` gate at seeds `620500--620599`, reversing seats per
seed. Require 200 clean games, zero invalid terminal/timeout/stall, score share
at least 55%, bootstrap CI95 lower bound strictly above 50%, positive mean
paired score margin, and at least 50% score share from each seat.

Only a direct-parent pass authorizes three independent fresh 50-pair blocks:

- gate-`030` shallow: `620700--620749`;
- historical legal scorer `016`: `620800--620849`;
- loop-safe canonical `023`: `620900--620949`.

Each block independently requires the same clean-game conditions, at least 55%
score share, CI95 lower bound above 50%, positive mean margin, and at least 50%
from each seat. Aggregate performance cannot rescue a failed opponent.

## Decision

A full validation, fresh-confirmation, parent, and broad pass promotes the
residual scorer as the new fast checkpoint champion and authorizes only a
separately preregistered next synchronous resolver collection. Any failure
rejects the proposal, retains raw `035`, and stops at that boundary.

`053` does not authorize additional candidates, hyperparameter changes, use of
the old `052` confirmation corpus, open-ended self-play, or GPU rental. CPU is
appropriate because the only training is a 24,769-parameter head; fresh
resolver collection remains rollout-bound.
