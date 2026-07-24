# Resolver Public-History Representation Feasibility (`054`)

Frozen after `053` and before implementing either diagnostic model, fitting any
`054` parameter, or computing any `054` train/validation metric. This is a
CPU-only matched representation experiment. It performs no confirmation
collection, gameplay, policy deployment, or promotion.

## Question and prerequisites

The verified public-belief resolver remains substantially stronger than raw
`035`, but `052` and `053` transferred probability shape without transferring
decisive action rankings. `053` froze the parent and trained only a small
action-specific correction, yet the correction still could not distinguish
resolver-switch states from anchor states.

`054` tests whether the missing discriminator is legally available public
history. A static Splendor snapshot does not express every fact an observer
can remember, including the tier and timing of prior blind reserves and the
sequence of multi-turn commitments. The `047` Deep-CFR work already validated
a chronological public-action encoding that hides blind-reserved identity
until public purchase.

Require these exact artifacts:

- `047` result SHA-256
  `7e59f4b8d91c2b2c3be2c593820706cc4167987aa482469b93570abd9e8dfd44`,
  with its representation privacy/invariance and schema gates passed;
- `053` result SHA-256
  `7610c56789cf3491cc889e4f9689957bf06a769fb906f5314731903c933339ef`,
  with decision `validation_residual_gate_failed`;
- `052` train-target SHA-256
  `6827848ca5773799a698c62712ee04f3c40b2e2a33fd1ec3c9b5e3828d3af7e5`;
- `052` validation-target SHA-256
  `865926c3e3221d8840bcbb1cdac43913bbd6efcd48b65637efca68e315a2d052`;
- raw `035` SHA-256
  `4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`.

The old `052` confirmation corpus is forbidden. No fresh simulation is
authorized.

## Information-legal reconstruction

Use only completed resolver-search rows: 4,397 train and 1,135 validation.
Group rows by game seed and require turns to be exactly chronological and
contiguous from zero. Before each acting row, reconstruct the public event for
every preceding recorded action through the validated
`public_action_event`/`encode_public_history` contract.

History exposes public action type; publicly visible card or purchased-reserve
identity; blind-reserve tier but not card identity; token take, return, and
spend flows; noble; actor relative to the observer; and turn. It may not use
hidden deck order, opponent blind-reserved identity before reveal, future
actions, final outcome beyond breakdown labels, resolver branch truth, or
continuation outcomes.

Require every recorded action to regenerate as a unique legal action, history
length to equal the turn index, observer identity to equal the acting player,
and all 047 privacy checks to remain true.

Each resolved row contains exactly four canonical successor groups and their
adversarial normalized score-margin values. Predict those four values and
their within-state ordering. If a group has multiple concrete aliases, its
prediction is the mean score of those aliases.

## Matched model arms

Both arms embed the exact frozen `035` legal-action attention scorer and keep
every parent parameter in evaluation mode. Extract its 256-dimensional state
hidden vector, 128-dimensional contextual legal-action vector, and raw parent
action score.

Both arms instantiate the same additional modules under seed `54`:

- history action-type embedding width 8;
- history card embedding width 16;
- history noble embedding width 8;
- history numeric projection `Linear(34,32)`;
- concatenated event projection `Linear(64,64)`, GELU, LayerNorm;
- one-layer chronological GRU with input/hidden width 64;
- action-value head `Linear(449,128)`, GELU, `Linear(128,1)`.

The **snapshot arm** forces the 64-dimensional history context to exact zeros
and never calls the history encoder. The **history arm** uses the final masked
GRU state; empty histories map to exact zero. Parent features and action-value
head initialization are otherwise matched. Only added modules are trainable.
This is a representation diagnostic, not a residual policy.

## Fixed target and optimization

For the four canonical group predictions, optimize:

`SmoothL1(predicted_value, resolved_value, beta=0.1)`

plus `0.5` times mean pairwise softplus ranking loss over all non-tied group
pairs:

`softplus(-sign(value_i-value_j) * (prediction_i-prediction_j))`.

No hard policy, blended distribution, parent anchor, outcome, trajectory,
WDL, or gameplay loss is allowed.

Train each arm independently from its matched initialization on the same
4,397 training rows, in snapshot-then-history order:

- AdamW, learning rate `1e-3`, weight decay `1e-4`;
- batch size 128, exactly 20 epochs;
- deterministic shuffle seed `54`, CPU, one PyTorch thread.

Do not select an epoch, stop early, interpolate, sweep a coefficient, or train
a second seed. Save the fixed epoch-20 models. Compute training metrics for
both final models. Only after both models and train metrics are fixed may the
validation file be reconstructed and evaluated exactly once.

## Fixed metrics

Report for train and validation:

- four-value MAE;
- per-state best canonical action accuracy;
- mean within-state Spearman ordering correlation;
- all non-tied pairwise sign accuracy;
- best-action capture where resolver and raw `035` disagree;
- phase, seat, and eventual-winner/loser breakdowns;
- finite/alignment/privacy integrity.

On validation, compute the paired history-minus-snapshot best-action accuracy
difference with a 20,000-sample game-cluster bootstrap, seed `540054`.

An arm is **learnable** only if:

- training best-action accuracy is at least 60%;
- validation best-action accuracy is at least 40%;
- validation mean Spearman is at least `0.25`;
- validation pairwise sign accuracy is at least 60%;
- train-minus-validation best-action gap is at most 20 percentage points;
- every phase, seat, and winner/loser validation stratum has at least 50 rows
  and at least 30% best-action accuracy.

Public history **materially helps** only if the history arm is learnable and:

- validation best-action accuracy exceeds snapshot by at least five percentage
  points;
- paired game-bootstrap CI95 lower bound is strictly above zero;
- validation Spearman exceeds snapshot by at least `0.05`;
- resolver-disagreement capture exceeds snapshot by at least five percentage
  points;
- no required validation stratum is more than two percentage points worse
  under history.

## Decision

Use the fixed priority:

1. If history materially helps, decision is
   `history_representation_promising_authorize_scaled_corpus_protocol`.
2. Otherwise, if snapshot is learnable, decision is
   `snapshot_representation_sufficient_authorize_scaled_corpus_protocol`.
3. Otherwise, if history is learnable, decision is
   `history_representation_promising_authorize_scaled_corpus_protocol`.
4. Otherwise, if either arm reaches at least 70% training best-action accuracy
   while exceeding a 20-point train/validation gap, decision is
   `data_scale_primary_authorize_scaled_corpus_protocol`.
5. Otherwise, decision is
   `resolver_targets_not_learnable_with_current_frozen_features`.

Any authorization is only permission to preregister a substantially larger,
game-disjoint resolver corpus and an appropriately sized policy/value model.
It does not itself authorize collection, GPU rental, training on confirmation,
gameplay, promotion, or self-play deployment. GPU rental becomes worth
considering only after a scaled corpus protocol passes collection integrity;
this small two-arm probe remains CPU-appropriate.
