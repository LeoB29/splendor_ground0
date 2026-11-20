# Experiment 142: stopped at the value-bootstrap gate

2026-09-08. The authorized pilot ran through its first learning stage and stopped
at its prospectively specified generalization gate. **035's move policy was
preserved exactly, but the new value head overfit.** No searched self-play,
end-to-end successor, or playing-strength arena followed. The central hypothesis
that strong initialization enables a productive search-and-learning cycle remains
untested. Champions remain 084 and 035; no promotion or continuation is authorized
automatically.

## What ran

All 256 fixed bootstrap games completed naturally, producing 16,328 positions;
156 games were won by seat 0 and 100 by seat 1. There were no cutoffs, exclusions,
replacement games, draws, or forced-pass adjudications. The whole-game split used
205 training games (13,112 positions) and 51 validation games (3,216 positions).
These are 256 independent games, not 16,328 independent outcome observations.

The original 035 model has 666,114 parameters. Only its 33,025-parameter value
head was trained, for the fixed eight epochs. The other 633,089 parameters,
including every policy-affecting tensor, remained bit-identical. The adapter's
complete legal-action scores and greedy choices also matched 035 in the tests
and off-schedule probes. The untouched source checkpoint remains available.

| Mean squared outcome error | Training games | Unseen validation games |
|---|---:|---:|
| Neutral value prediction | 1.0000 | 1.0000 |
| Final fitted value head | 0.8131 | 1.1328 |

Lower error is better. The final head fit its training games while its validation
error rose 13.28% above the neutral predictor. Its outcome-sign accuracy was
68.12% on training positions and 52.21% on validation positions; increasing
confidence without dependable generalization also hurt squared error. These
position-level metrics are diagnostics, not estimates of playing strength.

The validation trajectory supports overfitting: epoch 2 briefly reached .9964,
then epochs 4, 6, and 8 reached 1.0304, 1.0708, and 1.1328. The fixed final epoch
was the prespecified parent candidate. **Do not retroactively select epoch 2 or
reinterpret its tiny temporary gain as a successful bootstrap.** No settings or
data were changed after the gate failed.

## Diagnostic provenance and verification

The frozen trainer raised its exception before saving failed-epoch diagnostics.
The original log records all eight epochs and the gate failure, but not the loss
values. A separate diagnostic script therefore replayed only the identical
warmup, using the same data, split, seed and settings, to recover the metrics. It
reproduced the failed gate and the unchanged policy hash. The numbers above come
from that diagnostic reconstruction. This added no data, hyperparameter search,
new generation, eligible parent, or arena. The reconstructed failed checkpoint
is saved only under `bootstrap_failure_diagnostic/`, marked ineligible for use.

The independent audit replayed **every action in all 256 games**, matched every
saved state hash, verified legal moves and natural endpoints, reconstructed all
16,328 observer-legal feature/group tensors and outcome labels, and checked the
whole-game split and exact policy retention. All 271 current frozen dependencies
and all 268 dependencies from experiment 141 still matched. This checks the
repository engine and recorded execution; it is not an independent implementation
of Splendor's rules.

The full suite passed 970/970 in 283.76 seconds. The final focused suite passed
17/17, including an added JSON round-trip verifier regression. Before protected
collection, two complete off-schedule probe games passed replay/feature checks,
and timing probes verified the search adapter. An initial failed preparation is
preserved separately; its verifier comparison was corrected and the final
identity re-frozen before any experiment games or training.

## Scientific meaning and next direction

This establishes an engineering success—an exact bridge from 035 into the
policy/value architecture—and a failure of this small-corpus, fixed-epoch value
bootstrap. It does **not** establish that 035 knowledge is unsuitable for value
learning, that all value heads fail, or that strong-policy self-play cannot
improve. Search and end-to-end learning were never reached. Exact raw-policy
retention also does not certify that adding the proposed search would preserve
035's playing strength; that external comparison remains outstanding.

The immediate bottleneck is value generalization. More independent games,
regularization and predeclared early stopping remain plausible remedies; a value
representation that can learn separately from the frozen policy also remains
open. The current result does not isolate these explanations. Another full
search-and-self-play cycle is premature until a value bootstrap generalizes.

A useful next proposal is a **value-only generalization pilot** using a larger
corpus of independent 035 games, predeclared validation selection and a separate
untouched test set, while retaining 035's policy exactly. The 142 validation set
is now development evidence. Do not keep tuning against it or automatically run
the next proposal.

## Artifacts

- Protocol: `docs/engineering/strong_policy_pilot_142.md`.
- Result: `outputs/strong_policy_pilot_142/result.json`, SHA256
  `e2158ecb8c79e68563b920fbd46ebf3ec42f9993eebb3a79bff7fb71865af33d`.
- Fit diagnostics: `outputs/strong_policy_pilot_142/bootstrap_failure_diagnostic/fit_diagnostics.json`,
  SHA256 `aec937f79652bea8532bdd9c3e506326abdca3a8996efbd3d0614a42b386fa63`.
- Full bootstrap audit: `outputs/strong_policy_pilot_142/bootstrap_failure_diagnostic/verification.json`,
  SHA256 `9200af3341dcd260b207ac8581e3c8b66a57ed080154626b20e1f396dbc2b4a5`.
- Final manifest digest:
  `71b52489ee0dd15d412738819bd67dd81386a20f4ce52d7c6c8107f1ce7fda17`.
- Current checkpoint, corpus, code, tests, logs and failed preparation are all
  preserved. Original execution session 83234 and diagnostic session 90028 have
  exited. No process should be resumed and no parent checkpoint should be installed.
