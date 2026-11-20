# Fixed representation, existing score head only

Owner authorized this experiment on 2026-10-09 after the completed independent-game
coverage recovery01 screen. This protocol is fixed before any new production
fit or new-model diagnostic. It authorizes one CPU development experiment, no
new games, rollouts, arena, promotion, parameter sweep or automatic successor.
All prior dirty work and frozen evidence remain untouched.

## Question

The broad coverage models removed about 97% of recorded-score regret on their
720 added training examples but did not improve fresh planning. Does preserving
the existing learned representation while fitting the current scoring head
improve generalization? This tests a specific constraint on learning. It cannot
separate preservation of features from reduced fitting capacity. Earlier053
used a different parent, residual and objective; its failure lowers confidence
but is not the same intervention. No representation deficiency is assumed.

## Exact training contrast

Use the three original161 initializers and the exact BROAD batch streams from
`independent_game_coverage_128_recovery_01`, including the narrow partitions
needed to reproduce its paired length bucketing. Do not recalculate a new
one-arm sampling order. Old9,461 TRAIN rows from159 source pairs and all720
added rows retain their existing memberships and tensor encodings. Each fit
receives114,044 exposures over1,784 updates:57,022 old and57,022 added, with
1,772 batches of64 and12 of53. Every row order and exposure receipt must match
the saved BROAD fit. Added720 panels remain retired from future testing.

Only parameters named `score_head.*` may change: existing384->256->1 scoring
head,98,817 parameters. State/action encoders, projection, attention, layer
norms and value head remain bitwise equal to the original initializer. Keep
the representation in evaluation mode. All initializers must have the same
zero-dropout semantics as the matched BROAD controls. No new residual or feature
is added. Train with freshAdamW, lr1e-4, weight decay1e-4, norm clipping1,
float32 CPU, one Torch thread, deterministic operations and the identical three
saved training seeds. The target remains softmax(q/.05) across recorded
alternatives outside the actual first three, q=(128+sumWDL)/256. Temperature
is a fixed score scale, not calibrated uncertainty. All legal actions remain
present in the model input even though the loss uses recorded alternatives.

Always select update1,784; save only initial/final diagnostic endpoints and
ordinary progress. No checkpoint, seed, subset or coefficient selection.
All three new optimizer trajectories are independently reconstructed using
independently rebuilt batches, targets and canonical group reductions. Verify
exact final tensors, all diagnostics and frozen tensors before sealing models.
Reuse the completed BROAD models as the matched unrestricted-training control.

## Development assessment

After all model seals, compute original2,489 VALIDATION diagnostics for every
initializer, BROAD and head-only model. These42 source pairs were excluded
from all optimization. Report equal-pair recorded-score regret, target KL,
tied-best/first-best agreement and movement; label agreement alone is not the
decision criterion. Full old/added TRAIN acquisition remains separate.

Use the existing336-root teacher-signal panel:168 TRAIN roots from42 pairs and
168 VALIDATION roots from42 different pairs. Each root retains exactly its
assigned source-fit model and original teacher shortlist; three model fits do
not triple the sample size. Within each root, retain the original first three
and let each scorer pick one recorded outsider using full-legal inputs and
canonical max-logit grouping. Choose among these up-to-four actions using the
saved128 selection worlds with the original tie rule. Compare original anchor,
matched BROAD, head-only, and the teacher's full recorded shortlist.

Seal all supported proposals and selection choices before reading independent
assessment outcomes. Score them using the existing128 independent assessment
worlds for every recorded candidate. Preserve unknown-outcome bounds; never
impute a draw. No unrecorded full-legal proposal can be evaluated from this
panel. No new continuation is run. TRAIN conditional results describe fitting;
VALIDATION is the primary development decision panel. Both panels and the
2,489-row validation corpus are reused DEVELOPMENT evidence, not fresh tests.

Bootstrap whole pairs within the existing assigned-fit strata (14 pairs per
fit per split),50,000 replicates, seed2263000000, equal pair weighting. Report
paired95% intervals and separate per-fit point contrasts. Independently
reconstruct every new prediction, canonical membership, supported choice,
outcome reduction and final statistic. Historical simulation assurance is
inherited by hash from the completed studies, not replayed again.

## Fixed investment screen

All conditions must hold; no strong training or secondary result can rescue
a failed primary planning result:

1. Each head-only fit removes at least50% of its original recorded-score regret
   on the added720 TRAIN rows. This modest acquisition floor prevents calling
   an almost unchanged model a successful remedy.
2. On all2,489 excluded validation rows, each head-only fit's equal-source-pair
   recorded-score regret is strictly lower than both its initializer and its
   matched BROAD model.
3. On the168 VALIDATION planning roots, head-only minus original anchor has
   mean lower bound at least+.005 expected score (+.5 percentage points) and
   paired95% lower bound strictly positive. Head-only minus matched BROAD must
   also have a strictly positive paired95% lower bound.
4. Each assigned-fit validation stratum has nonnegative mean lower planning
   differences against both comparators.

Report each gate separately. If acquisition fails, this fixed head/schedule
did not demonstrate the necessary learning, and its null cannot establish
adequate representation. If acquisition passes but transfer fails, this
constraint did not solve the observed generalization problem. A pass is a
reason to consider separately specified fresh full-legal confirmation, not a
strength result or automatic authorization. No freezing-depth, learning-rate,
regularization or checkpoint rescue follows this run.

## Operations and verification

Use a new `tools/frozen_representation_128` namespace and hidden detached
pythonw supervisor with at most three CPU workers. No console windows,
recurring monitor or GPU. Inhibit system sleep while running; display sleep is
allowed. Estimated execution10-30 minutes after preparation; hard operational
ceiling two hours,5GiB free disk reserve. Preserve partial artifacts on failure
or interruption; never silently replace a completed/failed run.

Before freezing, pass targeted source, actual optimizer, assessment/statistics,
orchestration and real hidden-worker checks. Authenticate immediate completed
parent receipts and transitive frozen code identities. Tests must isolate
failure markers in fixture directories. Check production runtime cleanliness
before and after preflight, prepare and launch; resolve output paths at call
time. Independent optimizer reproduction and all changed inference/statistics
are required. No full historical simulation replay is justified by this
learning-only intervention; follow the existing verification policy.

Report actual elapsed time, all three fits and audit status, gate components,
supported conditional scores with paired intervals, acquisition diagnostics,
and the limits of reused data and recorded support. These scores are not
full-game win rates. Only unbound handoff files may be updated after freeze;
the experiment's bound code, tests, protocol and inputs remain immutable.
