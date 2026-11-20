# AlphaZero Fresh-Dose Head Attribution 111

Date frozen: 2026-08-24

## Purpose

Experiment 110 ruled out fresh-data dose alone at 896 games. Its high-dose
candidate improved common-holdout policy cross-entropy but worsened value MSE
and remained flat against gen11 in raw and searched play. Experiment 111 asks
whether the failed transfer is primarily caused by the high-dose value update,
the high-dose policy update, or neither head in isolation.

This is an attribution experiment. It performs no neural-network training and
generates no self-play corpus. The functional swaps below use the complete
policy computation from one checkpoint and the complete value computation from
the other, including their respective trunks; they are not literal final-layer
tensor swaps.

## Frozen inputs

- completed 110 result SHA-256:
  `df35d3893f4c184ee24906b884aef33542c954380bf03c0b45f14a6d829f34c2`;
- exact gen11 checkpoint SHA-256:
  `c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0`;
- exact high-dose gen12 checkpoint SHA-256:
  `a9553311c5e49df4c809ad9238e9f769aa2307f562816c78e0efcadeb282fa74`;
- high-dose 896-game replay SHA-256:
  `8cc5a04f7fca7a76e946605f99b270ae7a3e9e7440c9b97e9ee9479255ea0fdc`;
- common 128-game holdout SHA-256:
  `bdaf7d8015b0d5295904724191e04c2a3081cdb6e200203452ea5c00b1e6c717`;
- retained installed gen2 SHA-256:
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`.

Verify hashes, exact gen11/high-dose lineage, selected high-dose epoch 1, the
complete 110 decision, and retained installation before evidence.

## Stage 1: streaming value audit

Evaluate exact gen11 and the high-dose candidate on every row of the exact 110
common holdout without materializing the corpus. Report overall and fixed phase
slices: early turns 0--19, middle 20--49, and late 50+. For each model report
value MSE, MAE, bias, mean prediction, sign accuracy, decisive win/loss ROC AUC,
and ten fixed calibration bins. Report the zero predictor's MSE and MAE in each
slice. Also fit one fixed ridge score/tempo baseline only on the exact high-dose
training replay, never the common holdout. Its five features are intercept,
self-minus-opponent score divided by 15, purchased-card-count difference divided
by 15, total-bonus difference divided by 15, and turn divided by 100; ridge
lambda is `0.001`, excluding the intercept, and predictions are clipped to
`[-1,1]`. Evaluate it identically on the holdout. Report policy cross-entropy
and target-top agreement to reproduce the 110 direction.

This audit is descriptive. The holdout has already been inspected in 110 and
cannot select a future architecture as though it were untouched.

## Stage 2: shared-gradient conflict audit

Index the exact high-dose replay. Select 2,048 rows without replacement using
seed `1110111`, then process sixteen 128-row batches. At both exact gen11 and
the high-dose checkpoint, compute separate unweighted policy-cross-entropy and
value-MSE gradients over only parameters shared by both heads: entity/card/
noble embeddings, entity input, and entity transformer. Report per-batch and
aggregate cosine similarity, policy and value gradient norms, the value/policy
norm ratio before and after the frozen 0.5 value-objective weight, and the
fraction of batches with negative cosine.

Negative cosine is evidence of local gradient conflict, not by itself a causal
strength result.

## Stage 3: powered same-seed head attribution

Run three searched panels concurrently on identical fresh seeds
`2060000..2060199`, both seats, for 400 games per panel:

1. full high-dose policy and value versus full exact gen11;
2. high-dose policy with exact gen11 value versus full exact gen11;
3. exact gen11 policy with high-dose value versus full exact gen11.

All searches use 256 simulations, `c_puct=2.25`, real mode, zero root noise,
and the established deterministic per-game/per-seat search seed
`2 * game_seed + seat`. Use at most three one-thread workers. Every panel has
a sidecar identity written before its first game that binds protocol,
policy/value/baseline hashes, search configuration, and schedule. It then
resumes by exact game prefix and reports W-D-L, score share, margin, seats,
terminations, and a 50,000-sample paired bootstrap.

Compute same-seed paired score-share differences for each restored-parent-head
variant minus the full high-dose replication. With 200 paired seeds, the panel
is intended to have at least roughly 80% power for a large attribution effect
of about eight percentage points; smaller effects may remain unresolved.

## Primary endpoint and decisions

The primary endpoint is the paired score-share difference from restoring exact
gen11 value while retaining the high-dose policy.

- If its CI95 lower bound is above zero and restoring gen11 policy is not
  positive by the same rule, classify the high-dose value update as the primary
  blocker.
- If only restoring gen11 policy is positive, classify the policy update as the
  primary blocker.
- If both are positive, classify both updates or joint incompatibility.
- If neither is positive, record no clear head attribution and investigate
  representation/capacity or fixed-teacher quality rather than assuming a value
  redesign is justified.

Direct practical passes (>=55% share, positive margin, both seats >=50%, all
normal) are reported but do not override the paired attribution rule.

Estimated wall time is 3--6 hours with three parallel panels. No result
authorizes `035`, training, installation, promotion, or automatic architecture
work. Exact gen2 remains installed.
