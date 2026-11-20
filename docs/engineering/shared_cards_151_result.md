# Experiment151 result: development screen failed

2026-09-09. Complete and independently audited. All three main4096-update fits
and all three complete audit refits finished. Histories and selected/final weights
match exactly. Every development feature, frozen control prediction, selected
prediction and statistic passed audit. Final status records development failure
and no fresh stage. No experiment processes remain.

Protocol `docs/engineering/shared_cards_151.md`; output `outputs/shared_cards_151`.
Owner authorized one shared-card recipe and the conditional fresh384 test only
on audited development pass. No arena, promotion or automatic successor.

## Main finding

The327297-parameter shared-card learner did not turn the existing action labels
into a material development move improvement. Its equal-seed advantage MSE
was0.095864088 versus0.096116394 flat, a0.262500% reduction (required3%). Its
selected WDL minus035 was+.000189887 (required+.02), effectively no mean gain
on this reused validation panel. Two of four development criteria failed.

| Development criterion | Observed | Frozen requirement | Result |
|---|---:|---:|---|
| MSE reduction versus flat |0.262500%|at least3%|Fail|
| MSE reduction versus zero |3.658717%|at least3%|Pass|
| Selected WDL minus035 |+.000189887|at least+.02|Fail|
| Selected WDL minus flat |+.004828559|strictly positive|Pass|

Zero-advantage MSE0.099504683. Flat selected WDL minus035-.004638672. Shared
selected WDL+.000189887 corresponds to+.009494 percentage points of expected
score under the stored greedy035 continuations. This is a descriptive conversion,
not a win-rate or whole-game strength estimate. Every seed remains in the mean.
The shared model changes about59.90% of baseline choices, so merely changing
moves is insufficient. No confidence interval or confirmatory interpretation
is attached to this development panel used for checkpoint selection.

| Seed | Shared selected update | Shared MSE | Shared WDL minus035 | Flat selected update | Flat MSE | Flat WDL minus035 |
|---|---:|---:|---:|---:|---:|---:|
|149001|16|.094938405|+.010986328|30|.095150011|-.001464844|
|149002|17|.096055543|-.008463542|13|.097369950|-.010416667|
|149003|24|.096598316|-.001953125|16|.095829221|-.002034505|
|Equal-seed mean|—|.095864088|+.000189887|—|.096116394|-.004638672|

## Learning behavior and interpretation

Every main fit ran all4096 updates. Validation selected early updates16/17/24
under the frozen earliest-strict-minimum rule; this is not early stopping or a
failure to train. At4096 the training/validation MSEs were approximately:

| Seed | Final training MSE | Final validation MSE |
|---|---:|---:|
|149001|.000237902|.104410857|
|149002|.000010870|.124457352|
|149003|.000399335|.106139816|

Training labels can be closely fitted while validation worsens. This is evidence
of a substantial generalization gap for this recipe; it does not identify whether
limited coverage, noisy targets, objective, optimization or a different inductive
bias is the dominant cause. Histories use original float32 training reductions;
headline selected statistics use independently checked float64 reductions.

The controlled result weighs against this small shared-card/attention substitution
being sufficient on the current768-state learning task. It does not rule out
other entity architectures, afterstates, more diverse data or learned value in
general. The representation retains all original information and adds public
catalog semantics, but this bundled architecture intervention cannot separately
attribute failure to attention or card sharing. Existing post150 opposite-half
simulation-selected headroom remains suggestive, not a promise that a network
can learn those choices from the available inputs/examples.

Do not rescue a seed, lengthen training, relax thresholds or spend the conditional
test budget on a recipe that failed its screen. No fresh result or superhuman
performance claim follows. Champions084 research/035 fast remain unchanged.

## Verification and provenance

- Focused151/150/149:60 tests passed15.73s. Full suite:1156 passed309.04s.
- Input preflight passed. Exact feature inverse, public-catalog uniqueness,
  observer privacy, token-order invariance, all legal actions in representative
  positions, gradients, synthetic independent training, all screen gates and
  run-sequence prevention of unapproved fresh work tested.
- Three main fits and three independent complete refits passed, including all
  160 history rows per fit and every selected/final tensor. Auditor passed all960
  rebuilt features, independent target aggregation, exact flat150 prediction
  reproduction and independently recomputed statistics.
- Frozen151 binding
  `e8a62808167ee99ceb757bf43487461322b3bd288b0908d4d469eaf5c767dc74`.
- Parent150 binding
  `04c878af0d94292c1b4b834d4a19f63760d6f7ffbf29fc9934e0aa89026308a3`.
  Explicit150/149/148 inventories and all controls preserved; no frozen edits.
- Run22:58:30--23:08:44 Paris,613.83s (10m14s), excluding preparation/tests.
  Main fits/selection296.15s; audit314.86s. Parent3520/session14541 and auditor4872
  exited successfully; no Python experiment processes remain. No failure artifact.
- Manifest file SHA
  `27c1b1f65edc448aa11a72606598a560494a5e62a2d32cd9af417fac8ae15b1c`.
- Development result SHA
  `034b377f6c487aeea468a9b957039d6727a5d463745c5502257d6c7a683bc434`.
- Development audit SHA
  `fc84b9616bb3c60abdbe82d638d8c048aec62c56bc8091f2f4311c46c2537eec`.
- Selection lock SHA
  `337de54f707f3f56f6a64cf93b6d25eebd176590893f645ff6422e073fa3d969`.
- No new continuation, source game, protected-test prediction, fresh test,
  arena, promotion or automatic successor. Preserve intentionally dirty work.
- Derived histories: `outputs/shared_cards_151/analysis/learning_histories.json`.

![Audited151 learning curves](C:/Users/leoba/Desktop/splendor_ground0/outputs/shared_cards_151/analysis/learning_curves_verified.png)
