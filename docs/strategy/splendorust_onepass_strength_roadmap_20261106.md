# Test whether shorter training improves play

6 November 2026. The original EF8A network remains the reference. The next
prepared study is a fresh strength comparison of both fixed one-pass models
against that reference. No new games have launched.

The project aims at substantially stronger, ultimately superhuman two-player
Splendor using lawful information. More search improved research play, and the
collection work established practical ways to generate data. Neither achievement
proves that a small training update improves the network itself.

The saved-corpus diagnostic showed that repeatedly training on the small dataset
could lower training loss while worsening held-out predictions. After one pass,
both fixed repetitions instead showed a small policy-prediction improvement;
value changes remained uncertain. This supports testing the shorter update in
actual play. It does not establish that one pass is optimal, that data scarcity
is the only bottleneck, or that either checkpoint is stronger.

## The prepared comparison

Use the [bounded comparison design](../engineering/splendorust_onepass_strength_preparation_20261106.md):
400 fresh main games, split equally between the two final one-pass checkpoints,
plus six operational games per checkpoint. Both players use128 simulations,
matching earlier training assessments. Keep paired seats, the existing useful-gain
criteria, all verification and separate reporting of timing. The sample and models
are fixed before play; no selection between training repetitions is allowed.

The primary result concerns the average of these two fixed models. Report both
individual results so a pooled gain cannot hide differing behavior. Even a positive
result does not automatically promote a checkpoint or establish superhuman play.

## Decisions after verification

- A useful pooled gain would justify a separate confirmation or generalization
  proposal. It would show that this short update can improve this matched setting;
  it would not identify a new single reference without further judgment.
- An inconclusive result would leave a small gain possible. Finish the fixed sample
  without extending it, then weigh further confirmation against an independently
  varied data study, using effect size, uncertainty and cost.
- A clear deterioration would argue against scaling this exact recipe unchanged.
  Use saved evidence to choose a specific training change before more collection.

More independent training games remain plausible future work. Their collection,
storage and assessment costs need justification from this evidence. Larger-data,
external-opponent, six-world and further GPU work remain deferred. One second is
an eventual deployment target;700ms is descriptive. Forecasts and available RAM
are advisory, while actual execution bounds and preservation remain mandatory.

The concrete next action is to implement and qualify the separate checkpoint
import and paired-game runner, then execution-freeze and run this prepared design
when that scope is authorized. No training or larger-data pipeline starts with it.
