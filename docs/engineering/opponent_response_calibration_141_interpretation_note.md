# Outcome-blind interpretation note for 141

Recorded 2026-09-08 during the existing run, after the 01:36 Europe/Paris health
check (64/120 state files saved). No protected gains, scores or winners have been
inspected. This note changes no source, protocol, sampling, budget, estimator,
confidence interval, threshold, execution gate or frozen decision classification.

Review of reporting logic identified overlap in two descriptive conditions.
A primary CI can both exclude zero and lie wholly inside [-.025,+.025]. For
example, the hypothetical interval [.005,.015] detects a difference and bounds
its size below the prespecified material threshold. These are compatible facts.

The frozen classifier checks nonzero detection before the within-margin condition,
so in that overlap it emits `response_rule_difference_detected_materiality_uncertain`.
The label's last words are too broad for this specific overlap. Preserve the raw
label and separately explain both interval facts in the report. Do not call the
material size uncertain when the reported CI actually excludes it. This is a
wording limitation, not a reason to invalidate or change the experiment.

For all results interpret the same frozen CI directly: zero exclusion, containment
within the material interval, and complete exclusion beyond either material bound.
The two absolute-gain intervals remain prespecified secondary estimates. No new
significance test, alternative interval or outcome-driven rule is introduced.
