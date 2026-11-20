# Splendor: review of the full-corpus learning study

10 November 2026. This answers `splendor_full_corpus_auditor_update_20261110.md`. It comes from the author of the 8 November audit (`splendor_project_independent_audit_response_20261108.md`).

This is advisory. I read the update, the frozen protocol (`docs/engineering/splendorust_full_corpus_20261110.md`) and upstream's training script. I changed no process, source, setting, clock or evidence, and I opened no result from the running study.

**Labels:** **[F]** fact, checked. **[I]** interpretation. **[S]** speculation or prior.

## Verdict

**Finish the study as frozen.** It is essentially the experiment the audit recommended: train on the full corpus, with the value target as the controlled variable. None of the four disclosed departures justifies refreezing. No change to the running design is essential.

## A fact that changes the 8 November audit

**[F]** Upstream's `research/training_strategy/strategy_train.py` at commit `1828507b8d2f0e02bc03d102b9c6d29a54fa8d57` trains with:

- AdamW, learning rate **1e-4**, weight decay 1e-4.
- `CosineAnnealingLR(optimizer, epochs, eta_min=1e-5)`, stepped once per epoch, with a default of **4 epochs**.
- Batch **512**, gradient clip norm **5**.
- Policy and value losses at weight **1:1**.
- Value target `((outcome*2-1) + .837*(root*2-1)) / 1.837`.

Source: a fetched copy of the public file, read on 10 November. The lines above were quoted from it. The protocol records a verified local SHA256 for this file (`06caf53b…`); confirm the settings against that copy before relying on them.

**[I]** With Adam, a common rule of thumb scales the learning rate with the square root of batch size. On that rule, upstream's per-pass rates at batch-32 equivalence are about 2.5e-5, 2.2e-5, 1.4e-5 and 0.6e-5. The study's 1e-5 at batch 32 therefore sits around upstream's third-to-fourth-pass rate.

**The audit's suggestion that LR 1e-5 was itself too small is withdrawn.** The earlier updates were small mainly because of few steps and little data. This study runs 1,782 updates on fresh positions per fit, 18.6× the 96-step fits. The remaining difference from upstream is **one pass versus four**. Batch size and clip also differ. Upstream's dataset size and teacher remain unknown.

## The four departures

1. **Search mixture rather than parent anchoring: agreed, and preferable.**
   - **[F]** EF8A's value head was trained toward this exact mixed target.
   - **[I]** So the terminal-only arm is the one that changes what the value output means. A visit-weighted 512-search root value is far less noisy than a single game result, whose outcome is shared by about 56 correlated positions.
   - Anchoring was the audit's fallback for when no root value was stored.
   - Inherited network bias in the search value is real, but it is standard practice in AlphaZero-style training. The protocol already reports value drift and search-value error.

2. **Three 400-game screens: acceptable, not ideal.**
   - **[I]** Each comparison resolves only effects of about 5 points or more (95% half-width of about 4.5 points at 400 paired games). Gains in the +15 to +25 Elo range may stay unresolved.
   - Re-dividing the same 1,200 games would not change that much, and refreezing costs real time.
   - Keeping the 55% rule is consistent with a 400-game sample. The 53% criterion only makes sense with a sample of about 2,000 games.

3. **Fixed learning rate: acceptable for this study**, given the finding above.
   - A sweep is justified only after a null result.
   - The first variable to try then is the number of passes (upstream used four), monitored by held-out loss, rather than the rate.

4. **No calibrated proxy or human anchor yet: neither needs to come before this live screen.**
   - Live games are the ground truth.
   - The proxy mainly makes future studies cheaper, so do it after this study.
   - Human testing still matters most for locating the goal (see below).

## Answers to the update's questions

- **Should anchoring, learning-rate selection or proxy calibration come before the live screen?** No.
- **Is the three-comparison allocation worthwhile?** Acceptable. Not worth refreezing to change.
- **Finish or replace?** Finish.

## Essential, recommended and later

**Essential changes to the running design:** none.

**Recommended additions that change nothing frozen:**

- **Report the share of the teacher gap each fit closes,** as a descriptive diagnostic: (parent CE − fit CE) ÷ (parent CE − teacher entropy), on held-out positions and on the fixed training panel.
  - This is what distinguishes "trained too little" from "no useful signal" if the result is null.
  - The protocol already computes the cross-entropy; the teacher-entropy floor comes from the saved targets.
  - It selects nothing, so label it as an added descriptive measure.
- **Start organising the human-calibration games now.** This uses people's time, not machine time, so it does not compete with the running study. It remains the largest unknown relative to the goal.

**Later work:**

- Calibrate the decision-quality proxy against the known search-budget gains (64/128/256/512).
- Qualify concurrent strength games at fixed simulation counts, so that roughly 2,000-game samples become affordable.
- Run a multi-pass training follow-up only if this study is null.

## Reading the outcomes

None of these should trigger promotion, extra games or follow-up training automatically.

| Result | Reading | Next |
|---|---|---|
| Mixed beats EF8A (≥55%, Bonferroni lower bound >50%, seats ≥50%) | First demonstrated learning gain | Confirm with a larger, preferably concurrent sample and a descriptive check at 512 before promotion; then a second cycle |
| Mixed beats terminal directly; neither beats EF8A | Value target matters; terminal-only probably harms | Adopt the mixed target; the open question becomes training amount |
| Neither; small gap closure (under ~10%) | Under-trained | Upstream-like multiple passes, watching held-out loss |
| Neither; large gap closure | Better predictions are not becoming better 128-search decisions | Target/decision link (proxy calibration, target design), not more data |

## Prior, for later calibration

**[S]**

- Mixed beats terminal on value diagnostics.
- Roughly a 1-in-4 chance that mixed clears the 55% rule against EF8A.
- Upstream's own later "target and iterative controls" reportedly also failed to improve this checkpoint. That came from a fetched summary of upstream `EXPERIMENTS.md`, not checked verbatim.
