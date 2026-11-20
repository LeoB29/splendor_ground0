# Splendor: current learning study for independent review

10 November 2026; status checked at 14:13 Paris. Update for the author of the 8 November audit, with departures disclosed for review.

## Goal and latest evidence

Our goal remains superhuman strength in corrected base two-player Splendor. SplendoRust's original EF8A policy/value network with search remains the reference. Its strength against strong humans is unknown.

The independent-data study passed recovered verification: its 96-update broad-data models scored **48.625% against EF8A**, paired 95% interval **44.00 to 53.25%**, across 400 games. This neither established a gain nor excluded a small one. The subsequent 96-update mixed-target study was cancelled before live games, with evidence preserved; cancellation is not a scientific negative.

## What is running now

**Does full-corpus training help, and does an outcome/search value target help more than outcomes alone?** This frozen study collects no new self-play games.

- Training: all **57,022 eligible positions from 1,024 games**, collected with original EF8A at 512 simulations and three sampled hidden worlds under corrected rules.
- Diagnostics: 64 validation and 64 test games, 7,086 positions total. Both splits were inspected previously: they are reused diagnostics, not independent confirmation.
- Four fits: two value-target treatments, each with two fixed paired training seeds. Every fit starts from pristine EF8A.
- Each fit uses every position once: **1,782 updates**, batch 32, final batch 30. Both heads and shared layers train jointly. Everything except value targets matches between treatments.
- AdamW learning rate stays **1e-5**. No learning-rate sweep, checkpoint selection or best-seed selection.

**Terminal treatment:** regress the actor's value toward the eventual game outcome, `z` = -1, 0 or +1.

**Mixed treatment:** regress toward `(z + 0.837*q) / 1.837`, where `q` is the saved 512-search root value mapped to [-1,1]. This is the authenticated visit-weighted root estimate, not the chosen action's value or a raw network prediction. Both treatments use the same search-visit policy targets.

New since the audit: saved root estimates are available, and this formula matches the verified upstream EF8A recipe. Our policy targets and optimization settings differ; this is not a reproduction of that whole recipe.

## Evaluation and present status

| Fresh live comparison | Main games | Purpose |
|---|---:|---|
| Mixed vs original EF8A | 400 | Learning gain from mixed recipe |
| Terminal vs original EF8A | 400 | Learning gain from outcome-only recipe |
| Mixed vs terminal | 400 | Primary controlled value-target contrast |

Each comparison has 200 games per training seed in swapped-seat pairs. All players use 128 simulations and three worlds; 36 operational games are excluded. One second/700 ms are descriptive. Completed late moves count normally; existing 9-second internal/10-second full-response guards remain, with hard timeouts counted as losses.

The primary target contrast uses a paired 95% interval. Parent-improvement claims use 97.5% intervals for each of the two comparisons. The retained useful-gain rule requires a score of at least 55%, a lower bound above 50%, and at least 50% in each seat, alongside existing verification safeguards. No automatic promotion.

Training qualification and cache construction passed; independent cache auditing is running. **No full-corpus scientific fit or live game has started; no current failure is recorded.** Subsequent stages run automatically, including replays and independent acceptance. Forecast: 40 to 55 hours total; caps: 120 hours overall, 96 hours for model work.

## Departures requiring your assessment

1. **Search mixture rather than parent anchoring.** Available targets and upstream precedent make this credible, but it tests a different hypothesis. It does not directly isolate protection against value drift; search estimates can inherit network bias. There is no evidence yet that it is preferable to anchoring.
2. **Three 400-game screens rather than a powered confirmation.** They distinguish recipes but sacrifice precision: no single direct contrast has 1,200 games. Small useful gains may remain unresolved, especially with the retained 55% rule rather than your suggested 53% criterion.
3. **Fixed learning rate.** This simplifies the contrast but leaves optimization adequacy untested. A prospectively specified validation-based sweep could also be fair; the fixed rate is not scientifically necessary.
4. **No calibrated decision proxy or human anchor yet.** Live games avoid relying on an unvalidated proxy, but your proposed positive controls address that concern and could improve research efficiency. Human testing requires separate organization and remains necessary to locate us relative to the ultimate goal.

**Questions:** Should anchoring, learning-rate selection or proxy calibration precede the live screen? Is the three-comparison allocation worthwhile? At this early stage, would you finish this study or replace it with a separately frozen design? Please distinguish essential changes from later work; being frozen is not scientific justification.

## Records

- Current protocol: `docs/engineering/splendorust_full_corpus_20261110.md`
- Current evidence: `outputs/splendorust_full_corpus_20261110_01`
- Prior audit: `docs/strategy/splendor_project_independent_audit_response_20261108.md`
- Accepted previous study: `outputs/splendorust_independent_data_disk_recovery_20261109_02`

The current run remains unchanged while this review is requested. Further collection, external-engine testing, six-world work and rentals remain deferred.
