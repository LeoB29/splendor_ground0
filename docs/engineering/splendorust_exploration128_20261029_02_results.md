# Exploration at 128 simulations: attempt 02

29 October 2026. **The complete attempt is independently accepted.** Neither
challenger qualified for confirmation. The unchanged 128-simulation player at
coefficient 0.4 remains the reference. No new experiment has been launched.

## Verified development result

The study compared two exploration coefficients against the unchanged
128-simulation reference at coefficient 0.4. All arms retained three sampled
root worlds, the same inherited network and weights, precision, search quota,
first-play urgency, corrected rules and auxiliary decision behavior.

| Challenger coefficient | Wins | Draws | Losses | Score | Seat 0 | Seat 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.2 | 53 | 1 | 46 | 53.5% | 61% | 46% |
| 0.8 | 50 | 1 | 49 | 50.5% | 49% | 52% |

A draw contributes half a point. Each challenger played 100 games on 50
swapped-seat setups; both comparisons shared those setups. They are not two
independent confirmation studies and must not be pooled as one candidate score.

Neither challenger met the prespecified 55% overall score and at least 50%
from each seat. There were no administrative losses, so official and conservative
scores agree. Independent selection arithmetic passed; `selected_candidate` is
null. The frozen plan therefore ends without the conditional 400-game
confirmation. No new setting is promoted and no sample extension follows.

## Verification and preservation

The 200 development games, 11,034 development moves and 400 development game
worker closures passed the phase checks. All 96 selected exact reproductions
passed, 48 for each matchup. The earlier qualification and 12 operational games
also passed their recorded checks. All existing failures from attempt 01 remain
preserved; its resource-admission refusal is not retrospectively accepted.

The final stage began at 20:16:41 UTC on 29 October under the original clock.
The main saved-evidence audit passed at 20:31:50 UTC; the runner and watchdog
both exited with code 0 and empty cleanup errors. The independent saved-only
reader approved the complete evidence and exited with code 0 at 20:51:31 UTC
(21:51 Paris). The retained owning session 45073 then reported actual guardian
exit code 0. No model decision was retried and no new game was played during
finalization.

The accepted workload reconciles 212 games, 11,708 live-game decisions, 364
qualification scientific decisions, 96 exact reproductions, 524 worker trees,
5,764 graph cases and 12,428 parent requests. The source inventory contains
752 pins. The final evidence seal contains 85,321 files totaling 3,079,054,627
bytes (2.868 GiB). All 3,148 recorded owner, reader, worker and helper identities
were independently checked absent after successful exits. No experimental
runtime failure, hard-guard forfeit or failed-decision retry occurred.

The study used about 6 hours 32 minutes through reader cleanup, within its
original 48-hour allowance and the original final-stage deadline. No clock,
sample, player, threshold, source pin or frozen artifact was changed. Acceptance
is recorded separately in
`outputs/splendorust_exploration128_20261029_02_postrun/acceptance.json`.

## Descriptive live-game timing

The saved timing summary covers all 12 operational and 200 development games,
11,708 completed decisions. It excludes qualification and exact reproductions.
The clock runs from request preparation to trusted complete-response arrival,
before subsequent logging, decoding and verification.

| Player | Decisions | Median | 95th percentile | Maximum | Over one second |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unchanged128, coefficient 0.4 | 5,854 | 0.762 s | 1.032 s | 1.705 s | 359 |
| Lower exploration, coefficient 0.2 | 2,930 | 0.757 s | 0.991 s | 7.019 s | 138 |
| Higher exploration, coefficient 0.8 | 2,924 | 0.760 s | 1.079 s | 1.848 s | 218 |

No decision failed the 9-second internal / 10-second response guard. The late
completed moves remain in their games, as specified. These distributions reflect
different trajectories and host conditions; they do not establish a causal speed
advantage from changing the coefficient. In particular, the long low-exploration
outlier is preserved, not dismissed or removed. The study does not certify
strict one-second reliability. The independent final reader verified this
summary against the saved evidence.

## What this result establishes

The two tested exploration changes did not earn confirmation at the fixed
128-simulation budget. This is a valid negative screening outcome rather than
a software failure. It does not establish that coefficient 0.4 is globally
optimal, prove that either alternative is weaker, or exclude a small benefit.
The 0.2 score is only a development estimate; choosing it because its observed
score is above 50% would disregard the frozen selection rule.

## The next step

Retain unchanged128 and recommend one preselected
256-versus-128 comparison with coefficient 0.4, three worlds and the same weights.
The purpose is to measure remaining strength gains from more search and their
actual time cost. The earlier saved panel measured a 256-simulation median of
1.423 seconds, with 20 of 24 responses above one second: this would be a research
strength experiment, not immediate qualification for one-second deployment.

The proposed comparison should use a separate implementation and freeze,
correctness qualification, six operational games, an outcome-blind resource
forecast, then 400 fresh games on 200 swapped-seat setups. Preserve the existing
useful-gain criteria, independent verification and conservative treatment of
administrative losses. Retain correct completed late moves under the symmetric
9/10-second research guards; report timing separately. Neither 700 ms nor one
second is an admission gate. No experiment is launched by this recommendation.
There is no automatic promotion or subsequent 512-simulation experiment. An
inconclusive result leaves the reference unchanged and closes this comparison.

## The broader picture

The verified gains from 32 to 64 and from 64 to 128 simulations show that this
pretrained network still benefits from additional search. The six-world and
exploration-coefficient studies have not demonstrated a cheaper improvement.
A single further quota comparison would test how far the successful direction
extends without beginning a broad setting sweep or a training programme.

If 256 earns a useful gain, it becomes evidence for later efficiency work or a
possible stronger teacher; it does not automatically replace the practical
reference or establish that distillation/self-play will work. If it does not,
the case for further simple quota increases weakens. Local parent comparisons
remain narrower than exceptional strength against diverse strong or human
opponents. External testing, six-world work, training and larger-data collection
remain deferred.

## Evidence

- Protocol: `docs/engineering/splendorust_exploration128_20261029_02.md`.
- Output: `outputs/splendorust_exploration128_20261029_02`.
- Freeze SHA256: `3d899628cf0ba1a50688d53abe9a4b329e364131e269729eeac3b1f03bf87cbd`.
- Verified phase: `phase_verification/development.json`.
- Selection: `selection.json` and `selection_verification.json`.
- Independent approval: `independent_review.json`, SHA256
  `457e15277e12d20e9908593fc2ad13bdbb9ef156d0d892f25e27272eee67e0ad`.
- Actual reader exit: `independent_review_os_exit.json`, SHA256
  `11090fd2d0a8c42ac8314a7709f851131e536df7c433fadf1aa2c043960ebebb`.
- External acceptance receipt SHA256:
  `8113aeee97cfa9f05f5bb44124eeaad497fd2ecc46c833d0ea472e5aec994a40`.
- Earlier reference result: `docs/engineering/splendorust_search128_saved_audit_20261029_results.md`.
- Earlier deeper-search panel: `docs/engineering/splendorust_decision_quality_20261027_attempt02_results.md`.

The pinned roadmap, protocols, sources and experimental evidence remain
unchanged. Scheduled chat monitoring remains cancelled by the user's request.
