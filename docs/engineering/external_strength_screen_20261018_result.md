# Corrected-rules external screen result — 18 October 2026

The six card-cost disagreements are resolved in favor of the licensed BGA and
pinned external catalogue. The adapted external PPO passed compatibility and
timing qualification. The fixed 60-game screen then completed: **raw 035 won
60 games, with no draws or forfeits**, across 30 initial worlds played in both
seats. All 3,774 game transitions passed replay and every move arrived within
the one-second allowance.

This opponent is too weak in this matchup to justify an expensive planner
benchmark. Retain it as a verified inexpensive control, and seek a stronger
independently qualified opponent on the corrected rules. The larger-data study
remains deferred; no training or 402-game benchmark was launched.

## Authoritative component resolution

The [licensed Board Game Arena tutorial](https://en.boardgamearena.com/tutorial?game=splendor)
exposes all 90 card faces and ten base nobles. Its complete semantic catalogue
matches the pinned external source. The historical local catalogue contains
these six cost errors; every affected card has a white bonus:

| Card | Historical local cost | Corrected cost |
|---|---|---|
| T1_05 | blue 2, green 2 | blue 2, black 2 |
| T1_07 | blue 1, black 1 | white 3, blue 1, black 1 |
| T2_10 | white 2, blue 3, green 4, red 3 | white 2, blue 3, red 3 |
| T2_12 | green 4, red 5 | red 5 |
| T2_13 | green 4, red 5, black 3 | red 5, black 3 |
| T2_14 | white 6, green 4 | white 6 |

[Asmodee identifies BGA's adaptation as a joint project with Space Cowboys](https://www.asmodee.co.uk/blogs/news/splendor-released-on-board-game-arena).
The [publisher's classic rulebook](https://cdn.svc.asmodee.net/production-unboxnowcom/uploads/2022/02/Splendor-EN.pdf)
also visibly shows T2_12 costing five red only, in the page-two setup image.
The independently extracted 90-card/10-noble correspondence and source hashes
are in `outputs/component_adjudication_20261018_01/independent_component_adjudication.json`.

One provenance limitation is explicit: the tutorial embeds release 210916-2237,
whose version-specific scripts returned HTTP 403. Its color encoding was
decoded using the saved current release 260627-0953 scripts/CSS. Applying that
decoder to the older database is an inference supported by the publisher
illustration and complete 90-face correspondence; all six cards were not
individually photographed from a publisher component sheet.

The [publisher's current rules](https://cdn.svc.asmodee.net/production-spacecowboys/uploads/2025/10/SCSPL01EN_SPLENDOR_RULES_LIGHT.pdf)
also establish compulsory noble visits after any turn. The historical local
engine only awarded them after a purchase. The new referee corrects that rule,
including multiple eligible nobles and full-round termination after a
non-purchase noble award.

The corrections live in a separate versioned referee,
`splendor_base_qualified_rules_20261018_v1`, catalogue SHA256
`db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312`.
Historical engine files, models, data and results remain intact. The existing
application has not been globally migrated. Earlier four/eight-candidate
strength and timing results describe the historical rules; they do not prove
playing strength or identical decisions under the corrected rules. This screen
tests transfer of frozen raw 035 without retraining.

## External qualification and preserved failures

The external checkpoint is from `roeey777/Splendor-AI` commit
`95f84d2e6e839c0ef09ca97bdc3b3048a792fb0b`. Its SHA256 is
`7fdb58b46d00305af42fad818b05549692e27ff1243cb63a76c6a858ba348df4`.
The qualified profile is `official_rules_buy_and_token_alias_v1`.

The published implementation had three compatibility problems: its entry point
unpacked two outputs from a three-output network; a nonstandard seven-card
purchase cap could remove the only legal move; and some source token actions
required translation into official take-and-return actions with exactly the
same complete successor state. The adapter repairs these explicitly, retaining
weights, feature arithmetic and all 3,510 output indices. It has no fallback.
The original failure and intermediate purchase-only profile's return-support
failure remain saved alongside the final successful receipt.

Qualification passed 115 preflight tests, 20 exact pinned-source reference
fixtures and 1,101 complete successor comparisons for token aliases. Features,
unmasked logits/value, and probabilities/first argmax under the independently
verified adapted mask matched exactly. This is an adapted opponent, not a claim
that all its masks or choices equal the unchanged published agent. During the
screen, 820 of its 1,887 decisions selected an equivalent token alias, spanning
all 60 games. Blind reserves and some payment variants remain outside its
selected legal subset. Empty support remains fatal; none occurred during the
timed panel or games.

Both players then passed all 92 real-clock requests on the 18 predetermined
core positions plus five edges, each with an original and hidden-state variant.
All 46 privacy pairs produced identical decisions. Maximum qualification
response time was 0.1173 seconds. These were explicitly converted
corrected-rules fixtures, not claims to reproduce old choices after costs changed.

## Fixed screen and timing

The pre-outcome schedule contained 30 fresh 256-bit seed pairs, with swapped
seats and reset players. Collision checking covered 226 discovered historical
manifests/schedules/registries; 22 files larger than the declared 8 MiB scan
ceiling were inventoried but not searched. This was not an exhaustive scan of
all historical game logs. The namespace and every omission were frozen before
play. All paired initial worlds were independently reconstructed and matched.

| Outcome from PPO's perspective | Result |
|---|---:|
| Wins / draws / losses | 0 / 0 / 60 |
| Mean game score | 0% |
| Score as first player / second player | 0% / 0% |
| Timeout, crash or illegal-action forfeits | 0 |
| Accepted game moves | 3,774 |

The declared 50,000-resample paired percentile bootstrap yields [0%, 0%]
because all 30 observed pair scores are zero. This is a degenerate empirical
resampling interval, not evidence of zero uncertainty or a zero future winning
probability. There was no sample extension or post-outcome success threshold.

Each player made 1,887 game decisions under the real one-second clock:

| Player | Mean | Median | 95th percentile | Maximum |
|---|---:|---:|---:|---:|
| Raw 035 | 24.56 ms | 24.45 ms | 32.96 ms | 206.67 ms |
| Adapted PPO | 30.39 ms | 28.13 ms | 44.19 ms | 103.14 ms |

Timing includes request serialization and complete response receipt.
Initialization was separate: 61 workers, median 2.776 seconds, maximum 3.930
seconds, below the 60-second limit. Each worker used one numerical thread on
verified performance-core logical CPU 0 at normal priority. All 60 games ended
naturally; no repetition, no-progress or forced-pass draw was substituted.

The bounded qualification, games and supervisor replay took **650.34 seconds
(10 minutes 50 seconds)**, within four hours. Peak sampled worker RSS was
215.02 MiB. Screen evidence including the independent analysis occupies about
32.03 MiB; separately saved source-adjudication evidence occupies 19.65 MiB.
RSS/CPU are sampled measurements. Runtime guards enforced the disk/memory
bounds, but a continuous free-memory/free-disk history was not saved. These
measurements do not price a 128-world planner or guarantee future latency.

## Independent acceptance and next step

The independent analyzer verified all 840 frozen input hashes and byte sizes,
including the prior 563-input qualification freeze and its 451 original
dependencies. It reconstructed initial worlds, bound every saved move to its
original clock receipt, checked legal observations and selected PPO index/alias
semantics, replayed all 3,774 transitions with the separate dictionary ledger,
and independently recomputed winners, paired scores and bootstrap statistics.
It did not load models or recompute every game's full neural outputs or masks;
neural equivalence was established on the declared qualification fixtures.

The result supports using raw 035 and this adapted PPO operationally at one
second on this sampled setup. It establishes raw 035's superiority in this
specific development matchup, not elite strength and not a ranking of all
published PPO agents or four/eight-candidate planners.

The next useful work is to make the corrected rules the explicitly versioned
baseline for future evaluations and qualify a stronger external control.
Before a planner benchmark under those rules, requalify its mechanics and
timing against the corrected referee. Preserve the historical experiment
lineage; do not relabel old results as official-rules results. This weak-control
screen does not justify a costly 402-game planner arena or larger-data study.

Protocol: `docs/engineering/external_strength_screen_20261018.md`.
Evidence: `outputs/external_strength_screen_20261018_01/`.
Freeze SHA256:
`0aa6103ca0d2662c181a86d1088563c56e0ee609f5d4144bce5476e8a81c27d5`.
Completion and both replay/analysis receipts report success; no process remains
running and the completed panel must not be relaunched.
