# Search256 versus unchanged128: accepted recovered result

Closed out 31 October 2026, Europe/Paris. The amended recovered benchmark is
complete and independently accepted. The original interrupted attempt and the
first recovery remain incomplete and preserved; this report does not relabel them.

## Strength result

Across all 400 main games (200 swapped-seat pairs), the 256-simulation player
won 242, lost 154 and drew four against unchanged128. Its score, counting a draw
as half a win, is **61.0%**, with a paired bootstrap 95% interval of
**56.5–65.375%**. Candidate-seat scores are **62.75% and 59.25%**.

All frozen success criteria passed: at least 55% overall score, paired interval
lower bound above 50%, at least 50% in each seat, and complete verification.
No games ended in research hard-guard forfeits.

The single previously authorized fresh-clock retry of an unresolved move in
game235 remains disclosed. Conservatively treating that entire game as a 256
loss gives **60.75%** (241 wins, 155 losses, four draws), interval
**56.25–65.125%**, and seat scores 62.25%/59.25%. All criteria still pass.
The final continuation introduced no further retries or replacement games.

The comparison isolates 128 versus 256 simulations with the same pretrained
network, three hidden worlds, corrected rules, exploration settings and frozen
implementation. It establishes a useful gain in this head-to-head comparison;
it does not establish universal superiority or one-second deployment readiness.

## Timing, reported separately

These figures cover only the 400 main games, with 11,066 completed decisions per
player. The clock ends on complete response arrival; subsequent parent
validation is excluded. Means are weighted from verified seat aggregates.

| Measurement | 256 simulations | 128 simulations |
|---|---:|---:|
| Mean complete response | 1.4057 s | 0.7506 s |
| Completed moves over one second | 10,073 (91.03%) | 983 (8.88%) |
| Longest completed response | 7.1987 s | 2.6655 s |

The mean response cost is about 1.87 times higher. The one-second exceedances
are descriptive, not losses: correct late moves were accepted under the frozen
nine-second internal / ten-second complete-response research guard. The 700 ms
margin remains descriptive. These live-game timings are not a causal timing
experiment on identical positions; hardware and scheduling contributions still
need measurement. They nevertheless show that 256's current latency issue is
common, rather than confined to a few isolated late moves.

## Verification and preservation

All 400 main games, six operational games, 48 model reproductions and 22,460
ordinary game decisions passed the saved checks. The independent reader
approved the sealed evidence. Runner, watchdog, reader and guardian actual
exit codes were zero, cleanup was clean and their exact process identities
were absent at closeout. The retained guardian handle was observed to exit zero
in exec session5563, chunkc07af9; its exact exit time was not recorded.

The independent reader finished on 30 October at 20:37:24 UTC / 21:37:24 Paris.
All original stage and overall deadlines were satisfied. At closeout, all 1,454
frozen source/input pins were rechecked. No new games or model calls were run
for closeout, and no frozen source or evidence was edited.

Freeze SHA256:
`6cf5f864e451926cf530c8b78468ac7f496fff159f51cab0d62d29b5c80d8149`.

Primary receipts:

- [Main analysis](../../outputs/splendorust_search256_recovery_20261030_02/main_analysis.json)
- [Historical retry sensitivity](../../outputs/splendorust_search256_recovery_20261030_02/recovery_sensitivity.json)
- [Performance](../../outputs/splendorust_search256_recovery_20261030_02/performance.json)
- [Independent saved-evidence review](../../outputs/splendorust_search256_recovery_20261030_02/independent_review.json)
- [Independent final acceptance](../../outputs/splendorust_search256_recovery_20261030_02_closeout/independent_acceptance_01.json)
- [Actual guardian exit and main-only timing derivation](../../outputs/splendorust_search256_recovery_20261030_02_closeout/actual_guardian_exit_01.json)

## Recommended next step

Retain 256 as a verified stronger research challenger and keep unchanged128 as
the current reference pending an explicit promotion decision. Prepare one
bounded efficiency diagnostic on a fixed representative panel of saved
positions, separating inference, search work and host/supervision overhead.
Qualify measurement overhead and exact outputs first; then choose one lossless
optimization only if the measurements support it. Do not infer that host
scheduling explains the whole difference, or that communication alone can
remove it. This recommendation launches no new work or 512-simulation sweep.

This follows the [28 October roadmap](../strategy/splendorust_roadmap_update_20261028.md):
establish a strength benefit first, then invest in practical execution when that
benefit warrants it. Earlier 64-to-128 improvement and this 128-to-256 gain show
that the fixed network can still benefit from additional search. The next
question is how much of the stronger play can be delivered within the eventual
one-second target. External testing, six-world work, training and larger-data
collection remain deferred. Scheduled chat monitors remain cancelled.
