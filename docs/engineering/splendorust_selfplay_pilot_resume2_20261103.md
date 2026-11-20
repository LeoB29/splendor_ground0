# Self-play pilot continuation after the omitted RAM guard

3 November 2026. The user instructed: “ok resume the pilot after your tests if
you havent already done it pleas,e thanks”. This follows the explicit request
to stop resource-forecast refusals and the recommendation to resume with the
tested correction and a generous actual completion window. No additional launch
confirmation is required after the checks below.

The original pilot and its first continuation remain incomplete, separately
preserved attempts. The original pilot completed games 0–31. The first
continuation completed games 32–47, then rejected a new worker because a
hard-coded available-RAM check had been missed. Graph qualification passed, but
the 67-case scientific suite had not begun and game 48 had not started. The
failed request stays rejected; this continuation does not relabel it as passed.

## Scientific design stays fixed

Keep all 256 original source specifications, seeds, corrected initial states and
splits: 192 training, 32 validation and 32 offline assessment games. The first
48 complete games remain at their original paths under their original hashes.
Continue from game 48 (the 49th game), block 6, with the remaining 208 games.
No sample, seed, target or result is replaced. Engineering fixtures are not
training examples, and no engineering checkpoint is reused.

The teacher remains the original full network with 512 simulations, three
worlds, exploration coefficient 0.4, depth 64, float32, one inference thread
and CPU0. The original checkpoint, architecture, fixed two-epoch joint
policy/value fitting recipe, losses, weighting, order and seeds are unchanged.
There is one final checkpoint, with no selection based on assessment results.

Keep six operational games followed by 400 paired games, with both the trained
candidate and original parent using 128 simulations. Keep all original seeds,
swapped seats, outcome rules, statistics, acceptance criteria and sensitivity
analysis. No automatic promotion, external-opponent tests or larger-data study.

## Correction and saved evidence

Apply the hash-bound correction in
`tools/splendorust_selfplay_pilot_ram_policy_20261103/patch.py` to both the source
and assessment worker before changing their package namespace. It replaces
only the available-RAM rejection inside the qualification boundary. Every other
worker source byte is unchanged apart from the disclosed namespace routing.
The actual scientific suite and all real request/RSS/disk/evidence guards remain.

Tests must reproduce the saved original failure and exercise the actual new
source and assessment handlers below 2 GiB available RAM. They must also cover
the active controller, shared assessment and retained-owner admissions, imported
supervision paths, fixed scientific counts, actual deadlines and process cleanup.
Do not launch through the historical workflow owner's old reserve-admission path.

Seal all four prior evidence trees: original pilot and launcher, first
continuation and retained owner. Reuse the passed original 32-game raw audit
only under its complete preserved inventory. Independently validate the next
16 complete games and both closed successful source workers using their own
frozen readers. Check the failed `collect_006` graph, raw error response,
profiles, forensic record, process exits and cleanup separately. Confirm zero
scientific cases and zero game decisions in that failed qualification attempt.

There are seven successful preserved source lifecycles and one failed startup
lifecycle. New workers `collect_006` through `collect_031`, then `replay_on`,
give 27 new successful lifecycles. Report 34 successful source lifecycles plus
the one preserved failed startup: 35 actual source lifecycles in total. The
failure contributes two parent requests and 11 graph fixtures, but zero
scientific-suite or ordinary-game decisions. Charge its evidence and lifecycle
to the cumulative limits. Its namesake new worker has a distinct output path
and exact process identity.

The complete final corpus loader still validates all 256 trajectories and
official terminal targets. Retain the exact original OFF/ON capture qualification
and fitter tensor counterexamples. Retain all 48 source and up to 96 assessment
reproductions, raw transport checks and independent final verification.

Every new collector, trainer and evaluator retains the corrected ruleset,
catalogue and turn contract and calls `verify_runtime_sources()` once at startup.
Every live inference lifecycle passes its actual graph and full corrected
entrypoint checks before game decisions begin. No failed game decision is retried.

## Advisory forecasts and explicit amended time windows

Forecasts and available-host-RAM reserves are information, not admission or stop
gates, in every phase. Keep their recorded arithmetic and failed predicates;
do not turn them into passed estimates. No RAM stability wait is required.
Keep the actual 5 GiB disk floor rather than a projected whole-study reservation.

The original pilot origins remain monotonic 904900.656 and Unix
1793649048.764625. This new plan explicitly extends the total window to
120 hours and the model-work window to 108 hours, including all elapsed time.
Their absolute ends are 7 November 2026 at 19:50:48.764625 UTC and
07:50:48.764625 UTC respectively (20:50 and 08:50 Paris).

The original outer-workflow origins remain monotonic 900586.578 and Unix
1793644734.6388845; its new allowance is 124 hours. The old 76-hour owner claim
is preserved as provenance, not rewritten. No active decision clock is reset.

The collection window is 72 hours from the original collection start, clipped
to the amended model cutoff. Keep that original start and old 34-hour deadline
as historical evidence; bind the amended deadline separately. Remaining stage
ceilings are: fit 8 hours, assessment qualification 2 hours, operational games
2 hours, main assessment 40 hours, reproduction 8 hours, saved final work
12 hours. Every stage is clipped to the amended total/model limits. These are
generous ceilings, not predicted durations or authority for extra samples.

Keep the actual cumulative 48 GiB evidence ceiling, 8 GiB final evidence reserve,
2 GiB combined inference-tree limit, 4 GiB fitter limit, at most two resident
inference trees and all process-ownership checks. Include all previous evidence
bytes and eight prior actual source lifecycles; the combined maximum is 839
actual source/assessment lifecycles. Allocation or actual execution failures
remain recorded failures.

Every game move still has the nine-second internal and ten-second complete
response guards. One second and 700 ms remain descriptive. No active or failed
game move is retried, no game is replaced and no sample is reduced. A genuine
correctness failure ends the attempt; a pessimistic forecast does not.

## Separate freeze, launch and completion

Use only `tools/splendorust_selfplay_pilot_resume2_20261103` and
`outputs/splendorust_selfplay_pilot_resume2_20261103_01`, with the separate
retained owner `tools/splendorust_selfplay_pilot_resume2_owner_20261103.py` and
`outputs/splendorust_selfplay_pilot_resume2_owner_20261103_01`.
Preserve every original source, freeze, failure and receipt unchanged.

Require the sealed 48-game declaration, passed incremental saved audit with
actual exit confirmation, exact-source regression tests and independent reviews.
Freeze before launch and use one retained owning process. Each stage proceeds
automatically. Completion still requires full data, fitting, operational and
paired assessment evidence, all replays, independent acceptance, actual clean
owner exits and cleanup. No scheduled monitor is created. A completion file
alone is insufficient, and neither previous attempt is retroactively accepted.
