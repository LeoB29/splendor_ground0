# Sampling stability: six root worlds at the same 64-simulation budget

27 October 2026. The authorized bounded study completed with full internal
verification. Pool6 showed less seed-dependent decision variation on this fixed
development panel, with very similar timing to pool3. **No playing-strength gain
or player promotion is established.** Optimized64 remains frozen.

## Execution

The attempt ran from 12:33:05 to 12:38:22 UTC (13:33:05 to 13:38:22 Paris),
**317.1 seconds / 5 minutes17 seconds**, inside the frozen one-hour cap.
Qualification took111.0s, the panel195.3s and finalization9.9s. All ordinary
requests retained the1.000s external deadline and0.900s internal limit.

- The separate offline bridge build passed in19.625s; no learned model or search
  ran during compilation. The native core and all predecessor sources stayed
  byte-identical. The only new native configuration choice is root pool3 or6.
- **144 model-free tests passed**, covering the fixed matrix, seed derivation,
  strict pool/profile identity, deadlines, corrected-rule routing, evidence
  preservation and complete/partial analysis.
- **129 qualification decisions passed:** the two52-decision actual-entrypoint
  suites and25 exact parity requests. The new route exercised pool6 as the
  challenger and retained the16-simulation/pool3 reference. Both corrected-rule
  suites, all90 native card faces, negative inputs and graph checks passed.
- **324/324 panel requests passed:** 24 unchanged-player historical
  reproductions,72 pilot requests,216 remaining comparison requests and12
  prescribed exact repeats. All24 roots have both pools and all six seeds.
- **453 scientific decisions**, **355 parent requests**, **44 graph cases** and
  four serial worker lifecycles reconcile with the frozen plan. The parent count
  differs because two requests bundle the104 entrypoint decisions.
- No timeout, unexpected rejection, output mismatch, retry or replacement root occurred.
  All three owning processes had actual exit code0, and cleanup reported no errors.
  The execution status is `completed_verified`.
- The cost-only pilot passed every gate. Peak sampled combined worker/helper
  RSS was332,095,488 bytes (316.7MiB). The final seal binds2,532 artifacts totaling
  81,470,786 bytes (77.7MiB); logical evidence accounting remained within the
  separate frozen ceiling. No evidence was deleted to fit.

The freeze binds3,706 inputs and the complete1,380-file accepted predecessor
attempt inventory. Earlier failed attempts remain preserved and incomplete where
originally recorded. Preparation test-fixture and review-reader errors are also
retained; neither caused an experimental retry or a change to a frozen artifact.
Independent postrun review **accepted the complete attempt**, recomputed the
saved science/parity/analysis and confirmed all27 recorded owner/worker identities
were absent. It authenticated the finite source/seal/preservation inventories and
reused the sealed passed raw, graph and startup receipts. That review took7.3s,
ran no models or games, and is recorded separately under `postrun`.

## What was compared

Both settings used the same pretrained network, optimized inference graph,
float32 precision, depth64, batch1, one thread, PUCT and auxiliary move handling.
Each main search used **64 total simulations**, not64 per world. The maximum
neural callback count was65 for both settings. Mean counts differed slightly
(55.29 versus55.85) because terminal paths and trajectories can differ.

The24 saved development positions cover six phase/seat cells. Each was evaluated
with six fixed observer-derived seed families A-F, at both pool3 and pool6. A-C
retain the earlier seed construction; D-F were fixed before execution. No seed,
position or candidate was selected after inspecting the new results.

The bridge's effective root pool changed from3 to6. Its native core remained
unchanged. Root-world construction and later hidden completions used to encode
neural inputs consume a shared RNG, so the setting changes both parts of the
sampling sequence. This does not isolate root-world count as the sole cause.
The nominal `chance_universes=3` setting remains unchanged and is inactive on
this canonical batch1 path.

## Stability result

For each position, the six decisions produce15 unordered seed pairs. Its
disagreement rate is the fraction of those pairs choosing different actions.
The primary result averages the24 paired position-level rates. The360 pairs are
not treated as360 independent observations.

| Measure | Pool3 | Pool6 |
|---|---:|---:|
| Mean main-action disagreement | 12.5% | 5.0% |
| Positions where all six seeds agree | 18/24 | 21/24 |
| Median complete response | 0.3623s | 0.3623s |
| 95th-percentile complete response | 0.3930s | 0.3941s |
| Maximum complete response | 0.4303s | 0.4465s |
| Comparison decisions | 144 | 144 |

The mean paired difference is **-7.5 percentage points**. Four positions became
less variable, twenty had equal disagreement, and none became more variable.
Complete compound-action stability gave the same result as main-action stability.
Matched-seed choices differed between pools in14/144 cases across six positions.

The four improved positions were:

| Root | Phase / seat | Disagreeing seed pairs, pool3 | Pool6 |
|---|---|---:|---:|
| root_02 | plies0-19 / seat0 | 11/15 | 5/15 |
| root_05 | plies20-39 / seat1 | 8/15 | 0/15 |
| root_17 | plies40+ / seat0 | 5/15 | 0/15 |
| root_18 | plies20-39 / seat1 | 8/15 | 0/15 |

All18 positions already unanimous at pool3 remained unanimous and kept their
action. Two other roots, root_12 and root_15, changed some moves without changing
their disagreement rate. Equal stability therefore does not always mean identical
choices. The improvement spans three of six phase/seat cells; the middle/seat1
cell accounts for16 of the27 removed pair disagreements. These breakdowns are
descriptive development observations, not new subgroup-selection criteria.

A postrun check of the prospective deployment seedA shows changed decisions on
only **2/24 roots**, root_12 and root_18. That check does not redefine the
prespecified six-seed primary measure. It illustrates why a real playing-strength
gain could be modest even with a noticeable stability change across seeds.

The maximum across all349 individually timed ordinary requests (25 parity plus
324 panel requests) was0.4465s. The24 unchanged-route historical reproductions
had a maximum of0.3941s. The bundled qualification decisions also passed their
internal deadline checks, but are not added to these external timing denominators.
These are saved-position timings on this machine, not a live-game reliability
guarantee or proof of exactly equal runtime.

## What this result establishes

The six-world configuration is a qualified, practical follow-up candidate for
this panel. At the same search-simulation ceiling it made choices less sensitive
to the prescribed sampling seeds, without a material timing difference in these
measurements. It passed the actual corrected-rule, exact-control, graph, repeat,
clock and evidence checks. This supports testing the candidate in real games.

It does **not** prove that the more consistent actions are better. A player can
be consistently wrong; all evaluations use the same inherited network. The24
positions are historical development evidence, and the seed-pair outcomes are
correlated. There is no population significance claim, independent oracle,
Elo estimate, causal model-capacity conclusion or proof that six is an optimal
pool size. Optimized64 is not replaced on the basis of this result.

## The next step

Recommend one separately frozen **fresh paired whole-game comparison of the
pool6 candidate against unchanged optimized64/pool3**, using the canonical
observer-derived seedA and the same64 simulations and one-second deadline.
First qualify the complete live-game route against these saved outputs and run
a small six-game operational gate. The saved-panel worker rejects ordinary roots
at turn124 or later; a live wrapper must preserve the incumbent's existing late-game
heuristic and complete-turn handling rather than directly reuse that restriction.
Then use a fixed assessment sample and swapped
seats, preserving timeouts as losses and all verification checks. No live-game
adapter or benchmark was launched by the present diagnostic.

An assessment of100 swapped-seat pairs/200 games is a reasonable bounded next
screen, subject to a concrete timing/storage plan before launch. It could identify
a substantial improvement; a result near50% would leave small effects unresolved,
not establish equivalence. Freeze the sample, acceptance criteria and stopping
rules before any fresh outcomes; do not extend the run because its result is
close. Do not tune additional world counts or seeds on this panel first.

## The broader picture

The project moved to a verified pretrained policy/value network with search,
then demonstrated an improvement from32 to64 simulations and addressed runtime
reliability. Weak external opponents did not provide a useful generalization
test, so that work remains paused. The previous saved-position diagnostic showed
that simply adding much deeper search was expensive and sometimes unstable.

This study tests a smaller intervention: use the existing affordable simulations
over a broader root-world pool. The measured stability gain and similar timing
give a concrete candidate for the next strength test without collecting a larger
training dataset or increasing the network. The missing link is whether these
different decisions win more complete games. Independent strong opposition and
eventually expert-human comparisons remain relevant to the wider goal of
exceptional, lawful-information Splendor play; this local diagnostic does not
settle that goal. Larger-data work and training remain deferred.

## Evidence

- Protocol: [bounded study](splendorust_sampling_stability_20261027.md).
- Source: `tools/splendorust_sampling_stability_20261027`.
- Execution: `outputs/splendorust_sampling_stability_20261027_01`.
- Freeze SHA256: `cc550ef7f48597840ed5e41166b317574ef78d8ad88d3ae11b6ef5baca5b1eaa`.
- Analysis SHA256: `b673a2b4ab01685307257b2050e44f6ff5b9297799f65602996170d9a7776f73`.
- Completion SHA256: `abb9ca0ae424a386bcb91c187f4e2ce3370d1196968b642ef8c3fc31acd29a58`.
- Verification SHA256: `43252446c91d978ae0beeacd26e08185b7c05b80ff3a77215a727e82ba798e69`.
- Evidence seal SHA256: `f08e1f9f79b6fef76c9335157c2114ae62b0ce3d195c77cdf239012f6ee4e664`.
- Actual outer exit SHA256: `05aee9ab524d906a4d2ef685a4b13ceca6c66cc2f8667e17b40eafba33cfc0da`.
- Prelaunch readback SHA256: `588efbc126d5e1f2aa32f6610e93c0d41487eadade3189d55f6e236e66126e1f`.
- Independent completion review: `postrun/independent_completion_review.json`,
  SHA256 `7c1fbf2740a7acfd83bf98eb8ffe2a57266b80271f763362d092542bb9f0a952`.
