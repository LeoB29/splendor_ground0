# Corrected practical network study: useful speedup, matching panel decisions

31 October 2026. Attempt 02 completed all qualification and 144 ordinary measurements,
passed independent saved-evidence verification, and its retained owning process
exited 0. The final-token feed-forward candidate passes the precommitted practical
speed screen and selects the same complete actions on all 18 saved positions.
This is a successful implementation diagnostic, not a playing-strength result
or one-second deployment qualification. Neither reference 128 nor challenger 256
has been replaced.

## Result

Both workers retained 256 simulations, three worlds, the same checkpoint,
float32 precision, exploration coefficient 0.4, depth 64 and CPU/thread placement.
Only the candidate narrows the last feed-forward calculation to summary token 0;
all attention, earlier blocks and output heads remain. The controller repair
corrected only its idle handoff identity lookup. Worker, adapter, panel and
analysis source files are byte-identical to attempt 01.

| Saved sample | Control mean | Candidate mean | Paired median candidate/control | Candidate moves over 1 s |
|---|---:|---:|---:|---:|
| 12 systematic roots, four paired repeats each | 1.459 s | 1.389 s | 0.9427 (5.73% faster) | 38/48 |
| 6 historical slow roots, four paired repeats each | 1.679 s | 1.580 s | 0.9486 (5.14% faster) | 24/24 |

The paired median measures the typical matched ratio; it is not the ratio of
the two means. Systematic mean saving was 69.9 ms and paired median saving 75.8 ms.
Control one-second exceedances were 40/48 systematic and 24/24 historical tails.
All measurements use the complete response, with no overhead subtraction.

All three fixed speed gates passed:

- Systematic paired median ratio 0.942742, below the 0.95 threshold.
- One-sided 95% position-cluster bootstrap upper ratio 0.959096, below 1.00.
- Historical-tail paired median ratio 0.948581, below 1.05.

The bootstrap uses the original 10,000 draws and fixed seed, resampling 12 position
clusters while retaining all four repetitions. The 72 paired repetitions are not
72 independently sampled positions, and historical tails are deliberately selected.
There is no pooled population latency estimate.

Candidate maximum response was 2.311 s versus 2.031 s for control in the systematic
group. The tail maxima were 1.845 s and 2.162 s respectively. Thus the typical speed
gain does not establish improved worst-case latency or reliable one-second play.
All ordinary responses stayed within the unchanged 10-second research guard.

## Numerical agreement and decisions

Each worker passed the 11 original compatibility fixtures,38 repeated numerical
fixtures and 67 actual corrected-rule qualification decisions. Candidate policy
differences remained within `1e-5 + 1e-6 * abs(reference)`, and value differences
within `1e-6`. Maximum errors were 3.8146973 e-6 and 1.7881393 e-7 respectively.
The candidate was non-bit-exact on all 49 fixtures; control was exact on all 49.
These numerical inputs were already assessed in attempt 01 and are explicitly
reported as repeated assessment, not fresh independent data.

All 18 distinct panel positions produced matching complete atomic actions and
matching ordered callback-input paths across the two players, on every repeat.
Each worker was exactly deterministic across its own four repetitions, and
control reproduced its historical complete outputs. Candidate full numerical
outputs were not generally identical to control.

Matching callback paths means this observed speed difference was not accompanied
by a changed searched trajectory on this panel. Counterbalanced interleaving
reduces host-order effects, but does not prove the isolated duration of a single
matrix operation or universal behavior on other positions.

## Verification and preservation

The corrected version passed 202 combined model-free tests and independent
worker/controller/design and exact-freeze reviews before launch. It then passed
22 graph cases,76 repeated numerical cases,134 scientific decisions and 144
ordinary measurements:150 successful parent requests in total. No failed request,
timeout, discarded measurement, replacement or retry was used.

The controller and watchdog exited 0; the separately owned saved-only reader
approved full source, raw transport, legality, clock and coverage checks and
exited 0. The retained guardian session 97229 returned actual exit 0, tool chunk
`dab90b`. The final owned reader receipt was written about 551.3 seconds after
launch, comfortably within the three-hour overall and two-hour model limits.
External acceptance confirmed all 16 exact process identities absent, all 2,785
frozen pins and 961 sealed files unchanged, and all 155 prior failed-attempt files
preserved. Every recorded owner, reader and worker/helper exit was zero. The
guardian was observed exited within 584.7 seconds of launch.

An independent saved-only review recomputed all 144 pairings, timings, action/path
agreement and the 10,000-draw bootstrap, as well as all 49 numerical arrays per
worker. It agrees with the saved results. All old frozen evidence remains intact,
including the original bitwise rejection and attempt 01's controller failure.

## What this result establishes

The implementation has a measurable, useful speed advantage under the fixed
panel criterion, with numerical differences inside the predefined bounds and
unchanged decisions on the tested roots. The strict bitwise failure was therefore
not sufficient grounds to discard its practical value.

However, this remains a modest speed improvement. The candidate still averaged
1.389 seconds on systematic positions, with 38 of 48 measurements above one second.
This study neither proves equal playing strength nor makes 256 simulations a
reliable one-second player. No promotion follows automatically.

## The next step

Recommend integrating this fixed implementation into a separately versioned live
player and running six fresh operational games against unchanged 256, with swapped
seats. Keep the corrected rules, current research guards, all verification and
one-second timing reported separately. Treat those games as operational
qualification, not a strength-equivalence test. Preserve any failures.

If that passes, design the next strength/timing experiment around an intermediate
search budget, such as 192 simulations, against unchanged 128. This tests whether
some of 256's established strength gain can be retained nearer the one-second
target, using the faster implementation. Freeze candidate, sample and criteria
before launch; this report does not launch either follow-up.

## The broader picture

Earlier whole-game evidence established that 256 search simulations outperform
128 at a substantial timing cost. This experiment addresses execution efficiency
while retaining weights and search settings. It provides a small useful gain,
but the remaining timing gap is larger than this change alone can close.
128 remains the reference;256 remains the verified stronger research challenger.
External testing, six-world work, training and larger-data work remain deferred.
Scheduled monitors remain cancelled.

## Evidence

- Protocol: `docs/engineering/splendorust_network_practical_v2_20261031.md`.
- Attempt: `outputs/splendorust_network_practical_20261031_02`.
- Freeze SHA256: `a7c95305d16c3d96de30e58b16f0b699aeb709ee8fb5eddacb6160ea64bfb23d`.
- Main analysis: `evaluation_analysis.json` inside the attempt.
- External acceptance, independent recomputation, observed exit and final summary:
  `outputs/splendorust_network_practical_20261031_02_closeout`.
