# Practical numerical agreement and speed of the final-token FFN candidate

31 October 2026. The user authorized this separate follow-up after the exact-bit
attempt failed. Preserve that failed attempt unchanged. This study asks whether
the same implementation has small bounded numerical differences, stable playing
decisions and useful speed. It does not retroactively relax the earlier protocol,
prove equal playing strength, replace the reference player or authorize training.

New source: `tools/splendorust_network_practical_20261031`.
New evidence: `outputs/splendorust_network_practical_20261031_01`.
Only one implementation is assessed. Import the existing held
`candidate_final_token_ffn.py`, SHA256
`e10ff696e66dfe1cbc788c7968e053d7c07e2b0be3bf8f5ae411734c2906f870`.
No second candidate, parameter changes or automatic retry are permitted.

## Scientific conditions and explicit numerical criteria

Both workers retain the accepted256 checkpoint, float32 precision, one inference
thread and one interop thread, batch-one live search, three worlds, 256 simulations,
exploration coefficient0.4, depth64, native binary and auxiliary decisions. Keep
player CPU0 and supervisors CPUs2-19. The candidate only narrows the last encoder
feed-forward calculation to token0, preserving all attention and earlier blocks.
No new trace, model invocation or warmup is allowed inside its factory.

Every new evaluator worker calls `verify_runtime_sources()` once at startup.
Bind corrected ruleset `splendor_base_qualified_rules_20261018_v1`, catalogue
SHA256 `db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312`,
and turn contract `atomic_prereveal_completed_turn_v1`. Retain the actual-entrypoint
corrected-card, noble, payment, private-information and transition counterexamples.
This is evaluation of saved corrected-rule positions, not a training entry point.

The following candidate tolerances are fixed before new model results:

- For each of the81 raw policy outputs: absolute difference at most
  `1e-5 + 1e-6 * abs(reference)`.
- For each of the2 value outputs: absolute difference at most `1e-6`.
- All outputs must be finite, have the original shape and float32 precision.
  Retain every exact-bit comparison and every actual output, including failures.
- The unchanged control still requires exact output and float32-bit equality.

These are exploratory numerical screening bounds, informed by the earlier
development failure and fixed before fresh assessment. They are not a theorem
about search decisions or strength. Never compare values from different input
tensors as if their paths were aligned; any cross-worker callback comparison
requires identical complete input identity. Do not tune a tolerance after results.

## Fresh saved assessment and preparation

First precommit the deterministic source-selection rule, hash domain and exclusions,
then extract saved evidence without running a model. Select18 timing roots from
the verified400-game256-versus128 corpus:12 systematic positions and6 deliberately
selected historical slow tails, balanced across early/middle/late progress and
both seats. Exclude the earlier18-position panel's games and all constituent
batch-one input hashes from the previous11 compatibility fixtures and six
localization fixtures. Use distinct games and retain selection/provenance evidence.
The tail group remains separate; it is not representative population latency.

For fresh numerical assessment, retain the first full saved callback tensor from
each of those18 roots, plus18 further predetermined roots in distinct other games,
balanced across the same progress/seat cells. Historical middle callbacks retain
hashes rather than full arrays, so they are not reconstructed or invented. Require
36 distinct batch-one input hashes and add one fixed batch2 and one fixed batch8
composite:38 fresh fixtures in total. Freeze the exact input arrays and source
identities before launch. Previous11 graph fixtures remain development/compatibility
checks and are reported separately from this new assessment.

Before launch, pass model-free tests for numerical boundary behavior, malformed
inputs, deterministic selection/exclusions, complete request coverage, legitimate
decision disagreement, unchanged-control fidelity, independent transitions and
early-rejection cleanup. Independent implementation and exact-freeze reviews are
required. Freeze the candidate, protocol, all sources/runtime, input fixtures,
tests, selection evidence, evaluation order and acceptance criteria together.

## Qualification and ordinary decision timing

Start only two worker lifecycles, control then candidate. Each first runs the11
original graph cases, then one `numerical_checks` request on all38 fresh fixtures,
then the original67-decision actual-entrypoint corrected-rule suite. Direct/eager
and candidate outputs in numerical checks receive identical frozen input tensors.
The36 fresh batch-one eager outputs must also exactly reproduce their pinned
historical values and float32 bits. Composite batches use their own eager
references, since changing batch shape may change arithmetic.
The extra numerical request has a45-second internal/60-second complete-response
allowance. Ordinary decisions retain9-second internal/10-second complete-response
research guards. One second and700ms remain descriptive, never admission gates.

The control's complete scientific suite and ordinary results must reproduce the
historical originals exactly. Candidate scientific qualification retains all
within-worker exact repeatability, private-state invariance, legal action,
corrected-rule and native/host checks. It does not demand historical equality of
every search trace against the control. No candidate scientific suite or ordinary
timing follows a failed numerical bound. A complete valid numerical rejection is
a separately verified negative outcome with no retry; malformed evidence or an
infrastructure failure remains an incomplete attempt.

If both workers qualify, run four paired repetitions of each18-root position:
144 ordinary requests. Rotate roots and counterbalance control/candidate order
in the frozen matrix. Keep the extra profiler off. Time the complete response
without subtracting estimated overhead. Each full atomic action and its resulting
state must be lawful and independently verified using that worker's own action;
never substitute the historical control afterstate for a different candidate move.

Legal differences between candidate and control actions do not stop the panel.
Keep and report every one. Require each worker's repeated result on the same
position/seed to remain exactly deterministic. Report agreement in full atomic
actions and distinguish agreement from numerical or full-trace equality. The18
distinct positions are the decision sample;72 paired repeats are not72 independent
positions. Disagreement alone proves neither better nor worse play.
Report whether the complete ordered callback-input paths also agree. If those
paths differ, the response-time comparison includes both execution cost and the
changed search trajectory; it cannot isolate a matrix-kernel speedup alone.

## Fixed interpretation and useful-speed screen

After all144 valid measurements, report complete response means, medians, maxima,
one-second exceedances, paired ratios and differences for systematic roots and
historical tails separately. Retain all matched and mismatched decisions.

The useful-speed screen is the same size as the preceding planned screen:

- Systematic paired median candidate/control ratio at most0.95.
- One-sided95% position-cluster bootstrap upper ratio below1.00.
- Historical-tail paired median ratio at most1.05.

Use10,000 PCG64 bootstrap draws with seed20261031258. Resample12 systematic position
clusters, retaining all four paired repeats. Do not select positions or drop
outliers after seeing performance. No population timing or strength claim follows.

Report numerical qualification, full-panel decision agreement and useful speed as
separate outcomes.100% decision agreement is a provisional substitution screen,
not a prerequisite for finishing timing or proof of identical future behavior.
A fast candidate with disagreements is a potentially different player requiring
later strength assessment, not an automatically weak or invalid player. A failure
to improve useful speed is a valid result. No fresh operational games, reference
promotion, second implementation or additional study is launched automatically.

## Bounds, verification and preservation

One original owning launch; overall3h, model work cutoff2h, up to1h reserved for
saved-only final verification. Stage caps: qualification30min, evaluation60min,
final60min, each also clipped to its original overall/model deadline. All waits
consume that clock. No active decision may receive a new clock or be retried.
After all startup requests complete and before any ordinary request begins, each
idle qualified worker's absolute phase deadline moves to the evaluation deadline,
still clipped to the original model/global cutoff. A saved handoff receipt binds
this transition and is independently checked; no active move clock is changed.

Maximum2 worker lifecycles,2 resident trees and1 active request;22 original graph
cases,76 fresh numerical cases,134 actual-suite decisions,144 ordinary decisions,
150 parent requests. Numerical cases each call reference and chosen execution
once; no extra numerical sweeps. Only the original graph construction/zero warmups
are additionally allowed. No profiling, training or new games.

Require4GiB available RAM stable60s initially and2GiB before later starts/requests;
cumulative RAM wait at most30min. Aggregate worker-tree RSS at most2GiB. Evidence
at most4GiB with1GiB finalization reserve; at least5GiB disk space remains free.
Preserve every failure, source version, raw transport record and exit receipt.

Final acceptance requires independent saved-evidence readback, original/fresh
source integrity, all prescribed coverage or explicit valid numerical rejection,
legal transitions, clocks, actual owner exit and cleanup. A completion file alone
is insufficient. Report a valid early negative separately from completed timing.
Keep the old exact-bit attempt incomplete and unchanged in its original folder.

128 remains the reference and256 the verified stronger research challenger.
The proposed192 study is superseded for now. External testing, six-world work,
training and larger-data work remain deferred. Scheduled monitors remain cancelled.
