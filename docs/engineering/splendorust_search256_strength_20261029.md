# Fixed 256 versus unchanged128: strength and computation cost

29 October 2026. The user authorized implementation, qualification, freezing and
execution of the next study recommended in the accepted
[exploration report](splendorust_exploration128_20261029_02_results.md).
This is one preset challenger, not a parameter-selection study or an automatic
promotion. Output: `outputs/splendorust_search256_strength_20261029_01`.

## Question and immutable players

Does doubling the search quota from 128 to 256 simulations improve complete-game
play enough to justify its extra cost? The earlier saved 256 panel had a median
1.423-second response; this study does not assume one-second deployability.
Correct completed late moves remain in their games, with timing reported
separately. Neither 700 ms nor one second is an admission or strength gate.

Keep the exact original128 reference profile and original native binary:
`outputs/splendorust_search128_strength_20261028_01/profiles/puct128.json`, SHA256
`048fea7e9d9377f5b3e477d4bafc96d3e96321661d2146e299a95cdc51b36edb`.
The new research profile changes only `iterations` scientifically. It uses the
same original quota-capable native executable, not the configurable exploration
bridge. No native rebuild is needed. New identifiers describe the new player;
the unchanged reference's identifiers and profile bytes are preserved.

Both players use three root worlds, coefficient 0.4, dynamic first-play urgency
with reduction 0.02965, depth64, batch1, one numerical thread, the same optimized
float32 graph, canonical observer-derived seedA and the same pretrained Entity
checkpoint. Its SHA256 remains
`ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb`.
Keep all auxiliary actions, lawful-information handling, atomic pre-reveal turn
semantics, internal forced-pass convention and turn124 heuristic unchanged.
There is no retained game tree, fitting, larger network, new world pool or extra
search when time remains. Actual neural calls may differ by trajectory.

## Rules, freeze and qualification

Bind ruleset `splendor_base_qualified_rules_20261018_v1`, catalogue SHA256
`db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312`, and turn contract
`atomic_prereveal_completed_turn_v1`. Every new evaluator worker must invoke
`verify_runtime_sources()` at startup, and its actual route must pass the
[corrected-rules counterexamples](future_training_rules_20261021.md).
No historical targets are relabelled as corrected data. The checkpoint's
historical training origin remains explicit.

Before any learned-model call, freeze source/runtime/binary/profile identities,
this protocol, all 203 fresh game setups and seat order, qualification cases,
replay selector, statistics, clocks, resources and failure handling. Model-free
regressions and independent implementation review precede the freeze. The accepted
exploration study and its unsuccessful predecessor remain immutable and preserved.

Qualification uses two routes: original128 (`frozen`) and 256 (`research`). Each
receives the full 67-decision actual-entrypoint suite, including direct same-quota
native comparisons. Exercise the corrected catalogue/all90 native faces, noble
and terminal counterexamples, payments/returns, private reserves, lawful support,
encoders, late-game/forced-pass behavior and rejection of legacy/bad identities.
Retain the original16 route solely for the inherited internal reference checks.

Each route evaluates the same 18 predetermined saved positions and repeats six
of them exactly. This totals 182 scientific decisions: 134 in two bundled suites
plus 48 ordinary requests. Two graph requests and two suite requests produce 52
parent requests and 22 graph cases across two qualification trees. Require exact
128 scientific reproduction against the accepted saved suite/panel, exact repeat
outputs, actual 256 work counts and direct-native parity. Raw execution identities,
quotas and clocks are validated before any declared metadata normalization;
scientific work and numerical results cannot be normalized away. No timing-margin
gate or repeat-until-pass is permitted.

## Operational rehearsal and fixed main sample

Run six operational games on three fresh swapped-seat setups. Their scores do
not choose settings, alter sample size or enter the main strength estimate.
Every game must finish naturally and pass its checks; a genuine research-guard
failure here stops the attempt incomplete without a replacement.

Use these six games for an outcome-blind, conservative forecast of the complete
remaining workload. It includes all 400 main games, startup/closure, up to48
exact reproductions, graph cases, complete ledger and raw-evidence checks,
source preservation, file readback, statistical verification, independent review
and cleanup. Require each stage and the whole plan to fit their remaining
original allowances before admitting any main worker. Save and independently
verify an immutable admission receipt before main worker creation and requests.

The main sample is exactly 400 games on 200 fresh paired setups, swapping seats
within each pair and counterbalancing first-seat order across pairs. It is a
direct test of one preset candidate, not a development screen followed by
selection. Setups use the fixed `splendorust_search256_strength_20261029_v1`
domain, disjoint from the operational setups and checked historical registers,
including all reserved exploration setups. The collision audit lists checked
manifests/schedules/registries and discloses files above its 8MiB reading bound;
it does not claim to inspect every arbitrary historical log.

Finish the fixed sample without inspecting intermediate scores to change the
plan. No pooling with earlier studies, optional extension, alternative candidate,
replacement game or automatic subsequent 512-simulation study is authorized.

## Outcome and independent assessment

Score wins1, draws0.5 and losses0. Report the 400-game score, seat-specific scores
and paired uncertainty using 50,000 PCG64 bootstrap resamples of the 200 whole
pairs, seed20261029256 and linear percentile quantiles. Preserve the inherited
Hoeffding safeguard for a degenerate empirical bootstrap interval. Verify the
integer score arithmetic and bootstrap construction independently.

A useful-gain result requires all of: score at least55%, paired95% lower bound
strictly above50%, score at least50% from each seat and complete evidence
acceptance. Apply the same tests to a conservative sensitivity analysis that
counts every administrative-forfeit game as a candidate loss, even if the
reference caused it. Use identical resample indices in the two analyses.
Opponent failures cannot by themselves establish the candidate's gain.

All completed game ledgers are independently replayed. Reproduce up to48 main
decisions selected by the frozen outcome-blind hash/role/seat/phase/private-reserve
selector. Preserve absent strata and its established fill rules; do not choose
interesting moves after outcomes or create additional model calls. Exact action,
trace, afterstate and scientific result checks remain mandatory.

This comparison establishes only relative performance against unchanged128 under
the stated rules and costs. It does not certify universal or human strength,
strict one-second reliability, a network-capacity bottleneck or successful
future learning from deeper-search targets. An inconclusive result leaves128
unchanged; passing criteria still requires a separate explicit reference decision.

## Clocks and failures

Both players retain9-second internal and10-second complete-response research
guards, measured from before parent preparation/dispatch to trusted complete-line
arrival. Preserve raw timestamps and boundary observations. Report700ms,900ms
and one-second exceedances separately; none changes admission, stopping, scoring,
retained moves or sample selection. No added timing observer or host sampler.

In main games, an attributable player research-guard failure is an administrative
loss; retain the failed request and continue the fixed schedule. Infrastructure,
source, identity, rules, evidence or resource failure stops incomplete and is not
relabeled a player loss. Preserve the2000-ply ceiling and unresolved actual-game
deadlock behavior. Never reset an active move's clock or retry a failed decision.
This frozen attempt has no automatic technical-recovery or relaunch path.

## Bounded resources and final acceptance

Use one non-resettable48-hour clock. Stop all model work by hour36 and reserve
up to12hours for saved-only final verification, acceptance and cleanup. All stage
caps are clipped to the original global/model-work clocks and cannot add time.

| Stage | Maximum |
| --- | ---: |
| Correctness qualification | 2 hours |
| Six operational games and resource admission | 3 hours |
| 400 main games | 28 hours |
| Main verification and selected model reproductions | 6 hours |
| Saved-only final verification, independent review and cleanup | 12 hours |

The fresh independent reader has a three-hour local cap within the same final
and global allowances. The maximum workload is406 games, two qualification
trees and48 reproduction trees:862 actual worker trees/graph builds and9,482
graph cases, with no more than two resident trees and one active decision.
The lifecycle allowlist may name both possible routes for each replay slot;
only the selected48 actual slots may execute.

Cap evidence at24GiB, reserving8GiB for finalization (16GiB pre-final). Require
32GiB free disk initially and maintain5GiB. Cap aggregate worker/helper RSS at
2GiB. Require4GiB available RAM stable60seconds before initial admission, then
2GiB between requests and at startup. A low-RAM wait requires60seconds of stable
reserve before resuming, has a cumulative1800-second ceiling and consumes the
same original clock. It cannot reset or restart an active decision.

Use the existing CPU0 player/helper placement and supervisors on CPUs2–19,
normal priority and verified topology. Retain lossless bounded compressed logs,
bounded complete response packets and all raw/ownership/graph checks. Freeze the
exact symmetric capacities in the execution contract before qualification.

The operational forecast keeps the existing1.5 contingency and accounts for the
actual larger-quota costs. It must fit stage, whole-run, final-verification and
storage reserves simultaneously. A failed forecast stops incomplete; it does not
authorize weaker checks, fewer games, a longer clock or erased evidence.

Final acceptance requires complete saved-game/transport/science/clock/placement/
source checks, reconciled attempted requests and worker lifecycles, a finite
evidence seal, a separate saved-only independent reader and successful actual
runner, watchdog, reader and guardian exits with confirmed cleanup. A completion
file alone is insufficient. Reuse the selected model reproductions during final
saved readback; run no new models or games there.

## Scope and reporting

Scheduled chat monitoring remains cancelled; do not recreate it. Internal process
ownership and resource safeguards remain part of this run. Status comes from
small saved stage/terminal receipts and exact process identities. Preserve every
failure and report a blocker promptly if an authorized continuation is impossible.

External testing, six-world work, training, larger-data collection and a timing
optimization campaign remain deferred. Report the eventual outcome with
"What this result establishes", "The next step" and "The broader picture".
