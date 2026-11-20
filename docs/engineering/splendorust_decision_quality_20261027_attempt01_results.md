# Saved-decision diagnostic: first execution attempt

27 October 2026. **Incomplete: qualification stopped; the 24-position diagnostic did not start.**

The separate evaluator and its native telemetry bridge were implemented without
changing optimized64, unchanged32, the checkpoint, precision or search. All 105
model-free checks passed, followed by independent implementation and frozen-input
reviews. The user-authorized attempt launched under the prepared two-hour cap and
stopped after 101.8 seconds on its first rejected qualification response. No retry,
replacement, new game or training occurred. External candidates remain paused;
larger-data work remains deferred.

## What ran

| Check | Observed outcome |
|---|---|
| Model-free tests |105 passed; earlier development receipts retained |
| Actual entry-point suites |Both 52-decision suites returned success |
| Graph checks |11 cases in each of two qualification workers passed |
| Exact parity requests |3 of 19 attempted: frozen64 and research64-OFF accepted; research64-ON rejected |
| Saved-position diagnostic |0 of 156 requests; none of the 24 roots entered collection |
| Owner exits |Runner, watchdog and guardian each returned actual exit code 1 |
| Cleanup |No errors; all 15 recorded owner/worker identities absent |

Startup success receipts are evidence about those checks, not acceptance of the
whole qualification. Final verification failed too; the original completion
status remains `incomplete`.

## Why qualification stopped

The first telemetry-ON response arrived in 0.4252405 seconds under its 30-second
research allowance. It was rejected by the new semantic checker, not by a clock.
The native field named `SearchTargets.network_value` is copied from `node.value`
after search. With the unchanged dynamic-FPU setting, search updates that value
as a running mean. The checker incorrectly equated it with the initial network
prediction.

For this exact saved response:

- Initial signed network output: −0.02492569386959076.
- Initial acting-player win credit: 0.4875371530652046 before float32 projection.
- Mean backed-up root value: 0.5197411179542542.
- Updated root-node value: 0.5192456841468811.

The updated value matches the native running-mean relation
`(initial_value + 64 * backed_up_mean) / 65`, to the recorded float32 precision.
It is not supposed to equal the initial prediction. The selected-edge value is
a third, distinct statistic.

A second checker defect compounded the rejection. The unchanged receiver retains
the rejected payload with `status=illegal` and `accepted=false`, so it can be
inspected. The new failure audit wrongly required every failed response to have
no retained result. This caused the subsequent closed-stream and final audit
failures. The raw response remains preserved and unaccepted.

## What this establishes

Read-only inspection bound the saved response packets to their original clocks,
raw compressed logs and parent/child records. The frozen, research-OFF and
rejected research-ON responses have exactly equal scientific content and select
the same move. OFF and ON also have identical full diagnostics and all 65 callback
bit records. Thus the observed rejection is explained by two checker assumptions;
it is not evidence of a changed playing decision on this fixture.

That finding does not qualify unrun quotas or seeds, finish the research
qualification, recover the failed attempt, or tell us whether deeper search
improves decisions. The 105 tests missed the actual mutable-field convention and
the real rejected-response preservation contract; future regression tests must
use these saved records.

## The next step

Prepare a separate Python-only repair and new evidence directory. Preserve the
native field and label it as the root-node value after search; derive a separately
named initial network win credit from the first recorded callback. Validate the
running-mean relation with an explicit bound for rounded telemetry. Update the
analysis so it uses the intended initial prediction. Allow rejected payloads as
forensic evidence while keeping them excluded from accepted science.

Use the saved OFF/ON/rejected packets as regression fixtures, including corruption
and failure-path cases. A newly frozen attempt must still pass the full actual
qualification before executing the same fixed panel and matrix. No new attempt
was launched and no frozen source was repaired in place.

## The broader picture

The optimized64 versus unchanged32 comparison established an improvement within
the SplendoRust family, and subsequent reliability work addressed the one-second
operating constraint. The two screened independent policies were too weak to
provide a useful stronger-opponent comparison. With that work paused, this small
saved-position study is intended to identify one promising decision-quality
hypothesis before spending resources on training or another strength trial.
Its deeper searches are diagnostic references using the same network, not an
oracle. Any eventual improvement still needs fresh whole-game confirmation at
one second.

SplendoRust remains our strongest verified starting point. The screens do not
establish that its network or full player is the strongest available anywhere.
Other public leads remain unqualified; for example, AhinLendor's author reports
strong human results, which are not a matched comparison with our setup:
[AhinLendor](https://github.com/inhabae/AhinLendor). No external testing was resumed.

## Preserved evidence

Execution folder: `outputs/splendorust_decision_quality_20261027_01`.

- Freeze: `a07c1d54dbbf9a67eb6a202e5c6d08717949ef50de8bcfee5413581e8d4d20b0`.
- Saved diagnosis: `postrun/telemetry_value_diagnosis.json`, SHA256
  `96ac0c445b8da15fe73505167afc679885b8a7794d6caacd95bb6033bcf8814b`.
- Independent incomplete-attempt review: `postrun/independent_incomplete_review.json`,
  SHA256 `d1d7be3ae133b6dc711eef8e9022860115e0657712d8cb64470538659c9fbe3e`.
- Preservation inventory: `postrun/attempt_preservation_inventory.json`, SHA256
  `f54e7767540d8143718ecc76680cad5f89c36c9d35bb67c2c3526bc65663c772`.

All 1,776 frozen inputs rehashed unchanged. The inventory preserves 142 original
attempt files, excluding later postrun supplements to avoid self-reference.
The independent review confirms preservation and diagnosis only; it does not
replace the failed completion or supply missing qualification acceptance.
