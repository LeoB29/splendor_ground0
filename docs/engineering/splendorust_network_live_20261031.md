# Held network implementation: six fresh operational games

31 October 2026. Authorized by the user's “ok do it” after the accepted practical
attempt02 recommendation. This protocol prepares a separate player integration
and an operational qualification. It does not launch a strength comparison or
promote a reference player.

## Purpose and fixed players

The accepted saved-position experiment found a 5.73% median paired speed gain,
with matching actions and callback-input paths on all 18 positions. It did not
establish universal action equality, playing strength, or one-second deployment.
The next question is whether that same implementation works through complete
fresh games and reproduces its own decisions in a fresh process.

The candidate is exactly `final_token_only_last_ffn_v1`, source SHA256
`e10ff696e66dfe1cbc788c7968e053d7c07e2b0be3bf8f5ae411734c2906f870`.
It skips unused token outputs in the final feed-forward block. The control is
unchanged256. Both retain the accepted checkpoint, float32, 256 simulations,
three sampled worlds, cpuct0.4, depth64, one Torch thread/inter-op thread, and
the same search, action mapping, private information and RNG implementation.
Actual search seeds still derive from the canonical legal observer; there is
no added RNG policy. Player CPU0 and supervisory CPUs2–19 stay unchanged.

Source is `tools/splendorust_network_live_20261031`; evidence is
`outputs/splendorust_network_live_20261031_01`. The accepted practical output
and both previous failed attempts remain immutable and hash-bound.

## Correctness before play

Every new worker calls `verify_runtime_sources()` once before loading models.
All games and reconstructions use the corrected two-player environment,
catalogue `db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312`,
ruleset `splendor_base_qualified_rules_20261018_v1`, and
`atomic_prereveal_completed_turn_v1`. No historical default environment or
legacy decoder is authorized. Search deadlock conventions and pre-reveal
auxiliary choices retain their previously disclosed meanings.

Each of four worker lifecycles performs the same prefix: 11 graph cases,
38 repeated numerical fixtures, and the full 67-decision actual-entrypoint
suite. This tests corrected faces, noble handling, termination, payments and
returns, private reserves, legal support, hidden-world reconstruction and
scientific response/trace integrity. The two live workers qualify before any
game. They are fully closed before two fresh replay workers qualify.

The candidate retains the predeclared bounds for all raw policy outputs:
`abs(error) <= 1e-5 + 1e-6*abs(reference)`, and raw values `abs(error) <= 1e-6`.
Shapes, float32 and finiteness remain mandatory. Control qualification requires
exact original outputs. The 38 fixtures are repeated qualification, explicitly
not a new independent numerical assessment. A numerical rejection is a valid
negative qualification result. No tolerance adjustment or alternative candidate
is allowed. The strict earlier exact-output rejection remains valid.

## Six fresh games and timing

Freeze three deterministic, previously unused initial-state seeds in a new
domain before models run. Each pair uses the same complete initial state, with
candidate and control swapping seats. Play all six in fixed order. The schedule,
full initial states, bounded prior-seed inventory and replay-selection rule are
frozen inputs. These six games are operational evidence; their score is
descriptive and cannot establish strength or equivalence.

Retain the research guard of nine seconds inside the player and ten seconds to
receive its complete response. One second and 700 ms are descriptive thresholds,
never gates or forfeits. Count and report overruns, mean, median, upper quantiles
and maximum separately for each variant. The comparisons are not paired timing
measurements because game positions can differ.

No failed decision is retried or game replaced. A hard timeout, invalid move,
transport failure, failed check, resource refusal, unresolved actual deadlock or
the original 2,000-ply safety cap stops the attempt and preserves its records.
An unfinished game is not assigned a natural result or fabricated draw. Live
qualification requires six natural terminal games and all checks passing.

## Verification and replay

For every move retain the complete clock receipt, compressed raw request and
response, action, trace, legal observer, full before/after states, sidecars,
placement and exact process identity. Verify the corrected transition against
the independent dictionary ledger, observer/private-information legality and
full scientific response. Independently replay every complete game to verify
terminal status, final score, tie-break and winner.

Select at most 48 successful original decisions without using scores, winners
or timing: one per nonempty game × variant × absolute-ply phase cell (plies
0–19,20–39,40+, up to36),
then up to12 fixed-hash private-reserve representatives as specified in the
frozen executable selection rule. Re-evaluate with freshly qualified processes
of the same variant. Require exact full scientific result, action, trace and
afterstate; no relaxed replay equality and no cross-variant equivalence claim.
Every selected original must have exactly one reproduction, with no extra
model requests. Reconcile complete worker request inventories and actual exits.

An independent saved-only reader runs only after the model owners exit. It
checks frozen hashes, sealed evidence, all game records, reproductions, stage
clocks, resource/accounting receipts and cleanup. Completion also requires
observing the actual outer-owner exit and confirming exact process absence.
A completion file alone is insufficient.

## Bounded execution

One exclusive owning launch, one nonresettable monotonic and UTC origin:

- Overall four hours; all model activity stops within three hours.
- Initial qualification30min; live games90min; fresh-worker qualification and
  replays60min; saved-only finalization60min. Each is clamped to original clocks.
- Live admission requires at least30min left in its effective phase/model clock.
  Replay admission reserves two startup allowances of480s, ten seconds per
  selected replay and120s closure margin, all within the original clock.
- Four total worker lifecycles, at most two resident trees and one active request.
  At most44 graph,152 repeated numerical and268 actual-suite cases; at most
  12,000 live decisions and48 reproductions, further bounded by stage clocks.
- Initial available RAM4GiB stable60s, then2GiB between requests. Maximum30min
  cumulative RAM waits, all charged to the original clock. No waiting resets
  an active move clock. Combined resident workers/helpers RSS at most2GiB.
- Total evidence at most4GiB, including a1GiB saved-only finalization reserve;
  free disk at least5GiB. Lossless compression and bounded raw transport remain.
- No automatic retries, budget extensions, changed samples, alternate settings
  or silent continuation after failed qualification. Preserve and diagnose any
  failure; a subsequent corrected launch requires a separate authorized freeze.

Independent code/design review and passed model-free regression tests precede
the freeze; a separate exact-freeze prelaunch review precedes the launch.
The initial stable-RAM admission is enforced by the owning runtime.

## Interpretation and follow-up

Passing establishes operational integration and sampled fresh-process
repeatability for this fixed candidate under the generous research guard.
It does not make the non-bit-exact candidate universally equivalent, prove a
strength gain, or certify one-second deployment. The previous fixed-position
experiment remains the controlled estimate of speed gain.

If accepted, recommend a separately frozen intermediate-budget study, such as
192 simulations against unchanged128, measuring strength and timing separately.
Do not launch it automatically.128 remains the reference and256 the verified
stronger research challenger. External testing, six-world work, training and
larger-data work remain deferred. Scheduled monitors remain cancelled.
