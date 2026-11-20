# Bounded sampling-stability study at 64 simulations

27 October 2026. The user authorizes preparation, implementation, qualification
and execution of this bounded study now. Optimized64 stays frozen. External
testing is paused; larger-data collection and training remain deferred.

## Question and one intervention

Does using a root pool of **six sampled hidden-card arrangements instead of
three**, while retaining **64 total simulations and one second per decision**,
make this player's decisions less sensitive to the sampling seed?

There is one candidate, pool6, and one configuration control, pool3. There is no
parameter sweep, seed selection, fitting, training or automatic promotion.
This is a saved-position development diagnostic. It launches no new whole games
and cannot establish improved playing strength.

Both settings use the same inherited checkpoint, optimized graph, float32
precision, single thread, inference batch1, search depth64, PUCT settings,
dynamic first-play urgency, legal/safe-action restrictions and compound-turn
handling. **64 means total simulations, not 64 simulations per world.** Each
main search has at most 65 neural callbacks; terminal paths may use fewer.
Extra world generation still costs time and is charged to the same deadline.

The native core remains byte-identical to the accepted diagnostic build.
An additive bridge exposes a strict `world_pool` choice of 3 or 6, default3.
The legacy bridge and optimized64 sources/binaries/profiles remain untouched.
The new bridge in pool3 mode must reproduce the frozen player exactly.

The current batch1 canonical path cycles the root-world pool. Its nominal
`chance_universes=3` setting is unchanged but inactive on this path. Root-world
construction and subsequent hidden completions used by `transfer::encode`
consume the same RNG. Increasing the pool advances that stream, so later neural
inputs can also differ. Canonical game transitions ignore this RNG, and PUCT
selection/ties are deterministic here. This experiment compares the complete
sampling configurations; it does not isolate a pure causal effect of root-world
count. Splitting streams would be a second intervention and is excluded.

## Fixed development panel and seeds

Reuse all 24 roots in
`outputs/decision_quality_preparation_20261027_01/panel.json`. They were selected
without outcomes, actions, values or timing, four per absolute-ply band
0-19/20-39/40+ crossed with seat0/1, from 24 distinct paired worlds. The now-known
diagnostic results make them development evidence, not fresh confirmation.
Only lawful observations reach the players; private snapshots remain with the
auditor. Corrected ruleset/catalogue and the existing atomic prereveal turn
contract remain mandatory. Internal forced-pass/deadlock conventions remain
unchanged and do not become official outcome labels.

Use six fixed observer-derived seed families A through F. A is the first16 hex
digits of the canonical observer SHA256. B/C retain the exact domain
`decision_quality_saved_panel_20261027_v1`. D/E/F use the new fixed domain
`sampling_stability_20261027_v1`, hashed as `domain|label|canonical_observer`,
again taking the first16 hex digits. No winner, private deck or elapsed time enters
seed generation. A-F and both pools are used on every root; no missing cell is
replaced or selectively rerun.

The frozen matrix contains **324 ordinary requests**:

1. 24 exact historical optimized64 reproductions using the unchanged route.
2. A fixed six-root pilot: one original designated root per phase/seat cell,
   each at both pools and all six seed families, **72 requests**.
3. The remaining18 roots, both pools and all six seed families, **216 requests**.
4. Exact repeats of seedB for both pools on the six pilot roots, **12 requests**.

Root order reuses the previous prepared round-robin order. Within each root,
the twelve `(seed, pool)` conditions are rotated by its root-order index. The
pilot and repeat subset are fixed before execution and cannot be selected from
observed stability or timing. The final matrix binds every seed and observer.
Pool3-A must agree exactly with all24 historical scientific results; every repeat
must reproduce exact decision, numerical outputs and callback evidence.

## Qualification before the panel

Use four serial worker lifecycles: frozen qualification, sampling qualification,
frozen panel, sampling panel. Only one player tree may reside at once. A shared
model in the sampling worker serves the reference and search bridge processes.
Three explicit profiles identify reference16/pool3, search64/pool3 and
search64/pool6. Neither the model nor a profile is silently relabelled.

Run the existing **52 actual-entrypoint decisions** on each route. On the new
route, ordinary challenger decisions use **pool6**, while the historical
16-simulation reference remains **pool3**. Test all six corrected card faces,
all90 native cards, noble-after-nonpurchase and multiple/terminal noble cases,
gold/returns, private reserves, legal support and malformed identity/request
rejections through the real worker. Its startup calls
`verify_runtime_sources()` before model loading. Bind raw pool/profile/native
identity before any explicit identity-only old-checker projection. Corrected
rules must hold inside the native reconstruction and encoded neural input.

Run **25 parity decisions**: one unchanged frozen64-A request, then both pools,
all six labels, with telemetry OFF and ON. OFF/ON scientific results and callback
bits must match within each setting; pool3-A must match the frozen route. Pool3
and pool6 are allowed to choose different moves. Qualification totals **129
decisions**; the full study totals **453 scientific decisions**. Eleven fixed
graph fixtures per worker give **44 graph cases**. The expected parent request
count is **355** (25 parity +324 panel +4 graph +2 bundled entrypoint requests).

Model-free tests cover the bridge's strict parameter handling, seed identity,
exact matrix coverage, complete/partial analysis, raw retained rejection handling,
clock/resource contracts and corrected-rule qualification routing. Preserve every
test, build, review and execution failure. No actual model runs before an
execution freeze and independent prelaunch readback.

## Primary measurement and interpretation

For each root and each pool, compare the six main actions across all **15
unordered pairs** of seed families. Its disagreement rate is the number of
unequal pairs divided by15. The primary descriptive contrast is the mean of the
24 paired root differences, **pool6 minus pool3**; a negative value means less
sampling disagreement. Report both pool means, roots with lower/equal/higher
disagreement, and roots where all six seeds agree.

Report complete compound-action stability separately, matched-seed action changes
between pools, the six fixed phase/seat tables, complete-response timing and
callback counts. Do not mix Main-only and complete-turn disagreements. Retain
missing denominators in a failed prefix. A primary paired comparison requires
all six seeds for both pools on the root; never silently treat partial sets as
full coverage. The 15 pairs within a root are correlated, not15 independent
observations. Do not attach a population p-value, confidence claim or strength
threshold to this small development panel.

Q values and initial/post-search values remain available in the audited raw
evidence, correctly labelled. They are not ground truth and do not label one
move correct. Stability can improve by becoming consistently wrong. There is
no automatic promotion or claim of improved play from this result.

## Bounded execution and first-failure policy

One owning guardian supplies non-resettable monotonic and UTC clocks. The whole
attempt has a **3,600-second/one-hour cap**. Qualification is capped at900s,
diagnostic at1,200s and finalization at900s. Stop collection no later than elapsed
2,700s. Initial RAM waiting, startup, graph checks, decisions, closure, proof and
final verification all consume the applicable clocks. No restart, allowance reset,
automatic retry, replacement root or extension is permitted.

Every ordinary decision—including parity and exact repetitions—has a real
**1.000-second complete-response deadline** and **0.900-second internal limit**.
Startup graph checks have explicit60s service caps; a bundled actual-entrypoint
suite has a300s outer cap, with each constituent decision enforcing the internal
limit. These setup bundles are qualification work, not one-second deployment
measurements. Ordinary parity and panel requests supply the external measurements.
Stop at the first timeout, correctness, evidence, identity, process or resource
failure. Preserve the failed response/prefix and incomplete disposition. A saved
request timeout is a diagnostic failure, not a new game loss.

Keep the qualified CPU placement: active player/helpers on logical CPU0,
supervisors on2-19, normal priority. Require2GiB available RAM stable60s before
play, check between requests, and cap cumulative RAM admission waits at300s.
Combined worker/helper RSS is capped at2GiB. Require7GiB free disk initially and
5GiB maintained. Execution evidence is capped at1GiB, reserving512MiB for
finalization; charge logical raw streams and physical/sidecar/proof files with
the established accounting. The frozen route retains16MiB per-log and256KiB
response caps; the new diagnostic route uses128MiB per-log and4MiB response
caps for its larger number of recorded requests. No evidence deletion to fit.

Before expansion, the 72-request pilot must pass a **cost-only gate**. Use the
maximum observed total request/evidence cost within each pool and a1.5 multiplier
to forecast the remaining228 requests, closure and final checks against all
remaining clocks and storage reserves. Decisions, agreement and Q values cannot
influence this gate. Insufficient room ends the attempt with its truthful prefix.

The separate offline native build has a600s compiler cap and no model/search
execution. Build files are inventoried separately before the execution freeze.
No general training or old-default environment route is introduced.

## Acceptance, preservation and follow-up

Verify corrected reconstruction, profile/native/seed bindings, exact control
outputs and repeats, all trusted clocks, raw wire records, sidecars, placement,
compressed-stream hashes, graph identity, actual owner exits and cleanup. Bind
the previous accepted freeze and finite attempt inventory, including its saved
failure notes. All older attempts and frozen dependencies remain unchanged.
Require independent prelaunch and finite postrun receipt review. Distinguish full
verified completion from a verified partial diagnostic and an incomplete failure.

After the run, recommend at most one next action from the joint stability and
timing evidence. If pool6 offers a credible signal and meets practical timing,
it may justify separately frozen fresh paired whole games against optimized64;
these development roots cannot provide that confirmation. If it fails timing,
has mixed/worse stability or insufficient evidence, do not promote it or tune
additional pool sizes on this attempt. External testing remains paused and
larger-data work remains deferred throughout.

Source: `tools/splendorust_sampling_stability_20261027`.
Evidence: `outputs/splendorust_sampling_stability_20261027_01`.
Build: `outputs/splendorust_sampling_stability_build_20261027_01`.
