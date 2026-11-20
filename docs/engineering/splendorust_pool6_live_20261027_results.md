# Six-world live admission: correctness passed; timing margin failed

27 October2026. The separately frozen admission attempt finished and was independently verified. It did **not** admit the players to operational games: three successful decisions exceeded the prospective700ms engineering margin. **Every individually timed request finished under the actual one-second deadline. There were no timeouts, illegal decisions or forfeits.** No operational or assessment games were launched, and optimized64 remains unchanged.

## What ran and passed

The separate live wrapper fixes the candidate to six worlds, seedA and64 total main-search simulations, retaining the accepted network, precision, inference backend, late-game heuristic and auxiliary handling. The incumbent is the unchanged optimized64 route. Before launch,94 model-free tests and independent implementation/frozen-input reviews passed.

All194 scientific qualification decisions passed:104 decisions in the two actual-entrypoint suites,48 exact saved reproductions (24 per player), six explicit live edge cases, and36 requests across18 fresh reserved positions. The90 ordinary requests were individually timed under1s/.9s clocks. Recorded actions, numerical outputs and candidate callback bits reproduced exactly. The live turn124 heuristic matched the unchanged incumbent. All22 graph cases passed.

Both workers closed cleanly. Independent readback verified94 parent requests, all6391 frozen inputs,2593 prior-attempt files,616 sealed artifacts (30,645,302bytes), all three actual owner exits0, and absence of all15 recorded owned process identities. Total charged execution time202.667s (3m23s). Peak sampled worker/helper RSS327,446,528bytes. The earlier failed test fixture and all historical experimental evidence remain preserved.

## Why the gate stopped advancement

| Player | Saved position | Complete response | Handler time | Response emission |
|---|---|---:|---:|---:|
| Unchanged optimized64 | reserved_06 |736.9ms|687.8ms|1.1ms|
| Six-world candidate | gold_returns |772.3ms|668.4ms|54.6ms|
| Six-world candidate | reserved_00 |714.3ms|618.6ms|48.2ms|

These are valid completed decisions, not losses. The frozen plan required every ordinary admission request and operational move to stay below700ms to retain headroom beneath the hard1s deadline. One of42 incumbent requests and two of48 candidate requests crossed that margin. The candidate inventory includes six extra edge cases, so aggregate timing means are not a paired comparison. The gate was not relaxed after observing the results.

The three slow decisions each used one64-simulation Main search and65 neural callbacks. The additional payment/return phases in two cases were heuristics with zero neural calls. Repeated neural searches therefore did not cause these three delays. Most elapsed time was inside the handler; candidate evidence emission adds measurable cost but cannot explain the incumbent's slow request. Differences between wall time and sampled CPU, and faster matched counterparts, make scheduling or IPC waits plausible explanations. The existing minimal observer does not identify the cause within the handler.

## What this result establishes

The new live wrapper preserved the required decisions, rules and endgame behavior, and met the actual one-second deadline on this finite panel. Its conservative engineering admission failed. This does not show that six worlds make the player weaker or that it cannot operate within one second. It also does not establish live-game reliability or improved strength: all six operational games and all200 assessment games remain unplayed.

The accepted outcome is `not_admitted_verified`, with `admission_passed:false` and `benchmark_authorized_to_launch:false`. The result must remain distinct from a timeout, an incomplete verification, or a positive qualification.

## The next step

Prepare one small bounded repeat-timing diagnostic on these three saved positions and fixed ordinary controls. Balance and interleave both unchanged players, freeze the repeat count before execution, retain the1s/.9s clocks and exact-output checks, and record every outcome. Separate the already observed emission cost from handler variability before choosing an optimization or another admission attempt. No retry-until-pass, retrospective threshold change, fresh game or new study has been launched.

## The broader picture

The previous saved-position study showed that the six-world configuration made decisions more consistent across sampling seeds at the same64-simulation budget. Whole-game confirmation is needed to learn whether that consistency makes the player stronger. This admission was the bridge to that comparison.

The project still aims to improve a strong, verified SplendoRust-based player within one second per move. This result identifies a timing-headroom issue shared by both routes; it does not call for different weights, more training or a larger network. Optimized64 stays frozen, external-opponent testing stays paused, and larger-data work remains deferred. A future positive live admission would enable the already proposed fresh paired strength comparison.

## Evidence

- Protocol: `docs/engineering/splendorust_pool6_live_20261027.md`.
- Output: `outputs/splendorust_pool6_live_20261027_qualification_01`.
- Freeze SHA256: `247b143d1f5256048b336231fb3ba3c544f74ceb7580e7087a1e69b967d480a0`.
- Independent completion review: `postrun/independent_completion_review.json`, SHA256 `01405c2631723accafda75a642a64d481ead73c31fdd1307e9fde80295198feb`.
- Saved timing derivation: `postrun/admission_results.json`. Its creation-time pending-review wording is superseded by the independent negative-result acceptance above.
- No experiment remains running. No assessment freeze or game was created.
