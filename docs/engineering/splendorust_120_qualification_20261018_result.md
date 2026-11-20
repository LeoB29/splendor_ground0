# SplendoRust at 120 seconds: qualified on the corrected-rules panel

18 October 2026. The fixed 6,400-simulation SplendoRust profile passed all
46 final requests and all 23 hidden-state response pairs. The independent
analysis passed, with no timeout, internal exhaustion, initialization failure,
illegal decision or resource failure. All 17,205 frozen inputs retained their
recorded sizes and hashes. No strength games ran; the larger-data study remains
deferred.

**Keep 120 seconds per complete move for both players. The next step is a
versioned live-game adapter and transfer checks for our four-candidate planner,
then a separately frozen, bounded pilot of three operational pairs: six games,
one pair per proposal model.** Complete the remaining nine operational pairs
only if operational checks and measured costs support continuation. These
operational games are excluded from strength analysis.

## Exact qualified opponent

The selected profile is
`entity_onehot_puct6400_canonical_atomic_prereveal_cpu_120s_v1`.
Its SHA256 is
`6f74c3b6d77707c42619801d1b1bef3fbf203d1ee6ff548afbe935169c257599`.
The output is `outputs/splendorust_120_qualification_20261018_01/`.

The observer-only native binary, Entity model and canonical atomic pre-reveal
adapter are unchanged from the accepted external02 qualification. Only the
fixed quota, allowances and explicitly versioned profile validation changed.
The profile retains depth 64, three sampled worlds, batch one, cpuct 0.4,
dynamic FPU with reduction 0.02965, zero uniform prior and no noise. The
chance-universes setting is inactive on this canonical path. More iterations
do not provide additional independent sampled worlds. Auxiliary, single-choice,
late-turn and forced-pass branches retain their declared behavior.

| Identity | SHA256 or pinned identifier |
|---|---|
| Corrected rules | `splendor_base_qualified_rules_20261018_v1` |
| Corrected catalogue | `db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312` |
| Upstream source commit | `1828507b8d2f0e02bc03d102b9c6d29a54fa8d57` |
| Entity checkpoint | `ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb` |
| Native executable | `b15322f1c6aac1034638bdc52d91524639c63cbc2b61d361a9fef79dfdea728c` |
| Qualification freeze | `657018d8a7ba56c8413beae722c16fe06e025105b1804cc520d3a5a09f9caf4c` |
| Accepted independent analysis | `9ffa3eccd4f70b3e188de3f8b8bd2ca8423fd28526bc8f244fb19b7bc43c5370` |

This is our authenticated local adaptation, not a claim to reproduce the
published native competition entrant. Historical rules, training data,
one-second PUCT16 results and all earlier failures remain preserved.

## Predetermined selection and final gate

The prospective protocol fixed four quotas and seven calibration fixtures.
Each quota required seven correct, complete responses within 96 seconds.
All four passed; 6,400 was the predeclared ceiling. No action value or game
outcome entered selection, and the ladder was not extended to consume spare time.

| Fixed simulations per searched Main phase | Passed / attempted | Mean receipt time | Maximum |
|---:|---:|---:|---:|
| 256 | 7 / 7 | 1.465s | 1.681s |
| 1,024 | 7 / 7 | 5.066s | 5.278s |
| 4,096 | 7 / 7 | 19.381s | 22.511s |
| 6,400 | 7 / 7 | 32.935s | 39.542s |

The final gate used the unchanged 18 core positions, spanning the three local
proposal-model groups, two acting seats and three phases, plus five edge
positions. Each was followed by its predetermined hidden-state variant, giving
46 fresh requests. This is one fixed SplendoRust model across that fixture
panel, not three opponent models. The 23 original snapshots equal those used
in the corrected local-player qualification. They are corrected-rules
counterfactual positions; historical actions are not acceptance ground truth.

All 46 completed decisions passed legal-support and corrected dictionary
successor checks. All 23 original/variant pairs had byte-equivalent scientific
response content while their actual hidden states differed. Calibration plus
qualification supplied 74 independently checked actual dictionary transitions.
There were no retries, replacements, quota reductions or partial-search returns.

| Complete-move timing | SplendoRust final panel | Corrected four-candidate planner, prior panel |
|---|---:|---:|
| Completed / attempted at 120s | 46 / 46 | 26 / 26 |
| Mean | 30.102s | 51.428s |
| Median | 33.511s | 46.868s |
| 95th percentile, linear interpolation | 38.627s | 98.310s |
| Maximum | 41.618s | 101.952s |
| Margin from observed maximum to 120s | 78.382s / 65.32% | 18.048s / 15.04% |

SplendoRust meets the prospective category of at least 20% observed margin.
Our planner retains its conditional qualification with thin observed margin.
Panel repetition differs, and these are descriptive fixture timings, not
future latency guarantees or a controlled relative-speed estimate. Retain the
120-second allowance; do not reduce it based on this panel. Equal ceilings do
not mean equal actual thinking time, simulation counts or world coverage.

Every final request completed exactly 6,400 simulations in its searched Main
phase: 294,400 simulations total, with 252,506 host neural calls and leaf calls,
plus 12 auxiliary heuristic phases. Across calibration and final qualification,
the corresponding totals were 376,832 simulations, 328,033 host calls/leaves
and 20 auxiliary heuristic phases. No single-choice, late-turn or forced-pass
branch was observed in this execution. Simulation and neural-call counts are
different quantities; fast cached/terminal searches still completed their quota.

## Clock, resources and audit boundary

The trusted supervisor used the same assigned performance-core logical thread,
one numerical thread and normal priority as the local qualification. Each
request had a fresh Python/Rust process tree. The external 120-second clock
covered input serialization, transport, search, auxiliary choices and complete
response delivery. The internal 108-second limit was fail-closed and did not
replace that clock. Initialization and fixed model warm-up had their own
60-second cap outside thinking time; all 74 initializations passed, averaging
2.647 seconds and peaking at 3.957 seconds. No position or search tree persisted.

Production took 35m21.5s. The final analyzer ran once, passed in 22.6s, and
finished 36m38.3s after the original launch, within the four-hour execution cap.
Preflight had passed all 256 tests in 24.33s. Peak sampled combined process-tree
RSS was 255.3 MiB against a 2 GiB cap; the sum of last observed worker CPU time
was 1,703.97 seconds. Evidence before the analyzer receipt occupied 9,783,150
bytes, about 9.33 MiB against a 256 MiB cap. Finalization records the later
inventory and closed process identities. Sampled telemetry is not a continuous
peak measurement.

The independent analyzer checked schedule, profile selection, actual receipts,
work diagnostics, legal moves, dictionary successors, deterministic pairs and
all 17,205 inputs totaling 373,117,853 bytes. It did not rerun the neural model
or replay every internal native search edge or PUCT backup. Those internals
retain the authenticated source, compatibility and numerical-parity trust
boundary from the prior qualification. Some observer/legal/model dependencies
are shared. Finite fixture correctness and timing do not establish strength.

## Next matched-benchmark step

The local entrant remains `four_candidate128_corrected_rules_20261018_v1`,
with all three proposal fits 161001, 161002 and 161003, four candidates and all
128 worlds. No fit is selected by results. The external entrant is the exact
6,400 profile above. Preserve corrected rules, complete-move clock scope,
fresh initialization policy and one-core/one-numerical-thread constraints.
Keep local full-trace writing and delivery inside its thinking clock.

One integration task is necessary before live games: the current local context
validator and independent verifier bind the historical `main` schedule, 201 old
pair identities, fit assignment and public random streams. Its worker also
binds trace output to the qualification directory. Reusing old labels for new
games would silently reuse historical randomness and misstate provenance.

Create a separately versioned live-game context, worker, evidence destination
and independent verifier. Freeze a new public benchmark namespace and balanced
pair-to-fit allocation, keeping each pair's fit fixed across swapped seats.
Keep actual referee seeds in the parent-only schedule. Reuse unchanged corrected
scoring, proposals, continuation primitives and full 128-world reduction;
preserve all frozen qualification files.

Before games, require bounded transfer checks: with identical observations,
fits and old public streams, reproduce saved candidates, world schedules,
continuation outcomes and choices, allowing only declared identity metadata
changes. Separately test the new namespace, invalid contexts, hidden-state
repeats and the actual timed evidence interface across all three fits. New
streams intentionally change samples; they are not expected to reproduce old
choices. Freeze this new interface and paired harness before any pilot.

Retain the roadmap's 12 operational pairs, four per fit, staged as 3 + 9.
Start with only three pairs, one per fit, using paired initial worlds and
swapped seats. Keep win totals closed and exclude all operational games from
strength analysis. Continue based solely on correctness, privacy, timing,
storage and measured cost. Assign this pilot its own explicit wall-clock and
storage caps before launch; the qualification's 256 MiB cap is inadequate.
On failures, preserve evidence and apply the frozen stop policy.

Preserve `docs/engineering/verification_policy.md`: check every actual game
move and decision, replay eight complete sampled worlds per 128-world root
with all its candidates, and fully replay the predetermined stress subset.
Do not weaken verification to make the benchmark appear cheap, or assume its
runtime is simply 8/128 of full-audit cost. Fix timeout/forfeit, draw, termination
and failure handling before games; never exclude a losing or slow game after
seeing its outcome.

## Cost scenarios for the decision after the pilot

The prior PUCT16-versus-raw035 screen averaged 51.933 plies per game. Combining
that historical length with fixture means of 51.428 seconds for our planner
and 30.102 for this opponent gives the following serial collection scenarios.
For N pairs and mean L plies per game, thinking hours are
`N * L * (mean_local_seconds + mean_external_seconds) / 3600`.

| Allocation | Thinking time alone | Local decision receipts + compressed traces alone |
|---|---:|---:|
| First 3 operational pairs / 6 games | 3.53h | 0.309 GiB / 317 MiB |
| All 12 operational pairs / 24 games | 14.11h | 1.238 GiB |
| Alternative fixed 30 main pairs / 60 games | 35.28h | 3.095 GiB |
| Roadmap 201 main pairs / 402 games | 236.40h | 20.736 GiB |

These are scenarios, not forecasts or resource caps. Stronger players can
change game lengths and the distribution of decision difficulty. Thinking
cost excludes fresh initialization, referee work, evidence overhead and
verification. Storage uses the prior 26 local decisions' mean 2.034 MiB per
receipt-plus-trace and excludes all other records, logs and audits. Operational
and main allocations are separate: a main allocation adds to pilot costs.
Even six operational games exceed the qualification storage cap in this
scenario. Preserve evidence; no cleanup or larger-data collection is implied.

After the pilot has measured these costs, either retain the roadmap's fixed
201 independent main pairs, 67 per fit, or explicitly choose a smaller fixed
30-pair/60-game screen, ten per fit, before strength outcomes are opened.
The latter has lower precision and is not equivalent to the planned main
benchmark. Preserve equal fit weighting, pair-level analysis, win/draw/loss
scoring and the prospectively fixed uncertainty procedure. Never add games
after a near miss. The 6,400 quota is timing-qualified at a declared ceiling;
neither its strength advantage nor its optimality within 120 seconds is known.

Supporting records: `independent_analysis.json`,
`independent_analysis_execution_01.json`, `selected_profile.json`,
`cross_player_panel_check.json`, `cost_and_timing_summary.json` and
`finalization.json` in the output directory. Protocol:
`docs/engineering/splendorust_120_qualification_20261018.md`.
Roadmap: `docs/strategy/external_strength_calibration_2026-10-18.md`.
