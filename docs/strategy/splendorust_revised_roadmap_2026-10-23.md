# Revised research roadmap from verified SplendoRust

23 October 2026. Prepared design and recommendation; no new experiment is launched by this document.

The programme's objective remains a demonstrably strong Splendor player under a declared thinking-time budget, eventually tested against other strong bots and expert humans. Use our exact, corrected-rules SplendoRust adaptation as the starting player for future improvements. Preserve its current version as the permanent reference.

**First recommended study: increase only its fixed search quota from 6,400 to 10,240 simulations, with both players retaining the 120-second complete-move allowance.** Qualify the candidate, rehearse the matchup, then run a fixed, fresh 60-game screen if the operational gates pass. This tests a modest investment in search before redesigning the architecture or collecting training data.

The illustrated companion is [Two ways to think ahead in Splendor](../../output/pdf/splendor_players_intro_20261023.pdf).

## What changes in the roadmap

The accepted amended benchmark ended 60-0 for SplendoRust against our four-candidate, 128-world planner. Each of the three local proposal fits lost all 20 games; 59 losses were ordinary completed games and one was a genuine timeout. This establishes a substantial gap between those complete players. It does not identify the separate contributions of the network, search and other design choices, or establish superhuman strength. The original interrupted benchmark remains incomplete; the accepted result includes one disclosed recovery retry. [R1]

The default question now becomes: **can a change improve the verified SplendoRust player at a useful cost?** Our earlier planner, learned proposals and information-set ideas remain available as research candidates and diagnostic controls. Their value must be demonstrated against this stronger starting point. The existing corrected referee, legal-action adapters, clocks, hidden-information checks and evidence infrastructure remain useful.

The recent A64/B64 study passed all 54 timed decisions across the three fits and all 36 core-output replays, covering 9,088 continuations. The maximum observed decision was 89.999 seconds, below its 96-second gate. The alternative changed 13 of 18 decisions. At 18:48 Paris on 23 October, however, the outcome-blind assessment forecast required another 5h35 with its prescribed margin; only about 4h37 remained before the collection cutoff. The run stopped and preserved the failure. Its independent quality assessment is incomplete, so this is engineering progress, not evidence of stronger decisions or play. These statements are a read-only receipt inspection, not a new final closeout of that run. [R2]

Do not extend, retry, reinterpret or merge that stopped study into this one. Its original criteria and evidence stand.

## Exact starting player

Start from the version we tested locally, rather than assuming the upstream published entrant is identical. [R3, R4]

| Component | Retained identity or setting |
| --- | --- |
| Reference profile | `entity_onehot_puct6400_canonical_atomic_prereveal_cpu_120s_v1` |
| Profile SHA256 | `6f74c3b6d77707c42619801d1b1bef3fbf203d1ee6ff548afbe935169c257599` |
| Upstream commit | `1828507b8d2f0e02bc03d102b9c6d29a54fa8d57` |
| Entity checkpoint SHA256 | `ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb` |
| Native executable SHA256 | `b15322f1c6aac1034638bdc52d91524639c63cbc2b61d361a9fef79dfdea728c` |
| Ruleset | `splendor_base_qualified_rules_20261018_v1` |
| Catalogue SHA256 | `db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312` |
| Turn contract | `atomic_prereveal_completed_turn_v1` |
| Search | 6,400 iterations, depth 64, three sampled root worlds, batch one, cpuct 0.4, dynamic FPU reduction 0.02965, no noise, zero uniform prior |
| Execution | One assigned performance-core logical thread, one numerical thread, fresh per-move process/search initialization, observation-derived seed |
| Clock | 120 seconds for a complete response; internal 108-second fail-closed guard; separate 60-second initialization cap |

The three root worlds are cycled during canonical search. The separate configured `chance_universes=3` mechanism is inactive on that path. Increasing iterations does not increase the number of independent hidden worlds. Auxiliary payment, return and noble choices retain the qualified upstream heuristic and atomic pre-reveal restrictions. [R3, R5]

## First study and why it comes first

The previous qualification stopped its quota ladder at 6,400 because that was its predeclared ceiling, not because this was the highest demonstrated affordable search count. In the accepted benchmark, SplendoRust completed 1,497 decisions with a mean of 24.867 seconds and a maximum of 49.680 seconds. [R1, R3]

| Scenario | Mean decision | Maximum decision | Interpretation |
| --- | ---: | ---: | --- |
| Verified 6,400 | 24.87s | 49.68s | Historical measurements on that matchup |
| Proposed 10,240 | 39.79s | 79.49s | Simple 1.6x extrapolation, not a measurement |
| Unselected 12,800 | 49.73s | 99.36s | Simple 2x extrapolation already exceeds the proposed 96s margin |

Choose 10,240 as one conservative operational candidate, not as an estimated optimum. The existing native bridge accepts an iteration count, so this can be implemented through a separate versioned profile and evaluator wrapper without changing the native search algorithm. More search may produce little benefit, or amplify biases from the same network and three-world pool. That is precisely what the study should establish. [R5]

Keep the model, search settings apart from quota, seed rule, world count, adapter, auxiliary choices, hardware and initialization policy fixed. The candidate will consume more actual computation under the same ceiling. This is an equal-allowance comparison, not an equal-simulation or equal-energy comparison. Report actual cost alongside strength.

There is only one SplendoRust checkpoint in this study. The three fits from our earlier planner are not three versions of SplendoRust and do not require artificial repetition of this matchup.

### Stage 1 Correctness and timing

Proposed cap: **four hours, with the final hour reserved for verification**. Before model execution, freeze a new additive source/profile inventory, the fixture manifest and all limits. Each new evaluator worker must call `verify_runtime_sources()` once at startup and pass the actual corrected-rules entry-point counterexamples; a manifest or helper self-check alone is insufficient. [R6]

Use the existing 18 core and five edge positions. Add one predetermined stress position: the accepted benchmark's slowest completed SplendoRust move, selected only by trusted elapsed time, with game index and ply as tie breakers. Materialize that receipt and observer in the execution manifest. Include a hidden-state variant for each of the 24 roots. Run both quotas in alternating order by root, giving **48 requests per arm, 96 total**, plus separately specified startup/counterexample tests. Do not select positions by candidate results.

Require all of the following:

- Baseline scientific outputs reproduce their original records, excluding declared transport/profile metadata and clocks. Candidate decisions may legitimately differ.
- Every complete decision is legal and its resulting turn passes the separate corrected dictionary referee, including costs, payment, returns, nobles and final-round effects.
- Original/hidden-state-variant scientific responses match within each arm; unavailable opponent cards and true deck order cannot affect the delivered input or seed.
- Searched Main phases report exactly 6,400 or 10,240 simulations as appropriate. Known single-choice, auxiliary, late-turn and forced-pass branches are checked against their declared behavior rather than incorrectly required to search.
- Every timed request completes within **96 seconds**. The real 120-second deadline and internal 108-second fail-closed limit still apply. A 96-second margin failure is a failed qualification, not a game loss.
- Sources, weights and all unchanged settings match their pinned identities. No fallback quota, partial-search choice, replacement fixture or automatic retry is permitted.

Retain the existing source and numerical-parity assurance for native internals. Independently check actual delivered actions, transitions and reported work; do not describe this as replaying every internal PUCT edge.

### Stage 2 Operational rehearsal

If qualification passes, run **six games, three swapped-seat pairs**, under a separate **six-hour cap including one hour for verification**. Use fresh predetermined initial worlds and the unchanged 120-second complete-move limits. These games establish reliable operation and provide timing and storage forecasts. Their scores are excluded from the strength screen and cannot be used to tune the quota or other settings.

All moves and endings must verify, with no deadline, legality, source, privacy or infrastructure failure. Require every completed decision to stay within 96 seconds before advancing to the screen; a slower but timely move remains a legal game move, while failing this engineering gate. Use this rehearsal's outcome-blind cost measurements, with a 35% margin, to check that all 60 screen games and evidence will fit the next stage. A failed projection stops the proposed study without resizing it.

### Stage 3 Fresh strength screen

Use **60 games, 30 fresh initial worlds with seats swapped**, at **120 seconds per move for both players**. Freeze all seeds before launch; exclude every prior benchmark, diagnostic and rehearsal seed identified in the existing registries, and disclose any limits of that collision check. No games from the old 60-0 result enter this comparison.

Proposed total cap: **48 hours**, with collection ending by hour 42 and the final six hours reserved for verification and analysis. Use both monotonic and wall-clock bounds, and preserve all failures. Do not pause an active move's clock, reset it after interruption, add games, replace seeds or silently resume an interrupted attempt. An infrastructure interruption makes the affected frozen study incomplete; it is not automatically attributed to a player's skill.

An attributable player timeout, illegal output or player failure to deliver a valid complete move counts as a loss under the frozen forfeit classification. A timely complete move taking more than 96 seconds remains valid in this strength screen: 96 seconds is the earlier engineering margin, and 120 seconds is the game deadline. Natural draws earn half credit. An unresolved referee deadlock/cutoff is not an official terminal target and must be retained as an incomplete study condition.

Verify every actual move, terminal result, forfeit classification, trusted clock receipt, player identity and applicable quota. Retain a clear boundary around native-search internals: source authentication and work checks do not amount to reexecution of all neural search edges. Seal completed evidence before the independent final analysis.

### Resource envelope and estimated cost

Use one worker process tree at a time, at most **2 GiB combined worker/helper RSS**, and require **2 GiB available RAM stable for 60 seconds** before the initial admission. Later RAM waits occur only between requests, count against both stage and overall clocks, and total no more than 30 minutes per stage. Never suspend an active move to recover memory.

Propose **2 GiB of new evidence across all stages**, at least **7 GiB free disk at launch** and a maintained **5 GiB reserve**. Save compact move/receipt evidence rather than every internal tree edge. Existing frozen evidence is not deleted to meet these limits. Any resource breach stops dependent work and preserves the failure.

At historical game lengths of roughly 50-52 plies, the screen scenario is about 27-28 hours of thinking, another 2-3 hours of fresh initialization, and additional verification/bookkeeping: **approximately 30-36 hours total**. Rehearsal is roughly three hours plus checks; the 96-request timing panel is roughly one hour of thinking plus fixtures and audit. These are planning scenarios, not execution promises. Proposed stage caps total **58 hours**, including their reserves; one overall origin must prevent repeated stages from extending that total. Preparation time is separate and no cap has started here.

### Decision rule

Freeze the primary statistic as candidate score share: win 1, draw 0.5, loss 0, with each swapped-seat pair retained as a single resampling cluster. Report the overall estimate, a paired 95% interval, both seat means, timeouts/failures, and thinking-time distribution. A concrete execution protocol should use 50,000 pair-bootstrap resamples, a fixed seed (proposed 2026102302), and declared percentile conventions before launch. A degenerate bootstrap is not proof of zero population uncertainty.

Recommend advancing only if all evidence gates pass, score share is at least **55%**, the paired 95% lower bound is above **50%**, and the candidate scores at least **50% from each seat**. This is a screen for a substantial gain. Sixty games often cannot resolve a small improvement; missing the gate can mean inconclusive evidence rather than proof of no benefit. No post-result extension is built into this design.

A pass makes the candidate promising. Promotion requires a separately planned fresh confirmation, including another opponent or opponent family to check that progress is not limited to exploiting its ancestor. Select confirmation sample size and cost before its outcomes, using a practically worthwhile effect and observed pair variability. Retain the 6,400 reference regardless of the outcome.

## Subsequent research priorities

1. **Measure the value of search budget.** Complete the single quota study above before selecting a second intervention. If it fails operationally, preserve that result and redesign explicitly; do not walk down an automatic quota ladder.
2. **Investigate hidden-world diversity at fixed compute.** Cycling three root worlds is a concrete candidate for improvement, not a proven flaw. A later study could spread a fixed search budget across more lawful worlds while holding the network and other settings fixed. It needs new checks of sampling, cache behavior, privacy and stochastic refills.
3. **Consider reuse of work between turns.** This needs more than a persistence toggle. The current bridge constructs a fresh agent; keeping it alive also retains RNG/history and can enable purchase-only cycle-escape behavior. Isolate tree reuse from those changes, then qualify sequences, memory and clocks. [R5]
4. **Revisit interface restrictions as a separate player change.** Sequential reveal/return/noble choices and deadlock conventions remain documented follow-ups. Do not mix them into the search-quota comparison. [R6]
5. **Train only against a justified target.** Stronger targets, adequate storage, a qualified corrected-rules collection/loader/trainer chain and an independent assessment design must precede larger-data investment. Historical weights may be identified warm starts; historical targets cannot be relabelled corrected data.

Maintain distinct tracks for **120-second research strength** and **one-second deployment**. A larger search quota does not meet the one-second objective. Once a stronger research player is confirmed, investigate compression, efficient inference or lower-budget search with their own matched-clock tests. Final strength claims require diverse strong opponents and prospective expert-human play, not only victories over our older player.

## Evidence and architecture references

These references also identify the source key used in the illustrated report.

- **R1 / PDF 5:** [Accepted amended finalization](../../outputs/matched_strength_recovery_20261022_01/finalization.json) and [independent analysis](../../outputs/matched_strength_recovery_20261022_01/independent_analysis.json). Finalization establishes acceptance; the analysis file predates that acceptance.
- **R2 / PDF 6:** [A64 qualification](../../outputs/move_evaluation_64_20261023_01/qualification_complete.json), [breadth](../../outputs/move_evaluation_64_20261023_01/breadth.json), [throughput gate](../../outputs/move_evaluation_64_20261023_01/throughput_gate.json), [failure](../../outputs/move_evaluation_64_20261023_01/failure.json) and [execution](../../outputs/move_evaluation_64_20261023_01/execution.json).
- **R3 / PDF 3:** [SplendoRust 120-second qualification](../engineering/splendorust_120_qualification_20261018_result.md) and [earlier corrected adaptation](../engineering/external_strength_screen_20261018_02_result.md).
- **R4 / PDF 4:** [Pinned upstream champion and public weights](https://github.com/paytonjjones/splendorust/blob/1828507b8d2f0e02bc03d102b9c6d29a54fa8d57/docs/CHAMPION.md), [Entity architecture](../../outputs/external_opponent_search_20261018_02/splendorust/research/architecture_pivots/models.py) and [qualified host model adapter](../../tools/external_strength_screen_20261018_02/splendorust_adapter.py). The upstream result is not the evidence for our local strength comparison.
- **R5 / PDF 4:** [Native bridge](../../tools/external_strength_screen_20261018_02/native_bridge/src/main.rs) and [pinned neural search](../../outputs/external_opponent_search_20261018_02/splendorust/crates/splendor-agents/src/neural_search.rs), especially fresh construction, history/cycle handling, canonical iteration, three-world cycling and final root selection.
- **R6:** [Mandatory corrected-rules policy](../engineering/future_training_rules_20261021.md) and [additive helper](../../tools/future_training_rules_20261021.py).
- **PDF 1:** [Corrected local planner qualification](../engineering/corrected_planner_qualification_20261018_result.md), [frozen three-fit model registry](../../outputs/proposal_breadth_arena_128/models_lock.json), [teacher learning design](../engineering/teacher_budget_161.md), [action-scorer representation](../engineering/fast_selector_transfer_157.md), [neural architecture](../../src/splendor_ai/training/model.py), [actual matched-game selector](../../tools/matched_strength_20261020/local.py) and [signed terminal reward](../../src/splendor_ai/termination.py).
- **PDF 2:** [Saved-game postmortem](saved_game_postmortem_2026-10-22.md). Pace and evaluation mismatch are hypotheses supported by diagnostics, not proven causal explanations of the 60-0 result.

Both systems use attention-based neural models. Our verified planner uses Raw035 for its first two proposals and greedy continuations, a seeded exploration action, and one learned proposal from the selected 161 fit. Its action scorers encode 295 observation features and 79 features per legal action; their checkpoint metadata specifies two attention layers and four heads. Learned value heads do not determine the deployed rollout selection. The terminal win/draw/loss average over 128 lawful sampled worlds does, with historical internal deadlock conventions retained and disclosed.

The verified Entity Transformer represents 31 semantic tokens with 48 features each, width 256, six attention layers and eight heads, and has 4,780,883 parameters. It emits 81 Main-action logits and two signed values. The search uses priors and leaf values and chooses the most visited root action with a prior-based tie break. Auxiliary turn decisions are a separate heuristic path. These are architectural facts, not a causal attribution of its strength. Our historical training data is not certified corrected-rules data merely because those checkpoints now play through a corrected adapter.
