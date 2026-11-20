# Prepared one-second external comparison

26 October 2026. **Status: design prepared; no opponent qualified, execution freeze, or game launch yet.** The user authorized opponent research and preparation. This document specifies the next bounded study; player execution, training and game launches have not been requested in this preparation step. Preserve optimized64 as the candidate, unchanged32 as its control, and all earlier evidence. Larger-data work remains deferred.

## Question and recommended order

Does optimized64 deliver useful one-second playing strength against separately developed opposition, and does its advantage over unchanged32 extend to that opposition?

Investigate at most two independent players in this fixed order:

1. **BreckEmert's exported NumPy policy**, from `Splendor-Webapp` commit `6ec66efb02406e982510d2279ae1f29b84f80ace`. Proposed adapted profile: `breck_numpy_policy_corrected_public_v1`. Use the released policy weights only, with no critic or search.
2. **domdomegg's PPO512 policy**, from `splendor` commit `426fa50e06fe99b40620131ca962f418824bb73a`. Proposed adapted profile: `dom_ppo512_policy_corrected_public_v1`. Use the released browser policy weights, with no training or added search.

Both have public weight artifacts and MIT license files. Both need privacy and rules adaptation; neither is already a suitable unchanged entrant. The order is based on implementation feasibility and independent provenance, not results against optimized64. Their local strength and timing remain unknown. The detailed [opponent assessment](independent_opponents_20261026.md) records exact sources, limitations and excluded alternatives.

Qualify and screen the first profile. Consider the reserve only if the first is unavailable, fails qualification, or fails the predeclared competitiveness screen. Select the **first** passing profile, not the one producing the most favorable optimized64 score. Stop after two profiles if neither passes. No third opponent, tuned checkpoint, added search or repeated failed attempt is automatically substituted.

## Existing players and evidence to bind

The accepted recovered reliability evidence covers 200 games and 11,020 moves without a player failure. The earlier completed strength comparison already gave optimized64 a 64.25% score against unchanged32, including two deadline losses. This new study is not another direct 64-versus-32 match and does not erase those failures.

| Property | Candidate | Control |
|---|---|---|
| Identity | Optimized64 | Unchanged32 |
| Decision worker | `tools.splendorust_inference_opt_20261025.worker` | `tools.splendorust_search_scale_20261024.worker` |
| Inference | Qualified graph version | Qualified eager version |
| Search simulations | 64 | 32 |
| Profile SHA256 | `2b60b56419c938ecd5e96d5b9c9a5153f36b2796d4e4b35219b746dc7dfed6d9` | `ecaa11d2bfebc4701bf6129ab1f00ff2a31fff35ce0e590ebb683e1367002db5` |

Shared checkpoint SHA256: `ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb`; upstream SplendoRust commit `1828507b8d2f0e02bc03d102b9c6d29a54fa8d57`. Preserve float32, single numerical thread, three worlds, depth64, cpuct0.4, dynamic FPU reduction0.02965, batch1, no noise/uniform prior, fresh observation-seeded search per turn, qualified response reader and lighter observer. Retain the exact native binary, graph and weight identities from the accepted reliability freeze. No reference pointer is promoted by this preparation.

Preserve player/helper CPU0 and supervisors CPUs2–19 at normal priority. Only one move executes at a time; no pondering, GPU or concurrent arena. Every player receives a **1.000-second complete-response allowance**. Preserve the local players' 0.900-second internal limit. The external policy must also finish all encoding, inference, masking, auxiliary choices and response delivery within the same whole-turn limit. Startup loading/warmup is declared, identical in accounting, and charged to the job; it cannot hide per-decision work.

## Corrected rules and the external adapter

Use `splendor_base_qualified_rules_20261018_v1`, catalogue SHA256 `db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312`, and `atomic_prereveal_completed_turn_v1`. Follow [the mandatory policy](future_training_rules_20261021.md), including the additive factory/decoder and one startup `verify_runtime_sources()` call in every new collector/evaluator worker. No historical default environment, decoder or dataset loader may silently re-enter the evaluation chain.

The corrected referee alone determines legality, transitions, nobles, final-round completion and ties. The external source engine is not the rules authority. Neither proposed policy performs internal game search, so the adaptation need not preserve an incorrect upstream forward simulator. Historical weights remain explicitly historical; running them under the corrected contract does not relabel their training data.

The worker receives only lawful observation and public history. Both proposed source encoders expect features for opposing reserved cards without a suitable unknown-card sentinel. Use **one deterministic lawful completion** of hidden opponent reservations: sample without replacement from the corrected unseen catalogue, constrained by public tier/slot history, excluding all known cards. Seed solely from canonical lawful observation/history plus the frozen profile salt. Preserve own private cards and publicly known reservations. Never supply the real hidden identities, remaining deck composition/order, referee world seed or future refill. Do not average multiple completions or add search. Identical lawful observations must yield identical completion, features and decision despite different hidden referee states. See the [precise completion design](../../outputs/external_opponent_preparation_20261026_01/scout_independent/privacy_completion_design.json).

Before timing qualification, freeze the complete source-index-to-corrected-action correspondence table, including legal subsets, aliases, payments, returns, nobles, pass and tie order. Preserve source feature arithmetic and float32 inference. Select the first maximum among admissible source policy outputs in source index order. Auxiliary behavior must be fully specified before outcomes: retain a valid source auxiliary routine where it has defined semantics; any needed one-noble choice, gold-return correction or other new rule is an explicit adapter component, with fixed observation-derived RNG and reference fixtures. No silent fallback, guessed index, invented payment, substituted local planner or post-outcome auxiliary adjustment is allowed. Empty support or ambiguous mapping blocks admission. Card identity uses a verified tier/bonus/prestige/cost correspondence, not matching integer IDs.

All auxiliary choices share the original turn clock and pre-reveal information boundary. A sequential source interface must not see replacement cards, blind-reserve identities or other newly revealed information before the atomic turn's remaining choices are fixed. This is a test of explicitly adapted opponents, not a claim that the published agents or reported human performance transfer unchanged.

## Stage A: qualification and 60-game screen

Implement and test the adapter in new versioned files before a single bounded attempt. Pin source, weights, runtime, action mapping, hidden completion, every auxiliary choice, test fixtures, seeds and this design. Downloaded weight bytes must match the recorded Git blob and receive a local SHA256; metadata availability is not weight verification.

Qualification must include:

- The actual entrypoint on all six corrected card faces, nobles after non-purchases, multiple eligible nobles, noble-triggered final round, gold payments/returns, private reservations, scarce/exhausted decks, tie break, forced pass/deadlock and legacy-identity rejection.
- Encoder and unmasked numerical-output parity against the pinned source on the **same lawful surrogate state**. Check legal masks, first-maximum/tie behavior and complete translated actions separately; do not demand original choices where corrected rules deliberately change support. Define numerical tolerances before model execution and bind them in the execution freeze; action/tie/legality requirements remain exact. Do not loosen tolerances after failure.
- Privacy pairs that vary actual hidden states and referee seeds while retaining identical legal observation/history; complete decisions and surrogate provenance must match. Exercise sequential auxiliary phases and verify each complete successor independently.
- Existing local actual-entrypoint qualification suites, plus external representation-specific cases. Reserve at most 64 external cases, each with two hidden-state variants and two exact repetitions: at most256 external timed requests. Freeze their actual inventory before running. Existing local startup checks retain their52-decision suites and graph parity requirements.
- Strict one-second enforcement with no grace or failed-move retry. Require no fault and every ordinary timed qualification decision below0.700s. Separately bounded model loading, graph-parity fixtures and bundled startup checks are not ordinary move requests; their cost still belongs to the job. A profile needing more decision time does not qualify under this design; do not silently switch to another checkpoint or precision.

Then run **six fresh operational games against unchanged32**, three swapped-seat pairs, excluded from the screen. Require correct evidence and no player fault. All timely moves remain legal results; the0.700s level is descriptive in games, not an extra timeout rule. Operational forecasts must show the remaining screen and final audit fit their caps before expansion.

The competitiveness screen is **60 fresh games, 30 swapped-seat pairs against unchanged32 only**. No optimized64 game score is used to select the opponent. Score draws as half. The opponent must earn at least **15 points from natural games** out of the fixed60 (25%), with no external player fault and all verification passing. This practical threshold avoids spending the larger budget on an opponent already heavily outmatched by the control; it is not a statistically certified minimum strength. Forfeit wins do not help it meet this threshold; record all outcomes separately. There is no upper score cutoff: a strong opponent is welcome. A control failure stops admission for diagnosis rather than prompting another opponent selection. Incomplete or invalid screens cannot pass. A completed but weak screen remains a valid negative result; do not add games to move it across the threshold.

This is a coarse development screen, not a significance test or a strength claim. Its seeds, operational games and qualification positions are excluded from confirmation. Do not alter the selected opponent after inspecting results.

## Stage B: one fixed external comparison

After selection, freeze a new execution inventory and independent seed schedule. Run **12 fresh operational games**, three swapped-seat pairs for each local arm against the opponent. These are excluded from scoring. Gate only on correctness, faults, resources and timing delivery; keep their optimized64 outcomes out of opponent selection. A failed operational gate ends this attempt; it does not trigger choosing a more favorable opponent.

Main sample: **100 independent starting setups, four games per setup, 400 scored games**.

| Game within a block | Seat0 | Seat1 |
|---|---|---|
| A | Optimized64 | External opponent |
| B | External opponent | Optimized64 |
| C | Unchanged32 | External opponent |
| D | External opponent | Unchanged32 |

All four share the same initial world; player states reset between games. Freeze a balanced, interleaved order before outcomes, so arm/seat is not confounded with run time. The external player's policy, RNG rule, hardware allowance and auxiliary routines are identical in both matchups. Keep all referee randomness outside workers. Generate fresh256-bit seed namespaces, inventory/check known historical seed registries with any bounded omissions disclosed, and reconstruct paired worlds independently. Do not claim an exhaustive history scan if one was not performed.

An attributable timeout, worker error or illegal action loses that game; preserve the request, clock evidence and forfeit, then continue to the next scheduled game. Never retry the move, replace the game or change the player. This fixed strength sample differs explicitly from the preceding reliability trial's first-failure stopping rule. Infrastructure, rule, privacy, evidence or resource failure stops the study as incomplete. A stopped prefix is diagnostic only. Preserve forced-pass/deadlock conventions and the2,000-ply ceiling; a decision-limit/deadlock outcome cannot be invented as an official win or draw.

## Analysis fixed before outcomes

For block i, let c_i be optimized64's mean score over its two games, b_i the corresponding unchanged32 mean, and d_i=c_i-b_i. A win scores1, a draw0.5, a retained forfeit follows the frozen classifier. Analyze all100 blocks.

- **Primary:** S64=mean(c_i), strength against the selected opponent.
- **Secondary:** delta=mean(d_i), incremental performance over unchanged32 against common opposition. Report S32, seats, natural-game scores, forfeits, response-time distributions and slow moves as descriptive measures.
- Use50,000 joint resamples of whole four-game blocks with a frozen analysis seed. Report two-sided97.5% percentile intervals for S64 and delta: Bonferroni gives nominal/approximate95% family coverage, not a finite-sample guarantee. Also show ordinary95% intervals for interpretation. Never resample400 games as independent observations.
- Useful external superiority requires S64>=55%, adjusted lower bound>50%, each seat>=50%, and all integrity checks. Useful incremental gain requires delta>=5 percentage points and its adjusted lower bound>0. Report every negative delta; adjusted upper bound<0 supports regression. Otherwise the difference remains unresolved, not equivalence or noninferiority.
- If an endpoint's empirical resampling interval degenerates, do not report zero uncertainty: additionally use the bounded-block Hoeffding interval at the same confidence level, with support[0,1] for c and[-1,1] for d, and require its lower bound to pass before making that endpoint's superiority claim.

One hundred blocks may resolve a substantial effect; it is not reliably powered for a5-point difference. Correlation and paired covariance determine precision. As a conservative illustration, normal97.5% half-widths can reach roughly11.2points for S64 and22.4points for delta at their maximum variances; these are planning approximations, not guaranteed interval widths. No sample extension, opponent replacement or pooled screen/confirmation estimate follows an unfavorable or uncertain result. Generalization remains limited to one selected adapted opponent; no human rank or superhuman claim follows.

## Bounds, cost and verification

There are at most **two4-hour qualification/screen attempts**, followed by **one12-hour confirmation**: at most20 charged job-hours across the prepared sequence. No automatic relaunch or clock reset. Implementation/prelaunch model-free testing is separate preparation, not hidden game or inference work. Each launched job uses an owning watchdog, UTC and monotonic deadlines, a kill-on-close process job, captured child AND watchdog exit records, and cleanup charged within the same allowance.

| Job | Collection cutoff and stage maxima | Evidence cap |
|---|---|---|
| Each qualification/screen | 4h global;3h collection cutoff; qualification60min, six operational games15min, screen105min, final60min; clamp every stage to remaining global time | 1GiB including128MiB reserved for final verification |
| Confirmation | 12h global;10h collection cutoff; startup15min,12 operational games30min, main9h15min, final2h; clamp every stage to remaining global time | 4GiB including512MiB reserved for final verification |

Cap newly acquired opponent source/weight assets at256MiB, excluding existing immutable dependencies; do not install a large new stack under this plan. Maximum new footprint is therefore6.25GiB, with no automatic enlargement. Require13GiB free disk before the sequence and maintain5GiB. Require2GiB available RAM stable for60s at initial admission, then check that floor before each move; there is no routine60s wait per move. If RAM falls below the floor before a move, require60s restored stability before resuming, with at most600s cumulative waiting charged to each job. Pause only before a move, never extend an active move. Cap aggregate owned-worker RSS at2GiB; supervise resource and placement limits throughout. No unrelated models run concurrently as part of this study.

Measured local evidence gives47.15s/game collection and55.1plies/game. Assuming similar lengths and overhead, an external average move cost of0.2–0.9s projects approximately45–65s/game, or **5–7.2h for400 games**. Including operational and final verification, plan approximately **6–9h** for confirmation, plus qualification/screening. This is an extrapolation, not a local timing result for either proposed opponent. Slow policy implementations or different game lengths can invalidate it.

After the12 operational games, an outcome-blind forecast with at least1.5x allowance must fit remaining collection time, evidence and the2h final reserve. Include all412 game audits, every worker startup, raw transport, lossless compressed logs, privacy/placement checks, graph parity, sources/runtime, evidence seals, analysis and up to**48 completed-decision reproductions per matchup (96 total)**. Freeze the selection cells and missing-cell reporting in advance. Reproduce exact decisions with the same surrogate/RNG/auxiliary behavior; never repeat a failed move as if it had succeeded. If the forecast does not fit, stop before main collection rather than reducing checks or extending clocks.

Preserve all existing success criteria. Reuse accepted ancestry receipts through a finite pinned inventory; do not recursively re-audit old experiments during every guard. New actual-entrypoint checks and full new evidence verification remain mandatory. Use bounded streaming preservation and explicit source/inventory checks. Successful acceptance needs all checks, complete workload, cleanup and independent receipt reconciliation. Any later recovery requires a new separately authorized plan and retains the original incomplete attempt.

## Preparation deliverables and next action

Saved source inventories and the machine-readable [prepared design](../../outputs/external_opponent_preparation_20261026_01/prepared_design.json) pin research choices without pretending the adapters exist. Weight bytes, exact action/auxiliary tables, tolerances, reserved test cases, dependency inventory, actual seeds and executable hashes remain required execution-freeze fields. Their values must be fixed before the relevant model qualification or game outcomes; they cannot be chosen after failures or scores.

Recommended next action: implement and qualify the first adapted Breck policy, then run only its bounded unchanged32 screen if qualification passes. The reserve profile is not a request to train a replacement. No larger-data collection, fitting, self-play training, new SplendoRust architecture or search quota is part of this study.

Evidence: [accepted reliability report](splendorust_reliability_finalization_20261026_results.md), [completed64-versus32 strength result](splendorust_graph64_strength_20261025_results.md), [roadmap](../strategy/splendorust_revised_roadmap_2026-10-24.md), [source inventory](../../outputs/external_opponent_preparation_20261026_01/scout_independent/source_inventory.json).
