# Research roadmap: strong starting player, practical search, then justified learning

24 October 2026. Updated at the user's request after reconsidering the alternatives. This is a roadmap and a prepared study design, not an execution authorization or a new strength result.

This revision supersedes the recommendations in the [23 October roadmap](splendorust_revised_roadmap_2026-10-23.md). That document, its illustrated PDF, validation receipt and all experiment evidence remain unchanged. The former 6,400-versus-10,240 search proposal is deferred; it has not been launched or disproved.

## Objective and choice

The objective remains demonstrably exceptional two-player base-game Splendor using only lawful information, ultimately tested against diverse strong opponents and expert humans. Roughly **one second per move** remains the practical target. The **120-second allowance** is a separate research budget, not a replacement deployment goal. Multi-day training can be useful when justified, but larger-data work remains deferred while storage and its expected value are unresolved. See the [original goals](../../README.md).

Make a pretrained policy-and-value network with adaptive tree search the main development direction, starting from our exact verified SplendoRust adaptation. Preserve its weights and player versions as references. Retain our earlier four-candidate/full-rollout planner as a diagnostic control and source of ideas, rather than the default next investment.

This adopts a strong existing starting point; it does not establish that every SplendoRust design choice is optimal. Do not automatically increase model size, world count, search quota or training volume. The eventual fast player may use the inherited model, a more efficient implementation, or a smaller model taught by a stronger research player. The teacher and the deployed player need not have the same thinking budget or architecture.

## Evidence and uncertainty

The accepted amended 120-second benchmark ended 60-0 for SplendoRust against our four-candidate/128-world player: 59 natural losses and one retained timeout loss. All three local proposal fits and both seats lost every game. This compares complete players; it does not identify the causal share of network architecture, training, search, candidate support or implementation. The original interrupted attempt remains incomplete, and the accepted amended result retains its disclosed recovery. [Accepted finalization](../../outputs/matched_strength_recovery_20261022_01/finalization.json)

The [saved-game postmortem](saved_game_postmortem_2026-10-22.md) examined the narrow shortlist and found a plausible evaluation problem. Immediate maximum-prestige actions were always offered; inadequate longer-term candidate coverage remains unresolved. Our greedy-035 continuations usually allowed more remaining turns than the actual stronger opponent did. More samples can reduce sampling error without correcting that policy mismatch. This is diagnostic evidence, not proof that every engine-building move was wrong or that flat rollouts can never work.

The A64/B64 experiment passed its 54 timed decisions and 36 core-output replay checks, then stopped at its outcome-blind assessment-throughput gate. Its independent quality assessment is incomplete. Preserve that engineering success and failure without reclassifying the experiment as either a strength gain or a quality rejection. [Qualification](../../outputs/move_evaluation_64_20261023_01/qualification_complete.json), [throughput gate](../../outputs/move_evaluation_64_20261023_01/throughput_gate.json), [execution](../../outputs/move_evaluation_64_20261023_01/execution.json)

There is already a useful fast reference: the qualified Entity PUCT16 adaptation beat raw035 58-2 under a one-second ceiling. That establishes strength against raw035, not against the previous searched planner, expert humans or all independent bots. [One-second screen](../engineering/external_strength_screen_20261018_02_result.md)

Published SplendoRust evidence also separates equal-search-count strength from actual response cost. Its larger Entity parent beat smaller alternatives at a fixed search budget, while the smaller E81 won the tested comparisons at similar native response cost. Those are other profiles, hardware and an earlier checkpoint; they do not decide our local one-second comparison. They caution against assuming that a larger network or more search is always the best use of time. [Upstream architecture study](https://github.com/paytonjjones/splendorust/blob/main/research/architecture_pivots/REPORT.md)

One upstream offline learning update improved the Entity parent, but further iterative learning was not established by the later exploratory comparison. We inherit a strong player, not a proven unlimited improvement process. [Closed training study](https://github.com/paytonjjones/splendorust/blob/main/research/training_strategy/RESULTS.md)

## First prepared study: network alone versus small search

Follow the [bounded one-second comparison](../engineering/splendorust_one_second_comparison_20261024.md). The study asks: **does the already qualified small search improve the same strong network within our practical allowance, and what does that improvement cost?**

| Arm | Main decision | Shared components |
| --- | --- | --- |
| Network alone | First maximum of the inherited Entity policy logits on the qualified shared Main support; explicit greedy route with no tree search | Exact checkpoint, observation encoder, qualified Main support/order and safety filter, corrected referee, atomic pre-reveal adapter and common auxiliary/fallback behavior |
| Network with search | Existing PUCT16 behavior with the same Entity policy priors and leaf values | Same components and one-second whole-turn ceiling |

The network-only arm is a complete legal player including the same payment, return, noble, forced-pass and declared fallback behavior. It is not a bare network without rules or heuristics. The value output can be computed by the shared model forward but must not affect its ordinary greedy Main choice. Setting a search quota to zero is not an adequate implementation: existing native zero-quota behavior has different tie/branch semantics. Qualification must verify the explicit greedy path and reproduce the existing PUCT16 reference.

The exact inherited checkpoint remains `ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb`, from pinned upstream commit `1828507b8d2f0e02bc03d102b9c6d29a54fa8d57`. No fitting or checkpoint selection is part of this comparison. Its historical training origin is retained; using it through a corrected adapter does not relabel its original data.

The prepared plan specifies 200 fresh games, 100 paired initial worlds with swapped seats, after correctness/timing qualification and six operational games excluded from the score. Both arms receive a real one-second complete-move limit on the same declared CPU resource. The proposed execution cap is six hours across qualification, rehearsal, collection and final audit, with 512 MiB of new evidence and a 2 GiB combined-worker memory bound. Operational forecasts must fit the fixed collection and evidence envelopes before main games begin. The detailed study is authoritative for clocks, margins, seeding, failure classification, resources and analysis.

This is a development comparison, not a superhuman benchmark or automatic promotion. Two hundred games can identify a substantial difference but do not guarantee a precise answer about a small gain. Preserve uncertainty; do not enlarge the sample after seeing its score, equate an inconclusive result with equivalence, or use these games as fresh confirmation for a subsequently tuned candidate.

## How the result will guide the next investment

Keep the frozen PUCT16 and PUCT6400 references regardless of the outcome. The study changes no reference pointer by itself.

- **Useful, supported search gain:** retain small adaptive search as the deployment reference. Use measured costs to select one efficiency or search-allocation hypothesis. This does not prove that another 60% of search or a different world pool will improve it further.
- **Useful, supported greedy advantage:** investigate where shallow search or its value estimates cause regression before increasing search or collecting search-labelled training data. It does not establish that all deeper search is harmful.
- **Unresolved small difference:** report the interval and practical trade-off. Retain the existing qualified reference while deciding whether the question merits a separately designed higher-precision study. There is no automatic extension or equivalence claim.
- **Correctness, operational or resource failure:** preserve it and distinguish it from a valid playing-strength outcome. No automatic quota fallback, restart, discarded timeout or replacement game is permitted.

The direct comparison measures the value of this search package for these fixed weights, including its use of the value head. It does not isolate tree structure from leaf evaluation, compare neural architectures at matched training, or establish the benefit of the 6,400-iteration research player. No large causal factorial experiment is required merely to keep the already stronger starting player.

## Subsequent programme

### 1. Improve useful strength per second

Measure delivered whole-turn time, inference work and actual game strength before choosing the next intervention. Qualification and a narrow performance diagnostic should distinguish avoidable transport/input overhead from essential neural/search work. A runtime change needs a separate version and appropriate numerical, action, privacy and clock checks; do not assume that changing backend or batching preserves behavior.

If model evaluation is the limiting cost and a stronger teacher offers useful headroom, consider a smaller distilled policy-and-value network. If search adds value but spends its budget poorly, consider a single search-allocation or work-reuse change instead. There is no automatic compression campaign or parameter sweep.

### 2. Demonstrate a learning cycle before scaling one

Only when storage and the expected benefit justify it, prepare a bounded corrected-rules learning pilot. The proposed targets must have a credible relationship to stronger complete play. Start from the strong checkpoint when compatible rather than unnecessarily relearning basic strategy.

Keep a frozen parent, independent assessment games and whole-game evaluation at the intended time budget. Separate better fit to targets from better decisions and stronger play. A useful successor should improve strength at comparable cost, or preserve useful strength with a worthwhile speed reduction. A second successful cycle is needed before assuming improvements will compound. Teacher generation and evaluation costs belong in the investment estimate, not just fitting time.

Every future collector, evaluator and trainer must satisfy the [corrected-rules policy](../engineering/future_training_rules_20261021.md) end to end, including startup `verify_runtime_sources()` and the actual-entry-point counterexamples. There is currently no generally migrated training entry point. Historical checkpoints may be explicit warm starts; historical targets cannot be relabelled corrected data.

### 3. Keep alternative ideas as bounded challengers

The earlier planner's paired-world comparisons may still be useful. A short-horizon rollout with a strong leaf evaluator is a legitimate challenger, but is a materially different algorithm from hundreds of full greedy continuations. Pursue it only with a concrete hypothesis and an affordable, matched-clock test against the current reference.

Likewise, three sampled root worlds are an approximation, not a proven dominant defect. World diversity, belief modelling, tree persistence and sequential auxiliary decisions remain separate possible studies. Do not combine these changes and then attribute the result to one of them. Corrected catalogue and noble handling are mandatory; sequential interface improvements remain separately scoped follow-ups.

### 4. Keep research strength and external calibration visible

Retain the 120-second PUCT6400 player as a strong research reference and possible teacher. A future 6,400-versus-10,240 comparison remains reasonable if the next explicit purpose is better high-budget strength or better training targets. It is deferred because it is less directly informative about one-second play and repeatable learning than the first study above.

Progress must eventually survive opponents beyond an immediate parent. Add qualified independent bot comparisons and prospective expert-human play under declared rules, hardware and clocks before any superhuman claim. Do not infer a human ranking from the old 60-0 result or a chain of internal wins.

## Preservation and execution status

No model, player source, frozen profile, game record, timeout, termination convention or existing evidence is changed by this revision. Our corrected referee, clocks, legal-action mappings, privacy checks and evidence tools remain useful assets. The prior roadmap and PDF remain historical artifacts with their original hashes.

The user authorized updating this roadmap and preparing the comparison. No evaluator implementation, model inference, simulation, training or game launch has been performed for this preparation. The prepared scientific design must be implemented in additive versioned files, reviewed, bound to an execution inventory and qualified before a separately authorized run. No current execution clock has started.
