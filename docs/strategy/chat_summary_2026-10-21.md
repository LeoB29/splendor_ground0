# Splendor research — summary of this chat

**As of 21 October 2026. The new strength benchmark is prepared and frozen, but has not started.**

We began with a learning experiment already running. The central question was whether improved learned move proposals would translate into stronger play. This chat completed that assessment, investigated the alternatives, and prepared a fair external strength comparison.

## What we completed

1. **Finished the main learning study unchanged.** All six fits passed acquisition and the prescribed audits. The relation-aware model's fresh gain over its anchor was about **+0.13 percentage points**, with uncertainty spanning zero; the original practical-success criterion was not met. The stronger teacher still showed useful headroom. These are simulated conditional-score differences, not whole-game win-rate gains.

2. **Evaluated both alternative rankers and their effect on planner choices.** Ridge and boosting were inconclusive on the sealed 720-position assessment. A subsequent saved-world diagnostic found small, uncertain gains of approximately **+0.09 and +0.15 points** after the planner's final selection. It reused existing simulations and was development evidence, not fresh confirmation. Neither ranker was promoted. See the [assessment](../engineering/alternative_ranker_128_result.md) and [diagnostic](../engineering/ranker_planner_bridge_128_result.md).

3. **Prepared the bridge to gameplay and priced larger data.** We designed both shortlist-based and full-legal-action proposal integration routes. We also priced increasing independent training positions from 11,520 to 46,080: roughly **63.5 GiB additional raw evidence and 13–14 active days including contingency**, before gameplay confirmation. Storage review and authorized cleanup recovered about **15 GiB** of temporary or verified redundant files. The larger-data study remains deferred pending storage and justification.

4. **Established practical players and corrected the rules.** Real-clock qualification showed that neither searched planner met the one-second deployment target; the eight-candidate planner missed eight research deadlines. The four-candidate planner became the practical 120-second candidate. We resolved six card-cost errors using authoritative sources, preserved historical evidence, and requalified the four-candidate planner under corrected rules: **26/26 research decisions passed across all three fits**. This was finite correctness/timing evidence, not a universal deadline guarantee.

5. **Found and qualified a stronger external opponent.** The first external PPO lost its cheap screen **60–0** to raw035. Adapted **SplendoRust**, a Rust search bot guided by a neural model, then beat raw035 **58–2** at one second. We subsequently qualified its fixed 6,400-simulation profile at 120 seconds. Those screens do not establish its strength against our searched planner. See the [external screen](../engineering/external_strength_screen_20261018_02_result.md).

6. **Validated live operation and investigated failures.** We completed six pilot games and eighteen further operational games, recovering from RAM shortages and repairing validation/process-supervision issues under bounded plans. All failures were preserved. Required game/replay checks passed, but four local moves genuinely timed out. None delivered a complete move within the deadline. We therefore chose timeout losses, with no grace, retry or partial-search fallback. RAM waiting is permitted between workers only. Operational matchup outcomes remain closed, and these games are excluded from the new strength sample. See the [accepted validation](../engineering/live_game_operational_final_validation_repair_20261020_result.md).

## New recommendation and frozen next step

Run a **fixed 60-game strength screen**: 30 fresh swapped-seat pairs, ten pairs per local model, using the unchanged four-candidate/128-world planner and qualified SplendoRust.

- Both receive **120 seconds per move**; timeouts count as losses.
- Twelve mandatory adapter/transfer checks precede the games; complete-sample verification and paired statistics precede acceptance.
- Estimated duration: **40–50 serial hours**; hard cap **72 hours**, with **8 GiB** of new evidence allowed.
- All **426 preparation tests passed**; 31,853 frozen inputs were verified.

We recommended sixty games because the earlier 402-game option would cost roughly **11–13 days**. This smaller screen can reveal a large difference, but may leave a close contest unresolved. It will not expand according to its results. The larger-data study stays deferred.

**Next action, when authorized: launch the [frozen benchmark](../engineering/matched_strength_20261020.md). No matched strength result against SplendoRust is established yet.**
