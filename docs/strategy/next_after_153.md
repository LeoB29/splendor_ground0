# Recommendation after 153: fresh validation of direct rollout move selection

**Execution update, 2026-09-10:** the owner authorized this proposal as 154.
It is now complete and fully audited: fresh paired WDL gain +0.048868815,
CI95 [+0.036814372,+0.061350759], all statistical and integrity gates passed.
All 247,136 main continuations and all 1,616 source games independently replayed;
all 1,214 tests passed. See the
[154 result](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/rollout_selector_154_result.md).
This supports the fixed conditional selector, not a full-game strength or
learning claim. No arena, training, promotion or successor ran. The original
recommendation-only statements below are historical and are preserved.

2026-09-10. Recommendation only. No experiment implementation, new source game,
model inference, fitting, continuation, test runner or arena launched. The owner
asked for the next step. Preserve all dirty work and frozen predecessors.

## Decision

Recommend one experiment, provisionally 154: **does a fixed direct rollout
selector find useful deviations from raw 035 on fresh positions, when judged
using entirely independent possible futures?** Use exact terminal simulations
to choose the move, without a newly trained value or advantage model.

This targets a remaining uncertainty in the current action-advantage programme.
The post-150 split-world analysis suggested positive selection headroom, while
149–153 did not turn the labels into useful learned choices. Before choosing
another learner, establish whether this particular finite-budget source of
move improvements works prospectively outside reused development positions.

This is not a test of whether search can ever improve 035: 084 already established
that, with 75.875% score share. Nor would a pass establish a better player than
084, a stable exact best-action label, a learnability guarantee or a successful
self-play loop. It validates only this fixed conditional move selector. Actual
games and any subsequent learning remain separate decisions.

## Why this takes priority

| Evidence | Consequence |
|---|---|
| 153 fails all four gates; 122/192 validation states have feature aliases | Close that quadratic public-count control. A weaker penalty or added feature would be a local rescue, not the strongest next test. |
| 149–152 fail material learned move improvement despite representation, checkpoint and data-allocation changes | Lower the priority of another nearby learner adjustment or unchanged scale-up. These results do not reject different learning regimes generally. |
| Post-150 stored-world selector: +0.087240 validation WDL on the opposite half, with 32 discovery worlds | Positive local motivation for a fixed 32-world selector, but the states and analysis were reused and post hoc. |
| 055 found poor repeatability of old resolver labels; 138–141 exposed limits of conditional expanded-root gains | Use fresh worlds, prospectively frozen choices and actual greedy-035 WDL continuations. Do not call a noisy sampled minimum a truth label. |
| 148 delivered fourfold PUCT compute but did not establish its strength gate | This proposal changes the evaluation mechanism; it does not extend the consumed 148 panel or tune its budget. |
| 142 stopped before searched self-play; earlier 041 and 052–054 already tried policy updates/distillation | Strong-policy learning remains open, but plain PPO or distillation is not an untried shortcut. |

Direct rollout selection has precedent in stochastic game planning: evaluate
actions by continuing a fixed base policy, then select from estimated returns.
That motivates the method, not a guarantee for finite samples, this candidate
set, hidden information or Splendor gameplay.
[Tesauro and Galperin, original research](https://research.ibm.com/publications/on-line-policy-improvement-using-monte-carlo-search).

## Proposed fixed experiment

1. Generate **768 fresh positions, one per independent source game**, 128 in each
   early/middle/late by acting-seat cell. Use exact greedy 035 source games,
   as in 149's fresh conditional panel; this is closer to baseline play than
   the exploratory 143/152 development sources. The population difference is
   explicit: this is not an exact replication of the post-150 mean.
2. Select states by a deterministic, outcome-independent first-feasible balanced
   matching rule. Proposed source schedule ceiling: 4,096 games. Freeze source
   and position hash rules before collection; stop if the schedule cannot fill
   the panel. No replacement for invalid/cutoff games. Source filtering and all
   attempted games, including any prefetched suffix, must be accounted for.
3. Retain 149's four-candidate contract: exact greedy-035 action, highest-ranked
   distinct canonical alternative and two uniformly sampled remaining canonical
   actions. Keep fewer-action and baseline-only states. Freeze all concrete
   actions, ordering and semantic keys before their discovery outcomes.
4. For each state, use **32 independent observer-legal discovery worlds**, shared
   across its candidates. Take each fixed root action and continue both players
   with exact greedy 035 to a valid terminal WDL result. Select the highest mean
   WDL candidate, preferring baseline on exact ties. No confidence filter, action
   expansion, extra penalty, trained evaluator or chosen rollout budget.
5. **Seal every selected action before collecting any evaluation continuation.**
   Use **128 entirely new observer-legal evaluation worlds per state**, shared
   between that locked choice and baseline. Evaluate only those two fixed moves
   under the same exact greedy-035 continuation law. When choice equals baseline,
   share the identical computation and retain a zero-gain state. Never reselect
   from evaluation outcomes or use those worlds as discovery data.

Retain the existing restriction of no already-held blind reservation at the
source snapshot, with explicit coverage reporting. Blind reservation remains
legal at the root and throughout continuations. Policies see only their legal
observations. World generation must not use the actual hidden order as an input
to the selector. Payments, returns, nobles, final-round rules, terminal roots,
deck exhaustion and valid forced-pass deadlocks follow the audited engine.

All source, candidate, discovery, evaluation, bootstrap and off-panel probe
namespaces must be fixed and checked for overlap before any new outcomes.
Do not reopen 149's consumed test or any other protected panel. Existing
development results motivate this proposal, but are not pooled into it.

## Primary decision and interpretation

For each source state compute mean evaluation WDL(selected) minus WDL(035).
Average states equally. Use 50,000 paired source-game bootstrap replicates,
stratified within the six fixed cells, with one frozen seed. Worlds are nested
Monte Carlo observations, not additional independent source games.

Proposed pass requires all of:

- mean WDL gain at least **+0.02**;
- paired 95% confidence interval lower bound above zero;
- interval half-width no greater than **0.02**;
- all integrity and natural-termination checks passed.

The +0.02 WDL target equals one percentage point of expected terminal score
under these conditional continuations, not a predicted arena win-rate increase.
The 768-state size is intended to improve precision: scaling the old validation
state standard error gives a rough half-width near .017, but that old statistic
averaged two split directions and used different source states. This is not a
power guarantee. A wider interval remains precision-limited; do not extend the
panel or change the gate after inspecting results.

Report apparent discovery gain, independent evaluation gain, their difference,
switch fraction, baseline-only/coincident choices, phase/seat summaries and all
actual work counts. Exact top-action agreement is not a success endpoint.
There is no learner here, so do not impose a prediction-MSE gate or claim a
result about generalization of a fitted network.

A pass would establish that this fixed simulator-based selector supplies useful
conditional improvements on fresh baseline positions. The next question would
be whether applying it repeatedly improves complete games; passing does not
automatically authorize that arena or another learner. A failed or imprecise
result would not support using this selector as a validated improvement source.
It would not show that the target contains no signal, reject richer search,
invalidate 084 or prove that all learning approaches fail. Both outcomes stop.

## Cost and audit

At four candidates per state, discovery needs at most
`768 * 4 * 32 = 98,304` terminal continuations. Independent evaluation needs
at most `768 * 2 * 128 = 196,608`. Total main collection ceiling is **294,912**,
with savings when the chosen action equals baseline. Add source games,
off-panel preparation probes and a complete independent replay audit.
There are zero new fitted models and no neural refits.

This is a simulation experiment, substantially more expensive than 153's
42-second prepared execution. 149's 245,760-continuation pipeline took about
2h41 including six main fits and their reproduction; its replay audit alone
took 75 minutes. Those are historical scale references, not a measured runtime
for 154. Expect an order of hours including audit; establish the actual estimate
with an outcome-blind timing preflight before main collection. Do not silently
reduce the proposed population or replication to meet a time target.

Freeze isolated runner, protocol, verifier, tests, all inputs and schedules.
Reuse the frozen engine and exact 035 continuation code without editing them;
follow explicit parent identities instead of expanding 149's old dynamic one.
Independently verify every source game and selection rule, candidate support,
legal observation/world construction, discovery path, action lock, evaluation
path, termination, target arithmetic, bootstrap and decision. Verify that
evaluation files were produced after the complete action-selection lock.
Complete the full audit even if statistical gates fail. Preserve failures.

This recommendation includes no automatic implementation or launch, new learner,
full-game arena, model promotion, large-scale self-play or successor.

## Local evidence

- [153 result](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/public_feature_advantage_153_result.md)
- [Post-150 split-world diagnostic and caveats](C:/Users/leoba/Desktop/splendor_ground0/docs/strategy/next_after_150.md)
- [Stored post-150 diagnostic](C:/Users/leoba/Desktop/splendor_ground0/outputs/early_learning_150/analysis/post150_development_target_split_diagnostic.json)
- [149 candidate, target and uncertainty results](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/afterstate_advantage_149_result.md)
- [139 conditional/gameplay warning](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/independent_confirmation_root_policy_139_result.md)
- [141 response calibration](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/opponent_response_calibration_141_result.md)
- [148 search-compute result](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/search_teacher_headroom_148_result.md)
- [142 unreached strong-policy learning cycle](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/strong_policy_pilot_142_result.md)

Champions remain 084 research / 035 fast. Experiment 153 remains closed and
unchanged; no 154 code, output directory or experiment process was created.
