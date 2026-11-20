# Recommendation after 148: controlled afterstate advantage-learning pilot

Execution update2026-09-09: the owner authorized this as149. The bounded pilot is
complete and fully audited, with all245760 continuations and all6 fits reproduced.
It failed the advance rule: afterstate-selected WDL minus035-.042169,
CI95[-.070055,-.016845]; afterstate-control+.007094, CI95[-.006510,+.020061];
advantage MSE3.712% worse than control. See the
[149 result](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/afterstate_advantage_149_result.md).
No arena, promotion or automatic successor. The recommendation below is preserved
as pre-experiment reasoning; its not-launched statements are historical.

2026-09-08. Recommendation only. No experiment, model inference, training,
continuations, arena or test runner was launched in this review. Preserve all
dirty work and frozen artifacts. Champions remain 084 research / 035 fast.

## Decision

Recommend one bounded architectural experiment, provisionally 149: **does
explicitly encoding a move's deterministic consequences help a learner predict
and select useful deviations from a fixed policy?** Compare two new matched
learners on exactly the same state/action examples and outcome estimates:

- Control: the observable board before the move, plus the complete action.
- Treatment: the observable afterstate, after applying the known consequences
  of that action but before revealing a random refill or blind-reserved card.

Use paired action advantages as the common target: expected terminal WDL after
the candidate action minus expected terminal WDL after raw 035's action, with
both sides subsequently following exact greedy 035. This makes the target a
decision comparison, while the matched control isolates the new representation
within this recipe. Neither representation receives the actual future deck.

For example, buying a card has known token costs, a known bonus and known points.
The replacement card is uncertain. Let the exact engine supply those known
changes explicitly, and test whether the learner handles the remaining problem
better. This is an inductive-bias hypothesis, not a demonstrated missing-information
bug or a claim that representing an afterstate removes future outcome variance.

## Why this is the next useful intervention

147 exercised the available training exposure and found substantial overfit.
148 delivered almost four times the computation, but its 53.75% score share,
CI95 [49.5%,58.0%], established no search-strength gain. Both leave useful effects
possible. Neither supplies a validated teacher for an automatic learning cycle,
nor identifies a unique bottleneck. Another budget or epoch increase has low
priority relative to a substantive change in what the network must learn.

135 explicitly preserved the afterstate hypothesis while rejecting broad action
fragmentation as the reason to build a hierarchical action head. 136 showed
sufficient public information to reconstruct refill pools; 137 tested sampling
coverage, not a learned afterstate representation. 146/147 trained ordinary
pre-decision value encoders against terminal outcomes, not this controlled
afterstate/action-advantage comparison.

Action-return prediction and paired targets are NOT new to this programme.
The counterfactual action-value pilot, 002/003, 019, 052/053 and related audits
already caution against noisy action labels and limited generalization. 130/132
also warn that averaging more returns alone is not a sufficient remedy. The
new intervention here is explicitly computed afterstates, with a new conventional
learner as its matched control. Do not advertise another generic Q fit, more
samples, or paired randomness as an untested discovery.

The architectural idea also has precedent in stochastic planning: Stochastic
MuZero separates deterministic action effects from subsequent chance events.
That supports its plausibility, not its effectiveness in Splendor. Here we
already have the exact simulator, so there is no reason to learn replacement
game rules or a latent dynamics model for this pilot. See the
[author's explanation](https://www.julian.ac/blog/2022/05/15/planning-in-stochastic-environments-with-a-learned-model/).

## Proposed scope

First implement an isolated afterstate/chance interface. Keep frozen engine and
search files unchanged. Prove that composing the deterministic part with each
specified legal chance outcome reproduces the original engine step, including
payment/returns, gold, purchases/reservations, nobles, turn/final-round handling
and deck exhaustion. A pending afterstate is a distinct object, not a malformed
ordinary game state. No legal-action policy acts before its pending chance is
resolved. Do not obtain features by stepping with the real hidden deck and then
merely masking the drawn card: features must be invariant across compatible
hidden worlds before that draw.

The initial learning panel should use positions without an already-held blind
reserve, a clearly labeled scope restriction. Keep RESERVE_DECK legal among
candidate actions, with a pending private draw that is revealed only to its owner
after resolution. Verify that path explicitly. Naturally occurring later blind
reserves during continuations must also remain legal and observation-correct.
This does not validate general private-card belief learning or justify CFR.

Suggested fixed data scale: 768 training positions, 192 validation positions
and 192 fresh test positions, one per source game, balanced across early/middle/
late and acting seat. Training and validation sources can come from the existing
143 development games, with disjoint source-game splits. Exclude all previously
protected games. Generate new raw-035 source games for the test only after all
model choices are locked. Selection uses phase/seat/eligibility, never winners,
estimated gains or model disagreement. Record source filtering and its coverage.

For each position retain the greedy-035 canonical action, its highest-ranked
distinct alternative and two distinct actions sampled uniformly from the
remaining complete canonical support. Use all actions when fewer than four
exist. This deliberately reaches outside the policy's top prefix, but remains a
four-candidate learning panel, not a test of optimal ranking over every legal
move. Freeze all action keys without consulting continuation results.

Generate 32 independent observer-legal initial worlds per training position,
64 per validation position, and 128 per test position. On each world, take each
fixed action and continue both players with exact greedy 035 to a valid natural
terminal result. Couple the starting world across actions; do not preserve
discovery worlds in validation/test. Label from the snapshot actor's perspective,
using terminal WDL (+1/0/-1), not score margin or a sampled opponent minimum.
The advantage target lies in [-2,2]. Use no public-value/search estimate as a
truth label. With four actions everywhere, the full design contains at most
245,760 continuations, plus source generation, preflight and independent audit.

These are Monte Carlo estimates of outcomes under a specified continuation policy,
not exact optimal-play values. Pairing can reduce variance but need not do so;
record uncertainty and actual coupling. Exact greedy continuation aligns the
baseline action and subsequent policy, avoiding the sampled-minimum/behavior
mismatch examined in 140/141. It does not itself prove an eventual deployed gain.

Train both learners from fresh matched initializations, with the same trainable
encoder family, parameter budget, legal information, optimizer, update schedule,
validation rule and target data. Give both the same legal refill-pool information
and complete action details; the treatment is explicit state transformation.
Use three paired training seeds and retain all three under a fixed aggregation
rule, never select the best seed on test. An action's predicted advantage can be
formed by subtracting the model's prediction for the baseline action, ensuring
exact zero baseline advantage. Fix tie-breaking to prefer the baseline action.
No 146/147 checkpoint is reused, no frozen-035 feature bottleneck is imposed,
and no anchor/KL intervention differs between the two new learners.

The exact feature schema, model sizes, optimizer schedule, selection rule, seed
namespaces, audit coverage and execution bounds must be specified before any
new target outcomes. Use an outcome-blind timing preflight to establish cost;
do not promise the one-hour duration of 148. This pilot includes engineering,
controlled offline training and evaluation only, not a full new search engine,
multi-generation self-play programme or protected gameplay arena.

## What would count as progress

The primary scientific question is whether the afterstate representation improves
the expected outcomes of the moves selected by the learners. On the fresh test
positions, lock each model's choice among the fixed candidates before opening
new continuation outcomes. Compare afterstate-selected versus control-selected
terminal WDL, and afterstate-selected versus the greedy-035 baseline. Average
over the three paired training seeds without treating them as extra games;
resample whole source games. Report each seed as a robustness diagnostic.

A proposed advance rule is mean WDL gain >=.02 and positive paired CI95 lower
bounds for BOTH selection contrasts, together with improved held-out advantage
prediction MSE versus the matched control (paired CI upper below zero), valid
transitions/privacy and complete natural-termination accounting. A .02 WDL gain
means one percentage point of expected terminal score under these fixed
continuations; it is not a predicted full-game improvement. Require CI half-width
<=.02 for each selection contrast; wider intervals are precision-limited. Freeze
these criteria in the execution protocol before targets. Do not infer a useful afterstate
advantage from lower training loss or prediction MSE alone. Do not evaluate
"best-action agreement" against a noisy empirical test maximum as if it were an
optimal oracle. Model choices are evaluated on independently sampled futures.

- If only the afterstate learner improves prediction and selected outcomes,
  the new representation earns a subsequent integration/gameplay proposal.
- If both learners improve similarly over the baseline, the new data/advantage
  recipe may be useful, but an afterstate-specific benefit is not established.
- If prediction improves without decision benefit, retain that narrower finding;
  do not open a strength arena on prediction fit alone.
- If neither generalizes or intervals remain wide, report the failure or lack of
  precision. Do not expand samples, retune or add training seeds to pass.

Even a positive result would be conditional move-level evidence. 134 and 138-141
already show why it cannot substitute for actual full-game improvement under
replanning and changed state distributions. A later fresh gameplay test would be
needed, followed by evidence of repeated search/self-play learning. No promotion,
automatic successor or superhuman claim follows from this pilot.

This is one proposed architecture experiment in service of an eventual learning
loop, not a new indefinite 035-distillation programme. No work beyond read-only
review and these proposal/handoff documents ran in the current turn.

Local evidence: `docs/engineering/search_teacher_headroom_148_result.md`,
`docs/engineering/value_training_exposure_147_result.md`,
`docs/engineering/alphazero_action_space_fragmentation_audit_135_result.md`,
`docs/strategy/stochastic_refill_vs_blind_reserve_architecture_correction.md`,
`docs/engineering/alphazero_deadlock_consistent_repeated_return_value_ab_132_result.md`,
`docs/experiments/resolver_residual_ranking_distillation_053.md`, and the historical
counterfactual action-value pilot recorded in AGENT_CURRENT.md.
