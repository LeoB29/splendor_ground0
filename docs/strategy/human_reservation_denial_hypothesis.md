# Human observation about reserving contested cards

Recorded 2026-10-03 from the owner's description of two-player Splendor. This is
an untested diagnostic lead for the research programme, not a finding about the
AI or a selected next experiment. The running equal-budget breadth arena retains
its frozen rules. No new inference, simulation, training or gameplay was run.

## The reported situation and its limits

One especially attractive visible card requires many blue tokens, while other
attractive visible cards have little use for that many blue tokens. With four
blue tokens in the bank, taking two now is tempting because that action may be
unavailable on the next turn. Doing so can reveal the intended purchase. The
opponent may reserve the target, leaving the collected tokens with much less
immediate value. Reserving the target before collecting could protect the plan.

The owner explicitly does not know how widespread this situation is. Blocking
can harm the blocker by consuming a turn and reserve capacity needed for another
card. It is more attractive when the reserved card also contributes to the
blocker's own plan. The owner considers reservation of an obviously valuable
card the principal way players can disrupt an opponent's plan. This is their
experience, not a measured frequency or a universal rule to reserve first.

## Research interpretation

The proposed vulnerability is commitment to an exposed target: tokens have few
good alternative uses, the opponent has a worthwhile way to deny the target,
and the resulting loss of progress exceeds the cost of securing it first.
Double-taking itself is not diagnosed as an error. Tokens are not necessarily
useless after denial; later refills or another plan may restore their value.

The relevant response must be credible for the opponent. Forcing an opponent to
reserve at any cost could manufacture an apparent weakness. A test must allow
the original player to pivot after denial and compare the blocker with their
other useful moves. Gold availability, reserve capacity, purchases that also
remove the target, replenishment and the race to finish can alter the comparison.
Reserving defensively also consumes a turn and a slot, and may lose the chance
to double-take. Both sides' opportunity costs belong in the comparison.

The story combines urgency to take tokens, dependence on one target, and possible
revelation of intent. The target may already be apparent from the board, so this
example does not by itself establish a missing history or opponent-belief input.

## What static inspection establishes

The current arena forces each proposed first action, then calls the mature
terminal trajectory routine. Subsequent moves by both players use the frozen
035 policy. A missed reply in those continuations could bias a move estimate;
no missed reply has been observed or demonstrated in this review.

The observation encoder represents market cards, bank tokens, player resources
and reserve counts. The action-feature code already supplies target affordability
for both players and a reserve-denial flag when the opponent is at most two
tokens short, accounting for bonuses and gold. The code inspection does not
show how effectively a checkpoint uses these signals. It does not justify a
claim that blocking is absent from the representation or the trained policy.
Dependence on a single attractive target and the opportunity cost of blocking
require more than reading that one flag.

Sources inspected: src/splendor_ai/encoding/action_features.py,
src/splendor_ai/encoding/observation.py, src/splendor_ai/alphazero/strong_policy.py,
src/splendor_ai/alphazero/run_afterstate_advantage.py, and
tools/equal_budget_breadth_arena_128/arena.py. All files remain unchanged.

## Possible bounded diagnostic

If this lead is pursued, first inspect a small set of lawful saved positions
using public information, without showing the human engine scores or outcomes.
Ask which reply they would consider, why it helps the opponent and how the
original player could recover. User judgments generate hypotheses, not labels
declaring an optimal move.

Distinguish three possible problems: the protective reserve is missing from the
candidate set; a plausible denial response is underused by the continuation
policy; or useful search behavior is not captured by the learned proposer.
Each would suggest a different intervention. The AI may already handle all of
them correctly.

Use comparison cases with several attractive uses for the tokens and cases where
blocking is costly or unavailable. A selected collection of striking positions
can test a mechanism but cannot measure prevalence. Before elevating this lead
to a programme priority, estimate frequency on a prospectively defined broader
sample and independently assess the effect on play. Do not infer a broad
learning bottleneck from this anecdote or force a blocking heuristic into the
running experiment. No diagnostic execution or successor is authorized here.
