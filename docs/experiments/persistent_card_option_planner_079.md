# Persistent Card-Option Planner (`079`)

Frozen before implementing the planner, runner, telemetry, or tests, and before
generating any `079` game. No human replay is an input. No training, behavior
cloning, coefficient sweep, checkpoint change, or GPU is involved.

## Question and separation from `077`

`078` established causally that funded-prestige value for one high-efficiency
reserved target improves otherwise identical one-ply full-action search. `077`
itself remains closed: it may not be tuned, deepened, masked, phase-limited,
reweighted, cloned, or promoted.

`079` tests a different control architecture: an explicit card option persists
across turns, is visible to the planner as state, and terminates when its target
is bought or lost. Search compares lexicographic option outcomes rather than
adding the `077` scalar bonus. This asks whether the supported principle can be
turned into a direct challenger to champion `035`.

## Frozen option controller

The bot holds `target_card_id | None`.

- A target already visible or in the bot's reserved hand persists.
- A target terminates successfully when the bot buys it and unsuccessfully when
  it disappears without entering the bot's hand or purchases.
- With no target, qualifying own reserved cards are considered first, otherwise
  qualifying visible cards. A card qualifies at effective value
  `points / effective_cost >= 0.40`, with infinity when already free.
- Candidates are ordered by descending effective value, descending prestige,
  fewer nonzero cost colours, then card ID. The first becomes the option.
- If no card qualifies, the bot uses the exact frozen no-bonus full-action
  one-ply search from `078` for that decision.

This is the only target-selection rule. There is no stale timer, target switch,
phase rule, deck mask, opponent-value target, or alternative threshold.

## Frozen target-conditioned planner

When a target exists, use depth-3 minimax: own root action, adversarial opponent
response, then own action. Four public-observation determinizations are averaged
componentwise. Hidden cards from the true state are never read.

At each state, evaluate from the root player's perspective with this exact
lexicographic tuple:

1. terminal result: root win `+1`, nonterminal/draw `0`, root loss `-1`;
2. target status: purchased by root `3`, reserved by root `2`, still visible
   `1`, otherwise lost `0`;
3. funded target prestige: `card_points * progress`, where progress is the exact
   `078` funded fraction using bonuses, held colour tokens, and gold;
4. negative target deficit;
5. unchanged `GreedyHeuristicBot.evaluate_state` from the root perspective.

Tuple comparison is literal and lexicographic; no weighted scalar or tunable
coefficient is used. The opponent minimizes the tuple and can therefore model
buying or reserving an unsecured target.

For tractability, root candidates are the best eight by the mean immediate tuple
across the four determinizations. Internal maximizing/minimizing nodes retain the
best/worst four by their immediate tuple. All legal actions are simulated before
these fixed caps, so target buys/reserves, token payments, and opponent blocks
compete on exact engine successors. Ties use deterministic action representation.
These depth/breadth values are frozen once and may not be reduced, widened, or
swept after timing or outcomes.

## Comparator and dose

Opponent and same-state shadow comparator are raw legal champion `035`, SHA-256
`4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`.
At every candidate decision, fix the planner action first, then query `035` on
the same public state. Canonical movement ignores buy-payment aliases but retains
action type, card identity, deck tier, take/return multiset, and noble choice.
Report movement overall and by phase. Final movement must be at least 15%.

Also report option selections, option decisions, successful completions, losses,
completion rate, action mix, reserves and purchases by tier, deck reserves, gold
spent, mean purchases, mean turns, and terminations. Diagnostics never tune or
rescue the candidate.

## Stage A: mechanics and operational gate

Run 40 fresh paired seeds `773000--773039`, candidate once from each seat against
raw `035`: 80 games. Outcomes are not summarized or used. All required:

- exact seed/seat coverage and byte-identical first-cluster replay;
- every game `completed` or `forced_pass_deadlock`, with zero stalls, timeouts,
  illegal/null actions, or loop fallbacks;
- frozen checkpoint hash and exact planner constants;
- movement from raw `035` at least 15%;
- an active option on at least 50% of candidate decisions;
- at least half of selected options complete successfully;
- at least 5.0 purchases per candidate game; and
- Stage-A elapsed time at most 45 minutes and projected 200-pair Stage-B time at
  most three hours.

Any failure stops before Stage B and closes this exact architecture. No outcome
from Stage A is a gate or is pooled later.

## Stage B: strength

Only after Stage A passes, run 200 fresh paired seeds `774000--774199`, candidate
once from each seat against raw `035`: 400 games. Limits are `max_turns=400`,
`repetition_limit=4`, and `no_progress_limit=60`. No game may be replaced or
excluded.

Primary endpoint: candidate paired score share. Use a 20,000-resample paired-seed
bootstrap with seed `774079`. With paired-cluster SD about 0.32, 200 clusters
have about 82% power for the declared 56.5% expected share against the conjunction
of observed share at least 55% and CI95 lower bound above 50%.

## Decisions

- Integrity or operational failure:
  `persistent_card_option_planner_079_invalid_no_strength_decision`.
- Stage-A mechanics failure:
  `persistent_card_option_planner_079_mechanics_failed_close_exact_architecture`.
- Stage-B share at least 55% and CI95 lower bound strictly above 50%:
  `persistent_card_option_planner_079_stronger_authorize_deployability_preregistration`.
- Valid Stage B with CI95 upper bound below 50%:
  `persistent_card_option_planner_079_weaker_close_exact_architecture`.
- Any other valid Stage B:
  `persistent_card_option_planner_079_not_decisively_stronger_close_exact_architecture`.

A pass does not promote the planner. It authorizes only a separate deployability
preregistration addressing runtime and confirmation. Every non-pass closes this
exact persistent-option architecture. Do not tune target selection, tuple order,
depth, breadth, determinizations, or deck behavior; do not pool with `077`.

## Evidence

Single artifact:
`outputs/persistent_card_option_planner_079/result.json`. Record its SHA-256 in
`AGENT_CURRENT.md` after the single frozen run.

## Result (recorded after the frozen run)

The complete pre-evidence suite passed `612/612`. Stage A began at 15:45:12
local with ten workers. At the frozen 45-minute boundary (16:30:12), it had not
completed the 40-pair mechanics panel and had not reported even its first
ten-cluster progress block. The process tree was terminated at the preregistered
cutoff. No partial outcome was exposed, recovered, summarized, or used, and
Stage B never started.

Formal decision:
`persistent_card_option_planner_079_invalid_no_strength_decision`.

This is an operational invalidity, not a gameplay-strength result and not a
mechanics result. It closes the exact depth-3/root-8/internal-4/four-
determinization persistent-option architecture. The supported `078` principle
survives, but no reduced breadth, depth, determinization count, or smaller panel
is authorized as a rescue. A future architecture must make option state cheap by
construction rather than approximating this frozen planner after observing its
cost.

Evidence SHA-256:
`9473bced1d0104d9f682efb68a5f91858433f8498e02e7e37e74d1db400a43c0`.
