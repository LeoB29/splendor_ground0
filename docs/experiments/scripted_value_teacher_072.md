# Scripted Value-Reserve Teacher (`072`)

Frozen before implementing the bot, running any game, or inspecting any outcome.
No training, no checkpoint, no neural component. This is an existence test for a
strategy, and conditionally the qualification of a supervised teacher.

## Question

An experienced human player beat champion `035` 8-2 with mean margin +5.6, and
described their strategy as reserving high-efficiency cards early for a delayed
points payoff, while largely ignoring the opponent. Measured from the ten
recorded games, `035` never plays that strategy: 0 tier-3 reserves in 271
deterministic decisions, and 2 early tier-3 visible reserves in 63,622
epsilon-greedy self-play decisions.

**Does that strategy actually win in this engine, or did the human win for other
reasons?**

This matters because it is a question the local opponent league cannot answer.
Every league member -- `035`, `016`, `023`, and the shallow search bot -- traces
its strategic knowledge to one hand-written heuristic evaluator, and the
public-belief resolver takes its root candidates from `035`'s top four with `035`
continuations, so it inherits the same blind spot. See
`docs/project_review_response_070.md`.

## Why the strategy is mechanically plausible

Derived from `docs/splendor_base_cards.csv` and `engine/constants.py`, with no use
of the human replays:

- Points per token is monotone in cost **concentration** within each tier. Tier 3:
  1 cost colour `0.571`, 2 colours `0.500`, 3 colours `0.333`, 4 colours `0.214`.
  Tier 2 spans `0.083`--`0.500`, a 6x spread.
- For every colour X exactly two tier-3 cards cost `7X`, worth 4 and 5 points, and
  both grant the same bonus colour. One colour's accumulation therefore unlocks
  9 points across two cards.
- `BANK_GEMS_PER_COLOR_2P = 4`, so a `7X` cost can **never** be paid from tokens of
  colour X alone. At least 3 must come from gold or discounts.
- `GOLD_TOKENS = 5`, and gold is obtainable **only** by reserving, one per reserve.

So the five most efficient cards in the game are structurally reserve-gated
before a discount stack exists. Reserve-heavy play is the only early route to the
top of the value curve, which is consistent with the human's account and with the
champion's exclusion from it.

## Agent under test

`ValueReserveTeacherBot`: deterministic, no search, no neural network, no
randomness. Rules elicited verbatim from the expert and frozen here.

Two distinct quantities, deliberately separated:

- **effective cost** = `sum over colours of max(0, card cost - my bonus count)`.
  Discounts only, not tokens held.
- **effective value** = `points / effective cost`, infinite when the effective
  cost is zero.
- **deficit** = effective cost minus tokens held, minus gold held.

Targets are ranked by **effective value**, never by deficit. Ranking by deficit
would make the bot chase whatever it happens to hold tokens for, which is the
opposite of the elicited strategy.

**Efficiency filter:** effective value `>= 0.40`. At zero bonuses this admits
exactly 18 of 90 cards (the five `7X` tier-3s, the five 10-cost tier-3s, the four
`6X` tier-2s, the four `5X` tier-2s) and widens as bonuses accumulate.

Rules, applied in order; the first that fires plays:

1. Never reserve from the deck.
2. If any purchase reaches 15+ points, make it.
3. **Endgame block.** If the opponent's score plus the points of some visible card
   they can afford right now reaches 15, and I hold fewer than 3 reserves, reserve
   that card.
4. If a card in hand is affordable, buy it, highest points first.
5. If a visible card is affordable and its effective value is at least that of my
   best hand target, buy it.
6. If a visible card is affordable and its bonus colour reduces the effective cost
   of my best target, buy it regardless of its own points.
7. If I hold fewer than 3 reserves and a visible card passes the efficiency
   filter, reserve the best one, even if unaffordable, taking the gold.
8. Take tokens maximising useful tokens gained toward the best target's deficit,
   minus a penalty for returning tokens that target needs. This scoring prefers a
   double of a needed colour over three tokens of which one is needed, which is
   the intended default.
9. Drop a hand target only if its deficit has not shrunk in 10 of my own turns and
   a visible card has strictly better effective value.
10. Among action variants for the same card, prefer claiming a noble, then spend
    the least gold, then the fewest tokens.
11. Pass only if nothing else is legal.

Deliberately absent: any opponent model except rule 3, any noble targeting, and
any deck reserve.

## Panel

200 fresh paired seeds, `750000`--`750199`, each played twice with the teacher in
seat 0 then seat 1: **400 games**. The band is disjoint from every prior
experiment. Engine limits are the established `max_turns=400`,
`repetition_limit=4`, `no_progress_limit=60`. Valid terminals are exactly
`completed` and `forced_pass_deadlock`. No game may be replaced or excluded on
the basis of its outcome.

Paired score share is computed per seed cluster and its CI95 comes from a
paired-cluster bootstrap with 20,000 draws and seed `750072`. Individual games are
never independent observations.

## Declared power

Per-pair score-share standard deviation measured in the `043` robustness gate is
about `0.32`, so 200 clusters give a standard error near `0.023`. This panel
resolves a true effect of about `+6.4` points at 80% power. That is coarse, and it
is declared coarse in advance: the purpose is to separate "clearly stronger",
"clearly weaker", and "competitive", not to measure a small edge.

## Integrity gates

All required:

1. exactly 200 clusters and 400 games, seeds in the reserved band, each used once
   per seat;
2. every game ends `completed` or `forced_pass_deadlock`; zero stalls, zero
   `max_turns`, zero `repetition_cutoff`, zero `no_progress_cutoff`;
3. zero illegal actions; the teacher never returns a null action;
4. the teacher takes zero `RESERVE_DECK` actions across all 400 games;
5. the teacher is deterministic: replaying the first cluster reproduces it
   exactly;
6. the `035` checkpoint hash equals
   `4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`.

## Strength and decisions

Primary metric: teacher paired score share against raw legal `035`.

- Integrity failure: `scripted_value_teacher_invalid_no_strength_decision`.
- Score-share CI95 lower bound above `0.50`:
  `scripted_value_teacher_stronger_authorize_bc_teacher_preregistration`.
  The strategy wins in this engine. Authorizes only a separately preregistered
  behaviour-cloning corpus and student, with its own untouched gameplay gate. Does
  **not** promote the teacher, and does not make it the champion.
- Score-share CI95 upper bound below `0.50`:
  `scripted_value_teacher_weaker_close_scripted_strategy_line`.
  As encoded, the strategy loses. Do not tune the rules to rescue it; a rule sweep
  after seeing this result is exactly the post-result repair this project has
  avoided elsewhere. One re-elicitation from the expert is permitted **only** if
  the faithfulness diagnostics below show the encoding did not reproduce the
  described strategy, and it must be separately preregistered.
- Interval straddling `0.50`:
  `scripted_value_teacher_competitive_inconclusive`. Authorizes only a larger
  preregistered panel, sized from the observed variance.

## Reported diagnostics, gated on nothing

These are recorded to interpret the result and to check that the encoding is
faithful. **None is a gate, and none may be used as a tuning target.**

- Teacher action mix by type, and reserve counts by tier. The expert's mix was
  51.3% takes, 23.3% visible reserves, 18.8% reserved buys, 6.6% visible buys,
  with reserves split 14/21/28 across tiers 1/2/3. A teacher whose mix is far from
  this did not encode the described strategy, whatever its score.
- Nobles claimed by each side. The expert predicted their strategy is rarely near
  a noble. If the teacher wins while claiming none, that independently supports
  the account; if it loses while claiming none, nobles may be the gap.
- Cards purchased by tier, reserve conversion rate, mean game length, and mean
  gold spent per purchase.
- `035`'s own tier-3 reserve count on this panel, expected to be near zero.

Comparing the teacher's action mix with the expert's recorded mix is a
faithfulness check on an independently derived rule set. The ten human games are
**not** used to fit, select, or tune anything, so they remain held out.

## Prohibitions

Do not tune rules, thresholds, or the efficiency filter after seeing the result.
Do not promote the teacher as champion. Do not pool this panel with any future
confirmation. Do not build the behaviour-cloning corpus without its own frozen
protocol. Do not rent a GPU; there is no neural component.

## Evidence

`outputs/scripted_value_teacher_072/result.json`, with the SHA-256 recorded in
`AGENT_CURRENT.md` after the run.
