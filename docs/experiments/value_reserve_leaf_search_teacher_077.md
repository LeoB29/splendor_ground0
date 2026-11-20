# Value-Reserve Leaf Search Teacher (`077`)

Frozen before implementing the evaluator, runner, telemetry, or search-signature
repair, and before generating or inspecting any `077` game outcome. No human
replay is an input. No training, parameter sweep, checkpoint change, or GPU is
involved.

## Question and separation from `072`/`073`

`072` and `073` tested hand-written action sequences. Both failed to execute the
elicited strategy: the first spent target gold and the second made its gold rule
unsatisfiable. Their terminality clause closes further scripted-rule attempts,
not a state-value search instrument.

`077` asks a genuinely different question: can exact one-ply engine search,
repeated at every turn, execute a useful policy when its leaf value recognizes a
high-efficiency card held in reserve and the progress already made toward buying
it? Search, rather than an ordered rule list, compares taking, reserving, and
buying after applying their exact legal token and gold consequences.

## Frozen candidate

The candidate is an information-legal `ShallowSearchBot` with depth `1`, four
determinization samples, and no practical action cap (`10,000` globally and per
action type). The candidate therefore evaluates every distinct legal successor;
the large caps are not a search budget to be filled. Seeds are deterministic
functions of game seed, turn, and actor. Hidden cards are never read from the
true state: four frozen public-observation determinizations are averaged.

The base leaf value is the unchanged `GreedyHeuristicBot.evaluate_state`. For
the root player only, define for each reserved card:

```text
effective_cost = sum(max(card_cost[colour] - owned_bonus[colour], 0))
deficit = max(effective_cost - held_colour_tokens - held_gold, 0)
progress = 1                                      if effective_cost == 0
           1 - deficit / effective_cost           otherwise
option_value = 1000 * card_points * progress      if points/effective_cost >= 0.40
               0                                  otherwise
```

For an already-free card, effective value is infinite. The strategic leaf bonus
is the **maximum** option value across the hand, not the sum. The full leaf value
is base value plus that bonus. `1000` is the existing Greedy prestige coefficient;
the formula recognizes only the fraction of eventual prestige already funded.
Taking the maximum prevents the evaluator from rewarding three-card hoarding.
Buying a target removes the option but adds its full prestige plus the unchanged
Greedy card/bonus terms, so a fully funded card remains better bought than held.
The `0.40` threshold is inherited unchanged from `072`/`073`.

No opponent option bonus, deck-reserve mask, action prior, phase switch, learned
value, terminal rollout, or rule-specific action override is used.

### Search-signature mechanics repair

The current shallow-search successor signature omits the addressed card slot,
so distinct visible reserves with equal immediate token totals collapse to one
candidate. Before `077`, add `(tier, market_index, reserved_index, noble_id)` to
the local strategic signature. This retains deduplication of payment aliases but
keeps distinct card targets and noble choices. Unit tests must show that all
twelve visible reserve targets survive candidate construction in a state where
they previously collapsed. This correction is part of the frozen instrument and
is not evidence by itself.

## Comparators and intervention dose

At every candidate decision, record two outcome-blind shadow choices on the same
public state:

1. the same full-action, depth-1, four-determinization search with the strategic
   leaf bonus removed; and
2. raw legal checkpoint `035`.

Use canonical action identities that ignore token-payment aliases but retain
action type, card identity, deck tier, take/return multiset, and noble choice.
The primary intervention dose is candidate movement from raw `035`. Also report
movement from the exact search ablation, which isolates the leaf bonus, by phase
and action type. Report the fraction of decisions with a nonzero leaf bonus,
qualifying visible reserves, all reserves by tier, deck reserves, reserved-card
conversion, purchases by tier, gold spent, mean purchases per game, and game
length. None may be used for tuning.

## Stage A: automatic mechanics gate

Run 40 fresh paired seeds `770000--770039`, candidate once from each seat: 80
games. The runner proceeds automatically to Stage B without exposing or using a
Stage-A strength summary. Stage-A outcomes are never a gate.

All mechanics conditions are required:

- exactly 40 seed clusters and 80 games, each seed once per candidate seat;
- every game ends `completed` or `forced_pass_deadlock`, with zero stalls,
  timeouts, illegal/null actions, or loop fallbacks;
- byte-identical replay of the first cluster using fresh bot instances;
- frozen `035` SHA-256
  `4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`;
- candidate movement from raw `035` at least `15%`;
- candidate movement from the exact no-bonus search ablation at least `15%`;
- qualifying visible reserves at least `5%` of candidate decisions;
- at least half of qualifying visible reserves are later bought; and
- at least `5.0` total purchases per candidate game on average.

The last three conditions distinguish a live reserve-and-convert strategy from
the paralysis observed in `072`. Failure stops before Stage B and closes this
exact evaluator/search candidate. It does not authorize threshold or coefficient
changes.

## Stage B: fresh paired strength panel

Only after all Stage-A mechanics conditions pass, run 200 fresh paired seeds
`771000--771199`, candidate once from each seat against raw legal `035`: 400
games. Limits are the established `max_turns=400`, `repetition_limit=4`, and
`no_progress_limit=60`. Valid terminals are exactly `completed` and
`forced_pass_deadlock`. No game may be replaced, excluded, or pooled with Stage A
or any earlier panel.

Primary endpoint: candidate paired score share against raw `035`. The CI is a
20,000-resample paired-seed bootstrap with seed `771077`. Final intervention dose
is reported alongside it and must still be at least `15%`; it is an integrity
condition, not a secondary route to success.

The prior paired-cluster standard deviation is about `0.32`. At 200 clusters the
standard error is about `2.26` percentage points. The declared expected share is
`56.5%` (a `+6.5` point effect); under the normal approximation the conjunction
of observed share at least `55%` and a two-sided 95% lower bound above `50%` has
about 82% power. Estimated CPU cost is 10--30 minutes on ten workers; stop if the
mechanics panel exceeds 30 minutes or projects the full panel above three hours.
Timing failure makes the instrument operationally invalid and authorizes no
reduced determinization or action panel.

## Decisions

- Any integrity or operational failure:
  `value_reserve_leaf_search_077_invalid_no_strength_decision`.
- Stage-A mechanics failure:
  `value_reserve_leaf_search_077_mechanics_failed_close_exact_candidate`.
- All gates pass, Stage-B share at least `55%`, and CI95 lower bound strictly
  above `50%`:
  `value_reserve_leaf_search_077_stronger_authorize_bc_preregistration`.
- Valid Stage B with CI95 upper bound below `50%`:
  `value_reserve_leaf_search_077_weaker_close_exact_candidate`.
- Any other valid Stage-B result:
  `value_reserve_leaf_search_077_not_decisively_stronger_close_exact_candidate`.

A pass authorizes only a separately preregistered behavior-cloning corpus and
student with a fresh gameplay gate. It does not promote the search teacher.
Anything else closes this exact leaf formula, threshold, depth, breadth, and
determinization contract. Do not tune it, add a phase, deepen it, or reinterpret
a large movement rate as strategic success after a strength failure.

## Evidence

Single final artifact:
`outputs/value_reserve_leaf_search_teacher_077/result.json`. Record its SHA-256
in `AGENT_CURRENT.md`. Stage A and Stage B telemetry and all clusters remain in
that artifact; no human replay path or hash may appear in the configuration.

## Result (recorded after the frozen run)

The complete pre-evidence suite passed `604/604`. Stage A completed in 1,032.0
seconds and passed every integrity, operational, and mechanics gate. Its movement
was 74.69% from raw `035` and 66.64% from the exact no-bonus search ablation;
qualifying visible reserves were 9.34% of decisions, 62.67% converted, and the
candidate bought 12.925 cards per game.

Stage B completed all 200 clusters / 400 games with every integrity gate true and
all games completed normally. The candidate went **218-182-0**, score share
**54.5%**, paired-bootstrap CI95 **[49.5%,59.5%]**, mean final-score margin
**+0.5175**, CI95 **[-0.155,+1.1825]**. It therefore missed both frozen success
requirements: point share was below 55% and the CI lower bound was not above
50%. Formal decision:
`value_reserve_leaf_search_077_not_decisively_stronger_close_exact_candidate`.

The null was not low-dose. Final movement was **74.98%** from raw `035` and
**66.41%** from the no-bonus search ablation. Qualifying visible reserves were
8.53% of decisions and converted at 54.64%; all-reserve conversion was 60.59%.
The candidate still made 729 deck reserves (6.02% of decisions), while 94.33% of
decisions occurred with a live option bonus. These are diagnostics, not a route
around the failed strength conjunction. Close this exact evaluator, threshold,
depth, breadth, and determinization contract; no BC or promotion is authorized.

Evidence SHA-256:
`646652602826de5756c4dce46e826446b3a4d3d14af3dd50b3a64db428f4d61c`.
