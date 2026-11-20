# Value-Reserve Leaf Bonus Causal Ablation (`078`)

Frozen before modifying the `077` bot, implementing this runner, running its
tests, or generating any `078` game. No human replay, learned checkpoint,
parameter sweep, rollout, or GPU is used.

## Question and non-rescue status

`077` cleanly created a high-dose strategic mode and finished 218-182 against
`035`, but failed its frozen conjunction at 54.5% share with CI95
[49.5%,59.5%]. That exact candidate is closed permanently. It may not be tuned,
deepened, masked, phase-limited, cloned, behavior-cloned, promoted, or rescued by
this result.

`077` combined exhaustive information-legal one-ply search with a
high-efficiency reserved-card leaf bonus. Its outcome therefore does not say
whether the bonus helped relative to the otherwise identical search. `078` asks
only that causal question, head-to-head. A pass supports the strategic principle
and authorizes a separately preregistered, genuinely different architecture; it
does not reopen `077`.

## Frozen players

Both players use the exact `077` search contract:

- depth `1`;
- four public-observation determinizations;
- deterministic seeds derived from game seed, turn, and actor;
- no practical action cap (`10,000` globally and per action type);
- the corrected action signature that preserves tier, market slot, reserved
  slot, and noble choice while deduplicating payment aliases;
- unchanged `GreedyHeuristicBot.evaluate_state` base value; and
- no checkpoint, action prior, opponent option term, phase switch, deck mask, or
  terminal rollout.

The **bonus player** adds exactly the frozen `077` own-player leaf bonus:

```text
effective_cost = sum(max(card_cost[colour] - owned_bonus[colour], 0))
deficit = max(effective_cost - held_colour_tokens - held_gold, 0)
progress = 1                            if effective_cost == 0
           1 - deficit/effective_cost  otherwise
option = 1000 * card_points * progress if points/effective_cost >= 0.40
         0                              otherwise
leaf_bonus = max(option over reserved cards, default=0)
```

The **ablation player** removes only `leaf_bonus`; its leaf is the unchanged base
value. There are no coefficients or alternatives to select.

## Panel and primary endpoint

Run 200 fresh paired seeds `772000--772199`, disjoint from `077` and all prior
panels. Each seed is played twice with the bonus player in seat 0 then seat 1:
400 games. Limits are `max_turns=400`, `repetition_limit=4`, and
`no_progress_limit=60`. Valid terminals are exactly `completed` and
`forced_pass_deadlock`; no game may be replaced or excluded.

Primary endpoint: bonus-player paired score share against the no-bonus search.
Use a 20,000-resample paired-seed bootstrap with seed `772078`. Game winners use
the engine's rule-correct prestige then fewest-development-cards tiebreak.

The prior paired-cluster standard deviation is about `0.32`. At 200 clusters the
standard error is about 2.26 percentage points. The declared expected bonus
effect is +6.5 points (56.5% share); the conjunction of observed share at least
55% and two-sided CI95 lower bound above 50% has about 82% power under the normal
approximation.

## Intervention dose and diagnostics

On every bonus-player decision, run an outcome-blind shadow choice from a fresh
exact ablation search on the same public state. Compute movement with the `077`
canonical action identity, ignoring buy-payment aliases but retaining action
type, card identity, deck tier, take/return multiset, and noble choice. The
shadow is evaluated only after the actual bonus action has been fixed and cannot
alter it.

Final movement from the ablation must be at least 15%; lower dose is an integrity
failure. Report movement overall and by phase, plus action mix, live-option rate,
qualifying visible reserves and conversion, all reserves by tier, deck reserves,
reserved purchases, purchases by tier, gold spent, mean turns, and termination
counts. These diagnostics are never alternate success criteria or tuning inputs.

## Integrity and operational stopping

All required:

- exactly 200 clusters / 400 games and the exact contiguous seed/seat contract;
- every game has a valid terminal, with zero stalls, timeouts, illegal/null
  actions, or loop fallbacks;
- byte-identical replay of the first cluster with fresh bot instances;
- exact frozen search constants and leaf formula covered by focused tests;
- final bonus-versus-ablation movement at least 15%; and
- no human path, hash, replay, checkpoint, or model appears in the configuration.

Run the first 20 reserved clusters as an automatic operational tranche. Use only
its elapsed time, never its outcomes, to project the 200-cluster wall time. Stop
invalid if the projection exceeds three hours. Otherwise run the remaining 180
without exposing or using interim strength. Estimated total is 100--140 minutes
on ten CPU workers. No lower determinization count, smaller panel, or pruned
action set is authorized if the timing gate fails.

## Decisions

- Integrity or operational failure:
  `value_reserve_leaf_bonus_078_invalid_no_causal_decision`.
- Bonus share at least 55% **and** CI95 lower bound strictly above 50%:
  `value_reserve_leaf_bonus_078_causally_stronger_supports_principle`.
- CI95 upper bound strictly below 50%:
  `value_reserve_leaf_bonus_078_causally_harmful_close_principle`.
- Any other valid result:
  `value_reserve_leaf_bonus_078_not_decisive_close_principle`.

Only the first outcome supports the narrow claim that the frozen own-only
high-efficiency reserve option adds strength to full-action one-ply search. It
authorizes only a separate preregistration for a mechanistically different
option-aware architecture. Every non-pass closes this own-only high-efficiency
reserve-value principle as a strength direction. No result promotes either
search player or authorizes BC, tuning, more seeds, or pooling with `077`.

## Evidence

Single artifact:
`outputs/value_reserve_leaf_bonus_causal_ablation_078/result.json`. Record its
SHA-256 in `AGENT_CURRENT.md` after the single frozen run.

## Result (recorded after the frozen run)

The complete pre-evidence suite passed `608/608`. The 20-cluster operational
tranche completed in 621.8 seconds and projected 6,218.1 seconds for the full
panel, below the frozen three-hour ceiling. Its outcomes were not used. All 200
clusters / 400 games then completed normally, the first cluster reproduced
byte-identically, and every integrity gate passed.

The bonus player went **228-172-0**, paired score share **57.0%**, CI95
**[52.0%,62.0%]**. Mean final-score margin was **+1.4775**, CI95
**[+0.855,+2.100]**. Both frozen causal pass conditions held: share was at least
55% and the CI lower bound was strictly above 50%. Formal decision:
`value_reserve_leaf_bonus_078_causally_stronger_supports_principle`.

Dose was high: the bonus action differed from the exact no-bonus shadow on
**66.55%** of 12,136 decisions, with phase doses 68.65% early, 67.54% mid, and
59.87% late. A live option was present on 94.31% of decisions. Qualifying visible
reserves were 8.72% of decisions and converted at 55.29%. The bonus player still
made 720 deck reserves (5.93% of decisions); this is diagnostic and does not
establish whether deck reserves help.

Interpretation is narrow but causal: the frozen own-only high-efficiency reserve
option adds strength to otherwise identical full-action one-ply search. `077`
remains closed and no search player is promoted. The result authorizes only a
separately preregistered, genuinely different option-aware architecture; it does
not authorize BC, parameter tuning, deeper search, a deck mask, more seeds, or
pooling with `077`.

Evidence SHA-256:
`c3acc219261115d15841808fb9b8e0089406603a0c0ffee270e39f1f40fd1f1b`.
