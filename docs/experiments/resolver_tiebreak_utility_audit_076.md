# Resolver Tiebreak-Utility Audit (`076`)

Frozen on 2026-08-04 before implementing a `076` runner, evaluating a selected
`076` state, or opening any `076` result. This is a bounded CPU-only correctness
audit on existing non-human `052` states. It generates no games, trains no model,
touches no human replay, changes no production resolver behavior, and authorizes
no promotion by itself.

## Question and motivating observation

The frozen public-belief resolver evaluates terminal continuations with normalized
prestige-score margin only. The engine, correctly, breaks equal-prestige games by
fewest purchased development cards. Therefore a 15--15 tiebreak loss and a true
draw both receive rollout value zero. The owner independently noticed a matching
late move in held-out seed `800007`: the resolver bought a free zero-point ruby
card instead of a free one-point tier-2 card and later lost 15--15 on cards. That
human state motivated this audit but is forbidden as an audit input.

A second, narrower defect was exposed at the same state. Root values separated by
only `1.39e-17` defeated the resolver's intended earlier-root preference on a
tie. A post-hoc census of stored `052` root values found 41 such switches at gap
at most `1e-12` among 6,625 resolved decisions (`0.619%`). `076` measures both
corrections separately and together on outcome-blind non-human states.

## Frozen sources and identity

Use only completed-search (`target_available=true`) rows from:

- `outputs/resolver_selfplay_distillation_052/train_targets.jsonl`, SHA-256
  `6827848ca5773799a698c62712ee04f3c40b2e2a33fd1ec3c9b5e3828d3af7e5`;
- `validation_targets.jsonl`, SHA-256
  `865926c3e3221d8840bcbb1cdac43913bbd6efcd48b65637efca68e315a2d052`;
- `confirmation_targets.jsonl`, SHA-256
  `8625d9049957ce266fa544896f0262cb37ccea6d168ed0c60a847ef5325cce8f`.

Require frozen `052/result.json` SHA-256
`60458c1bea4328b64f33ad8238529d074a25d1034fa2845e0b339deee3827153`,
all three collection-integrity blocks to pass, and frozen `035` checkpoint
SHA-256 `4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`.
The fixed completed-search counts are 4,397/1,135/1,093 rows from 80/20/20
unique games. `outputs/human_games/` is forbidden.

## Outcome-blind state selection

Pool the 120 unique game seeds and sort them ascending. Assign the game at
zero-based sorted position `i` to phase `(early, mid, late)[i % 3]`, giving
exactly 40 games per phase. Within each game, retain completed-search rows in its
assigned phase, sort by `(decision_index, turn_index)`, and select row
`floor(count / 2)`. Selection may inspect only source split, game seed, phase,
decision index, turn index, `target_available`, and state snapshot. It must not
inspect game outcome, action identity, stored action values, score margin, or
whether the resolver switched. Every game must supply an eligible row; otherwise
the audit is invalid and stops.

This yields exactly 120 states, one per unique game and 40 in each phase. It is a
fixed broad sample, not uncertainty enrichment or a search for known errors.

## Frozen continuation contract

For each selected row, reconstruct the exact `052` resolver decision using the
unchanged `ResolverGameplayConfig()` contract:

- frozen `035` top four distinct canonical root actions;
- three deterministic public-belief determinizations;
- frozen `035` top four opponent actions per information set;
- four stochastic continuations per opponent-response leaf;
- epsilon 0.1, temperature 0.5, top-k 3;
- score-margin scale 20, 400-turn limit, loop-safe fallback limit 64;
- the existing `gameplay_belief_root_index` and
  `gameplay_continuation_seed` namespaces using the stored game seed, actor seat,
  and decision index.

There are exactly `4 * 3 * 4 * 4 = 192` branches per state and 23,040 branches
overall. Run each stochastic continuation once and derive both utilities from the
same terminal state, so current and corrected arms are exactly branch-matched.

The original arm is `clamp(score_margin / 20, -1, 1)`. The tiebreak-aware arm is
identical when prestige margin is nonzero. When prestige margin is zero, assign
`+0.025` if the engine winner is the root player, `-0.025` if the opponent wins,
and `0` for a true draw. The magnitude is frozen at half of a one-prestige margin
step (`0.05`), preserving the ordering loss < draw < win without equating a
tiebreak win to a one-point win.

The numerical-tolerance arm treats minimizer or maximizer values within `1e-12`
of the best value as tied. Among tied opponent responses use the existing
lexicographic semantic-action rule; among tied roots use the existing earlier
raw-`035` root order. No other search, rollout, candidate, or utility parameter
changes.

## Comparisons and endpoints

Resolve four matched variants from every leaf table:

1. frozen original utility and exact floating comparison;
2. original utility with `1e-12` tolerant comparison only;
3. tiebreak-aware utility with exact floating comparison only;
4. tiebreak-aware utility plus `1e-12` tolerant comparison.

The primary descriptive endpoint is variant 4 versus variant 1 selected-root
change fraction over the 120 states, with an exact Clopper--Pearson 95% interval.
Report variants 2 and 3 separately to attribute changes. Also report changes by
phase, action-type transition, change relative to raw `035`, and whether the
corrected action has more immediate prestige or fewer purchased cards after the
root action. These are diagnostics with no strength claim.

Report branch-level equal-prestige frequency, equal-prestige root wins, opponent
wins, true draws, and hence the fraction whose utility changes, overall and by
selected-state phase. Because branches within a state are correlated, also report
the distribution and mean of per-state and per-game (identical here) branch
fractions; do not use 23,040 as an independent-state sample size.

## Integrity gates

Interpret the audit only if all are true:

- exact source/checkpoint hashes, completed-row counts, game counts, and passing
  `052` collection integrity;
- exactly 120 unique selected games/states and 40 states per phase, with selection
  depending only on the declared fields;
- every reconstructed target and frozen root action is legal and canonical;
- original root actions, action values (absolute tolerance `1e-12`), and selected
  root reproduce the stored `052` row at every selected state;
- exactly 23,040 continuations, all resolved, zero hard caps, at most 0.1% loop
  adjudications, and finite values;
- both utilities come from each same branch terminal state and all nonzero-margin
  branch values are byte-identical between arms;
- serial repeat of the first selected state is byte-identical for branch records
  and all four resolutions;
- no human-data path is configured or opened.

An integrity failure yields no correction-dose conclusion. Diagnose only; do not
change the contract or substitute states after seeing results.

## Interpretation and stopping rule

This audit measures correctness dose, not gameplay strength.

- Primary change fraction below 5%: classify the combined correction as an
  isolated low-dose correctness repair. It may be implemented separately with
  unit contracts, but is not a plausible programme-level strength breakthrough
  and does not authorize gameplay.
- From 5% inclusive to below 15%: classify it as a nontrivial but sub-threshold
  correction. Any gameplay claim needs a separately preregistered panel powered
  for its declared expected effect under the standing low-dose rule.
- At least 15%: classify it as a material resolver intervention. Require a
  separately preregistered fresh paired gameplay experiment of at least 200 seeds
  before using the corrected resolver as a teacher or strength claim.

Regardless of dose, do not train on `076`, reinterpret `074` or `075`, replay the
held-out human state as evidence, or promote anything. Stop after this one sample
and the four fixed resolutions; no tolerance, bonus magnitude, state-selection,
or resolver-budget tuning is permitted.

## Expected cost

The audit evaluates 23,040 existing-state branches, about 2.7% of the original
`052` continuation count, with ten persistent CPU workers. Expected wall time is
well below one hour after implementation. CPU remains appropriate because engine
simulation, not neural fitting, dominates.

## Result (2026-08-04)

The complete pre-evidence suite passed 597/597. The single frozen run completed
all 120 selected states, 23,040 primary branches, and the 192-branch serial repeat
in 563.4 seconds wall. All source, selection, canonical-reproduction, finiteness,
branch-accounting, resolution, hard-cap, loop, same-branch, serial/parallel, and
privacy integrity gates passed. All 23,040 primary continuations terminated and
resolved naturally, with zero hard caps, loop adjudications, or fallback triggers.

The combined correction changed 1/120 selected roots (`0.833%`), exact
Clopper--Pearson CI95 `[0.021%,4.556%]`. The entire interval is below the frozen
5% boundary, so the formal classification is
`isolated_low_dose_correctness_repair`.

Attribution is narrower still:

- tiebreak-aware utility alone changed 0/120 selected roots;
- `1e-12` numerical tolerance alone changed 1/120;
- the combined correction changed that same one state, seed `610148`, early
  turn 9 / resolver decision 4;
- the change was `BUY_VISIBLE->BUY_VISIBLE` for the same card `T1_37`, differing
  only in which coloured/gold tokens paid for it. Immediate prestige and purchased
  card count were identical.

Equal-prestige terminal branches were not vanishingly rare: 683/23,040 (`2.964%`).
They comprised 294 root-player tiebreak wins, 266 opponent tiebreak wins, and 123
true draws. Therefore 560 branches (`2.431%`) received a nonzero corrected value.
Ninety of 120 states contained at least one such branch and 42 root-value tables
changed numerically, but none changed the exact selected root. Mean per-state
affected-branch share was 2.431%, median 1.5625%, maximum 18.229%.

This validates both correctness defects while falsifying their importance as the
current strength bottleneck. A separate maintenance change may make terminal
utility rule-correct and comparisons numerically stable, but it is not a
programme-level intervention and does not authorize gameplay, teacher reuse,
training, or promotion.

Evidence: `outputs/resolver_tiebreak_utility_audit_076/result.json`, SHA-256
`6b01ce4b47c2d0bfd1d3f702111b8436fd3e9a4c9ad232e486ef86a9e07724c0`.
