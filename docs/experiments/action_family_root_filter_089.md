# Experiment 089: Action-Family Expanded-Root Filter

**Preregistered:** 2026-08-09, before computing any family-filter metric from
the stored `085` or `086` root tables.

## Motivation and scope

`088` found no static rank-prefix cap that simultaneously preserved expanded-
root value and reduced roots by at least 30%. Cap 16 used 69.11% of roots but
fell short on gain and regret; cap 24 preserved gain but used 89.58% of roots.
This implies useful roots can occur deep in raw-`035` rank, so rank alone is a
poor pruning signal.

`089` tests one interpretable alternative: always preserve the exact champion
top-four prefix, then add every stable optional root belonging to selected
public action families regardless of rank. This can retain a deep purchase or
visible reservation while discarding large families of token-successor
combinations.

The owner-waived uncapped `087` panel continues unchanged. `089` replays stored
tables only, generates no rollout/game/corpus/model fit, and inspects no `087`
strength outcome. It is a cross-experiment architecture audit, not fresh causal
or promotion evidence.

## Frozen evidence and action-family schema

Inputs are immutable finalized artifacts:

- `085` result SHA-256
  `74d2e1e968d95634e2c73061e0bc479489026c805120de6d2ca2847c3b1f9654`;
- `086` result SHA-256
  `8ae985f6747b5f3bbbcf4997357daf299ef0f46cc23534d55b6de91367910eb5`.

Invalid `085` is discovery only. Valid `086` is locked confirmation. `088`
opened no cap-specific `086` metric, and no family-filter `086` metric has been
inspected before this protocol.

Parse each stored canonical root-key representation with `ast.literal_eval`.
The first tuple item must be one of the engine's frozen action names:
`PASS`, `TAKE_TOKENS`, `RESERVE_VISIBLE`, `RESERVE_DECK`, `BUY_VISIBLE`, or
`BUY_RESERVED`. Define family `BUY` as both buy types. Optional
`RESERVE_DECK` roots were already excluded by `085`/`086`; champion-prefix
roots are always retained regardless of family.

The fixed discovery frontier and order are:

1. `buy_only` = `BUY_VISIBLE | BUY_RESERVED`;
2. `reserve_visible_only` = `RESERVE_VISIBLE`;
3. `take_tokens_only` = `TAKE_TOKENS`;
4. `card_actions` = `BUY_VISIBLE | BUY_RESERVED | RESERVE_VISIBLE`;
5. `buy_or_take` = `BUY_VISIBLE | BUY_RESERVED | TAKE_TOKENS`;
6. `reserve_or_take` = `RESERVE_VISIBLE | TAKE_TOKENS`;
7. `all_public_non_deck` = all three families, an uncapped reproduction
   control.

For each row/filter, retain indices `0--3` plus every optional index whose
action name belongs to the filter. Select the highest discovery value with the
established earliest-original-index tie rule. Evaluation values never affect
selection. Gain and full-frontier regret use the same definitions and tables as
`088`.

## Discovery selection on `085`

Compute all seven filters on `085`. Use 20,000 state-cluster bootstrap samples
with seed `890085 + frontier_index`; phase intervals add
`1000 + phase_index`, with early/mid/late indices `0/1/2`.

A filter is discovery-eligible only if all conditions hold:

1. mean evaluation gain is at least `+0.045` and at least 80% of reproduced
   uncapped `085` mean gain;
2. bootstrap CI95 lower bound is strictly above zero;
3. selection changes the champion action in at least 15% of states and at
   least one state in every phase;
4. mean gain is positive in early, mid, and late phases;
5. at least 60% of states have full-frontier regret at most `0.025`, mean
   regret is at most `0.025`, and p90 regret is at most `0.075`; and
6. mean retained roots divided by mean uncapped roots is at most 0.70.

Select the eligible filter with the lowest mean retained-root ratio. Break an
exact ratio tie by the frozen frontier order above. If none is eligible, do not
open any `086` family-filter metric and decide
`no_action_family_root_filter_selected`.

## Locked confirmation on `086`

Open exactly the selected filter on `086`, using 20,000 state-cluster bootstrap
samples with seed `890086 + selected_frontier_index` and the same phase-seed
derivation. Do not inspect or substitute another confirmation filter.

All confirmation gates are required:

1. mean pooled untouched gain at least `+0.040`, CI95 lower bound above zero,
   and positive mean gain in each of the three untouched evaluation halves;
2. at least 15% champion-action change and at least one change per phase;
3. positive pooled mean gain in every phase and no phase CI95 upper bound at or
   below `-0.05`;
4. full-frontier regret equivalence at least 60%, mean regret at most `0.025`,
   and p90 regret at most `0.075`; and
5. mean retained-root ratio at most 0.70.

Report retained action-family counts, root-count quantiles, root ratio,
theoretical/observed branches (`48*retained_roots`), cap/full discovery-action
agreement, champion-action change, effects, phases, halves, and regret.

## Integrity, decisions, and authorization

Require all `088` input-integrity reproductions plus valid canonical key
parsing, exact champion-prefix retention, no optional family outside each
filter, exact uncapped reproduction by `all_public_non_deck`, finite metrics,
and byte-identical bootstrap repetition.

- Integrity failure: `action_family_filter_089_invalid_no_decision`.
- No eligible discovery filter: `no_action_family_root_filter_selected`.
- Selected filter fails locked confirmation:
  `selected_action_family_filter_not_confirmed_close_family_frontier`.
- Full confirmation pass:
  `action_family_root_filter_confirmed_for_future_fresh_evaluation`.

A pass designates one interpretable pruning rule for future separately
preregistered evaluation. It does not change or stop `087`, promote a bot,
authorize gameplay, train a model, generate a corpus, or permit post-hoc filter
substitution.

## Cost and evidence

Expected runtime is under five minutes with negligible load beside `087`.
Write the result and pre-evidence manifest under
`outputs/action_family_root_filter_089`, with hashes for protocol,
implementation, tests, wrapper, reused `088` replay implementation, both input
results, manifest, and final result.
