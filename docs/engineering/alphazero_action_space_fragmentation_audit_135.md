# AlphaZero Action-Space Fragmentation Audit `135`

Frozen before implementation output or aggregate inspection on 2026-09-04.

## Purpose

Test whether the current AlphaZero policy/search target is unnecessarily split
across concrete payment, token-return, and noble-selection variants of the same
main strategic choice.

This is a lightweight, no-training audit over existing replay. It starts no
self-play, search, continuation, arena, or gameplay and does not inspect
experiment `134` outcomes. It may run while `134` is active because it performs
only one streaming pass over four existing JSONL files and uses no Torch model.

The separate afterstate-encoding hypothesis is recorded here but is not tested
causally: a future action encoder may benefit from representing the exact
deterministic result of an action before a random refill, rather than requiring
the network to infer that result from action fields and a pooled state vector.

## Frozen source

Use all four sanitized experiment-`114` self-play shards. They contain 256
complete exact-generation-11 self-play games generated with 1,024 simulations
per move and 15,916 position rows.

| shard | rows | bytes | SHA-256 |
|---:|---:|---:|---|
| 0 | 3,956 | 74,919,121 | `f22b29a3ce5dbfc2e18331dc10a9f30928fe8ee6cf93d65ad48b524a9835af8b` |
| 1 | 4,004 | 76,570,863 | `c3d333830f3d9e92557bcd762b5605580aa0af0b0f3b0d5fcddaaab77cde3e60` |
| 2 | 3,956 | 76,478,700 | `00e816d7c11c56e1dfc55a4c82198ffd6fceff7e57dfdf780239092b93ca00d8` |
| 3 | 4,000 | 76,509,765 | `7bd6b03227aa6e4e5ce1bb902adda6021d438b5e927d1da2b9f56793424dd95e` |

The audit may read only these replay fields:

- `game_seed`, `turn_index`, and `player_id`;
- `action_payloads`, `action_card_ids`, and `action_noble_ids`;
- `root_network_priors`, `root_visits`, `root_q_values`;
- `raw_policy_top_row` and `visit_argmax_row`.

Final scores, winner, selected self-play action, terminal value target, and game
outcome are not inputs.

## Three action levels

### Exact action

The stored canonical legal action, including every payment, return, and noble
choice. Exact duplicate successor aliases were already removed by search.

### Intent with noble preserved

Ignore token spending and token returns, but retain any chosen noble:

- `TAKE_TOKENS`: action type plus sorted tokens taken;
- `RESERVE_VISIBLE`: action type plus target card id;
- `RESERVE_DECK`: action type plus tier;
- `BUY_VISIBLE` / `BUY_RESERVED`: action type, target card id, and noble id;
- `PASS`: action type.

### Core strategic intent

Use the same grouping but also ignore noble id. Noble selection remains a legal
and potentially meaningful second-stage decision; this grouping asks whether
the primary card or token choice is stable before that detail is selected.

Never merge `BUY_VISIBLE` with `BUY_RESERVED`, because only the former causes a
market refill. Never merge different target cards, reserve tiers, or token sets.

## Frozen metrics

Report overall and for early (`0--19`), middle (`20--49`), and late (`50+`):

1. position and game counts;
2. mean, median, p90, and maximum exact-action count;
3. the same statistics for noble-preserving and core-intent counts;
4. mean group/exact compression ratio and fraction of positions containing at
   least one multi-variant group;
5. fraction of total visit mass assigned to multi-variant groups;
6. raw-prior versus visit-argmax agreement at the exact-action level;
7. agreement after asking only whether those two exact argmax actions share the
   same noble-preserving intent or core intent;
8. agreement between the largest aggregate raw-prior group and largest
   aggregate visit group at both grouping levels;
9. fraction of positions where aggregating visits changes the winning intent
   relative to the intent containing the single most-visited exact action;
10. concentration of visits on the most-visited exact variant inside the
    winning aggregate intent;
11. within-group Q range for groups with at least two visited exact variants;
12. counts of multi-variant groups in which return tokens, spend tokens, noble
    choice, or more than one of these fields varies;
13. core-intent visit-argmax family mix.

All distributions must be finite, nonnegative, aligned with the action list,
and have positive total mass. Root visits must total 1,024 on every row.

Use 5,000 deterministic game-cluster bootstrap resamples with seed `1,350,135`
for:

- noble-preserving agreement minus exact agreement;
- core-intent agreement minus exact agreement;
- aggregate core-intent agreement minus exact agreement;
- core-intent count divided by exact-action count.

## Interpretation fixed before results

Define:

- **material structural compression:** overall mean core-intent/exact-action
  ratio at most `0.80` and at least `20%` of visit mass lies in multi-variant
  core-intent groups;
- **material detail-only policy discrepancy:** core-intent agreement minus
  exact agreement is at least `5` percentage points and its game-bootstrap CI95
  lower bound exceeds `2` percentage points.

Decisions:

1. Both conditions hold:
   `material_action_fragmentation_supports_hierarchical_design_audit`.
2. Structural compression holds but detail-only discrepancy does not:
   `structural_action_fragmentation_without_policy_rescue_review_afterstate`.
3. Structural compression does not hold:
   `action_fragmentation_not_material_keep_current_encoding`.
4. Any source, schema, probability, visit, or accounting invariant fails:
   `action_fragmentation_audit_invalid_no_architecture_conclusion`.

Even a positive result authorizes only a separately designed representation
pilot. It does not authorize training, gameplay, checkpoint installation,
promotion, a change to experiment `134`, or automatic continuation.

## Durable architecture hypotheses

The two ideas to preserve are:

1. **Hierarchical action policy:** first choose the main card/token intent, then
   choose payment, returned tokens, and noble conditionally. Full legal actions
   remain available; the hierarchy organizes rather than prunes them.
2. **Deterministic afterstate encoding:** encode the legal public state after
   applying the chosen action but before the refill chance event, then evaluate
   possible refills from that afterstate. This may make action consequences and
   stochastic search easier to learn.

Direct action-to-state-entity cross-attention is a related secondary hypothesis.
The current scorer gives each action a pooled state summary and lets actions
self-attend, but does not let each action query individual state entities
directly.

