# AlphaZero Action-Space Fragmentation Audit `135` Result

Date: 2026-09-04

## Decision

`action_fragmentation_not_material_keep_current_encoding`

The audit completed cleanly, but the proposed hierarchical split between a
main strategic intent and payment/return/noble details is not supported as a
broad AlphaZero bottleneck on this corpus.

The current structured legal-action encoding should remain the default. Rare
high-action-count states may eventually benefit from adaptive handling, but a
new hierarchical policy head is not the next priority.

The separate deterministic-afterstate hypothesis remains open. This audit did
not test whether encoding the exact post-action, pre-refill state would make
action consequences and refill evaluation easier to learn.

## Frozen input and integrity

- Protocol:
  `docs/engineering/alphazero_action_space_fragmentation_audit_135.md`
- Protocol SHA-256:
  `0a4f24d673fd1f88033da6c5178b685d01aace8964745baf1746e4333944def7`
- Source: all four sanitized experiment-114 exact-generation-11 self-play
  shards, generated with 1,024 simulations per move.
- Coverage: 256 games, 15,916 positions, and 15,916 unique
  game/turn/player identities.
- All four source hashes, byte counts, and row counts matched the frozen
  contract.
- Every action-aligned vector matched; all priors were finite, nonnegative, and
  normalized; all Q values were finite; every root contained exactly 1,024
  visits.
- The audit read no final score, winner, selected self-play action, terminal
  target, experiment-134 file, or experiment-134 outcome.
- Runtime was 15.60 seconds. It loaded no model and ran no search, training,
  continuation, or gameplay.

## Main result

Across all 15,916 positions:

| metric | result |
|---|---:|
| mean exact legal actions | 17.882 |
| mean core strategic intents | 16.452 |
| mean per-position core/exact ratio | 0.9597 |
| total core/exact count ratio | 0.9200, game-bootstrap CI95 [0.9096, 0.9296] |
| positions with any multi-variant core intent | 7.778% |
| visit mass in multi-variant core intents | 2.888% |
| exact raw-prior/visit-argmax agreement | 63.483% |
| same-core-intent agreement | 64.288% |
| core-intent agreement gain | +0.804 percentage points, CI95 [+0.666, +0.947] |
| aggregate-group agreement gain over exact | +1.068 points, CI95 [+0.857, +1.282] |
| aggregation changed the visit-winning intent | 0.817% |
| mean winning-intent mass on its top exact variant | 98.446% |

Both frozen materiality gates failed by wide margins:

- structural compression required a mean core/exact ratio no greater than
  0.80 and at least 20% of visit mass in multi-variant groups; observed values
  were 0.9597 and 2.888%;
- detail-only policy discrepancy required at least a five-point agreement gain
  with bootstrap lower bound above two points; observed gain was 0.804 points
  with lower bound 0.666.

Thus exact-action disagreement between the raw policy and MCTS is almost never
explained merely by choosing different payment, return, or noble details for
the same main move.

## Phase results

Fragmentation was largest early but remained small:

| phase | exact actions | core intents | multi-variant visit mass | intent agreement gain |
|---|---:|---:|---:|---:|
| early | 20.946 | 19.210 | 4.707% | +1.680 points |
| middle | 14.744 | 13.024 | 2.244% | +0.469 points |
| late | 20.580 | 20.368 | 1.488% | +0.193 points |

The largest exact action set contained 253 actions in an early state, while
the largest core-intent set contained 38. This confirms that severe local
combinatorial outliers exist. However, the median exact counts were only 25,
13, and 18 by phase, and the grouped variants attracted little total visit
mass. A later adaptive search mechanism could target these tail states without
changing the whole policy architecture.

## What created the duplicate intents

Across multi-variant core groups, the varying fields were:

- returned tokens: 3,773 groups;
- token spending/payment: 528 groups;
- noble choice: 212 groups;
- payment and noble jointly: two groups.

Among groups with at least two visited variants, the median within-group Q
range was 0.0279 and 70.52% had range at most 0.05. Many variants were indeed
close in estimated value, but they were too rare and received too little visit
mass to explain the programme-level plateau.

## Interpretation

The current model's core action design is stronger than the raw 183,961-entry
fixed codec suggests. AlphaZero scores only legal structured actions, search
already removes exact canonical-successor aliases, and in ordinary positions
there are not enough remaining implementation variants to create a large
training or search problem.

Do not build or train a hierarchical action head from this result. Preserve
the idea for a future tail-state optimization only if profiling shows that
rare very-wide roots consume disproportionate wall time.

The more promising action-side hypothesis is now deterministic afterstate
encoding, ideally evaluated together with explicit refill chance handling:

`current state + action -> deterministic pre-refill afterstate -> chance refill`

This could simplify the relationship between actions, resulting resources,
bonuses, score, and the tier-specific refill distribution. It requires a new
causal test and is not authorized automatically.

## Evidence

- Raw result:
  `outputs/alphazero_action_space_fragmentation_audit_135/result.json`
- Raw result SHA-256:
  `d5c2e45941a9c12f99c2a9761c65a4eb44e953a42caba294d3bca697d6354c2c`
- Implementation:
  `src/splendor_ai/alphazero/action_fragmentation_audit.py`
- Runner:
  `run_alphazero_action_space_fragmentation_audit_135.py`
- Focused tests:
  `tests/test_alphazero_action_space_fragmentation_audit_135.py`
- Focused result: 5/5 passed.

No checkpoint was trained, installed, or promoted. Experiment 134 remained
active and unchanged.

