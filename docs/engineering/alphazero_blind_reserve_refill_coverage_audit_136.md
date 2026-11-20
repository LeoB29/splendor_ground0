# AlphaZero Blind-Reserve and Refill-Coverage Audit `136`

Frozen before implementation output or aggregate inspection on 2026-09-04.

## Purpose

Use existing AlphaZero self-play replay to answer three narrow questions:

1. Does the raw network already prefer blind deck-reserves, or does root search
   systematically promote them?
2. Given the number of visits assigned to visible-card moves, how completely
   can ordinary root determinization sample the possible public refill-card
   identities?
3. On states without unrepresented private information, does the public entity
   encoding retain everything needed to reconstruct the possible refill-card
   set?

This is a replay-only engineering diagnostic. It starts no model inference,
search, self-play, continuation, training, arena, or gameplay. It reads no
experiment-`134` file or outcome. It may run while `134` is active.

The refill calculation is an analytic coverage-pressure estimate, not a direct
measurement of value bias. Under the current search implementation, each root
simulation independently determinizes the observer's unseen card pool; root
action choice occurs before the refill is drawn. If an edge receives `n` visits
and has `N` information-legal possible refill identities, the probability that
a particular identity is never sampled is `(1 - 1/N)^n` and the expected
identity coverage fraction is `1 - (1 - 1/N)^n`.

## Frozen sources

Use the three 256-simulation sanitized generation-8--10 corpora from experiment
`104`, followed by all four 1,024-simulation exact-generation-11 shards from
experiment `114`.

| generation | source | rows | bytes | SHA-256 |
|---:|---|---:|---:|---|
| 8 | `outputs/alphazero_continued_scaling_104/selfplay/generation_008_sanitized.jsonl` | 16,030 | 307,848,775 | `96f0d346743f30bd29e329f4bf3e2a7a361e778391c6d6e5833ca2a06af9be10` |
| 9 | `outputs/alphazero_continued_scaling_104/selfplay/generation_009_sanitized.jsonl` | 16,000 | 306,518,964 | `1b13ef038ccc6b3f695d7c7860a2284272ae18c8e75d44028e4cbb6f69914f0c` |
| 10 | `outputs/alphazero_continued_scaling_104/selfplay/generation_010_sanitized.jsonl` | 15,936 | 307,318,318 | `09279d9593acfd944a32571716a6527c0be40e41055b6fa1f5bc25c88f43a60c` |
| 11 | `outputs/alphazero_high_budget_distillation_114/selfplay/shard_0/sanitized.jsonl` | 3,956 | 74,919,121 | `f22b29a3ce5dbfc2e18331dc10a9f30928fe8ee6cf93d65ad48b524a9835af8b` |
| 11 | `outputs/alphazero_high_budget_distillation_114/selfplay/shard_1/sanitized.jsonl` | 4,004 | 76,570,863 | `c3d333830f3d9e92557bcd762b5605580aa0af0b0f3b0d5fcddaaab77cde3e60` |
| 11 | `outputs/alphazero_high_budget_distillation_114/selfplay/shard_2/sanitized.jsonl` | 3,956 | 76,478,700 | `00e816d7c11c56e1dfc55a4c82198ffd6fceff7e57dfdf780239092b93ca00d8` |
| 11 | `outputs/alphazero_high_budget_distillation_114/selfplay/shard_3/sanitized.jsonl` | 4,000 | 76,509,765 | `7bd6b03227aa6e4e5ce1bb902adda6021d438b5e927d1da2b9f56793424dd95e` |

The audit may read only replay identity, public state encoding, legal-action,
root-statistic, selected-action, checkpoint-generation, and search-configuration
fields. It must not read winner, final score, terminal value target, or other
game-outcome fields.

## Reconstruction and integrity

Reconstruct every source game from `game_seed` with the exact simulator and the
stored selected legal action. At every row require:

- turn, player, generation, checkpoint generation, and root simulation count
  match the source contract;
- stored public entity encoding exactly matches a fresh encoding of the
  reconstructed state;
- every stored action payload matches exactly one legal canonical action and
  every action-aligned vector has the same nonzero length;
- network priors, post-noise search priors, visit probabilities, and Q values
  are finite; both prior vectors normalize and visits total the contracted
  simulation count;
- each game proceeds in exact turn order and ends in a simulator terminal
  state after its final replay row.

For every observer and tier, reconstruct the information-legal possible refill
set in two independent ways:

1. actual hidden tier-deck identities plus the identities of opponent blind
   reserves of that tier;
2. the fixed base tier deck minus every card identity exposed by the public
   entity encoding (board, either player's purchases, own reserves, and
   opponent visible reserves).

Require exact set equality. Also require that its size equals public deck count
plus the encoded number of opponent blind reserves in the tier. Group identical
public encodings across the corpus and require that no group maps to different
possible-refill sets.

## Blind-reserve metrics

Report by generation and overall phase (early `0--19`, middle `20--49`, late
`50+`):

- blind deck-reserve legality and selected-action rates;
- blind-reserve mass under raw network priors, post-Dirichlet search priors,
  and final root visits;
- raw-network, post-noise-prior, visit-argmax, and selected-action top-family
  rates;
- raw-to-search-prior and search-prior-to-visit mass shifts;
- promotion/demotion counts from raw top to visit top and from post-noise top to
  visit top;
- best visited reserve Q minus best visited non-reserve Q when both exist.

The generation sequence is descriptive only: checkpoint identity and the jump
from 256 to 1,024 simulations are confounded. It must not be called a causal
learning curve.

Define a possible search-driven blind-reserve pressure flag on generation 11
only when blind reserve is legal and both of these hold:

- mean visit mass minus post-noise search-prior mass is at least `+0.02`;
- visit-top blind-reserve rate exceeds raw-network-top rate by at least `0.02`.

The flag is diagnostic only. It does not establish that blind reserve is bad.

## Refill-coverage metrics

A refill edge is an exact `BUY_VISIBLE` or `RESERVE_VISIBLE` legal action whose
tier deck count is positive. A priority refill edge receives at least 5% of all
root visits. For every visited refill edge report:

- root visits, possible refill identities, visits per possible identity;
- expected distinct identities and expected identity coverage fraction;
- probability that any particular possible identity is missed.

Summarize these overall, by generation, phase, tier, and for:

- all visited refill edges;
- priority refill edges;
- visit-winning refill edges;
- selected refill edges.

Report both unweighted means and visit-weighted means where meaningful. Also
report the fraction of priority edges with expected coverage below 80% and 90%.

Define material refill identity-coverage pressure on generation 11 when both:

- at least 25% of priority refill edges have expected coverage below 80%;
- the visit-weighted mean probability of missing a particular identity across
  priority refill edges exceeds 10%.

This threshold deliberately asks whether even the expensive 1,024-simulation
search leaves a broad identity-level hole. Passing it supports a controlled
explicit/stratified chance-search experiment, but does not prove that such a
search will play better.

## Frozen decisions

1. Any reconstruction, source, encoding, schema, or accounting invariant fails:
   `audit_invalid_no_architecture_conclusion`.
2. Encoding cannot reconstruct the information-legal refill set:
   `public_refill_representation_defect_fix_before_search_pilot`.
3. Representation passes and material refill coverage pressure is detected:
   `refill_coverage_pressure_supports_stratified_chance_search_pilot`.
4. Representation passes without material coverage pressure:
   `no_broad_refill_identity_coverage_pressure_test_bootstrap_or_targeted_chance`.

The blind-reserve pressure flag is reported alongside the main decision. It can
motivate a later targeted opponent test, but cannot make private-information
machinery the primary architecture by itself.

No outcome authorizes training, gameplay, checkpoint installation, promotion,
a modification to experiment `134`, or automatic continuation.
