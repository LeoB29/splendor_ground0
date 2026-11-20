# AlphaZero Blind-Reserve and Refill-Coverage Audit `136` Result

Date: 2026-09-04

## Decision

`no_broad_refill_identity_coverage_pressure_test_bootstrap_or_targeted_chance`

The audit completed cleanly. The public entity representation retains the
information needed to reconstruct the observer's possible refill-card set, and
the expensive 1,024-simulation search covers important refill identities very
well. This rules out a broad remaining-deck representation defect and a broad
identity-coverage failure at 1,024 simulations.

There is nevertheless a real efficiency gap at the 256-simulation budget used
through most of self-play. This supports one small, controlled no-training test
of stratified refill chance sampling against the existing 1,024-simulation
reference. It does not support changing the architecture or launching training.

The high blind deck-reserve rate is primarily a learned-policy/self-play habit,
not something MCTS newly creates at generation 11. This reinforces the decision
not to let blind reserves determine the main architecture.

## Integrity and scope

- Frozen protocol SHA-256:
  `001692a92e5889b38133e24032601f2d0fbc8ec10fe939895d53deea960e08ce`
- Sources: generation 8, 9, and 10 sanitized 256-simulation corpora plus all
  four generation-11 experiment-114 1,024-simulation shards.
- Reconstructed 1,024 games and 63,882 positions from their seeds and stored
  selected legal actions.
- Every source hash, row count, generation, simulation budget, public state and
  action encoding, legal-action binding, probability, visit, turn-order, and
  terminal-state invariant passed.
- Runtime: 124.09 seconds; focused tests: 5/5 passed.
- No model inference, search, training, self-play, continuation, arena, or
  gameplay ran. No experiment-134 file or outcome was read or changed.

## Remaining-deck representation

All 63,882 reconstructed states passed the independent refill-set check. For
every observer and tier, the simulator's hidden deck plus opponent blind-card
identities exactly equalled the fixed tier deck minus all card identities
exposed in the public entity encoding. Its size also always equalled public
deck count plus encoded opponent blind-reserve count.

There were 63,882 distinct public encoding hashes and no collision mapping one
encoding to different refill sets. This does not prove the network learns to
use the information efficiently. It does show that an explicit remaining-deck
summary would be an optimization, not a repair for missing public information.

## Blind deck-reserves

| generation | selected, all positions | raw network top when legal | visit top when legal |
|---:|---:|---:|---:|
| 8 | 4.685% | 7.935% | 27.271% |
| 9 | 5.763% | 18.027% | 41.695% |
| 10 | 6.344% | 42.607% | 44.488% |
| 11 | 6.855% | 51.953% | 43.003% |

This trajectory is descriptive, not causal: state distributions changed and
generation 11 used 1,024 rather than 256 simulations. Still, the raw learned
policy clearly became much more likely to rank blind reserve first.

At generation 11, blind reserve was legal on 27.029% of positions. Conditional
on legality, raw network prior mass was 22.774%, post-noise prior mass 19.655%,
and visit mass 25.114%. It was selected on 25.360% of those legal states, or
6.855% of all positions. Mean best-visited reserve Q minus best non-reserve Q
was -0.0520; median was -0.0313.

Search increased aggregate reserve mass relative to the noise-adjusted prior by
5.459 points, but reduced its top-choice rate relative to the raw network by
8.950 points and gave it a negative Q comparison. The frozen search-pressure
flag therefore failed. The network/self-play ecosystem learned an overly strong
reserve prior; deeper MCTS partly corrected its ranking while still spending
visits on those prior-favoured alternatives.

Pooling the four corpora, blind reserves were selected on 13.447% of early
positions, 3.216% of middle positions, and 0.260% of late positions. This looks
like an opening habit, not evidence that private information drives the game.

## Refill identity coverage

For each visited `BUY_VISIBLE` or `RESERVE_VISIBLE` edge, the audit computed
expected identity coverage under the exact current root-determinization rule.
A priority edge received at least 5% of root visits.

| generation | simulations | priority edges | mean coverage | below 80% | visit-weighted miss | median visits/identity |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 256 | 32,069 | 79.786% | 41.510% | 6.380% | 2.36 |
| 9 | 256 | 28,586 | 82.859% | 34.979% | 5.118% | 3.15 |
| 10 | 256 | 27,148 | 84.117% | 32.286% | 4.661% | 3.40 |
| 11 | 1,024 | 23,851 | 98.635% | 0.658% | 0.255% | 17.17 |

At generation 11, visit-winning refill edges averaged 99.951% expected
coverage; none was below 80%, and visit-weighted miss probability was 0.008%.
Selected refill edges averaged 98.046% coverage.

The frozen generation-11 gate required at least 25% of priority edges below 80%
and a visit-weighted miss probability above 10%; observed values were 0.658%
and 0.255%. Broad high-budget coverage pressure is therefore not supported.

At 256 simulations, however, roughly one third to two fifths of priority
alternatives had less than 80% expected coverage. The winning action was usually
well covered—generation-10 visit winners averaged 98.175%—but some plausible
competitors were not. Stratification might therefore improve comparisons among
alternatives without paying for four times as many simulations.

## Scientific interpretation and next step

- A remaining-deck representation defect is not supported as information loss.
- Refill identity sampling failure at 1,024 simulations is not supported.
- Refill sampling inefficiency at 256 simulations remains plausible, mainly for
  important competing actions rather than the eventual winner.
- Blind reserve as the primary architectural issue is not supported; the agent
  has learned an opening preference that merits a later targeted test.

The best next evidence is a phase-balanced, no-blind-reserve search test:
compare current 256-simulation root determinization with matched-compute
stratified refill sampling, using 1,024-simulation search as the reference.
Measure action agreement, value/ranking stability, refill coverage, and wall
time. Do not train unless that mechanism test shows a meaningful gain.

## Evidence

- Protocol: `docs/engineering/alphazero_blind_reserve_refill_coverage_audit_136.md`
- Raw result: `outputs/alphazero_blind_reserve_refill_coverage_audit_136/result.json`
- Raw result SHA-256:
  `fe3ffda6bc429d8c5a24e0e0a8666c4c768ed57db9828af5994c18c51ad7c7ea`
- Implementation: `src/splendor_ai/alphazero/blind_reserve_refill_audit.py`
- Runner: `run_alphazero_blind_reserve_refill_coverage_audit_136.py`
- Focused tests: `tests/test_alphazero_blind_reserve_refill_audit_136.py`

No checkpoint was trained, installed, or promoted. No successor was started.
