# MCCFR Average-Policy Repair Result (`046`)

Experiment `046` passed all ten frozen gates. The corrected standard
two-player external-sampling average policy is deterministic, robust across five
sampling seeds, essentially unexploitable on the reduced Splendor reference,
and no longer dependent on rare or missing information-set substitutions.

## Integrity

- Protocol checkpoint: `e20f249`.
- Implementation checkpoint: `368df81`.
- Pre-evidence verification: 423 tests passed in 158.49 seconds.
- The complete evidence run finished normally in 119.7 seconds.
- The exact reference payload reproduced immutable `044` hash
  `b8184ec7c9dbfa3611d014f98c674efbeae27728429b2fa90c7939e1bbe043ab`.
- Corrected seed `440046` repeated byte-identically at payload hash
  `3a0850d16156869a453204dd9b68bbad234e0a4f0a10683c91afe17c2e7d4df3`.
- Every run had finite metrics, zero illegal actions, zero unstable action sets,
  normalized positive regret far below `0.05`, and coverage from `95.63%` to
  `99.03%` of exact-reference information sets.
- Result JSON SHA-256:
  `bc025be45cbb8afd1c907f4e74b587a3647f2414ac11e5d5aba55b2223622449`.

## Strategic results

Corrected average-policy exploitability by 50,000-iteration seed:

| Seed | Exploitability | P0 incentive | P1 incentive | Coverage |
|---:|---:|---:|---:|---:|
| 440046 | 0.000374336 | 0.000416566 | 0.000332107 | 98.54% |
| 440047 | 0.000383431 | 0.000419315 | 0.000347546 | 99.03% |
| 440048 | 0.000470736 | 0.000618184 | 0.000323288 | 97.57% |
| 440049 | 0.000336231 | 0.000286107 | 0.000386354 | 97.09% |
| 440050 | 0.000401162 | 0.000560111 | 0.000242212 | 95.63% |

The range was `0.000336231–0.000470736` and the median was `0.000383431`,
comfortably inside the frozen per-seed `0.020` and median `0.010` limits. The
exact reference exploitability was `0.000177429`. The largest corrected excess
over exact was only `0.000293307`, versus the allowed `0.015`.

All final regret-matched current policies measured zero exploitability on this
small game, but that non-guaranteed fact was only reported and was not used to
pass the average-policy gates.

## Rare-state and cross-play results

- Filling information sets entirely missing from a corrected solver changed
  exploitability by exactly zero for every seed.
- Replacing every sub-100-observation corrected strategy with the exact
  reference changed exploitability by only `0.00000214–0.00006302`, versus the
  frozen `0.005` maximum.
- The largest exact/corrected cross-play deviation from exact self-play was
  `0.000283000`, versus the `0.020` limit.
- The largest individual best-response deviation incentive was `0.000618184`,
  versus the `0.040` one-sided limit.

These adversarial checks demonstrate that the `045` exploitability of
`0.293314` came from the legacy averaging rule, not sampling variance, missing
coverage, regret failure, or the structure of CFR itself. Updating the sampled
opponent's strategy during the other player's regret pass fixes the defect while
leaving regret tables and current regret-matched policies unchanged.

## Decision

`046` authorizes a separately preregistered full-game neural Deep CFR
feasibility pilot. It does not promote a gameplay checkpoint. The pilot should
begin with information-state/action representation and traversal-memory
contracts, run a bounded CPU collection/shape smoke, then use a rented GPU only
for the neural advantage and average-strategy fits if the projected workload
materially benefits. It must validate held-out regret/strategy targets,
rare-state coverage, legal-action masking, runtime/memory, and an available
exploitability proxy before any gameplay gate.

Preserve the repaired opponent-node strategy-sampling semantics in the neural
strategy memory. Do not use the legacy `044/045` average accumulator. Raw legal
`035` remains champion. BGA/expert replays remain a separate future authorized-
data option and were not used in `046`.
