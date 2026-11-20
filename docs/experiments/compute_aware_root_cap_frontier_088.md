# Experiment 088: Compute-Aware Expanded-Root Cap Frontier

**Preregistered:** 2026-08-09, before computing any cap-specific metric from
the stored `085` or `086` root tables.

## Motivation and scope

The exact uncapped expanded resolver has replicated offline headroom, but its
first live smoke revealed a severe compute tail: 41.03 retained roots per
completed candidate search on average and as many as 221. At 48 terminal
continuations per retained root, this produced 248,160 candidate continuations
in four smoke games. The owner-authorized uncapped `087` strength continuation
is running unchanged and must not be affected by this audit.

`088` asks whether a fixed prefix cap can preserve most of the validated
expanded-root value while sharply limiting search cost and multiple-choice
noise. It replays already stored root action-value tables only. It generates no
new state, rollout, game, label, corpus, or model fit, and it inspects no `087`
strength outcome.

This is a prospectively specified cross-experiment architecture audit, not
fresh causal evidence. Invalid `085` is used only for cap discovery. The
cap-specific views of valid `086` are locked confirmation metrics: none has
been inspected before this protocol. No result can promote a champion or
retroactively alter `085`, `086`, or `087`.

## Frozen evidence and exact replay

Inputs are the immutable finalized artifacts:

- `085` result SHA-256
  `74d2e1e968d95634e2c73061e0bc479489026c805120de6d2ca2847c3b1f9654`;
- `086` result SHA-256
  `8ae985f6747b5f3bbbcf4997357daf299ef0f46cc23534d55b6de91367910eb5`.

Require exactly 300 ordered rows in each input, unique sample indices `0--299`,
four champion roots per row, identical lengths for every root-value table, and
the stored uncapped selections/gains to reproduce exactly before cap analysis.

Candidate total-root caps are fixed at `6, 8, 12, 16, 24, 32`. For cap `K` and
row root count `R`, retain the first `min(K,R)` roots in the original raw-`035`
rank/stability order. This always contains the exact champion prefix. Select
the maximum discovery value with the established earliest-index canonical tie
rule. Evaluation values never affect selection.

For `085`, use `expanded_discovery_values` for selection and
`expanded_evaluation_values` for evaluation. For `086`, use
`expanded_root_values.discovery` for selection, pooled untouched
`expanded_root_values.pooled` for the primary evaluation, and `eval0--eval2`
for independent-half checks. The stored champion discovery-selected index is
the comparison action. Cap gain is evaluation value of the cap-selected action
minus evaluation value of that champion-selected action.

Cap regret is the best evaluation value among **all** stored expanded roots
minus evaluation value of the discovery-selected capped action. Thus it
penalizes both discovery noise and valuable roots excluded by the cap.

## Discovery selection on `085`

Compute every candidate cap on `085`, with 20,000 state-cluster bootstrap
samples per cap and seed `880085 + K`. Phase intervals use that seed plus
`1000 + phase_index`, with early/mid/late indices `0/1/2`. A cap is
discovery-eligible only if all
conditions hold:

1. mean evaluation gain is at least `+0.045` and at least 80% of the reproduced
   uncapped `085` mean gain;
2. bootstrap CI95 lower bound is strictly above zero;
3. cap selection changes the champion action in at least 15% of states and at
   least one state in every phase;
4. mean gain is positive in early, mid, and late phases;
5. at least 60% of states have full-frontier regret at most `0.025`, mean
   regret is at most `0.025`, and p90 regret is at most `0.075`; and
6. mean retained roots divided by mean uncapped roots is at most 0.70.

Select the **smallest** eligible cap. This ordering prioritizes compute and is
frozen before results. If no cap is eligible, do not inspect confirmation cap
metrics; decide `no_compute_aware_cap_selected`.

## Locked confirmation on `086`

Evaluate exactly the one discovery-selected cap on `086`, using a 20,000-
sample state-cluster bootstrap with seed `880086 + K`; phase intervals use the
same `+1000+phase_index` derivation. Do not select or compare another cap after
opening confirmation.

All confirmation gates are required:

1. mean pooled untouched gain is at least `+0.040` and bootstrap CI95 lower
   bound is strictly above zero;
2. each of the three untouched four-continuation evaluation halves has positive
   mean gain;
3. selection changes the champion action in at least 15% of states and at least
   one state in every phase;
4. pooled mean gain is positive in early, mid, and late phases, and no phase
   has CI95 upper bound at or below `-0.05`;
5. at least 60% of states have full-frontier pooled regret at most `0.025`, mean
   regret is at most `0.025`, and p90 regret is at most `0.075`; and
6. mean retained-root ratio versus uncapped `086` is at most 0.70.

Report cap/full discovery-action agreement, cap/champion change, retained-root
quantiles, per-phase effects, three-half effects, full-frontier regret, branch
ratio, and theoretical maximum branches per completed search (`48*K`). Exact
agreement is diagnostic only.

## Integrity and falsification

All stored identities, hashes, row counts, schemas, ordering, finite values,
uncapped summaries, champion indices, phase labels, and original selection
fields must reproduce. Replaying cap `K >= max(R)` must exactly reproduce the
stored uncapped selection and evaluation gain for every row. Bootstrap output
must repeat byte-identically.

- Integrity failure: `compute_aware_cap_088_invalid_no_decision`.
- No discovery-eligible cap: `no_compute_aware_cap_selected`.
- Selected cap fails locked `086` confirmation:
  `selected_root_cap_not_confirmed_close_static_cap_frontier`.
- Selected cap passes every confirmation gate:
  `compute_aware_root_cap_confirmed_for_future_fresh_evaluation`.

A pass designates one capped architecture for a future separately
preregistered evaluation. It does not modify or stop `087`, promote a bot,
authorize training or corpus generation, or permit a post-hoc alternative cap.

## Cost

The audit reads roughly 15 MB of existing JSON and performs scalar table replay
plus deterministic bootstraps. Expected runtime is under five minutes and its
CPU load is negligible relative to the ten-worker `087` panel.

## Evidence

Write the final result and pre-evidence manifest under
`outputs/compute_aware_root_cap_frontier_088`. Record protocol,
implementation, tests, wrapper, both input results, manifest, and final-result
SHA-256 hashes.
