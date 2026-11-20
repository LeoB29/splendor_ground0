# AlphaZero Search-Value Distillation Pilot 128 Result

Date completed: 2026-08-29

Experiment 128 completed normally in 4m54s and stopped at the frozen offline
gate. It generated no self-play and played no arena game. Nothing was installed
or promoted.

## Search-value signal

The sealed 224-game experiment-114 training replay passed every integrity and
signal check. Across 13,960 positions, 1,024-simulation search changed the raw
network value by 0.12008 on average, and 66.05% of positions changed by at
least 0.05. Search-root MSE against the eventual terminal WDL was 0.88572,
slightly better than the raw network's 0.91937. Every stored root value exactly
reproduced the visit-weighted root-Q mean.

This confirms that the serialized search values contain a real, correctly
formed signal. It does not by itself establish that the fixed 50/50 target is
a better learning target.

## Training and sealed holdout

Parent-safe training selected epoch 1. On its internal whole-game validation
split, candidate-minus-generation-11 policy loss was -0.03493, blended-value
loss was -0.01023, and combined objective was -0.04005. The candidate changed
12.07% of top raw actions with mean policy TV 0.09375.

On the sealed 32-game / 1,956-row holdout:

| Model | Policy CE | Terminal MSE | Search-value MSE | 50/50 blended MSE |
|---|---:|---:|---:|---:|
| generation 11 | 1.77663 | 0.91004 | 0.03426 | 0.25375 |
| experiment-114 terminal control | 1.73991 | 0.89992 | 0.03305 | **0.24808** |
| experiment-128 blended candidate | **1.73896** | 0.90806 | **0.03279** | 0.25203 |

The new candidate improved generation 11 on policy CE, terminal MSE,
search-value MSE, and blended MSE. It also very slightly improved the 114
control on policy CE and search-value MSE. However, it was worse than the 114
control on the preregistered causal metric, blended MSE, by 0.00394. The 114
control's better terminal calibration outweighed the new candidate's small
search-value advantage.

That single failed check closed gameplay exactly as frozen. No fresh seed,
arena result, `035` measurement, or post-result coefficient choice exists.

Decision:
`search_value_blend_offline_failed_no_gameplay_no_install`.

The result rejects the exact fixed target
`0.5 * terminal WDL + 0.5 * search_root_value`. It does not show that search
values are useless; they were valid and modestly learnable. Rather, a simple
pointwise average did not improve the overall value target beyond ordinary
terminal training. Do not tune the blend coefficient on this opened holdout or
play the candidate.

Candidate checkpoint SHA-256:
`f30474ed09c01ad699d682b82dd6097a755198f1ee849eeeef731ff9bf3d29ec`.

Raw result SHA-256:
`21e506ea5792543330da90a7cd2880c0e85b1c7d46a721ade1ac57028a5c9170`.
