# Experiment 140 result: continuation-policy crossover remains inconclusive

Completed 2026-09-07. **Valid execution; inconclusive mechanism evidence.** Frozen
decision: `stochastic_control_gain_uncertain`. Exact 084 remains research
champion and 035 fast champion. No training, promotion or arena followed.

## Main result

The estimated expanded-086-minus-original-084 move advantage was smaller when
both players continued with exact 084, but the experiment did not establish
that decrease. The weak control's positive advantage was also uncertain.

Values below are paired state means of expanded-minus-default normalized score
margin, averaged over three fresh worlds per state. Confidence intervals use
50,000 paired source-game bootstrap draws within the six fixed cells.

| Quantity | Mean | CI95 |
|---|---:|---:|
| Stochastic-035 continuation gain | +0.043981481 | [-0.010185185, +0.105092593] |
| Exact-084 continuation gain | +0.021296296 | [-0.007407407, +0.054629630] |
| **084 gain minus stochastic-035 gain** | **-0.022685185** | **[-0.070844907, +0.026851852]** |
| Greedy-035 gain, secondary | +0.043981481 | [-0.012037037, +0.107407407] |
| 084 gain minus greedy gain, secondary | -0.022685185 | [-0.075000000, +0.029629630] |
| Greedy gain minus stochastic gain, secondary | 0.000000000 | [-0.015740741, +0.014814815] |

The control CI crosses zero, triggering the frozen decision before a mechanism
claim can pass. The change interval contains both the material-attenuation
threshold (-0.025) and zero; its point estimate also falls short of that
threshold. The strong-play gain interval crosses zero. Neither attenuation nor
a reliably surviving positive advantage is established. This is not evidence
of equivalence or absence of an effect.

In final score-margin units, point gains were **+0.87963 points with stochastic
035** and **+0.42593 with 084**. Their paired change was -0.45370 points, CI95
[-1.41690,+0.53704]. The apparent reduction must not be presented as a confirmed
fraction of 134's failure explained. The greedy control had the same mean as
the stochastic control; its difference interval is descriptive and does not
isolate planning as the cause of the primary change.

## Execution and validation

- **36 positions from 36 distinct games**, six per phase/seat cell, disjoint from
  138, 139 and the timing probes. Timing alone selected this nested panel.
  There were 22 differing move pairs and 14 coincident pairs. Coincident choices
  shared computation and remained zero-gain observations.
- **522 outer continuations**, 174 per policy, all naturally completed. Zero
  outer cutoffs, forced-pass deadlocks, illegal-action failures or exclusions.
  All **11 integrity and operator-dose gates** passed.
- Exact 084 made **4,837 subsequent decisions**, completed **4,413 searches**
  (91.23%), and switched away from its raw-035 default **2,793 times** (57.74%
  of decisions; 63.29% of completed searches). The intervention was substantial.
- Its **847,296 inner continuations** all resolved, with zero hard caps,
  adjudications or rollout fallback triggers. Searches by source phase numbered
  2,667 early, 1,436 middle and 310 late.
- Native structural fallbacks numbered **424**: 203 for fewer than four opponent
  actions, 174 for a terminal root action, 41 for fewer than four root actions,
  and six for unstable opponent action sets. These were preserved components
  of exact 084. The 139 reply-set adapter was not applied.
- Execution ran from **20:55:22 to 23:49:33 Europe/Paris**, taking **10,450.29
  seconds (2h54m10s)**. Ten workers each used one Torch thread. Summed trajectory
  CPU time was 22.6388 hours; summed trajectory wall time was 28.9731 hours,
  versus the conservative 34.0233-hour serial projection. Runtime after panel
  selection was telemetry. All workers and parent exited successfully; stderr
  was empty.
- Preparation separately ran 19 off-panel outer paths, including the exact
  repeat, with 27,840 inner continuations. Probe outcomes were discarded and
  contributed no statistical observations. The preparation record contains the
  timing-only selection rule and immutable manifest identities.

The full suite passed **939/939 tests in 257.13 seconds** before launch; initial
focused checks passed 50/50. The independent verifier replayed all **522
trajectories and 15,531 moves** through the engine and reconstructed all **54
reported means and CI endpoints** to 1e-12. It checked worlds, scoring and
card-count tiebreaks, telemetry, schedules and frozen hashes. The classification
was separately checked from the verified estimates. All 266 dependency files
frozen by 139 remained byte-identical. Existing work and checkpoints were preserved.

## Secondary outcomes and limits

WDL uses +1 for a win, 0 for a draw and -1 for a loss from the root actor's
perspective. Stochastic-035 move gain was +0.092593, CI95 [-0.074074,+0.277778];
084 gain was +0.055556, CI95 [-0.018519,+0.148148]. Their paired change was
-0.037037, CI95 [-0.212963,+0.138889]. These secondary conditional returns do
not establish a utility reversal or an arena win rate.

There are only 36 independent source games and three worlds per state. Hundreds
of outer games and 847,296 inner continuations do not increase that population
sample size. The primary interval is too wide to resolve the prespecified
material change. No worlds or states were added after outcomes.

The stochastic control chooses the immediate opponent reply itself, whereas
139 used a sampled, grouped opponent minimum. This is not an exact replication
of 139's estimator. Failure to establish the control gain cannot be attributed
specifically to 084 or treated as a contradiction of 139.

Phase, seat and private-state breakdowns remain descriptive. There were twelve
states per phase, eighteen per seat, and only eight states without existing
blind reserves. Do not select subgroups or new thresholds from these breakdowns.

## Programme interpretation

The exact strong operator was exercised heavily in every source phase, and
termination was clean. A missing intervention or failed termination cannot
explain this experiment's inconclusive estimate.

140 **does not resolve the offline/gameplay discrepancy**. Continuation-policy
mismatch remains plausible, alongside repeated replanning, horizon effects and
state-distribution shift. It neither rescues expanded 086 after its decisive
134 loss nor establishes that independent-world checking alone is sufficient.
The 138 world-generalization finding and 139's valid partial positive remain.

Before another costly continuation experiment is proposed, its precision and
control-estimator design need explicit review using this completed pilot. A
larger run or changed estimator needs a new prospective protocol. Do not repeat
140 merely to cross a gate, promote a bot, or train on these three-world move
values automatically.

## Provenance

- Protocol: `docs/engineering/continuation_policy_crossover_140.md`, SHA256
  `c71dd40e4756d40c787befd4ede98488d14e26d65ebc1b0820720068eeac5b8b`.
- Preparation: `docs/engineering/continuation_policy_crossover_140_preparation.md`.
- Raw result: `outputs/continuation_policy_crossover_140/result.json`, SHA256
  `cfc9c99e2db2b000a04e5003078ce10cb83dbc0ac1c8ebd1248b980736168df5`.
- Verification: `outputs/continuation_policy_crossover_140/independent_verification.json`,
  SHA256 `c712d577ce169ca785d32b65db7895f2bb32ddcccf53c4782edcf10150287b31`.
- Frozen verifier SHA256:
  `9e5ad51c8dc1a7e4b386e5942e5896cb6cc46416bfb5e0738e61f6558041ef62`.
- Manifest canonical digest:
  `b725ab75391ddf7f5b4df4cb4d68cfce63c6589544f04d05426bd0ad918c958e`.
- Preflight SHA256:
  `aa5529589e6b3fea26676090646a38d690f4ee7dcf45b5d997f23f7ce25792a4`.

Code: `src/splendor_ai/eval/run_continuation_policy_crossover.py`, wrapper
`run_continuation_policy_crossover_140.py`, tests
`tests/test_continuation_policy_crossover_140.py`, and independent verifier
`scripts/analysis/verify_continuation_policy_crossover_140.py`.
