# Experiment 141 result: response-rule effect is small; actual-policy gain remains uncertain

Completed 2026-09-08. **Valid execution; primary precision target met.** Frozen
decision: `response_rule_difference_bounded_below_material_size`. Exact 084 remains
research champion and 035 fast champion. No training, strong-player search, arena,
promotion or automatic successor followed.

## Main finding

On this fresh calibration panel, changing the immediate opponent-response rule
does not change the average expanded-086-minus-default-084 move advantage by the
prespecified material amount of half a final score point. The primary confidence
interval lies wholly within that material interval and includes zero. This bounds
the average effect at the tested scale; it does not prove exact equality.

The actual stochastic-035 policy-response arm did **not** establish a positive
move advantage. Its estimate is small and reasonably precise around zero. The
sampled-minimum arm has a small positive secondary estimate. Thus this experiment
calibrates the measurement successfully, but does not produce a reliable positive
actual-policy baseline or a validated improvement oracle.

| Expanded-minus-default move gain | Mean, score points | CI95, score points |
|---|---:|---:|
| Sampled opponent minimum, secondary | +0.193273 | [+0.007508, +0.379558] |
| Actual stochastic-035 response, secondary | +0.074045 | [-0.148048, +0.290625] |
| **Actual-response gain minus sampled-minimum gain, primary** | **-0.119227** | **[-0.283291, +0.042492]** |

These score units are 20 times the frozen normalized-margin endpoint. A separate
read-only endpoint check found **zero clipped returns**, so this conversion also
equals final score-margin units on the observed data. Gains concern a single
forced move followed by weak-policy continuation, not arena strength.

| Frozen normalized-margin endpoint | Mean | CI95 |
|---|---:|---:|
| Sampled-minimum gain | +0.009663628472 | [+0.000375379774, +0.018977918837] |
| Policy-response gain | +0.003702256944 | [-0.007402398003, +0.014531250000] |
| **Policy minus minimum** | **-0.005961371528** | **[-0.014164550781, +0.002124620226]** |

Primary CI half-width is **0.008144586**, beating the fixed .025 target. Both
absolute-gain CIs also exclude a +.025 gain on this panel, although these are
prespecified secondary estimates, not additional independent primary claims.
The small positive sampled-minimum result should not be described as a robust
large benefit: its lower bound is close to zero.

## What this changes in the programme

- Deprioritize the difference between 139's sampled reply minimum and 140's
  actual weak-policy reply as a **large average per-move** explanation at the
  tested half-point scale. Smaller differences could still matter when repeated;
  this does not rule out compounding effects in full games.
- Do not interpret 140's uncertain weak control as demonstrated continuation-
  policy mismatch. This larger, more replicated calibration also fails to
  establish an actual-policy gain, with a much narrower interval and a smaller
  point estimate. An uncertain baseline is not a reason to add strong-player
  compute automatically.
- The earlier 138/139 positive conditional results remain valid. Their intervals
  overlap these smaller estimates; the panels and, for 140, replication differ.
  There is no isolated evidence here that either earlier result was erroneous,
  nor a claim that every expanded move is harmful.
- Preserve 134's decisive rejection of exact expanded 086. This calibration
  provides no basis to promote it or train a successor on its values. Avoid
  simply adding worlds until an absolute gain becomes positive.
- Strong continuation quality, repeated replanning, source-position distribution,
  horizon and utility alignment remain unresolved. This study uses historical
  035 source positions and weak continuation in both arms. It does not identify
  an optimal evaluator, measure live 084/086 decision values, or prove a policy
  improvement that will compose across turns.

The sampled minimum remains a finite, noisy minimum over four proposed replies;
it is not an exact best response. The contrast includes the consequences of that
estimator, rather than isolating strategic opponent strength from minimum-selection
noise. Every realized policy reply lay inside the sampled-minimum reply set.

## Execution and verification

- **120 positions from 120 distinct games**, twenty per phase/seat cell, disjoint
  from the 138--140 calibration panels and all their timing probes. Selection
  used identifiers only. There were **78 differing move pairs and 42 coincident
  pairs**; all were retained, with coincident-root computation shared.
- **48 fresh worlds per state**, in sixteen three-world blocks, with four
  continuation RNG replicas. The grouped opponent minimum was resolved separately
  in every block, preserving 139's estimator. Both arms used the same worlds,
  fixed actions, weak continuation law and unchanged rollout routine.
- **190,076 natural continuations**: 152,060 sampled-minimum and 38,016 policy
  response. Zero hard caps, adjudications, loop fallback triggers, excluded states
  or illegal-action failures. One three-reply branch was retained and counted.
  All **eight integrity/dose gates** passed.
- Protected execution ran **01:11:12.80--01:54:36.81 Europe/Paris**, taking
  **2,604.01 seconds (43m24s)**. Ten workers used one Torch thread each. Summed
  arm CPU time was **6.3420 hours**; summed arm worker wall time was **7.1526
  hours**, exceeding the 6.35-hour preflight serial-wall projection. Runtime was
  telemetry after launch, and no sampling or stopping decision changed. All
  workers and parent exited successfully; both run and verification stderr empty.
- Full suite passed **954/954 tests in 249.12 seconds** before preflight. Focused
  checks covered 59 tests across 138--141, including fifteen new tests. All 1,920
  block structures passed before launch. Six off-panel policy trajectories matched
  140 from the same RNG seeds, and the repeated late-seat0 probe was exact.
- The independent verifier reconstructed all **190,076 terminal returns**, every
  leaf average, shared-information-set minimum, policy average, state gain, all
  **18 reported mean/CI numbers**, classifications and variance estimates to
  **1e-12**. It independently checked selection, original move bindings, prepared
  structures, seeds and hashes. It made no policy calls. This is endpoint and
  statistical reconstruction, **not full trajectory replay or independent rules
  validation**.
- All **267 dependency files frozen by 140 remain byte-identical**. All 268
  dependencies frozen by 141, its sources, checkpoint, protocol, manifest and
  preflight passed hash checks. The intentionally dirty worktree was preserved.

No partial gains or winners were inspected before all 120 states completed.
The outcome-blind interpretation note documented a possible overlap in descriptive
labels without changing any frozen calculation. This result includes zero, so
that label overlap does not arise here.

## Secondary WDL and precision evidence

WDL is +1 win, 0 draw, -1 loss, using the actual card-count tiebreak. Sampled-minimum
WDL follows the margin-minimizing reply. These conditional WDL gains are not arena
win-rate changes.

| Secondary WDL endpoint | Mean | CI95 |
|---|---:|---:|
| Sampled-minimum gain | +0.013802083 | [-0.011024306, +0.038108724] |
| Policy-response gain | +0.000868056 | [-0.033333333, +0.033029514] |
| Policy minus minimum | -0.012934028 | [-0.043619792, +0.016449653] |

All three WDL intervals span zero. They supply no additional positive gameplay
evidence or confirmed utility reversal.

The prespecified moment estimates below are for future design, not new mechanism
claims. Observed state variance averages the six within-cell variances. Estimated
between-state variance subtracts within-block variance divided by sixteen.

| Margin endpoint | Within-block variance | Observed state variance | Between-state moment estimate |
|---|---:|---:|---:|
| Sampled-minimum gain | 0.008260279 | 0.002834905 | 0.002318637 |
| Policy-response gain | 0.012573290 | 0.003953760 | 0.003167929 |
| Policy minus minimum | 0.011948125 | 0.002163015 | 0.001416257 |

Unlike the unstable three-world pilot decomposition in 140, these estimates are
positive. They suggest that differences between positions now account for most
remaining variance, particularly in the actual-policy arm. More worlds on the
same positions therefore offer diminishing precision gains; they cannot establish
a large mean benefit that is absent from the present estimate.

## Provenance

- Protocol: `docs/engineering/opponent_response_calibration_141.md`, SHA256
  `2832213514fe82b1c30b4a9b814e7c51ec0a25db19ea18d77e3b7576c7874033`.
- Preparation: `docs/engineering/opponent_response_calibration_141_preparation.md`.
- Outcome-blind note: `docs/engineering/opponent_response_calibration_141_interpretation_note.md`,
  SHA256 `9097634ecec696b16d12574f95020e0b100e9584fae303e5373767b1376f8b21`.
- Raw result: `outputs/opponent_response_calibration_141/result.json`, SHA256
  **`973d5dc1a068cd64e4888e6c0e6fc13658e669d4ae0bbb39b32333c12afb2567`**.
- Independent verification: `outputs/opponent_response_calibration_141/independent_verification.json`,
  SHA256 **`4e0a11f91ce4b5f274f4c3303e08190ab43c8d163a6426959e8240764e392c80`**.
- Preflight SHA256: `d7ad6c8fe089202ed55c0b7d85da622eee16cee9fdfd03ef87b4e3ae8abda161`.
- Manifest canonical digest: `c227063d255909e9fc6819cbb274362770573064e8315561d7c591e86d353fce`.
- Verifier code SHA256: `3368d2d41c8c8a00fc19ee7674d07161c470a2f89da4e0899fbd9d0a5d6a38b7`.

Implementation: `src/splendor_ai/eval/run_opponent_response_calibration.py`,
`run_opponent_response_calibration_141.py`,
`tests/test_opponent_response_calibration_141.py`, and
`scripts/analysis/verify_opponent_response_calibration_141.py`.
