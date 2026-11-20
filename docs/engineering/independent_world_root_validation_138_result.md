# Experiment 138 result: material attenuation under independent worlds

Completed 2026-09-06. Decision: `material_world_attenuation_supported`.

All 60 states / 60 distinct source games completed, ten per phase/seat cell.
All 39,552 continuations terminated normally: 19,776 in each condition, with
zero hard caps, loop adjudications, fallback triggers, or structural exclusions.
All integrity gates passed. Protected evaluation took 763.30 seconds (12m43s)
on ten CPU workers with one Torch thread each. The 43 differing fixed choices
were retained alongside 17 coincident choices; identical actions shared their
evaluation, explaining the reduction from the 46,080 maximum budget.

The original 084-contract and expanded 086 discovery actions were frozen.
Each condition used four separately resolved three-world/four-continuation
blocks. Only the fresh condition resampled complete observer-legal worlds;
rollout randomness was new in both conditions and paired across alternatives.
This preserves the same evaluation operator in both conditions. The control
does not reproduce the old pooled-twelve-continuation headline estimator.

| Primary measurement, normalized final-score margin | Mean | Paired CI95 |
|---|---:|---:|
| Fixed 086-minus-084 gain, original worlds | +0.056146 | [+0.038021, +0.075503] |
| Fixed 086-minus-084 gain, fresh worlds | +0.016736 | [+0.001424, +0.031892] |
| Fresh-minus-original change in gain | -0.039410 | [-0.061719, -0.019149] |

The conditional advantage reproduced on the original worlds. Approximately
**70.2% of that point-estimated advantage disappeared on fresh worlds**. The
paired attenuation interval excludes zero, and the point estimate exceeds the
preregistered 0.025 materiality threshold. The frozen decision rule therefore
passes. The interval does not establish that attenuation exceeds 0.025 at every
value inside CI95; the 70.2% ratio is descriptive and has no separately reported
confidence interval.

The fresh-world gain remains positive, with its interval just above zero.
Consequently, the result supports the missing-independent-world-validation
hypothesis without claiming that world sampling explains all of 134's large
gameplay loss or that expanded 086 has become a useful teacher.

## Scientific interpretation

Experiment 086's held-out continuation-action randomness overstated how much
of its selected-action advantage generalized to independent worlds. Its high
conditional correlation and positive offline payoff gain were insufficient
evidence of reliable policy improvement. The new test supports selection
overfitting to a small sampled-world set as a material part of this failure.

It does not validate optimal action values, a deployable broader-root policy,
or a new architectural choice. Both conditions still use the frozen stochastic
035 continuation policy, a shallow grouped opponent response, and score-margin
utility. A small positive conditional gain can coexist with 086's 31.875%
score share against 084 in experiment 134. Continuation-policy mismatch,
repeated replanning, limited horizon, and the changed gameplay-state
distribution remain open explanations for that remaining discrepancy.

The practical programme consequence is to require independent-world evaluation
of any proposed improvement signal before trusting it as a teacher or accepting
conditional value gains. Do not distill 086, promote it, or reopen an old
world-budget/PUCT tuning recipe from this result. Retain exact 084 as research
champion and 035 as fast champion. A next experiment needs separate design and
owner authorization; nothing launches automatically.

## Prespecified descriptive diagnostics

| Phase | States | Original-world gain | Fresh-world gain | Attenuation |
|---|---:|---:|---:|---:|
| Early | 20 | +0.054115 | +0.001667 | -0.052448 |
| Middle | 20 | +0.062917 | +0.012031 | -0.050885 |
| Late | 20 | +0.051406 | +0.036510 | -0.014896 |

The early/middle effect is larger descriptively. Seat-0 and seat-1 changes are
-0.037431 and -0.041389. These breakdowns are not separately powered gates and
do not authorize phase-specific tuning.

Under the margin-minimizing reply, mean WDL advantage in [-1,+1] fell from
+0.191667 to +0.031944. With WDL-minimizing replies it fell from +0.195139 to
+0.030903. These secondary point estimates show similar attenuation under
win/loss scoring; no secondary confidence claim is made. They are not arena
win rates or unbiased optimal opponent values. Margin and WDL selected
different minimizing replies in 797/1,950 sampled information-set choices.

The expanded action was TAKE_TOKENS on 47/60 states versus 19/60 for the
champion action; the remaining expanded choices were seven visible buys and
six visible reserves. This describes the selected population, not evidence
that token-taking itself is defective.

Only 11 states had no existing blind reserves. Their original/fresh gains were
+0.071307/+0.048864, a descriptive change of -0.022443. The other 49 states
contained one or more blind reserves. The experiment resamples both refill
uncertainty and opponent-private-card possibilities and does not isolate their
contributions. Do not use this result to reinstate a belief/CFR-first
architecture or infer a universal failure on ordinary public stochastic states.

## Verification and provenance

The focused tests passed 29/29 and the full repository suite passed 907/907
before protected evaluation. The frozen preflight verified all prepared world
hashes and exact original branch reproduction. The runner reverified frozen
source/checkpoint/dependency identities before final reporting.

A separate read-only calculation then revalidated every one of the 39,552
stored returns from final scores and purchased-card counts. It rebuilt grouped
margin minima directly from the leaf records without calling the experiment's
resolution or summary helpers. Maximum value discrepancy was 5.55e-17. It
independently reproduced all three primary means and every endpoint of the
50,000-sample paired bootstrap intervals to within 1e-12. Bootstrap units were
distinct source games resampled within the six fixed cells; blocks and
individual rollouts were not treated as independent population observations.

- Protocol: `docs/engineering/independent_world_root_validation_138.md`.
- Preparation: `docs/engineering/independent_world_root_validation_138_preparation.md`.
- Raw result: `outputs/independent_world_root_validation_138/result.json`.
- Raw per-state returns: `outputs/independent_world_root_validation_138/states/`.
- Raw result SHA256:
  `9b1e1ec893c2dcbb3284530e8166a262d0074e52ea17344cf049de9c8e87d020`.

No model was trained or promoted; no new source games, full-game arena,
installation, sealed 087 outcome use, or automatic successor occurred. All
pre-existing code, checkpoints, raw results, and intentional worktree changes
were preserved. This experiment added its isolated code/tests/documents and
artifacts and updated the current handoff and review pointer.
