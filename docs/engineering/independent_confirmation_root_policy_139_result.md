# Experiment 139 result: default improvement, confirmation benefit unproven

Completed 2026-09-07. Valid experiment; frozen decision:
`default_improvement_supported_confirmation_benefit_unproven`.

Independent confirmation selected moves with a positive held-out advantage over
the original four-root 084-contract choices. It did **not** demonstrate an
advantage over either the unfiltered expanded proposals or the equally costly
same-world confirmation control. The full conjunctive success criterion did not
pass. This is a partial positive offline result, not an invalid experiment,
evidence of equivalence between methods, or a stronger deployed agent.

## Primary held-out evidence

All 60 positions came from distinct source games, ten per phase/seat cell, with
no overlap with the experiment-138 panel. Original 086 discovery proposal and
default choices were reused; no new source games or discovery search were run.
Both confirmation policies used eight check blocks and the same fixed screen.
All choices were frozen before the eight independent test blocks began.

| Test contrast, normalized final-score margin | Mean | Paired CI95 |
|---|---:|---:|
| Independent confirmation minus default 084 choice | +0.024740 | [+0.013368, +0.038472] |
| Independent confirmation minus unfiltered proposal | +0.003802 | [-0.010078, +0.017734] |
| Independent confirmation minus same-world confirmation | -0.000911 | [-0.012266, +0.010981] |
| Same-world confirmation minus default | +0.025651 | [+0.009062, +0.042622] |
| Unfiltered proposal minus default | +0.020937 | [+0.001927, +0.040486] |

The independent policy changed 17/60 default decisions, exceeding the frozen
six-switch dose requirement. Its default-relative gain exceeded the +0.01
magnitude requirement and had a positive confidence interval. Both required
comparative intervals included zero, so neither established the additional
benefit of independent confirmation. The near-zero control contrast is not an
equivalence result; its interval still permits gains or losses of practical size.

The same-world policy switched on 30/60 states. Original proposal/default choices
differed on 42 states; the other 18 remained in every analysis. Descriptively,
both screens switched on 15 states, only the independent screen on two, only
the same-world screen on 15, and neither on 28. Independent confirmation was
more conservative, but conservatism alone was not the success criterion.

The bootstrap used 50,000 paired source-game resamples within six fixed cells,
seed 139300000. Displayed means and intervals use the same equal-state/game
weights. Rollouts and blocks were not treated as independent population samples.
The full-success claim required all three primary comparisons, rather than
selecting whichever contrast looked favorable.

## What this adds to 134 and 138

Experiment 138 established that much of the old conditional expanded-root gain
did not survive fresh worlds. Experiment 139 establishes that an independently
screened subset can retain a positive conditional gain on another untouched
world set. However, both the unfiltered proposals and the same-world screen also
retained positive gains on this panel. The evidence therefore does not credit
independent screening with a demonstrated improvement over those alternatives.

The remaining offline/gameplay discrepancy is especially important. Unfiltered
086 choices still look better than the default here under fresh-world
stochastic-035 continuations, while the deployed expanded-root player lost
decisively to 084 in experiment 134. Passing an independent-world check is not,
by itself, sufficient evidence of stronger gameplay or a trustworthy teacher.
This does not undo 138's sampling-overfit finding; it shows why that finding and
its proposed remedy must not be treated as the complete explanation.

Raise the priority of continuation-policy mismatch, limited planning horizon,
repeated replanning and the different distribution of positions reached in
actual games. Experiment 139 does not separate those explanations. It does not
justify retuning this screen on the test panel, relaxing gates, adding states,
distilling 086, or reopening the old allocation/PUCT recipes. Retain 084 as
research champion and 035 as fast champion. No arena or successor was launched.

These are values for fixed actions under a particular stochastic continuation
policy and sampled opponent minimum. They are not optimal action values, a
comparison of complete live 084 decisions, or gameplay win rates. The panel
contains existing raw-035 source states, rather than new on-policy 084/139 play.

## Secondary and descriptive checks

WDL under the margin-minimizing opponent reply gave independent-minus-default
+0.072222, independent-minus-unfiltered -0.002257, and independent-minus-same-world
-0.005903 in [-1,+1] units. Separately WDL-minimized values gave +0.072569,
-0.001389 and -0.009375 respectively. These descriptive estimates do not supply
additional evidence of a comparative screening benefit. No secondary confidence
claim is made, and these values must not be presented as arena win-rate changes.

| Phase | States | Independent switches | Independent minus default | Independent minus unfiltered |
|---|---:|---:|---:|---:|
| Early | 20 | 4 | +0.024609 | +0.018620 |
| Middle | 20 | 2 | +0.005573 | -0.009245 |
| Late | 20 | 11 | +0.044036 | +0.002031 |

Both seat point estimates were positive versus default (+0.017188 and +0.032292).
Only eight states lacked existing blind reserves. These unpowered breakdowns
cannot choose a phase-specific screen or isolate public-refill uncertainty from
private-card uncertainty, and do not justify a CFR-first architecture.

## Accounting, cost and verification

All **117,504 continuations completed naturally**, with exactly 39,168 in each
check arm and 39,168 in the test stage. Each condition had 2,448 opponent-scoring
calls. There were zero hard caps, adjudications, fallback triggers, fewer-than-
four-reply branches, or state exclusions. All 13 final integrity gates passed.

The two check arms were exactly matched in continuation and opponent-scoring
counts. Their aggregate worker wall times were 4,690.54 seconds for same-world
and 4,701.80 for independent checking, ratio 1.0024. The independent test used
4,950.60 aggregate worker wall seconds. These are summed worker wall times, not
measured CPU times or complete deployed-agent move latencies. The unfiltered
and default policies do not pay the additional confirmation cost.

Protected execution took **1,469.27 seconds (24m29s)** on ten CPU workers with
one Torch thread each. It started at 00:20:08 Europe/Paris; the all-state choices
file was sealed at 00:36:11; the result was written at 00:44:38. All workers
exited, the parent returned exit code zero, and stderr was empty. Only counts, time and
integrity were inspected during execution; no partial returns or switch counts
were used to steer the run.

Final focused tests passed 26/26; the full suite passed 921/921 in 199.55 seconds
before launch. Two preflight-only compatibility corrections and their preserved
initial manifests are detailed in the preparation record. They changed neither
the scientific protocol nor the selected positions, proposal/default pairs,
thresholds, seeds or underlying 138 helpers. All 262 pre-existing source
dependency files matched their initial preparation hashes.

The separate verifier reconstructed every terminal return from scores and
card-count tiebreaks; every grouped margin/WDL value; both screening decisions;
test-stage policy routing; all five primary means and every bootstrap endpoint;
and the secondary WDL contrasts. It used no experiment evaluation, resolution,
screening or summary helpers. All comparisons agreed within 1e-12, and source,
dependency, choice and result hashes were verified. No additional gameplay or
training was used for verification.

## Artifacts

- Protocol: `docs/engineering/independent_confirmation_root_policy_139.md`.
- Preparation: `docs/engineering/independent_confirmation_root_policy_139_preparation.md`.
- Result: `outputs/independent_confirmation_root_policy_139/result.json`.
  SHA256 `0df72eb600ad84d8404bc550935746f6de185493640865a65de95d63a41e11ac`.
- Frozen choices: `outputs/independent_confirmation_root_policy_139/frozen_choices.json`.
  SHA256 `dc43fd5c4b4a8ba0d5d76ca4ed858dfdb21109499ca1fea23158e887bce6961f`.
- Raw records: the output directory's `checks/` and `test/` subdirectories.
- Independent verifier: `scripts/analysis/verify_independent_confirmation_139.py`.
- Verification record: `outputs/independent_confirmation_root_policy_139/independent_verification.json`.
  SHA256 `faddb2886ee1c95f6f819b1f92418d2abc3ab5c6511282ea5fd63f29a494f935`.

No model was trained, promoted or installed; no new source games, full-game
arena, sealed 087 outcomes, human training data or automatic successor were used.
All earlier work and results were preserved.
