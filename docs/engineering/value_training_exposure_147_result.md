# Experiment 147: longer training missed the prediction gate

Completed 2026-09-08. Both fixed training schedules, the 512-game prediction
test and the independent audit finished successfully. The prediction gate
failed, so no protected gameplay ran. No promotion or successor is authorized.
Champions remain 084 research / 035 fast. Final status records `complete`,
`prediction_gate_failed_stop`, conditional arena false and promotion false.

## Result

The longer schedule selected a later trainable-encoder checkpoint, but its
fresh prediction gain was too small and its advantage over the short control
was not established. All 512 fresh games were natural, with 32,480 positions.
Means below give equal weight to games; lower MSE is better.

| Phase | Neutral | Public | Frozen short/long | Trainable short | Trainable long |
|---|---:|---:|---:|---:|---:|
| Early/middle <40 (primary) | .998047 | .940781 | .938507 | .938659 | .934217 |
| Overall | .998047 | .821568 | .816631 | .816523 | .813295 |
| Late >=40 | .998047 | .603061 | .593550 | .592986 | .591855 |

Both frozen schedules selected update 128, so their predictions are identical.
The two 146 short models were evaluated alongside the new models on the same
fresh games. Compare these paired contrasts, not aggregate MSE across the
different 146 and 147 test sets. Neither old test set entered model selection.

| Primary contrast | MSE difference | Paired CI95 | Relative reduction | Requirement |
|---|---:|---|---:|---|
| Trainable long minus public | -.006564 | [-.012020,-.001184] | .6977% | >=3%, CI upper <0: fail |
| Trainable long minus frozen long | -.004290 | [-.009677,+.001010] | .4571% | >=1%, CI upper <0: fail |
| Trainable long minus trainable short | -.004443 | [-.009631,+.000667] | .4733% | CI upper <0: fail |

The gain over public is statistically detectable but below the practical
threshold. The other two intervals include zero; they establish neither
superiority nor equivalence. Frozen long-minus-short is exactly zero. The
secondary exposure-by-trainability interaction equals the trainable exposure
contrast here, because the frozen selected model did not change; it is also
inconclusive. No secondary contrast can replace the failed primary gate.

Natural-completion, phase-coverage, overall/late noninferiority tolerances and
cost gates passed. The overall trainable-long improvement versus public was
1.0071%, and late improvement 1.8583%, without established overall/late gains
over the short trainable or long frozen controls. Early/middle remains primary.
The noninferiority tolerance was .01 absolute MSE, not a zero-regression claim.

## What the longer fits actually did

Both fits completed **16,384 updates**, compared with 640 completed in 146.
Each drew 2,097,152 training positions and encountered all 4,096 games and
260,797 of 260,904 unique positions (**99.959%**). This is measured coverage,
not a claim that every position received eight visits.

| Fit | Selected update | Selected unique positions | Selected training MSE | Selected validation MSE | Final training MSE | Final validation MSE |
|---|---:|---:|---:|---:|---:|---:|
| Frozen encoder | 128 | 15,855 | .946543 | .944132 | .908566 | .975694 |
| Trainable encoder | 1,152 | 112,437 | .938447 | .943141 | .126827 | 1.658038 |

All error columns in this training table use game-balanced turns <40. The
selected trainable model had 147,456 position draws across all 4,096 games and
43.095% unique-position coverage. It was a real later checkpoint, not the
original short model renamed. The separately saved update-16,384 checkpoint
was ineligible for protected testing or gameplay.

The terminal trainable model fit training outcomes closely while validation
error rose sharply: clear overfitting. Giving the original recipe substantial
extra exposure did not produce a useful final predictor. The unchanged .001
validation-improvement rule selected the earlier, better-generalizing model.
Both initial 640-update histories and selected update-128 tensors reproduced
146 exactly. Policy architecture, training data, optimizer, batch stream and
selection threshold were held fixed; only the available schedule was extended.

## Scientific meaning

147 closes the specific ambiguity that the recipe was never allowed to train
beyond 146's early stopping point. It did train much longer, broadly covered
the available positions, found a modest later validation improvement, and then
overfit. The selected later model still failed every primary improvement gate.
Plainly adding more updates to this fixed recipe is not supported as the next
practical remedy. Do not automatically extend it or choose another checkpoint.

This does not prove that more training has exactly zero benefit: the paired
long-minus-short interval allows a small benefit. Nor does it reject other
optimization, regularization, targets, state distributions or trainable value
networks universally. The result concerns one initialization, data source,
loss and selection rule. No gameplay strength conclusion is available.

The next strategic review should examine how to obtain a useful learning/search
signal and representative states, with the programme's offline/gameplay
mismatches in mind. A new architecture or data remedy needs its own causal
question and prospective gate; none is launched or approved by this report.

## Cost, verification and artifacts

New training took 137.67 seconds frozen and 215.41 seconds trainable (5.88
minutes combined, including validation and fit diagnostics). Combined inference
on the fixed 130-state probe was .002996 seconds trainable versus .002709
public; median ratio 1.11062 and p95 ratio 1.08176, both within the caps.
These are probe timings, not a gameplay measurement.

The final full suite passed **1,070 tests in 263.45 seconds**; 25 focused tests
passed. Preflight verified 130 exact initial predictions and two natural arena
probes with 128 roots and eight full search reproductions. Arena probe mean
time ratio was .996624. Probe outcomes are engineering checks, not strength data.

Independent audit passed in **328.36 seconds**. Both complete 16,384-update fits,
all selected and terminal tensors, the 146 prefix checks and independently
counted training coverage reproduced exactly. All 512 trajectories, 32,480
observations/latents/features/labels, six-model predictions, paired bootstrap
contrasts, interaction and gate checks matched. All four models' policy logits
matched the unchanged 035 bridge at 130 probe positions.

All 280 current and 278 explicitly listed prior dependencies, prior inputs and
short-control bindings stayed unchanged. Candidate selection preceded fresh
test collection. Parent session 24466 / PID 61544 and audit PID 51844 exited
successfully; no active experiment processes remain. The frozen pipeline's
manifest-to-final-status interval was 761.26 seconds (12m41s); preparation,
the full suite and preflight were additional.

Protocol: `docs/engineering/value_training_exposure_147.md`.
Runner: `run_value_training_exposure_147.py`.
Artifacts: `outputs/value_training_exposure_147/`, including both selected and
ineligible terminal models, training histories/coverage, the candidate lock,
six-model prediction tensors, result, cost and audit evidence.
Execution session 24466, parent PID 61544; audit child PID 51844.
Logs: `outputs/value_training_exposure_147/run.log`,
`outputs/value_training_exposure_147_tests.log`,
`outputs/value_training_exposure_147_preflight.log`.

Manifest digest:
`a8aae4bbdcd1f5c739b11933ad90ee1dac228b4f0fabb9e6d37b9f4a15cf0ebe`.
Result SHA256:
`91b1dd5cd4581b5484911b7af3dbf8b3c281aa385cee98d8ed0d9dfea0bd9ad4`.
Prediction-verification SHA256:
`b9193b03a22ec6b7c850b615580f4112497effccc038afd1f60eb6120a3ec0fa`.
Selection-lock SHA256:
`acccf05fdce7b2cd63c34744604698c3544bf77c9bd7f2d1829effff3e339f15`.
