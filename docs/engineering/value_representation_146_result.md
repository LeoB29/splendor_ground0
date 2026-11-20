# Experiment 146 result: prediction gate failed; stop

Completed 2026-09-08. The separately trainable value encoder did not deliver
the prespecified material early/middle improvement. No protected gameplay,
promotion or successor ran. Champions remain 084 research / 035 fast.

The pipeline exited successfully (execution session 59324, parent PID 58212).
`outputs/value_representation_146/final_status.json` records `complete`,
`prediction_gate_failed_stop`, `conditional_arena_ran: false`.

## Primary result

All 512 fresh games completed naturally, covering 32,680 positions. Both locked
residual models used the same public baseline and untouched original 035 move
policy. Lower MSE is better; games, rather than positions, receive equal weight.

| Phase | Neutral | Public baseline | Frozen encoder | Trainable encoder |
|---|---:|---:|---:|---:|
| Early/middle, turns <40 (primary) | .988281 | .954854 | .953001 | .953447 |
| Overall | .988281 | .839888 | .835029 | .835288 |
| Late, turns >=40 | .988281 | .628001 | .618148 | .618039 |
| Early, turns <20 (descriptive) | .988281 | .987260 | .986685 | .987656 |
| Middle, turns 20..39 (descriptive) | .988281 | .922447 | .919317 | .919238 |

Primary trainable-minus-public MSE difference: **-.001407**, paired whole-game
CI95 **[-.002019,-.000798]**. Relative improvement **0.1473%**, below the frozen
3% requirement. The improvement is statistically detectable but too small to
pass the practical gate.

Primary trainable-minus-frozen difference: **+.000446**, CI95
**[-.000054,+.000937]**. The trainable arm's error was 0.0468% higher on the point
estimate; it did not establish superiority or meet the required 1% improvement.
The interval does not establish a primary regression either.

Both primary requirements failed. Natural-completion, phase-coverage,
overall/late noninferiority and inference-cost requirements passed. No threshold
or sample size changed after outcomes. The conditional 800-game arena correctly
did not launch; its two earlier engineering probes are not strength evidence.

## Secondary results and cost

The trainable arm reduced overall error by 0.5477% against public (difference
-.004600, CI95 [-.005125,-.004078]) and late error by 1.5863% (difference
-.009962, CI95 [-.010557,-.009365]). Against the matched frozen learner, overall
and late differences were +.000259 and -.000109, respectively; both intervals
include zero. Most residual improvement over public again arrived late.

The noninferiority guardrail required interval upper bounds below .01 absolute
MSE against both comparators, for both overall and late. Passing this tolerance
does not prove exact equality or nonregression at a zero margin.

Combined policy/value inference on 138 fixed probe states, three repetitions:
trainable/public median ratio **1.09066**, p95 ratio **1.17805**, below the
1.5/2.0 caps. Mean inference times were .003329 seconds trainable, .003238 frozen,
.002969 public. All policy logits matched exactly. These are isolated probe
timings, not a gameplay speed/strength result.

## Training and limits on interpretation

Both models initialized at the public baseline. The frozen arm trained 33,025
head parameters; the trainable arm trained 241,921 encoder/head parameters in
its separate value model. All original policy parameters stayed untouched.

The available development corpus contained 4,096 training games / 260,904
positions and 512 validation games / 32,704 positions. Both fits stopped at
update 640 and selected update 128 under the frozen .001 minimum-improvement,
four-stale-validation rule. Selected early/middle validation MSE was .944132
frozen / .944664 trainable, versus initial .946480.

A post-result reconstruction of the fixed batch stream (no fitting or new
evaluation) shows that each selected model had sampled 16,384 positions with
replacement: 15,855 unique positions across 4,011 games. The completed 640-step
fit sampled 81,920 positions, 70,063 unique, across all 4,096 games. Thus available
corpus size is not the same as the selected model's actual exposure. These dose
counts are descriptive and did not alter selection.

Later validation losses sometimes decreased by less than the required .001;
for example trainable update 384 reached .944088. The frozen rule retained
update 128. Those later checkpoints were not substituted or tested. The pilot
tests this optimizer, initialization, residual/clipping recipe, outcome labels
and selection rule, not all ways to train a separate value encoder.

The evidence does not support unfreezing value features as a practical remedy
within this recipe. It also does not establish that early advantages are
intrinsically unpredictable, that the observations lack necessary information,
or that trainable value networks cannot work. Optimization, training exposure,
target quality and the sampled state distribution remain possible limitations.
Prediction gains cannot be treated as gameplay gains, particularly given the
programme's earlier offline/gameplay mismatches. Do not scale or promote this
unchanged recipe on the present result, or infer a search-depth explanation.

## Validation and provenance

- Final full suite: **1,045 passed in 200.13 seconds**; 22 focused tests passed.
- Corrected preflight: 138 exact initial predictions; two natural arena probes,
  136 audited roots and eight exact full-search reproductions. Timing gates
  passed, mean .405657/.404394 seconds, ratio 1.003121.
- Initial preflight detected 29 scalar/batched rounding mismatches among 138
  states, maximum 1.11e-16. Before any training/protected data, deployment was
  fixed to reuse the exact public scalar. Original evidence/source/identity
  and logs remain preserved. Corrected probes and the final full suite passed.
- Independent audit: all 512 behavior trajectories, 32,680 observations,
  original latents, public features, actor-relative labels and saved predictions
  reconstructed. Both 640-update fits and every selected tensor reproduced
  exactly; frozen encoder unchanged, selected trainable encoder changed.
- Independent paired bootstrap and gate checks passed. All 278 current frozen
  dependencies and the explicit previous manifest's dependencies remained
  unchanged. Candidate lock predates fresh test collection. Audit took 54.22s.
- No old protected data entered training or selection. Only 143 development
  was reused. No active experiment processes remained after completion.

Protocol: `docs/engineering/value_representation_146.md`.
Runner: `run_value_representation_146.py`.
Result, selection, complete history, cost, prediction audit and descriptive dose
data: `outputs/value_representation_146/`.
Logs: `outputs/value_representation_146/run.log`,
`outputs/value_representation_146_tests_v2.log`,
`outputs/value_representation_146_preflight_v2.log`.

Manifest digest:
`a5a12a2c689994e636a8860c805219f436dc2ff54d116329e34a656a5767d594`.
Result SHA256:
`923326e7af228e408f307e239d92c290fe06f749c791f3dc5189cf1e0c938d92`.
Prediction-verification SHA256:
`81c4e5586010389229ddd10af14f555cd4243cd1d7d3cd8a92728751c6ca8aa5`.
Selection-lock SHA256:
`9070718041bce431969ac1cc37038e88746f3d8c2ede7bd3b6605957e2546406`.
