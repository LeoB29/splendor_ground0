# Recommendation after 152: a compact public-feature advantage control

Completion update, 2026-09-10: authorized Experiment 153 is now closed and fully
audited. All four gates failed: validation MSE 0.097116269 (2.132% worse than C,
2.400% better than zero) and selected WDL minus 035 -0.005777995. All 26 fits and
all 3,264 public-feature/target rows independently verified. See
`docs/engineering/public_feature_advantage_153_result.md`. No successor ran.
The proposal below is preserved as the pre-experiment recommendation.

Execution update, 2026-09-10: the owner subsequently approved this proposal
("ok, please try it"). Experiment 153 is being prepared under
`docs/engineering/public_feature_advantage_153.md`. The recommendation and
historical no-launch statement below describe the earlier review, not the
current authorization. No expansion beyond the proposed control is included.

2026-09-10. Recommendation only. The owner asked for the next step. No new
experiment, feature extraction on experiment positions, model inference, fit,
simulation, test runner or promotion was launched. This review read existing
evidence and verified the frozen 152 identity. Preserve all dirty/frozen work.

## Decision

Recommend one inexpensive development experiment, provisionally 153: **can a
small model of explicit public game facts learn useful move advantages from
the existing data?** Use a fixed quadratic public-feature ridge model, with
135 coefficients, against the three frozen 152-C neural models and raw 035.
Reuse all 3,072 C training states and their existing eight-world labels.

This is a deliberately compact control for the learning recipe. It bundles a
coarser representation, much lower capacity, regularization and a direct solve;
it does not isolate which of those mechanisms is responsible. A pass would
support a subsequent fresh test, not establish a stronger player. A failure
would close this specific coarse control, not prove that the full inputs lack
learnable information. Do not automatically turn either outcome into a larger
training run or a feature/model sweep.

## Why this has higher value than another scale or network tweak

| Evidence | Consequence for the next decision |
|---|---|
| 152: C beats the same-precision B control, but only 0.81% MSE improvement versus A and +0.00136 WDL versus 035 | Coverage helps descriptively, but this fourfold allocation change did not deliver material move improvement. Another straightforward increase has weak support. |
| 149–152: training fits improve while validation worsens; early checkpoints remain best | More updates, denser checks or another small encoder substitution are low-priority remedies for this recipe. |
| 143: public score/resource ridge MSE 0.835858 versus 0.973134 for the larger neural value head | Explicit public summaries have positive local evidence for generalization, although this was ordinary outcome prediction, not action advantage. |
| 144/145: public-value search and public-versus-neutral comparisons did not establish gameplay gains | Better ordinary value prediction is insufficient. The proposed control must be trained and evaluated on actual move advantages. |
| Old preference probe 003 failed; 052/053/054 distillation/ranking attempts failed | A smaller model or ranking objective is not an untried general cure. This proposal uses explicit public afterstate quantities and the current paired terminal-WDL task, not frozen policy embeddings or the old resolver labels. |
| Post-150 opposite-half simulation selection gained +0.087240 validation WDL | The panel contains suggestive decision signal, but a selector with extra simulations is not a neural or coarse-feature learnability guarantee. |

The 143 reference is the positive reason to test this control. Its error reduction
versus neutral was 15.59%, much larger than its neural head's 1.73%, on fresh
ordinary-value test games. Most benefit occurred late. That result cannot be
transferred numerically to this task or taken as evidence of superior choices.
The new experiment asks whether explicit quantities also help with differences
between moves, where useful signal has repeatedly been harder to learn.

The larger programme still requires reliable policy improvement followed by
complete-game validation. The 134 loss and 138–141 diagnostics caution against
equating conditional gains with strength. The 137 dose failure and 148
inconclusive compute result do not rule out better chance search or deeper
planning. Strong-policy learning, different useful targets and substantially
different training regimes remain open, but need evidence beyond incremental
fit improvements before large-scale investment.

## Proposed fixed model

Start from the same 15 public quantities used by 143: actor-minus-opponent
score, five bonus counts, five ordinary token counts, gold, purchased-card
count, reserved-card count and noble count, with 143's fixed normalizations.
Always keep the original root actor's perspective, even after the turn changes.

Project these quantities from each move's deterministic afterstate, before a
refill/private card is drawn. Reuse the audited 149 decomposition. A pending
blind reservation increases the actor's reserved count by one without exposing
its identity. Payments, returns, bonuses, points and acquired nobles reflect
the known move effects. No actual future card, hidden identity or outcome is
used in feature construction. No new terminal-value override is introduced.

For the normalized vector phi, use the fixed basis psi consisting of its 15
entries plus every product phi_i*phi_j for i<=j: 120 products, 135 coefficients
total. No intercept is needed because the model predicts a difference:

`advantage(a) = w dot [psi(phi_after_a) - psi(phi_after_035)]`.

This permits basic interactions while retaining a small, directly fitted model.
The baseline prediction is exactly zero. Choose the maximum predicted advantage
among the original real candidates, retaining baseline-first exact ties and all
original masks. Do not clip predictions, change candidates, ensemble controls,
or replace fitted targets by a noisy empirical-best-action label.

The public summary is intentionally lossy: for example, different reserved
card identities can have the same feature vector. Report candidate-feature
collisions and zero differences on the entire panel. Keep those examples and
the existing baseline-only state; do not add features or exclude states after
seeing collisions or results. A negative result cannot reject richer public
representations or distinguish underfitting from inaccessible signal universally.

## Data, fitting and evaluation

- Use exactly 152-C's 3,072 training states, eight-world advantage labels and
  candidate masks. No new worlds, source games or outcome generation.
- Retain the original 192-state, 64-world validation panel for the development
  comparison. Do not open any consumed protected panel for new predictions or
  relabel existing validation as fresh evidence.
- Minimize equal-state mean squared advantage error over real nonbaseline
  candidates, plus lambda times squared coefficient norm. A baseline-only state
  contributes zero and stays in the state denominator, as in 152.
- Use the fixed penalty grid [0.001, 0.01, 0.1, 1, 10] from 143. Choose lambda
  using five deterministic source-game folds within C training data only,
  balanced by phase/actor. All candidates/worlds of a source stay together.
  Freeze the exact fold map, scaling and tie rule before any fit. Minimize the
  pooled out-of-fold state-mean MSE; break exact ties toward stronger penalty.
- Refit once on all C training states. This is 25 small fold/penalty fits plus
  one final fit, not a stochastic-seed search. Retain every CV result and all
  coefficients. Use Torch float64 linear algebra on this host; do not combine
  NumPy BLAS and Torch or install a new learner dependency.
- Compare the one deterministic model with zero/raw 035 and the equal-seed mean
  of all three frozen C models. Retain each C seed's comparison descriptively;
  do not pretend copies of the deterministic model are independent seeds.

Ridge regression uses squared prediction error with an L2 coefficient penalty.
The formula and direct-solver interpretation are standard; the proposed feature
basis and Splendor use are our hypothesis, not an externally established result.
[Method reference](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html).

Propose requiring all four development conditions, retaining the same practical
standards: MSE at least 3% below both frozen C's mean and zero; selected WDL minus
035 at least +0.02; selected WDL strictly above C's mean. Report all metrics,
collisions and contrasts even on failure. Training-only CV does not make the
repeatedly reused validation panel confirmatory. No confidence interval should
be used to claim a new strength result from it.

## Cost, integrity and stopping

There are zero new continuations and no neural refits. The computational work
is public-feature projection, 26 small linear solves and their independent
reproduction. This should be much cheaper than 152's simulation and neural-fit
pipeline; no measured runtime or exact end-to-end promise is available yet.
Implementation, tests, hashing and feature audits may dominate the fit time.

Freeze the implementation, exact basis, fold map, penalty/loss scaling, numeric
tolerances, input/control hashes and decision rules before fitting. Independently
reconstruct all public features, pending-reservation counts, perspectives and
target arithmetic. Reproduce every CV fit and final fit using an independent
solve/check, and verify all predictions, metrics and control hashes. Bind the
existing full source/continuation replay audits instead of rerunning unchanged
simulations. Complete the audit even if the development screen fails.

Both a pass and failure stop this experiment. A pass only supports a later,
separately frozen fresh conditional move test; gameplay would still follow
only if justified. No fresh test, arena, model promotion, automatic successor,
search-budget change or large-scale self-play is included here.

## Local sources

- [152 result](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/data_allocation_152_result.md)
- [143 result](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/value_generalization_scale_143_result.md)
- [143 public-feature definition](C:/Users/leoba/Desktop/splendor_ground0/src/splendor_ai/alphazero/run_value_generalization_scale.py:97)
- [149 afterstate projection](C:/Users/leoba/Desktop/splendor_ground0/src/splendor_ai/alphazero/afterstate_149.py:64)
- [149 result and target uncertainty](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/afterstate_advantage_149_result.md)
- [150 result](C:/Users/leoba/Desktop/splendor_ground0/docs/engineering/early_learning_150_result.md)
- [Post-150 split-world diagnostic](C:/Users/leoba/Desktop/splendor_ground0/docs/strategy/next_after_150.md)
- [Programme review and historical results](C:/Users/leoba/Desktop/splendor_ground0/docs/strategy/independent_superhuman_review_after_134_137.md)
- [003 and 052–054 records](C:/Users/leoba/Desktop/splendor_ground0/AGENT_CURRENT.md)

152's verified frozen binding remains
`015aef64dac1775b8e1403092d6f630514dcc7900e9b7348810be78d8499c1a8`.
Champions remain 084 research / 035 fast.
