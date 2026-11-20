# Short-prefix control-variate feasibility

Authorized by the owner on 2026-10-09. One bounded CPU study is authorized:
authenticate saved evidence, reconstruct short continuations, measure comparable
costs, calibrate a rule on TRAIN, assess it once on VALIDATION, and independently
verify the result. No neural fitting, new source games, new player, arena,
promotion, parameter rescue, recurring monitor or automatic successor.

Implementation and output use the new `control_variate_128` namespace. Preserve
all intentionally dirty work and frozen predecessors. Freeze this protocol,
source inventory, code and checks before production timing or prefix collection.

## Question

Can the existing frozen143 public evaluator lower the sampling variance of
comparisons between moves per unit CPU cost, when exact terminal simulations
correct its estimates? Ordinary outcome prediction is not the endpoint. The
latest learner captured the seen teaching examples but did not establish planning
transfer; this study explores a different use of learned predictions.

This is a feasibility measurement of a prospective estimator. It does not run
that estimator as a player, establish better decisions, or measure game win rates.
Historical143 value accuracy is motivation only; action differences may remain
poorly predicted. A null closes this fixed evaluator and calibration recipe,
without rejecting control variates generally.

## Saved population and terminal target

Use all 336 original main roots from the completed `teacher_signal_capture_128`:
168 TRAIN roots from42 source pairs and168 VALIDATION roots from42 different
pairs,14 pairs per original161 fit per split. Each pair contributes four roots.
Preserve phase/seat balance, small supports, private reservations and original
up-to-eight canonical candidates/concrete representatives. No outcome-dependent
subsets or additional games. These positions and outcomes are reused DEVELOPMENT,
even though the original learning splits remain intact.

Authenticate the completed source freeze, code, main inventories, original
selection seals and independent simulation audits. Use only the newer independent
128-world assessment matrix for terminal samples Y. The reference move at each
root is the teacher choice already sealed using the older disjoint128-world
selection block. Never choose a reference maximum from the Y samples used to
measure covariance. Check all world IDs, candidate ordering and outcome fields.
An unknown/censored terminal sample is not a draw; it prevents this finite-variance
calculation and holds the study without replacing a root. The inherited complete
panel has no unknown outcomes, but the implementation must verify this.

Y is terminal expected-score sample `(WDL+1)/2` for the original root actor.
Keep the audited observer-legal determinization and greedy035 continuation law,
including natural termination, tiebreaks, payments, nobles, private observations
and forced-pass deadlocks. This target is the existing rollout-policy return,
not an optimal-play value.

## Prefixes and fixed evaluator

At each saved assessment-world seed, reproduce the same sampled world and forced
root action. The forced root action counts as ply1. Continue both actors with
unchanged greedy035, measuring endpoints after4,8 and16 individual plies on TRAIN.
On VALIDATION collect only the horizon selected and sealed on TRAIN.

Load the selected coefficients from the authenticated143 public15-feature ridge
evaluator; do not train, reselect a coefficient set, or read143 protected-game
predictions as new evaluation data. Reconstruct its normalized root-actor-minus-
opponent scores, five bonuses, five ordinary tokens, gold, purchased-card count,
reserved-card count and noble count. Apply the original clipping to[-1,1], then
map to[0,1]. Maintain root perspective when turn ownership changes. Endpoint
features may use lawful public state only; neither hidden deck order nor opposing
private card identities enter this evaluator. A natural terminal reached before
the horizon receives its exact terminal score. Unknown or invalid cutoffs hold.

Record all endpoint public facts, features, predictions, prefix lengths and
trajectory/state hashes. Avoid duplicating full hidden snapshots at every endpoint.
Candidate futures within a world retain the historical common-world coupling.

## Finite nested estimator

For a fixed horizon h, coefficient beta and counts M>n, the prospective estimate is

`Qhat = mean(Y[0:n]) + beta * (mean(Z[0:M]) - mean(Z[0:n]))`.

The first n short estimates belong to the n full trajectories. The remaining M-n
require prefixes only. The cheap mean is estimated, never assumed known. Fixed
coefficients and allocations preserve the expectation of Y. The experiment
predicts this estimator's variance from sampled moments; it does not manufacture
additional realized Z samples or return estimates when M exceeds128.

For each nonreference candidate a, compute the paired samples
`D=Y_a-Y_teacher`, `U=Z_a-Z_teacher`. Sample variance/covariance use denominator127
and center within each root/contrast across the128 common worlds. Average these
moments across the real nonreference candidates, then equally across roots.
A one-candidate root contributes exact zeros and stays in the population.
Between-position covariance is not a substitute for within-position covariance.

For fixed beta,n,M:

`Var(Dhat) = Var(D)/n + (1/n - 1/M) * (beta^2*Var(U) - 2*beta*Cov(D,U))`.

Independent reconstruction must reproduce both covariance arithmetic and this
finite-M correction. A perfect or zero-variance control, zero covariance, unequal
support and the no-control limit require explicit tests. Nonfinite values or
material negative variances fail; only declared floating-point tolerances apply.

## Comparable cost calibration

Before main collection, identity-select18 TRAIN roots, one per fit/seat/phase
cell. Select eight assessment worlds per timing root by a separate fixed identity
hash. Outcomes, support size, measured execution time and disagreement do not
select roots or worlds. Retain every candidate in each selected world.

Use the same loaded035 backend and current host for seven timing arms: full
terminal only; full terminal with the143 estimate at each of h=4,8,16; and prefix
only at each horizon. Rotate execution order by fixed identity. Record actual
CPU and wall time and calls, including world construction, endpoint extraction
and evaluator overhead. Full timing trajectories must exactly reproduce their
saved outcomes and available trajectory/final-state accounting. These bounded
replays do not add new terminal evidence or independent samples.

Pool CPU seconds and actual candidate-world calls within each phase to obtain
F_phase (full only), B_h,phase (full with its endpoint), and P_h,phase (prefix).
Use `B=max(measured B,F)` to avoid accepting apparent free savings from timing
jitter. Preserve per-root timing distributions and phase diagnostics. This is a
TRAIN timing calibration; uncertainty intervals below condition on these rates.
They do not include timing estimation or future-hardware uncertainty. An actual
estimator test remains necessary before claiming a realized speedup.

## TRAIN-only choice and validation seal

For each h in ascending[4,8,16], choose one global
`beta_h = mean_TRAIN Cov(D,U) / mean_TRAIN Var(U)`; use zero if the denominator is
exactly zero. No separate phase-, fit- or position-specific coefficient.

Consider integer counts16<=n<=128 and n<M<=512. A fixed allocation is feasible
only if `n*B_h,phase + (M-n)*P_h,phase <=128*F_phase` in every phase. This is a
fixed finite analytic allocation search, not training or a validation sweep.
For each feasible rule calculate its TRAIN efficiency ratio:

`[mean_root predicted variance * mean_root vector CPU cost] /
 [mean_root Var(D)/128 * mean_root 128-full vector CPU cost]`.

Vector cost equals actual recorded support times the corresponding per-candidate
phase rate. Use the product of population means, not per-root variance ratios;
zero-variance roots must not be dropped. Rank TRAIN ratios rounded to ten decimal
places, then horizon,n,M ascending. Report the unrounded scientific quantities.
The declared rounding makes independent arithmetic deterministic near exact ties.

Seal one selected configuration, TRAIN moments, timing model, all calibration
artifact hashes and independent reconstruction in `selection_lock.json` and
`selection_audit.json` before any VALIDATION prefix calculation or scientific
outcome loading. Metadata/hash authentication is allowed earlier. Even if the
best feasible TRAIN ratio is unfavorable, assess that rule once. If TRAIN's
baseline variance is zero or no nontrivial allocation is feasible, finish with
that explicit nonpassing feasibility classification; do not invent a rule or
open VALIDATION for a rescue.

## Validation endpoint and stopping decision

Use all168 VALIDATION roots once with the fixed rule and TRAIN timing model.
The primary endpoint is the same population variance-times-CPU ratio versus128
full simulations. Below1 predicts lower variance at equal total CPU budget.
Pass requires ratio<=0.80 and paired95% upper endpoint strictly below1, plus all
scientific/integrity/operational conditions. A pass only supports a separately
specified actual estimator/decision experiment, not an automatic second stage.

Use50,000 whole-source-pair bootstrap resamples, stratified within the three
assigned fits (14 pairs per fit), seed2283000000. Keep all four roots and every
contrast/cost paired within each draw. Models and worlds are not extra independent
source pairs. Independently reproduce the bootstrap and selection arithmetic.
Undefined baseline variance or zero-denominator bootstrap draws prevent a pass;
report them without silently dropping, replacing or imputing draws.

Report TRAIN/VALIDATION efficiency, selected h/beta/n/M, raw variances/covariances,
finite-mean correction, per-fit and phase sensitivities, all support/no-op counts,
timing spread, actual work and audit coverage. Phase results do not select rules.
No alternative horizon, allocation, evaluator, coefficient or threshold rescue.

## Verification and operations

Verify every saved endpoint's public feature and value arithmetic independently.
Independently replay eight identity-chosen whole worlds at every main root, all
candidates in each world, plus all128 worlds at fixed full-stress roots. Within
each split stress coverage selects by identity: fit/seat/phase, fit/private-reserve
category and fit/short-support1/2/3 (at most39 roots per split). Report missing
cells without adding games. Independently reconstruct all timing prefixes and
compare every timing full trajectory to the inherited authenticated terminal
record. Test changed trajectory/evaluator paths exhaustively on bounded fixtures;
do not rerun every historical complete trajectory or any old training.

Require source/privacy/terminal tests, analytical variance and allocation tests,
independent statistical checks, production-versus-fixture failure isolation,
strict stage ordering and a real hidden-worker test before freeze. No preflight
fixture may leave production runtime markers. Failure is fatal to that frozen
run; preserve artifacts and do not silently retry or patch frozen files.

Use one hidden detached pythonw supervisor and at most eight one-thread CPU
workers. No terminal windows or GPU. Inhibit system sleep; display sleep is allowed.
After timing and its audit, produce an outcome-blind time/storage forecast before
main prefixes. Count elapsed work, worst selected horizon, full-stress/sample
prefix replays, endpoint reconstruction and finalization with a1.35 allowance.
Hard execution ceiling two hours and5GiB disk reserve. If the forecast fails,
hold before main work rather than changing the population or protocol. Expected
prepared execution1-2h is provisional; implementation and preflight are additional.

Finish with immutable selection, assessment, audit, readable report and final
status. Both pass and fail stop. Only unbound handoff notes may change after freeze.

Method context: Peherstorfer, Willcox and Gunzburger, *Optimal Model Management
for Multifidelity Monte Carlo Estimation*, SIAM2016, DOI10.1137/15M1046472. The
estimator principle supplies no empirical Splendor efficiency guarantee.
