# Experiment 153: compact public-feature advantage control

2026-09-10. Owner authorized implementation and execution of the recommendation
in `docs/strategy/next_after_152.md`. This protocol is frozen with the runner,
entrypoint, independent auditor and tests before any fit. Preserve all dirty
work and predecessor files. Champions remain 084 research / 035 fast.

## Question and fixed representation

Can a small explicit public potential generalize useful move advantages on the
existing 152-C data? This bundles representation, capacity, regularization and
direct optimization. It does not isolate a mechanism or test gameplay strength.

For each original candidate, use the frozen 149 deterministic afterstate,
before any public refill or blind-reserve draw. Always project the root actor
minus the other seat, regardless of next current player or terminal outcome.
The normalized vector phi has these exact indices:

| Index | Public quantity | Divisor |
|---|---|---|
| 0 | Score difference | 15 |
| 1–5 | Bonus differences, white/blue/green/red/black | 7 |
| 6–10 | Ordinary token differences, same color order | 7 |
| 11 | Gold difference | 5 |
| 12 | Purchased-card count difference | 20 |
| 13 | Reserved-card count difference | 3 |
| 14 | Noble count difference | 3 |

The pending blind reservation contributes one to the actor's reserve count.
Known payments, returns, purchases, points and nobles have already occurred.
Compute Python integer differences/divisors directly into Torch float64, with
no float32 intermediate, standardization, learned normalization or clipping.
This uses 143's quantities/scales, with the required fixed root perspective.
No hidden identity, future card identity, terminal override or intercept.

psi consists of phi followed by phi_i*phi_j for i=0..14 and j=i..14 in that
loop order. Width is 15+120=135. x_a = psi_a-psi_baseline. Prediction is x_a w;
baseline and its padded copies are exactly zero. Argmax over baseline followed
by original real alternates, first index on exact ties. No candidate changes.

Report real candidate-pair aliases, states with aliases, alternate-to-baseline
zero differences, all-candidate-identical states, and baseline-only states on
both panels. Exact phi equality defines aliases (equivalent to psi equality
because phi is included). Counts exclude padding. Keep all examples/actions.

## Data, CV and loss

Train: exactly 3,072 C states, first 768 original 149 train states with their
first eight of 32 worlds, followed by the 2,304 new 152 states with eight worlds.
Validation: original 192 states with all 64 worlds. Read only these panels.
q is terminal WDL mean; y_a=q_a-q_baseline. Retain all original masks and padding.
No new simulations, source games, world draws, continuation replays or neural
inference/refits. Bind the completed 152 audits and their original 149 lineage.

One state per unique source seed, disjoint across training and validation.
Within each of six phase/actor cells, sort training rows by SHA256 of compact
JSON `["153-fold-v1", source_seed]`, then numeric source seed on a digest tie.
Assign fold `(rank + 2*cell) % 5`, rank zero based. All candidates and worlds
stay with their source. Save every row, source seed, cell, sample digest and
fold in `fold_map.json` and seal its hash before features/fitting. Each cell
has 512 training rows, so fold cell counts differ by at most one.

For a fitting subset of N states, minimize

`(1/N) sum_s [sum_real_nonbaseline_a (x_sa w-y_sa)^2 / max(m_s,1)] + lambda ||w||^2`.

m_s is the real nonbaseline count. A baseline-only state contributes zero but
remains in N. Each real row has weight 1/(N*m_s). No intercept or extra penalty
rescaling. Main solve: `(X' W X + lambda I) w = X' W y`, Torch CPU float64,
one thread, deterministic algorithms. Fixed grid .001,.01,.1,1,10.

Fit each penalty leaving out each of five folds (25 solves). Pool the held-out
state losses over all 3,072 rows; minimize that mean, exact ties favor larger
lambda. One final all-training solve with selected lambda (26 total). Retain
every coefficient vector, residual, fold metric and full out-of-fold prediction.
Seal fits and selection before computing validation predictions. No tuning
on validation, stochastic seed, search expansion or alternative basis.

## Controls and gates

Compare one deterministic treatment with zero/raw 035 and all three cached
frozen 152-C predictions, using the arithmetic mean of their individual MSE,
selected WDL minus 035, and baseline fractions. This is not an ensemble of
predictions. Report each C seed and every treatment-minus-control contrast.
Also report final training metrics. Do not manufacture treatment seed repeats.

All four development gates are required:

1. MSE at least 3% below C's equal-seed mean.
2. MSE at least 3% below zero.
3. Selected mean WDL minus 035 at least +.02.
4. Selected mean WDL strictly greater than C's equal-seed mean.

This repeatedly reused validation is development evidence, not confirmatory.
No confidence interval or game win-rate claim is attached to these results.
Both outcomes stop after the full audit. Failure closes only this coarse
control. A pass could support a separately authorized fresh conditional test;
neither result automatically authorizes a successor, arena or promotion.

## Independent checks and numeric policy

Preflight and final auditor reconstruct all 3,264 states' features by direct
public move-effect arithmetic from the prestate and action, without calling
the main feature code, stepping the engine or reading any hidden card identity.
Recheck root actor, phase, source/sample lineage, pending-reserve counts,
polynomial ordering, baseline difference, masks, target record hashes, world
seeds, winner-to-WDL conversion and target means. Compare those tensors exactly.
No resimulation of unchanged paths: bind 152's full replay/refit audit hash
`e430a94ae2c307ab532ebf0314434eec5a81dbc6fad3ece059b10c8bc52e765f`
and parent binding `015aef64dac1775b8e1403092d6f630514dcc7900e9b7348810be78d8499c1a8`.

Independently reconstruct folds and all 26 fits using augmented least squares:
scale rows/targets by 1/sqrt(N*m_s), append sqrt(lambda) times identity and zero
targets, and call Torch float64 `lstsq(driver="gelsd", rcond=1e-14)`.
Assert full column rank. This avoids the main normal-equation implementation.
Both main and audit normal residual infinity norm must be <=1e-10. Coefficients
must agree with absolute tolerance 1e-9 and relative tolerance 1e-8; predictions
absolute tolerance 1e-9, CV/summary metrics absolute tolerance 1e-12. Choices,
selected penalty and gates must agree exactly. Positive ridge ensures full
rank; these tolerances allow floating-point algorithm differences while being
far below the practical decision thresholds. No tolerance change after fitting.

Verify all saved predictions/statistics independently with scalar dot products
and arithmetic, all collisions, and control metrics against the frozen 152
result. Final audit is mandatory on failed development gates as well as passes.
Any audit/identity failure stops without changing frozen code or retrying fits;
preserve failure evidence for an explicit correction proposal if needed.

## Execution and records

Run focused synthetic tests and the full repository suite before freeze.
`run_public_feature_advantage_153.py prepare` seals manifest and fold map,
projects features and completes independent data preflight. `run` executes
the 26 solves, seals selection, evaluates and runs the separate-process final
auditor. All experiment artifacts use refuse-overwrite writes and a process
lock. Final status records result/audit hashes and zero expanded-scope work.

Runtime of small solves should be low; feature projection, tests, file hashing
and independent checks may dominate. Record measured preparation, fit/audit
and end-to-end time without claiming an unmeasured duration in advance.
