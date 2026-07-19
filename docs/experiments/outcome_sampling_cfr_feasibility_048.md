# Outcome-Sampling CFR Feasibility Gate (`048`)

Frozen before implementing the outcome-sampling estimator, the finite full-game
adapter, or producing any `048` evidence. This is a CPU-only target-quality
diagnostic authorized by `047`. It performs no neural fitting, gameplay strength
test, checkpoint promotion, or GPU work.

## Questions

1. Is the single-trajectory outcome-sampling regret estimator unbiased and
   strategically usable on the exact reduced Splendor reference from `046`?
2. Can the same estimator produce finite, reproducible, adequately sampled
   full-game targets without enumerating the traverser's legal actions?
3. Does a formal finite-game contract resolve `047`'s nontermination without
   making adjudicated trajectories dominate the corpus?

## Fixed estimator

- At every player node, let the target strategy be `sigma` and sample from the
  full-support behavior strategy
  `q = 0.9 * sigma + 0.1 * Uniform(legal actions)`.
- Sample exactly one chance/action trajectory per traverser pass. Never enumerate
  unsampled successor states.
- For a sampled terminal `z` passing traverser information set `I` through
  selected action `a*`, use the standard outcome-sampling counterfactual weight
  `W = u(z) * pi_{-i,chance}^sigma(z[I]) *
  pi^sigma(z[I]a*, z) / q(z)`.
- Emit the regret estimate `(1-sigma[a*])*W` for `a*` and
  `-sigma[a*]*W` for every other legal action. Compute products in log space;
  do not clip, normalize, or silently discard a finite target.
- A traversal contributes rows only after reaching a valid terminal under the
  applicable game contract. Record target probabilities, behavior probabilities,
  terminal provenance, log importance weight, phase, and semantic legal actions.
- Average-strategy fitting is outside `048`. Reduced-game convergence is judged
  using the current regret-matched profile solely as an estimator diagnostic;
  it is not authorized for deployment.

## Reduced-game correctness

- Reconstruct the unchanged `ReducedSplendorCFRGame` and its exact uniform-policy
  single-pass counterfactual regret increments.
- At seed `480048`, collect exactly `200,000` independent fixed-uniform outcome
  samples for each traverser. Compare mean estimated increments, counting
  unvisited information sets as zero, against every exact information set with
  maximum absolute exact increment at least `1e-4`.
- Require at least 95% eligible-information-set sample coverage, mean absolute
  component error at most `0.010`, maximum component error at most `0.050`, and
  root component error at most `0.010` for both players.
- Separately run adaptive outcome-sampling MCCFR for exactly `100,000` iterations
  from zero regrets at seeds `480046`, `480047`, and `480048`, alternating one
  trajectory for each traverser per iteration. Require zero illegal/unstable
  action sets, finite regrets, at least 95% exact information-set coverage,
  normalized positive regret below `0.05`, current-profile exploitability at
  most `0.030` per seed and median at most `0.015`.
- Repeat adaptive seed `480046` from zero and require a byte-identical canonical
  payload.

## Formal finite full-game contract

`048` studies a declared finite learning game; operational cutoffs are not
mislabelled as official Splendor terminals. In priority order, a trajectory ends
through:

1. ordinary engine completion;
2. `forced_pass_deadlock` after two consecutive states whose sole legal action
   is `PASS`;
3. `repetition_adjudication` on the fourth visit to the same complete simulator
   state modulo absolute turn count;
4. `no_progress_adjudication` after 60 consecutive non-progress actions;
5. `turn_limit_adjudication` on reaching turn 200.

The three learning-game adjudications use current score, then fewest purchased
cards, otherwise draw. They are valid utilities only for this explicitly altered
finite game. Report them separately and require ordinary completion plus forced
deadlock to account for at least 95% of trajectories. Never describe an
adjudication as an official game result or use it in a strength gate.

## Fixed full-game collection

- Load the exact clean `023` checkpoint and SHA-256 frozen in `047`. At every
  node, use softmax over its canonical semantic action scores at temperature
  `1.0` as `sigma`, floored only at `1e-12` before renormalization. Apply the
  fixed 0.1 uniform behavior mixture above.
- Collect exactly 256 fresh trajectories at engine seeds `480000--480255`, with
  traverser `seed % 2`, giving 128 trajectories per seat. Sampling RNG seed is
  `seed * 10 + traverser`.
- Preserve the information-legal state/history/action schema from `047`. Permit
  semantic legal-action counts in `[1,600]`, covering the 550-action state found
  by `047` without changing the variable-shape model contract.
- Exercise a deterministic capacity-512 Algorithm-R advantage reservoir at seed
  `480048`, repeat it from zero, and require byte identity and serialization at
  most 64 MiB. Every stored row must contain outcome-sampling provenance and
  must not contain privileged entities.
- Repeat the first 16 full trajectories from zero and require exact equality
  after excluding elapsed time.

## Frozen full-game quality gates

All are required in addition to the reduced-game gates:

1. Exactly 256 finite trajectories, 128 per traverser, with zero illegal actions,
   no state/action alignment error, no NaN/Inf, and depth at most 200.
2. At least 95% ordinary-completion plus forced-deadlock terminals; every other
   terminal follows the frozen learning-game adjudication contract.
3. At least 2,048 emitted advantage rows, including at least 128 each in turns
   `0--19`, `20--49`, and `50+`; legal-action counts remain in `[1,600]`.
4. Absolute importance weights have overall and phase-specific ESS ratio at
   least `0.01`; maximum log weight is at most `log(1e12)` and the 99th
   percentile absolute weight is at most `1e6`.
5. Comparing the first and second 128 trajectories, median log absolute weight
   differs by at most `1.0` and the 95th percentile differs by at most `2.0`.
6. The repeated 16-trajectory block and capacity-512 reservoir are deterministic;
   reservoir round-trip, provenance, privacy, and size checks all pass.

## Decision

A complete pass authorizes a separately preregistered neural outcome-sampling
fit. GPU rental is then justified for representation-compatible advantage
models, but average-strategy memory semantics, held-out calibration/order,
iteration weighting, legality, and gameplay gates must be frozen first.

If the reduced estimator fails, repair its math before any full-game work. If
the reduced estimator passes but full-game ESS/variance fails, do not buy more
compute or clip targets post hoc: outcome sampling is statistically mismatched
to the horizon, and the next CFR branch must use bounded subgame resolving with
an independently validated leaf contract. If finite-game adjudications exceed
5%, revise the game abstraction before learning. Raw legal `035` remains
champion, and BGA data is not used.
