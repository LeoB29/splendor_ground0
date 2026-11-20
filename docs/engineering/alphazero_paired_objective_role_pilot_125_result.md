# AlphaZero Paired-Objective Start-Role Pilot 125 Result

Date completed: 2026-08-28

## Outcome

Experiment 125 completed normally and failed its fitting-only development gate.
It therefore stopped before protected confirmation, reachable-state safety, or
gameplay.

The pilot used the exact experiment-122 architecture from generation-11
identity, kept complete original/flipped pairs in every batch, and optimized
the experiment-124 joint objective. Only the 96-wide role input and final
entity-transformer block were trainable: 74,880 parameters. The remaining
281,666 parent parameters stayed bit-exact.

Decision:
`paired_objective_role_pilot_failed_development_no_confirmation`.

## Development result

The earliest minimum development joint objective was epoch 12. Ordinary
policy/value fitting improved substantially: the ordinary weighted objective
fell from 2.087908 to 2.029080, a 2.8175% relative improvement against the 0.5%
minimum.

However, the explicit role-difference targets barely improved:

- canonical policy-contrast error fell only 0.0451%, against 10% required;
- summed policy-contrast error fell only 0.0438%;
- value-contrast error fell only 1.3850%, against 10% required;
- raw policy change from flipping role was TV 0.000579, against 0.025 required;
- raw absolute value change from flipping role was 0.004634, against 0.05
  required.

The role input was nonzero with L2 norm 0.066304, and the final entity block
moved by L2 0.688148. The joint objective was finite and declined from
129.952411 to 129.779363. All scope, movement, and integrity checks passed, but
the four substantive role-uptake thresholds failed.

## Interpretation

The scale-native paired objective does not solve the role-uptake bottleneck
under this architecture and frozen optimizer schedule. Its coefficient made
the contrast term large in the objective and gave it a qualified gradient at
identity, but the trained model still produced almost role-blind outputs.

The role-input norm is close to experiment 122's 0.061051 after the same
12-epoch, `1e-4` AdamW schedule. This suggests an optimizer/parameterization
magnitude limit may be part of the problem: gradient-parity scaling controls
the raw gradient, while AdamW's adaptive normalization can greatly reduce the
effect of that scaling on actual parameter steps. This is an evidence-based
hypothesis, not yet a conclusion. The learned role direction could also be
poor.

Do not extend epochs, alter the learning rate, lower the gates, open protected
confirmation, play the checkpoint, or promote it as a post-hoc repair.

## Next step

Before another training pilot, freeze a fitting-only, no-training magnitude
audit using the saved experiment-125 checkpoint. Evaluate a small prospective
set of fixed multipliers on only its learned role direction, without writing a
new checkpoint. Measure ordinary loss, paired policy/value contrast errors,
and raw role sensitivity on the unchanged train/development split.

This directly distinguishes two cases:

1. If a larger fixed magnitude substantially improves both paired errors while
   preserving ordinary fit, the next minimal identity-preserving pilot should
   use a fixed-gain zero-initialized role channel.
2. If scaling does not help, the learned direction or its injection point is
   wrong, and another optimizer-only retry is not justified.

The audit must not read protected confirmation, launch training, generate
search targets, write a candidate checkpoint, or run gameplay.

## Guard behavior

- Preflight validated all 448 fitting pairs and exact generation-11 identity
  over 896 orientations.
- Both paired loss components reached the role input and final block in the
  gradient audit.
- Existing fitting targets were reused; no new search target was generated.
- Protected states were not reconstructed and the protected target manifest
  was not accessed.
- No confirmation, safety evaluation, arena, installation, promotion,
  self-play, or automatic continuation ran.
- Focused experiments 118, 124, and 125 tests pass 14/14. The full repository
  suite passes 825/825 in 224.16 seconds.

## Artifacts

- Frozen protocol SHA-256:
  `d5f5cced1e74255f981f6e1c8471ccae49244c7ee9bc5c212753e54a4988ed4d`.
- Candidate checkpoint SHA-256:
  `f41738c06b200161ba652903a83a214317539c35d388edcf66ec90c50ceaa54c`.
- Training metrics SHA-256:
  `c19b8650b1301e480fc6f95c6cb29d88088f9678226d1615a0419960dea679b5`.
- Raw result SHA-256:
  `7e11e49ed022bdb81a87a1f9be2e8c1c04a2c8e91be33297d56f7b0e7068c3f5`.
