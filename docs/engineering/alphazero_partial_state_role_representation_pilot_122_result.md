# AlphaZero Partial State-Role Representation Pilot 122 Result

Date completed: 2026-08-28

## Outcome

Experiment 122 completed normally and stopped at protected confirmation. It is
the first start-role-aware design in experiments 118--122 to pass the frozen
development gate, but it did not demonstrate that the gain came from learning
start role.

The model added one signed 96-wide start-role input to the canonical global
entity and jointly trained the final 74,784-parameter entity-transformer block.
Exactly 74,880 parameters were trainable; the other 281,666 parent parameters
remained bit-exact. No new fitting target or self-play was generated.

## Preflight and development fit

Preflight validated all 896 sealed fitting rows. Live epoch-zero logits and
values matched exact generation 11 across 1,024 orientations with maximum
absolute difference zero, as did checkpoint reload. The exact trainable-name
scope passed. Both policy and value produced nonzero gradients in the role input
and final entity block, no frozen parameter received a gradient, and the audit
did not mutate parameters.

Epoch 12 was selected. Development weighted objective improved from 2.087908 to
2.030171, a relative improvement of 2.7653% against the prospective 0.5% gate.
Policy loss improved from 1.907804 to 1.886492 and value MSE from 0.360207 to
0.287358. The role-input norm was 0.061051, final-block movement was 0.690659
L2, and frozen-parent maximum movement was exactly zero. Every development
selection check passed.

## Protected confirmation

The protected exact-search panel reproduced experiment 117 within `1e-9`: mean
visit TV 0.234634, top-action change 18.75%, and mean absolute search-root-value
change 0.312712.

Candidate 122 improved value MSE overall against the strongest role-blind value
baseline by -0.053461 with CI95 [-0.084287, -0.022881]. However, its start-role
subset regressed by +0.032660 while its non-start subset improved by -0.139582.
Policy CE regressed by +0.026795 against candidate 114, including regressions in
both roles; its CI95 was [-0.001434, 0.055154]. Weighted objective improved only
-0.001636 and its CI95 upper bound was +0.031638.

Most decisively, changing only the public start role changed the candidate's raw
policy by mean TV 0.000418 and value by mean absolute 0.003480. These are far
below the prospectively required 0.025 and 0.05, and their CI95 lower bounds
0.000382 and 0.003299 are below the required 0.01. Nine confirmation checks
failed. The candidate therefore did not qualify for reachable-state safety or
gameplay.

## Interpretation and next step

The partially trainable state representation has enough capacity to learn the
common deep-search targets, but the ordinary per-orientation policy/value loss
still gives too little effective weight to the paired role-dependent difference.
Aggregate development improvement was therefore not evidence of role uptake.
Do not enlarge or tune candidate 122, and do not weaken the protected gates.

The best next step is a no-training paired-objective audit on only the sealed 118
fitting panel. Measure and normalize explicit original-versus-flipped policy and
value contrast losses against the ordinary objective, verify that they deliver
usable gradients into both the role input and shared state block, and
prospectively define a development role-contrast gate before any further pilot.
The protected 117 panel must remain unused for that design work.

Decision:
`partial_state_representation_failed_protected_confirmation`.

No reachable safety evaluation, gameplay, external `035`, installation,
promotion, self-play, or automatic continuation ran. Exact generation 2 remains
installed.

Focused experiments 118--122 tests pass 19/19, the 102--122 AlphaZero band
passes 92/92, and the full repository suite passes 815/815 in 231.36 seconds.

## Artifacts

- Frozen protocol SHA-256:
  `4e4c179eaae1f2887e66b95b0f3eba101d6d299f331385a197da3c7dfd1b3550`.
- Preflight SHA-256:
  `7ad3764324e273f8df6064231565fee2497759195ea1ee58e1f5fd3cf4a453ca`.
- Candidate checkpoint SHA-256:
  `843e2f0d69cea9f799491552259bb867357a4a25b3faa08b78f9ef13e88961d6`.
- Training metrics SHA-256:
  `e044370ac7b2931630c6c99a026efa7e6f8dae468332c01fd34e561b99dcc514`.
- Raw result SHA-256:
  `a8eadd8650561ef3f3c20dbf18bb409add47aa862a49e2006b724518c23706d1`.
