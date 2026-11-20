# AlphaZero Start-Role Joint Pilot 118 Result

Date completed: 2026-08-27

Experiment 118 completed at its frozen role-only selection gate. It generated
all 448 paired exact-gen11 teacher targets (896 oriented rows) at 1,024
simulations from the sealed 224-game training partition, then trained only the
two-entry, 96-wide start-role embedding. The exact gen11 parent remained frozen;
the candidate had 356,642 total parameters and exactly 192 trainable parameters.

The paired teacher signal reproduced the material start-role dependence seen in
117 on a larger fitting panel:

- mean visit-distribution TV: 0.218436;
- visit-argmax action change: 20.759%;
- mean absolute search-root-value change: 0.275106;
- all 448 pairs used exactly 1,024 simulations in both orientations.

Training selected epoch 6, but uptake was much too small to pass the prospective
gate. Validation weighted objective moved from 2.087908 at epoch zero to
2.086905, a relative improvement of only 0.0480% against the required 0.5%.
Policy loss moved from 1.907804 to 1.907612 and value MSE from 0.360207 to
0.358586. Both embedding rows were nonzero (norms 0.102866 and 0.098403), but
their distance was only 0.015974. Thus most learned displacement was common-mode
rather than a strong role contrast.

Because selection failed, the runner correctly did not open the protected 117
confirmation panel, the 1,956-row reachable-state safety audit, or the
conditional 400-game arena. It performed no self-play, external `035`
comparison, installation, promotion, or automatic continuation. Exact
generation 2 remains installed.

The first fitting attempt stopped before any optimizer step because the loader
rebuilt every syntactically legal action while search targets use one
representative per canonical successor. The implementation was corrected to
reconstruct the exact search action set, a regression test was added, and all
448 already sealed target pairs validated as 896 oriented rows before the run
resumed. No teacher target or frozen protocol identity changed.

Decision:
`minimal_role_bridge_failed_direct_uptake_review_richer_interaction_no_gameplay`.

The result closes the constant global-row role-embedding design as an
insufficient direct repair under this protocol. The next step should be a
no-training representation/gradient design audit of a slightly richer but still
identity-preserving interaction (for example, antisymmetric role conditioning
at the policy/value readout paths), not more epochs, an arena, or promotion of
this checkpoint. Any new pilot requires a separately frozen protocol and
explicit launch authorization.

Raw result SHA-256:
`4846b79e1fbcd6344e759155befc469463a9100423f8785ff0e352fa9127fbc1`.
