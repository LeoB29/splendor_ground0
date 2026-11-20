# AlphaZero Start-Role Representation Audit 117 Result

Date completed: 2026-08-27

Experiment 117 completed normally in 3m36s. It reconstructed all 32 untouched
holdout games and selected exactly two late pre-round-end states per game,
yielding 64 balanced counterfactual pairs (32 observations from each start-role
class).

Every integrity check passed. In each pair, flipping only `start_player`
changed the canonical public-belief key while preserving the v1 entity tensor,
legal semantic action set, raw gen11 logits, and raw gen11 value. Both searches
used exactly 1,024 simulations. This confirms that v1 aliases a public variable
that can still affect downstream search through successor states and terminal
round semantics.

The coupled exact-gen11 teacher comparison found:

- mean visit-distribution TV: 0.234634, game-cluster CI95
  0.171844--0.305038;
- visit-argmax action change: 18.75%, game-cluster CI95
  9.375%--28.125%;
- mean absolute search-root-value change: 0.312712, game-cluster CI95
  0.213466--0.428585;
- mean signed original-minus-flipped root-value change: -0.031250.

The effect appeared in both original-role strata. For non-start-player
observations, mean TV was 0.257721, top-action change was 21.875%, and mean
absolute value change was 0.281462. For start-player observations, the
corresponding values were 0.211548, 15.625%, and 0.343963.

Both frozen interpretation gates passed: policy sensitivity passed all three
TV/top-action requirements, and value sensitivity passed both magnitude and
confidence-bound requirements. The descriptive holdout audit also reproduced
the earlier offline improvements: relative to gen11, candidate 114 improved
policy/value/weighted-objective losses by
-0.036725/-0.010119/-0.041785, while candidate 116 improved them by
-0.031705/-0.001153/-0.032281 across all 1,956 rows.

Decision:
`teacher_role_sensitivity_confirmed_review_identity_role_aware_pilot_no_training`.

The result authorizes design review of a minimal identity-preserving
start-role-aware joint policy/value pilot. It does not by itself authorize
training, new self-play, a gameplay arena, an external `035` comparison,
installation, promotion, or automatic continuation. Exact generation 2 remains
installed.

Raw result SHA-256:
`a95081b4170604f5e7a162a532016c0eb2c4b5a67b36e31643fe5c50caf2a283`.
