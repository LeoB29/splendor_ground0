# AlphaZero adaptive replication 098 result

098 completed its frozen 200-game searched panel on 2026-08-13.

- exact 097 `visit_argmax_aux` candidate versus exact generation 2: 107 wins, 2 draws, 91 losses;
- score share: 54.0%;
- mean candidate margin: +0.645;
- seat 0 score share: 54.5%;
- seat 1 score share: 53.5%;
- paired-bootstrap CI95: `[46.75%, 61.25%]`;
- all 200 games completed normally.

The candidate passed both seat checks and the positive-margin check, but failed the required 55% score share and the strict paired-bootstrap lower-bound-above-50% checks. Frozen decision: `replication_overall_failed_keep_generation_2`. The raw diagnostic did not run. Exact generation 2 remains installed.

Interpretation: the auxiliary candidate may contain a small positive effect, but it did not demonstrate a promotion-grade improvement. The decisive visit-argmax auxiliary recipe is closed rather than retuned or reconfirmed. The broader AlphaZero architecture is not rejected; 099 moves to a scaled multi-generation sandbox loop using the ordinary AlphaZero objective.
