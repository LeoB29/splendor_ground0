# AlphaZero Consolidation Repair 108 Result

Date completed: 2026-08-20

Experiment 108 reproduced the exact 107 parent and candidate holdout metrics
within `1e-5`, then evaluated five prospectively fixed parameter interpolations
and the policy/value hybrid without training. Four coherent interpolations
passed the unchanged retention gate. The predeclared selector chose lambda
`0.25`, the passing coherent model with the best aggregate held-out objective.

The selected single-network checkpoint improved objective loss relative to
gen11 on every held-out corpus: aggregate `-0.00157860`, gen7 `-0.00172502`,
gen8 `-0.00217852`, gen9 `-0.00087204`, and gen10 `-0.00175373`. Its checkpoint
SHA-256 is
`251c0ddca69925090bbe21fae75cb2616b6fdd63a88b9e2b34cfdb102c284552`.

Those offline improvements did not transfer to fresh gameplay:

- raw lambda-0.25 versus raw gen11: 98-1-101, 49.25%, margin -0.215,
  seats 53.5%/45%, paired CI95 `[44.75%,53.75%]`;
- searched lambda-0.25 versus searched gen11: 98-5-97, 50.25%, margin +0.125,
  seats 60%/40.5%, paired CI95 `[44%,56.5%]`.

All 400 games completed normally in 4 hours 39 minutes with empty stderr. The
searched practical gate failed on share and second-seat performance. No `035`
game or installation occurred. The full consolidated-policy/gen11-value hybrid
passed the offline retention gate but was ineligible while a coherent repair
passed; its gameplay transfer remains the final unresolved consolidation
hypothesis.

Final decision:
`coherent_repair_retained_but_internal_gameplay_failed_no_install`. Raw result
SHA-256:
`57392ef95e50b54ad239b870213240e37935a1a5a037cacb0442cf42f344aca4`.
