# AlphaZero Generation-11 Replay Consolidation 107 Result

Date completed: 2026-08-19

Experiment 107 trained a valid generation-12 consolidation candidate from
exact gen11 on all 63,794 positions in the four clean gen7--gen10 corpora.
Training stopped after four epochs in 14 minutes 53 seconds. The checkpoint
and metrics passed every engineering and lineage check; stderr was empty.

The candidate failed every predeclared held-out retention check. Objective
loss deltas, candidate minus gen11, were:

- aggregate: `+0.00611788`;
- gen7: `+0.00165868`;
- gen8: `+0.00386813`;
- gen9: `+0.00961813`;
- gen10: `+0.00818447`, above the allowed `+0.005` regression.

The failure was asymmetric. Aggregate held-out policy loss improved by
`-0.00325870`, including gen7 `-0.01143261` and gen8 `-0.00775062`, while
held-out value MSE worsened by `+0.01875317`. Training objective improved but
validation objective worsened monotonically after the first epoch. This is
evidence of value-side overfitting or update overshoot, not evidence that the
broader replay policy signal was uniformly harmful.

Final decision:
`consolidation_retention_gate_failed_no_gameplay_no_install`. No arena or
`035` game opened and no checkpoint was installed. Exact gen2 remains
installed. Raw result SHA-256:
`d0bba71d37eb6883096f7c532f5abd25cd0bbcf5623c9fc10980dbb24a52e157`.
Candidate checkpoint SHA-256:
`c21c0ed3765e09820eb415c7369d0be0733913915e29d288db825cbcf9461c89`.
