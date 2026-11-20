# AlphaZero Fresh-Dose Head Attribution 111 Result

Date completed: 2026-08-25

Experiment 111 completed normally in 10h07m. It trained no checkpoint, generated
no self-play, and left the exact generation-2 installation unchanged.

## Gameplay result

Each functional variant played exact generation 11 on the same 200 fresh paired
seeds (400 games), with 256-simulation search and both seats.

| Candidate path | W-D-L | Score share | Mean margin | Seat 0 / seat 1 | Paired CI95 |
|---|---:|---:|---:|---:|---:|
| full high-dose policy + value | 221-2-177 | 55.50% | +0.8875 | 61.50% / 49.50% | 50.375%--60.50% |
| high-dose policy + gen11 value | 194-1-205 | 48.625% | -0.0225 | 59.00% / 38.25% | 43.75%--53.375% |
| gen11 policy + high-dose value | 225-4-171 | 56.75% | +0.8150 | 64.75% / 48.75% | 51.875%--61.625% |

All games terminated normally. None passed the full practical gate because the
second-seat share was below 50%. Relative to the full high-dose candidate,
restoring gen11 value changed paired score share by -6.875 percentage points
(CI95 -14.0 to +0.125), while restoring gen11 policy changed it by +1.25 points
(CI95 -5.25 to +7.75). The preregistered strict attribution rule therefore did
not identify a single primary blocker: the value comparison narrowly included
zero and the policy comparison was not positive.

## Offline diagnostics

On the exact 7,986-row common holdout, high-dose policy cross-entropy improved
by 0.01068 and target-top agreement improved by 1.62 points. Its value MSE,
however, worsened from 0.98011 to 0.99495; a fixed score/tempo baseline reached
0.94653. Plain holdout MSE therefore did not explain the gameplay usefulness of
the high-dose value path.

The aggregate shared-trunk policy/value gradient cosine was positive at both
checkpoints (gen11 0.214; high dose 0.313). Negative-cosine batches occurred in
25.0% and 37.5% of batches respectively. This does not support a simple global
gradient-opposition explanation.

## Decision and lesson

Decision:
`no_clear_head_attribution_review_representation_capacity_or_teacher_no_install`.

The experiment does not authorize an architecture change by itself. It does,
however, expose a strong, directionally consistent causal clue: with the
high-dose policy computation held fixed, the high-dose value computation was
about 6.9 points better than gen11 value. Because 111's primary rule was framed
as identifying a harmful update, that beneficial effect was not itself a pass.
The correct follow-up is a prospectively frozen direct duel between those two
otherwise identical functional paths. No installation or promotion follows
from 111.

Raw result SHA-256:
`02fbd6cbf2a971c06adc435f1b537cc7aa24ac5d4251b7783efe8de943a7e2aa`.
