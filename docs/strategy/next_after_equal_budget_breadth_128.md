# Next step after the fresh equal budget breadth confirmation

2026-10-03. Recommendation only. Run one inexpensive saved-data test of adaptive
simulation allocation before funding the next complete-game comparison. The
question is whether concentrating some simulations on promising moves improves
the already-confirmed uniform eight-candidate rule at the same budget. No new
experiment is authorized by this recommendation.

AGENT_CURRENT.md was read first. Independent reviews considered direct gameplay,
adaptive allocation, the learning plateau, statistical resolution and cost.
Confidence in this priority is moderate. No empirical resampling, inference,
training, simulation, replay, test or implementation ran during this review.
Only this new recommendation and the unbound handoffs are changed. All frozen
evidence and the intentionally dirty worktree remain intact.

## What the latest result changes

The fresh 360-position confirmation passed all prescribed verification. Uniform
eight-by64 search improved on the four-by128 anchor by +0.631510 percentage
points in conditional decision score, with paired 95% interval
[+0.021701, +1.254340]. The point estimate clears the +0.5pp investment hurdle;
the interval narrowly excludes zero and does not establish a true gain above
+0.5pp. It corroborates the earlier reused-data result of +0.560997pp.

The eight-by128 teacher beat the anchor by +1.369358pp. Uniform breadth was
0.737847pp worse than that teacher, with interval [-1.163194, -0.323351]. Breadth
helps at the smaller budget, but reducing precision loses useful information.
This is evidence for testing allocation, not proof that early elimination can
recover that loss.

The expensive broader teacher already has gameplay evidence: 56.2189% score
against the anchor in 402 games, paired interval [51.6169%, 60.6965%]. The cheaper
uniform rule still lacks its own full-game test. Earlier 134/139 results warn
that conditional improvement need not survive repeated replanning, a planning
opponent and changed reached positions. Neither current positional result
predicts a particular arena win rate.

## Why one cheap allocation test precedes the arena

A direct uniform-versus-anchor arena is justified now. The proposed 402 games
would address the most important unresolved performance claim and should remain
the fallback. Read-only cost review estimates roughly 60–75 hours, including a
separate operational probe and prescribed verification, on eight CPU workers.
This is an extrapolation requiring a fresh outcome-blind forecast, not an ETA.

A single arithmetic-only allocation test has useful option value before that
investment. A miss costs little execution time and leaves the uniform candidate
available. A pass could lead to a better candidate for approximately the same
eventual arena cost, but only after fresh confirmation of its incremental gain.
There is no need to certify every intermediate player in games before testing
one inexpensive improvement. Conversely, repeated allocation sweeps would defer
the decisive gameplay question indefinitely; they are expressly outside this
recommendation.

The strongest objection is that an early noisy ranking may eliminate the best
move permanently. An older four-candidate halving screen found only +0.108805pp
at a different budget and on reused data. It is a precedent for feasibility,
not evidence that this new schedule will clear its hurdle. The new test should
be able to reject this specific idea cheaply.

Another nearby learning recipe has lower priority. Broader winner imitation
and the outsider-specific objective both changed models and improved their
development criteria without passing fresh proposal-utility gates. This does
not rule out learning: the original learned proposer improved gameplay. It
does mean that the evidence currently identifies simulation allocation more
concretely than another loss, dose or arbitrary architecture change.

## One fixed adaptive rule

Keep the original three 161 fits, original anchor candidates, two additional
ranked candidates, two identity-uniform candidates, lawful world distribution,
canonical representatives and greedy035 terminal continuations. Candidate
generation retains its existing 128-world convention. No model changes.

Compare three arms, with full support meaning at least eight canonical legal
groups:

| Arm | Selection at full support | Continuations |
| --- | --- | ---: |
| Anchor A | Four candidates, 128 each | 512 |
| Uniform U | Eight candidates, 64 each | 512 |
| Adaptive S | Eight candidates, 32 each; retain four and give each 64 more | 512 |

S retains the four highest first-stage mean scores. Break every tie by original
candidate order. Select the final move only among these four, using their
combined 96 outcomes each; final ties use original candidate order, not the
first-stage ranking order. Retaining the initial
32 outcomes is part of the frozen player. It is not a request to rank candidates
using held-out results or test several survivor counts.

For fewer than eight canonical groups, S delegates exactly to the existing U
allocation. For five, six and seven groups, U divides 512 continuations evenly
with the existing identity-assigned remainders; for at most four groups it
matches A's 128 outcomes per candidate. This preserves the validated handling
of short supports and avoids inventing a second allocation mechanism. Keep all
360 roots, including these cases. Their S-minus-U differences are exactly zero.

## Fair comparison using already saved outcomes

Use all 360 fixed main roots from outputs/proposal_breadth_128, authenticated
against its closed evidence and the completed saved_equal_budget_breadth screen.
The source has 128 common worlds for every offered candidate and independent
extension worlds for the original anchor candidates. No new rollouts are needed.

Reuse the completed saved uniform screen's exact 64 frozen common-world,
extension-world and remainder permutations per root. This preserves U and A's
selection choices and avoids choosing another partition scheme. Here first,
next and last refer to each partition's common-world permutation:

- At full support, S uses the first 32 worlds for all eight candidates and the
  next 64 for its four survivors. U uses the first 64 common worlds for all
  eight. A retains its original first 64 common plus 64 independent extension
  worlds per original candidate. Lock all three choices, then assess each on
  the last 32 common worlds, which none used for selection.
- For five, six and seven groups, S equals U. Preserve the prior U/A selection
  construction and its independent assessment suffix of 25, 42 and 54 worlds,
  respectively. For at most four groups all three choices coincide; retain
  exact zero paired differences without inventing an assessment score.

Preserving the old U/A choices does not preserve their published scores: at
full support the assessment suffix is now 32 worlds rather than 64. Recompute
every new contrast on the same new held-out suffix. Do not import the earlier
+0.560997pp as this study's U-minus-A control result or overwrite that result.

The latest fresh confirmation saved assessment outcomes only for its three
locked choices. Those sparse outcomes cannot generally score a newly chosen
adaptive action. The older complete outcome matrices and independent extension
block are what make this arithmetic-only design possible.
Do not add an eight-by128 teacher control evaluated on the same assessment
suffix: that teacher would already have used those outcomes to select its move.

Keep each partition's selection separate from its assessment. Average the
64 paired differences within each root, then weight all roots equally. Use
50,000 paired bootstrap resamples within the 18 original fit/seat/phase cells,
keeping all three arms together and freezing a new bootstrap seed before the
analysis. Roots, not partitions or worlds, are the resampling units. Preserve
exact rational arithmetic for unequal assessment denominators.

These are reused development data. New arithmetic and disjoint within-partition
assessment do not erase prior use of the corpus or create fresh confirmation.

## Fixed decision and stopping rules

Advance this exact adaptive rule only if all conditions hold:

- S minus U has mean gain at least +0.25pp and paired 95% lower endpoint above zero.
- S minus A has mean gain at least +0.5pp and paired 95% lower endpoint above zero.
- All prescribed arithmetic, provenance and statistical checks pass.

The +0.25pp incremental hurdle is a prospective investment judgment: half the
existing entry hurdle, required on top of an already-confirmed useful rule.
It is not a power calculation, prediction of gameplay strength, or proof that
smaller benefits are worthless. Report U-minus-A as context, with its newly
computed assessment. Do not select a fit, phase or support subgroup afterward.

A pass supports proposing one fresh S/U/A confirmation with independent
assessment and the same incremental and anchor hurdles. It does not authorize
an arena or permit replacing U based on reused evidence alone. Only a fresh
pass would justify recommending S rather than U for the complete-game test.

A miss ends this adaptive proposal. Do not tune 32, 64, the survivor count,
candidate mixture, short-support fallback, partitions or thresholds. Recommend
the already-supported U-versus-A arena as the next investment, subject to its
own authorization and operational forecast. A later inconclusive or failed
fresh adaptive confirmation has the same fallback; it does not erase U's
completed confirmation. No successor launches automatically.

## Cost and verification

Expect minutes of CPU execution once implemented and checked; preparation is
additional. The preceding saved-data screen completed in 117.86 seconds including
its independent verification. Allow roughly 5–20 minutes provisionally for this
new three-arm calculation, with a one-hour execution ceiling. No new model
inference, simulation, source games, training, GPU use or trajectory replay is
needed. Use hidden execution and no recurring monitor.

Authenticate the bounded source chain and inherit its existing prescribed
trajectory-audit evidence. Independently reconstruct every partition's budgets,
survivors, allowed outcome accesses, exact means, locked choices, assessment
differences, root averages and final statistics. Include fixtures for first-stage
and final ties, discarded candidates, short supports, changing denominators,
selection/assessment leakage and missing or tampered records. Do not rerun
mature simulations, historical fits or a full repository test suite for this
arithmetic-only study. Follow the accepted verification policy.

If later authorized, implement in new files, proposed tools/saved_adaptive_breadth_128
and run_saved_adaptive_breadth_128.py, with a new protocol and output directory.
Preserve every frozen predecessor. This recommendation itself launches nothing.

## Implications for the programme

This is an attempt to obtain more useful planning for the existing simulation
budget. It could make future teacher-data collection and experiments cheaper.
The eight-by128 teacher remains the strongest gameplay-validated breadth
variant; a cheaper candidate is not automatically its replacement. Equal
continuation counts also do not establish equal wall-clock cost.

The larger obstacle remains repeatable transfer from stronger search into
stronger learned guidance. This experiment does not solve that obstacle or
establish broad or superhuman strength. Complete-game transfer and independent
strong opposition remain necessary, with declared resources. The reason for
this one detour is its unusually low incremental cost and a concrete precision
gap, not a claim that allocation alone will reach the programme's final goal.

## Evidence

- outputs/equal_budget_breadth_128/report.md and final_status.json
- outputs/saved_equal_budget_breadth/report.md and final_status.json
- outputs/proposal_breadth_arena_128/report.md and operational_decision.json
- outputs/proposal_breadth_learning_128/report.md
- outputs/proposal_specialist_128/report.md
- outputs/learned_proposal_128/report.md
- docs/engineering/saved_halving_screen_result.md
- docs/strategy/next_after_saved_equal_budget_breadth.md
- docs/strategy/next_after_proposal_specialist_128.md
- docs/strategy/independent_superhuman_review_after_134_137.md
- docs/engineering/verification_policy.md
