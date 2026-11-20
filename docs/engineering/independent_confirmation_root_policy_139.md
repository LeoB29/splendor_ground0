# Experiment 139: independent confirmation of expanded-root proposals

Frozen before panel selection or new continuation outcomes on 2026-09-06.
The owner authorized preparation and execution of the proposed propose/check/test
experiment. This is a bounded offline policy experiment, with no training,
promotion, installation, full-game arena, or automatic successor.

## Question and population

Can an independently sampled confirmation step turn expanded-root proposals into
better choices under held-out stochastic-035 evaluation? Experiment 138 identified
material world-sampling overfit; it did not establish a successful remedy.

Use the unchanged 086 selection, original discovery choices, and legal 035
checkpoint, with the same source hashes as 138. Reconstruct the 60 experiment-138
IDs from its deterministic ID-only selector and exclude them. From the remaining
40 states per phase/seat cell, choose ten by SHA256 of UTF-8
`139-selection-v1|SOURCE_SEED|TURN_INDEX|ACTOR_SEAT|PHASE`, then source seed and
turn. This produces 60 different positions from 60 source games, ten in each of
six cells. Inclusion depends on identifiers only. Attach original 086 discovery
champion/expanded action keys only after selection; retain coincident choices.
No 138 returns, 087 outcomes, or human records are inputs.

This reuses existing 086 discovery proposals: it is not a new discovery search,
a fresh source-population study, or a reproduction of all live 084 decisions.
The default action is the original four-root 084-contract discovery choice.

## Three stages and four policies

1. **Propose:** reuse each original expanded 086 discovery action and its original
   four-root 084 comparator, frozen in the state manifest.
2. **Check:** evaluate those two fixed actions in eight blocks for each arm:
   `same_world_check` reuses the original three discovery worlds;
   `fresh_world_check` samples three new observer-legal worlds per block.
   Both use four stochastic-035 continuations per opponent reply, identical
   reply limits and paired continuation seed slots. Thus both checks have the
   same scheduled continuation budget; any differences due to terminal roots
   or legal reply counts are accounted for explicitly.
3. **Test:** freeze both policies' choices for all 60 states before any test
   continuation runs. Evaluate both original actions on eight further independent
   three-world/four-continuation blocks. These worlds and continuation seeds
   are disjoint from discovery and both checks. No policy can use test returns
   to select a move. Computing both actions allows a paired comparison of every
   policy using the same test returns.

For either check arm, define eight block gains as expanded minus default
normalized final-score margin. Switch to the expanded move iff the moves differ,
the mean gain is at least +0.025 (half a prestige point before clipping), and
`mean - 1.895 * sample_sd / sqrt(8) > 0`. Otherwise retain the default.
The constant is a fixed screening penalty motivated by a one-sided t threshold
with seven degrees of freedom. With only eight non-Gaussian blocks this is a
heuristic decision rule, not a guaranteed per-position confidence statement.
There is one frozen rule, no parameter sweep or adaptation after results.

The four policies are: always default (`champion`), always expanded
(`unfiltered`), same-world confirmation (`same_world`), and independent-world
confirmation (`independent`). Both confirmation policies have identical
proposal/check budgets and thresholds. The default/unfiltered policies have no
confirmation cost; report that additional cost rather than claiming cost parity
with them. This tests the confirmation mechanism, not a new world-allocation
recipe across every candidate root.

## Engine, worlds, and seeds

Reuse the verified 138 branch construction, observed legacy loop-safe rollouts,
and information-set-grouped opponent minimum, without modifying those helpers.
The root observer's legal information is fixed; fresh worlds resample the full
unknown pool, hidden reserve possibilities and deck orders. Regenerate legal
opponent replies after each root/refill. Shared information sets must have
identical reply sets and a shared minimizing reply. Terminal roots and fewer
than four legal replies are retained with exact accounting, never excluded.

Original worlds use the exact 086 schedule. Fresh-check root index is
`1_390_100_000 + sample_index*100 + block*10`; test root index is
`1_390_200_000 + sample_index*100 + block*10`, with the unchanged belief
function's seed start of 49,000,000. Blocks are 0..7 and worlds 0..2.
Continuation seed is
`1_390_000_000_000_000_000 + stage*100_000_000 + sample_index*100_000 + block*1_000 + world*10 + replicate`,
where stage=0 for both checks (paired) and stage=1 for test. Replicates are 0..3.
These namespaces are disjoint from 086/138 and from runtime probes.

Each block uses three worlds, up to four distinct legal 035 replies, and four
stochastic-035 continuations per reply (epsilon .1, temperature .5, top-3,
400-turn cap, 64 fallback triggers, score margin /20 clipped to [-1,+1]).
Resolve each block separately before averaging; never pool worlds before the
nonlinear opponent minimum. Record rule-correct WDL, score and card-count
tiebreaks, and explicit adjudication reasons from the same trajectories.
All continuations must resolve, no hard caps are allowed, and adjudications
must be <=0.1% overall and separately for each of the three stages/arms.

Maximum new primary work is 60*3*8*2*3*4*4 = 138,240 continuations.
Coincident actions share computation. The fixed eight blocks are still evaluated
for coincident choices so that schedule, accounting and zero gains are explicit.
Shared evaluation across policies is diagnostic reuse, not measured deployment
speed. Record structural scoring counts, continuations and per-stage time.

## Primary estimates, decisions and precision

Use all 60 positions, including defaults and coincident proposals, with equal
state/game weights. Primary test contrasts in normalized margin are:

- independent minus champion (does the resulting policy improve over default?);
- independent minus unfiltered (does filtering improve the original proposals?);
- independent minus same_world (does independence help beyond extra computation?).

Also report same_world minus champion, unfiltered minus champion, all policy
values, switching counts, and secondary WDL under margin-minimizing replies and
under separately WDL-minimizing replies. These are policy-conditioned offline
values, not arena win rates or optimal values. Phase, seat and private-reserve
breakdowns are descriptive and cannot select another rule.

Use 50,000 paired percentile bootstrap resamples, NumPy default_rng seed
139300000. Sort by selection_cell then source_seed, resample ten games within
each of six cells, and keep each state's choices and test estimates together.
No individual rollout or block is an independent population observation.

The single full-success decision requires all of:

- integrity passes;
- independent policy actually changes at least 6/60 default decisions;
- independent-minus-champion mean >=+0.01 and its CI95 lower bound >0;
- independent-minus-unfiltered and independent-minus-same_world CI95 lower
  bounds both >0.

This is a conjunctive claim: all three comparisons must pass; do not choose the
best contrast after seeing results. Report individual intervals without treating
them as a simultaneous confidence region. A positive gain versus default alone
does not establish the independent-check remedy or outperform unfiltered search.

Decision priority: integrity failure; insufficient switch dose (<6); full success
(`independent_confirmation_policy_improvement_supported`); independent-minus-
champion CI upper <=0 (`no_positive_default_improvement_supported`); positive
default gain meeting its magnitude/CI gates but not all comparative gates
(`default_improvement_supported_confirmation_benefit_unproven`); otherwise
`independent_confirmation_policy_inconclusive`. No failure triggers extra states,
threshold changes, training, or an arena. A full pass supports designing a later
gameplay test against 084; it does not establish a stronger deployed agent.

This is a mechanism pilot, not promised to be powered for small gains. With
60 independent games, normal CI half-width is approximately .0127/.0253/.0506
for state-contrast SD .05/.10/.20. Conjunctive success may be difficult; uncertainty
is an acceptable result and must not be relabeled as equivalence or architecture
failure. The existing stochastic-035 continuation-policy limitation remains.

## Preparation, execution and sealing

Verify frozen source/checkpoint hashes, source binding and disjoint panel IDs;
snapshot round trips; root information/inventory and hidden-deck permutation
invariance; original branch reproduction; seed separation; grouped minimum;
screening/held-out routing; complete accounting and resume rejection. Run focused
and full tests before protected evaluation. Freeze protocol, full src Python
dependencies, runner, test, verifier and state manifest in preparation identity.

Eight runtime-only probe continuations on an ID-selected state outside both
138 and 139 panels may use seed `1_399_000_000_000_000_000 + probe_index`.
Do not serialize probe outcomes. The preparation ceiling is 36 projected serial
CPU-hours (mean probe time * maximum work *2 engineering allowance); use ten
CPU workers, one Torch thread each. Runtime during the protected experiment is
telemetry. Stop for integrity failures or genuine hangs, not unfavorable outcomes.

Use a kernel process lock, atomic files and deterministic resume identities.
Save complete check-state records, then an immutable all-state choices manifest,
then complete test-state records. Test workers receive the frozen pair of original
actions but no check outcomes; policy routing occurs only in final reporting.
No test-stage continuation may begin until all choices are frozen. A resume must
validate hashes and reject altered or incomplete records. Do not display partial
returns, switching counts or contrasts; only progress/time/integrity until all
60 test states complete. Independently reconstruct final returns, screening
decisions, grouped values, policy routing and bootstrap intervals before reporting.

Preserve exact 084 as research champion, 035 as fast champion, all prior results
and the intentionally dirty worktree.
