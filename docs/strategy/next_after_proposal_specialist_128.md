# Next step: test broader search at the existing simulation budget

2026-10-03. Recommendation only, following the completed specialist proposal
pilot. Run one saved-data development screen of eight candidates with 64
simulations each against four candidates with 128 each. Both spend 512
continuations at full support. Use the original successful 161 proposals,
unchanged candidate mixture and greedy035 continuations. No new training or
simulation is needed for this screen.

AGENT_CURRENT.md was read first. Three independent reviewers considered the
latest learning results, frozen-model shortlist diagnosis, uniform breadth,
adaptive allocation and wider programme priorities. After checking the saved
data construction below, all three favored this inexpensive first screen.
Confidence is moderate: agreement about information per cost is not evidence
that the proposed player will win. This review launched no experiment, model
inference, fit, rollout, statistical resampling or tests. Only this new note
and the unbound handoffs are changed; frozen evidence and dirty work remain.

## Why change direction now

The specialist gained only +0.0521 percentage points over the anchor on the
fresh position panel, paired95% interval [-0.2181,+0.3331]. Its difference from
the matched overall-winner learner was -0.1204, interval [-0.3331,+0.0770].
All prescribed verification passed. Its primary upper endpoint is below the
fixed +0.5-point investment hurdle. This closes that fixed recipe, without
establishing exact zero or a universal limit on learning.

Several useful distinctions now matter:

- The first learned proposer worked: learned128 scored65.796% against the
  original128 planner. The evidence contradicts a blanket claim that learning
  cannot improve this search.
- Repeating winner imitation did not establish a useful second generation.
  Gap-weighted ranking, broader comparisons and the latest exact outsider
  objective also failed their respective practical screens. Broader data alone
  and this role-specific target alone were insufficient under the tested dose,
  models and selection rules. Another neighboring loss or dose sweep has low
  immediate priority.
- The specialist changed119 fourth proposals and38 final choices. All three
  fits improved their validation criterion. An unchanged model or failure to
  lower that criterion is not a good leading explanation.
- A bounded descriptive read of already-used development validation records
  found new-row outsider CE improved about2.05-2.53%, while pair-weighted noisy
  outsider-winner accuracy changed40.90 to40.65%,41.63 to41.15%, and40.89 to41.67%.
  These are not fresh utility estimates. They weaken the assumption that strong
  improvements already exist on the observed support and are merely hidden by
  unrestricted deployment.
- The broader teacher has real gameplay evidence:56.2189% score in402 games
  against the original four-candidate learned planner, interval[51.6169,60.6965]%.
  It used about1.94 times the decision time. In the preceding fresh position
  screen, breadth beat the anchor by+1.2316pp and equal-budget extra precision
  by+1.0829pp. Precision alone gained+0.1487pp with an interval including zero.

This suggests testing whether coverage remains useful when each move gets less
precision. It does not establish that halving128 to64 worlds is harmless:
the marginal value of precision can be larger at the lower budget.

The specific question is whether the successful wider candidate set can improve
the cheaper four-candidate planner at the same simulation count. It is not a
claim that eight-by64 matches the strongest eight-by128 player, halves actual
wall time, or fixes the stalled repeatable learning loop. Affordable stronger
planning would nevertheless improve the programme's research budget and could
make future teacher-data collection cheaper.

## The single proposed screen

Use all360 previously fixed main roots from outputs/proposal_breadth_128, with
all three original161 fits and all18 fit-by-seat-by-phase cells retained. Do
not select disagreements, favorable phases or large supports. The main root
files total244,860,358 bytes. Authenticate them against their sealed collection
inventory, closure, source identities and existing audit receipts before use.

The original experiment saved two useful outcome matrices at every root:

1.128 common worlds evaluated for every candidate in the up-to-eight mixture.
2.An independent precision-extension block evaluated for the original up-to-four
  candidates. For five through eight candidates, this has32,64,96 or128 worlds.

These extra precision outcomes make the exact128-world reference possible
without consuming the common worlds reserved for assessment. A naive split of
only the common128 worlds would not do that.

Let G be the canonical legal support, m=min(G,4), n=min(G,8), B=128m. For G>4,
define q=floor(B/n), r=B mod n, L=ceil(B/n), and H=128-L. Freeze64 partitions
before calculating comparisons. Within each partition, independently permute
the common world indices and the available extension indices using fixed
identity-keyed seeds; preserve each world's complete candidate outcome vector.
Use a separately keyed, outcome-independent candidate priority for remainders.

- Broad uniform arm: give every candidate the firstq common worlds, and give
  the firstr candidates in that priority one additional common world. Divide
  each score sum by its own actual count. No early elimination or learned filter.
- Reference arm: give each original candidate the firstL common worlds and
  firstH extension worlds, for exactly128 outcomes each.
- Lock both choices, preserving original candidate-order ties. Assess them on
  the remainingH common worlds, which neither arm used to choose its move.

| Candidates | Broad allocation | Reference selection | Assessment | Extension available |
|---|---|---|---|---|
|5|Two103, three102|103 common +25 extension|25 common|32|
|6|Two86, four85|86 common +42 extension|42 common|64|
|7|One74, six73|74 common +54 extension|54 common|96|
|8|Eight64|64 common +64 extension|64 common|128|

Both arms spend exactly512 continuations in every row. Required extension
length never exceeds the saved amount. Compare rational means exactly when
implementing unequal sample counts so numerical noise cannot alter ties.

For G<=4 both rules evaluate the identical complete support on the same128
worlds and must make the same choice. Verify that identity and include their
zero paired differences. Do not discard these roots or invent assessment
outcomes to produce an absolute mean. The primary result needs paired gains,
not absolute selected-action scores for those identical cases.

Each partition must choose its own actions from its own selection cells only.
An outcome can be selection in one partition and assessment in another; do
not pool these outcomes to make a single supposedly held-out choice. Average
the64 paired differences within each root, then equally across all360 roots.
Use50,000 paired bootstrap resamples within the original18 cells, preserving
the20 source games per cell. Freeze seeds, source list and all rules before
analysis. Roots are the resampling units; worlds and partitions are not extra
independent observations. Intervals condition on these three original fits.

The reconstructed reference follows the same four-by128 decision rule but uses
a different lawful128-world sample than the old published reference. Its choices
may differ. This neither changes nor replaces the old breadth result. Check
that common and extension blocks use the same world law and continuation rules
with distinct identities; their block labels do not confer different information.

Do not manufacture an eight-by128 positive control using the same128 common
worlds and then score it on their held-out suffix. It would have seen those
outcomes. Existing fresh breadth and gameplay results remain contextual evidence;
a later fresh screen can include that player with truly independent assessment.

## Decision rule and scientific limits

Keep the existing practical hurdle: broader uniform minus reference mean at
least+0.5 percentage points, with the paired95% lower endpoint above zero, and
all prescribed checks passing. This is a development investment rule, not a
fresh significance claim, a power guarantee or a conversion to arena win rate.

A pass supports proposing one fresh bounded confirmation, with eight-by64,
four-by128 and the proven eight-by128 context arm, using separate assessment
worlds. Fresh confirmation would be required before a costly complete-game
comparison. Neither stage launches automatically from this recommendation.

A miss closes this fixed screen. Do not change candidate mixtures, sample
counts, partitions, phase filters or the hurdle after seeing outcomes. Record
the size and uncertainty, then reconsider the next mechanism. No learner or
planner is promoted from reused data.

These roots and outcomes have already demonstrated a breadth effect and helped
motivate this proposal. Within-partition separation prevents choosing and
judging a move on the same simulated outcomes; it does not erase that history
or turn the analysis into fresh confirmation. The assessment is under035
continuations on fast161 source positions, not direct full-game strength or an
external/human benchmark. Equal branch counts do not establish equal time.

## Why this precedes the strongest alternatives

The frozen-specialist shortlist diagnostic remains a valid second choice:
compare the original anchor, unrestricted specialist, the same specialist
restricted to outsiders in the original eight-candidate mixture, and the full
teacher. It would need both a practical restricted-over-anchor gain and a direct
restricted-over-unrestricted gain; merely undoing harmful deviations is not
progress. One reviewer initially favored it, then reduced its priority after
the modest in-support learning evidence and saved-data opportunity became clear.
Action-support extrapolation is still plausible, not ruled out.

Adaptive eight-by32 followed by four survivors receiving64 more worlds also
spends512 continuations at full support. It adds an early-elimination risk that
the simple uniform comparison avoids. The old halving result was only+0.1088pp
on reused data at a different budget and candidate set. Fixed-budget exploration
research provides context, not a guarantee for these correlated noisy game
outcomes ([Karnin et al.,2013](https://proceedings.mlr.press/v28/karnin13.html)).
Test the directly supported breadth-versus-precision question before adding
that extra assumption. This recommendation contains no adaptive schedule sweep.

Larger models, different representations, new value estimators and better
continuations remain open. The past nulls constrain their tested recipes, not
the entire classes. Independent strong opposition remains necessary for broader
strength and eventual superhuman claims. Neither a new arbitrary architecture
nor a benchmark tournament has the same immediate information-per-compute case
as this already-paid controlled comparison.

## Cost and verification

No new model inference, source game, training or continuation is required.
Arithmetic and file checking should take minutes once implemented; roughly5-20
minutes of execution is a provisional estimate, not a measured ETA. Preparation
and preflight are additional. The earlier saved-halving analysis took32.1 seconds
for its arithmetic, but that is not a benchmark of this new implementation.

Propose a one-hour execution ceiling after launch, with no automatic extension.
Independently reconstruct every partition's candidate budgets, selection-cell
accesses, choices, held-out scores, root aggregates and bootstrap statistics.
Use hand-checkable fixtures for all support sizes, unequal counts, ties,
cross-partition leakage and missing/tampered records. The old fixed128-world
integer/256 endpoint code cannot be reused unchanged: assessment counts25/42/54/64
and partition averages
require their correct rational denominators and independently checked numerical
aggregation. Do not quantize these effects back to the old score grid. Inherit
the authenticated original trajectory audits; do not repeat mature simulation
or historical fits. Clearly report that no new full inner replay was performed.
Implement only in
new files outside existing frozen inventories if later authorized.

No experiment, promotion, monitoring or successor has been started.

## Evidence

- outputs/proposal_specialist_128/{report.md,result.json,final_audit.json,
  fits/fit_161001..161003/{specialist,control}/validation_*.json}
- outputs/proposal_breadth_learning_128/report.md
- outputs/proposal_breadth_128/{report.md,result.json,final_audit.json,
  main_collection_lock.json,main/roots/}; tools/proposal_breadth_128/engine.py
- outputs/proposal_breadth_arena_128/report.md
- outputs/proposal_iteration_128/report.md; outputs/proposal_ranking_128/report.md
- outputs/learned_proposal_128/report.md
- docs/engineering/saved_halving_screen_result.md; verification_policy.md
- docs/strategy/next_after_proposal_breadth_learning_128.md
- docs/strategy/next_after_proposal_ranking_128.md
- docs/strategy/next_after_generalization_panel.md
- docs/strategy/independent_superhuman_review_after_134_137.md
