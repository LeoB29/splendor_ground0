# Next after the positive breadth screen

2026-09-28. Recommendation only. AGENT_CURRENT.md was read first. Three
independent read-only reviews challenged the comparator, learning-first option,
independent-opponent alternative, gameplay adapter, verification and cost. No
new experiment, inference, simulation, fitting, replay, tests, implementation,
freeze, launch or monitor ran. Existing dirty and frozen work remains unchanged.

## Recommendation

Run **one fresh full-game comparison of the fixed breadth planner against the
current validated learned128 anchor**: up to8 candidates at128 worlds versus
the original4 at128 worlds. Use402 games,201 starting-seed pairs, all three
original161 fits and both seats. Keep the exact two-ranked-plus-two-uniform
expansion package, original incumbent candidates, models and continuations.

The question is whether the better decisions measured in the screen produce a
better complete player when both sides repeatedly replan. Confidence is moderate
that this is the best next investment; the result itself remains uncertain.
The recommendation is a single arena, not automatic training, a second arena or
an open-ended improvement programme.

## Why the result warrants this step

The completed screen passed every predefined gate on360 fresh positions:

| Contrast | Conditional score gain, percentage points | Paired95% interval |
|---|---:|---:|
| Breadth minus equal-budget precision | +1.0829 | [+0.5089,+1.7166] |
| Breadth minus anchor | +1.2316 | [+0.6608,+1.8620] |
| Precision minus anchor | +0.1487 | [-0.0944,+0.3982] |

Breadth retained the useful fourth proposal and used exactly the same aggregate
selection continuation count as precision:356,864 in each arm. All360 positions
and prescribed audits passed. Main replay was sampled:83,696 of599,936 recorded
continuations, including34 full stress roots. The18-root probe was fully replayed.
The final independent statistical calculations agree. See
`outputs/proposal_breadth_128/{report.md,result.json,final_audit.json}`.

Breadth changed126 decisions; precision changed59. The added ranked candidates
were chosen77 times and random additions49 times. This does not identify their
separate causal contributions or justify changing their mixture. The positive
point estimates occur in all three fits and both seats, with phases and privacy
categories also positive descriptively. These summaries are not separately
confirmed subgroup effects and supply no selection rule.

This establishes useful broader candidate coverage for the tested conditional
operator and source population. It does not show that candidate coverage was
the unique cause of earlier learning failures, that precision can never help,
or that still more candidates will improve the system. The tested extra precision
has no established positive contribution, but its interval is not equivalence.

The remaining uncertainty is consequential. The screen chooses one action and
judges it under greedy035 continuations from fast161-generated positions. Actual
play repeatedly chooses actions, changes the reached positions, and faces an
opponent that replans. Earlier134/139 showed that positive fresh-world move
assessments can coexist with poor full-game performance.134's recorded runtime
flag is interpreted under its contemporaneous owner amendment; the substantive
31.875% gameplay loss is not erased. Those earlier operators differ from this
one, so they warn against an inference rather than predict another failure.

Conversely, learned128's65.796% score against original128 demonstrates that
better proposals can improve actual search. The current recipe has earned a
direct test, with both a positive precedent and a known transfer risk.

## Why this comparator and not another experiment

**Compare with the current anchor first.** A breadth-versus4x256 arena would
test equal-continuation allocation in full games, which remains valuable. But
4x256 is not itself a validated improvement over the4x128 incumbent. Beating
that alternative alone would not establish a stronger current teacher without
assuming transitivity or monotonic benefit from more sampling. The conditional
equal-budget result already justifies selecting breadth for this investment.
The direct anchor match tests the missing practical advancement claim.

The cost difference matters: breadth versus anchor has approximately8+4 search
branches per matching pair of decisions, while breadth versus precision has
8+8 branch-equivalents. Funding both complete arenas now would approach nine
days. The single proposed arena can establish improvement with additional
compute; it cannot establish equal-budget or equal-wall-time gameplay efficiency.
That limitation is intentional and must remain explicit in its report.

**Do not distill first.** Learning from wider comparisons is a promising later
direction, especially after the second CE generation and ranking pilot failed
their progress gates. However, a negative new learning result would conflate
teacher quality, data coverage, target construction and learning transfer.
The present main screen contains only360 distinct positions, versus23,936 rows
in the preceding learning corpus. Many action pairs from one state do not create
many independent source positions. A large new breadth corpus would itself
require substantial simulation. The arena directly tests teacher validity while
retaining its naturally reached decisions for potential later learning.

**The strongest alternative is an independent-opponent panel.** With fast enough
opponents, evaluating both planners separately could cost approximately the same
search work as their duel, while testing a broader kind of transfer. This deserves
serious weight because another family matchup can reveal an exploit that does
not generalize. But a difference between two opponent-panel score rates is not
automatically as precise as a direct duel; outcome correlation, opponent strength
and ceiling effects matter. Raw035 also shares the exact continuation behavior
used in this screen, making it a weak independence claim. The existing related
fast-student panel is not an established strong, cheap, unsaturated benchmark
for these planners;084 is interesting, but its informative difficulty and cost
against the present planners have not been established here.

The direct anchor match is therefore the cleanest first gameplay test with the
available evidence. Broader opposition remains required before broad-strength,
strongest-agent or superhuman claims. If a suitable independent benchmark were
already demonstrated, the balance could favor it instead.

Another source-position screen, a candidate-mixture sweep, new continuation
controller, larger model, loss variant, or batching change would answer a
different question while postponing this one. Keep those interventions separate.

## Proposed fixed gameplay design

- 402 fresh games:201 environment seeds played with reversed seats,67 pairs per
  fixed original161 fit. Assign pairs prospectively by identity; both players use
  the corresponding unchanged fit, retained across both seat games. No best-fit
  selection or reuse of the inspected positional panel as fresh confirmation.
- Breadth preserves the exact current anchor four and appends the same fixed
  ranked/random mixture, with supports below8 truncated naturally. The reference
  is the unchanged learned128 anchor. Preserve lawful inputs, canonical concrete
  representatives, incumbent-first ties and original greedy035 continuations.
- In actual play, each actor evaluates only its own offered candidates in128
  worlds and plays its selected action. The arena does not run the diagnostic's
  precision extension or separate assessment blocks. Actual game outcomes are
  the endpoint.
- Give all201 seed pairs equal weight, retain all three fits, and bootstrap
  whole seat pairs within fit with50,000 frozen resamples. Wins count1, draws0.5,
  losses0. The interval conditions on these fixed fits, not a population of
  independently trained agents.
- Require observed breadth score at least55% and paired95% lower bound strictly
  above50%. This is a practical investment threshold, not proof the true score
  exceeds55%, nor guaranteed power for every useful improvement. Distinguish
  modest superiority below the target, inconclusive evidence and demonstrated
  harm. Do not call every failed practical gate "no effect."
- Freeze sample size and analysis before production. No score-based interim
  stopping, extension, subset rescue, mixture change or automatic follow-up.
  Stop for integrity/resource failures and preserve incomplete evidence.

The gameplay adapter needs explicit bounded validation. The screen's panel index
and three integer world blocks are not a complete game policy. Preserve the
established lawful belief-based128-world schedule and common draws at matching
lawful positions, using fresh pair streams. Freeze a fresh public identity for
the added-candidate hash order at every decision, such as stage/game/turn with
an injective mapping under the400-turn guard. The random-order law and expansion
semantics remain unchanged; new namespace values must be explicit. Neither
private hidden state, worker completion order nor outcomes may define randomness.
Check original-anchor delegation, rank/random ordering, short supports, ties,
private-input invariance and fresh identities independently before launch.

## Verification and cost

Follow the accepted `docs/engineering/verification_policy.md`: reconstruct every
actual game and decision; verify all lawful inputs, proposals, model identities,
worlds, budgets, stored outcome fields, aggregates and chosen actions. Replay
all candidates in eight prespecified worlds at each root, plus a deterministic
full-stress sample over roles/fits/phases/seats/privacy and short supports. Include
the new supports5/6/7 in bounded fixtures and prospectively declared stress
coverage. Freeze the exact finite stress inventory and cost allowance before
collection. Disclose sampled inner replay; do not rerun historical training or
automatically fully replay every mature simulation.

Use12 separate operational probe games, two seat pairs per fit, excluded from
strength statistics. Their prescribed audit must pass. Their timing/storage,
not scores, determines whether the fixed main run fits the budget. Use worker
wall time and separate ordinary/stress verification accounting, rather than
multiplying the short probe's unusually high full-stress fraction across main.

Observed costs from the two recent402-game4x128-versus4x128 arenas:

| Completed arena | Main collection | Sampled audit |
|---|---:|---:|
| Learned128 versus original128 |63.084h|4.524h|
| Proposal iteration Stage A |57.838h|4.148h|

Multiplying combined search work by approximately1.5 gives86.8-94.6h collection
and6.2-6.8h audit. Including the probe and overhead, budget **roughly4-5 days on
eight CPU workers**. This is an extrapolation, not a measured ETA; the changed
reached positions, trajectory lengths, supports and operating conditions matter.
The three-hour positional screen is not a valid runtime estimate for the arena.

Recommend a main forecast ceiling of120h including a25% allowance, a144h total
execution ceiling from the new launch, and5GiB disk reserve after projected
storage. Preparation is additional; the ceiling is not the expected duration.
Freeze exact resource checks before launch, hold if the probe fails, and do not
silently extend the budget. Current read-only free-space check found about47GiB;
fresh projection is still required before main. There is no present external
blocker identified by this recommendation.

Run as a hidden detached supervisor with eight CPU workers, one deterministic
Torch thread each, serial mature continuation backend and robust best-effort
status reporting. Keep scientific writes strict. No GPU, visible terminal or
recurring monitor is needed; the chat need not stay active. Preserve all prior
code, experiments and output trees in their frozen state.

## What a result would mean

A pass establishes a stronger complete planner in this fresh matchup at its
higher compute allocation. Save all actual trajectories and candidate outcomes,
including losses, coincident decisions and private/short-support positions.
Those data could support a separately designed broader-teacher learning study,
with retained old examples and controls for added training/data. Saving evidence
does not automatically authorize its training use or another experiment.

A miss ends this fixed gameplay test. Depending on its interval, distinguish a
small possible benefit from a clear failure to transfer or harm. Use its reached
states and decisions to investigate the discrepancy; do not automatically tune
the recipe, append more games or train from an unvalidated improvement claim.

This sequencing advances the search-learning-search programme: first establish
that the expanded teacher really plays better, then ask whether a learner can
retain that improvement more cheaply. Neither step is guaranteed. Stronger
independent opposition and declared-resource human evaluation remain necessary
for the eventual superhuman objective.

## Sources and external context

Primary local evidence: completed breadth result/report/audit; learned128
report/result and protocol; proposal_iteration_128/stage_a_result.json;
next_after_proposal_ranking_128.md; next_after_proposal_iteration_128.md;
independent_confirmation_root_policy_139_result.md;134 owner runtime amendment;
student_generalization_panel/report.md; verification_policy.md.

External primary context was checked: [Sampled MuZero (Hubert et al.,2021)](https://proceedings.mlr.press/v139/hubert21a.html) studies policy improvement over
sampled action subsets. It supports the relevance of candidate coverage as a
research direction, not a guarantee for this different imperfect-information
planner. The next-step decision above rests on this programme's measured evidence.

Recommendation only. Nothing new has been prepared or launched.
