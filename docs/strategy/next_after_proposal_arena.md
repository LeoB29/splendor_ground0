# Next after the successful proposal arena: compare the original-ranking alternative

2026-09-19. Recommendation only. Read AGENT_CURRENT first, then the completed
arena, proposal screen and relevant160/161/162 strategy/evidence. Two independent
read-only challenges considered training,128-world transfer, broader opposition
and efficiency work. No experiment, new statistical analysis, implementation,
inference, training, profiling or launch occurred. Preserve all existing work.

## Recommendation

Run **402 fresh full games of the learned fourth-proposal planner against the
same planner using the next-highest035-ranked unused group as its fourth
proposal**. Both use32 worlds. Retain all three frozen161 models, the paired
design,12-game timing/integrity probe and full replay of the completed arena.

Confidence is moderate. This is one actionable comparison to guide investment
in learned proposals, not a requirement to repeat confirmation indefinitely.

## Why this question now

The completed arena established a substantial improvement over unchanged155:
259/402 wins,64.4279%, paired95% interval59.7015--69.1542%. Both seats and all
three fixed fits had positive point estimates, and every continuation was
replayed. The question is no longer whether this proposal package can improve
complete play. It can, against155 under the tested budgets.

The earlier single-decision screen also tested a straightforward alternative:
use another high-ranked035 group instead of the fourth random candidate.
The learned variants exceeded that control by0.4209 percentage points, but
the95% interval was only0.0235--0.8409. That is positive conditional evidence;
it is not full-game evidence about these two planners facing each other.

This control was deliberately deferred when first testing practical usefulness.
Now that usefulness is established, comparing against it can redirect the
programme: how much does the newly distilled ranking add beyond better use of
the original035 ranking? Both rankings come from learned models. Describe this
as newer learned knowledge versus the original learned policy, not learning
versus no learning.

The control's runtime advantage is probably small because simulation dominates
the approximately18.5-second decisions. Its value here is explanatory and
strategic, not a presumed large deployment speedup.

## Bounded design

- Candidate: exactly the completed learned32 planner, including its assigned
  frozen161 checkpoint. Opponent: change that planner's fourth proposal to the
  highest035-ranked canonical group outside the original first three.
- Build the same original155 candidates at each lawful observation. Preserve
  top two plus the first random alternative in their original order. Keep
  original concrete representatives, group maxima, tie ordering, uniform
  hidden-world law,32-world selection and greedy035 continuations unchanged.
  Raw ranking uses descending concrete035 scores, stable legal-action ties and
  first occurrence of each canonical group, exactly as in the prior screen.
  Both arms keep full original support below four. Do not reorder the common
  random third candidate behind the new fourth candidate: that changes ties.
- Main:201 new distinct starting-seed pairs, both candidate seats,402 games.
  Assign67 pairs prospectively to each frozen161 fit; hold that fit fixed
  across both games. No best model, ensemble, phase filter or conditional use.
- Include private reserves and every ordinary game position. Keep coupled
  search streams at matching lawful observations and all three fits in the
  final estimate. Use fresh seeds, never the previous arena's outcomes as part
  of this new endpoint.
- Probe:12 separate games, two seat-swapped seed pairs per fit. Inspect timing,
  storage and integrity only; probe strength cannot change design or sample size.
- Primary: learned candidate's full-game score against the original-ranking
  control, win1/draw0.5/loss0. Use50,000 complete-pair bootstrap resamples within
  fit, equal weighting of the three fixed fits. Report95% interval, win/draw/loss
  counts, and descriptive fit, seat, margin and latency results.
- Separate superiority (lower bound>50%) from observed practical progress
  (score>=55% plus superiority). Retain positive-below-target, below-material-
  target and inconclusive interpretations as in the completed arena. An upper
  bound below55% argues against a five-point advantage; it does not prove
  equivalence. No outcome-based extension or subgroup rescue.

This changes the opponent relative to the preceding study; it does not modify
the successful learned candidate. A roughly400-game study is a bounded test
for a sizeable difference and may not resolve a small learned advantage.
Neither previous effect can supply a reliable power calculation for this match.

## Interpretation

A clear learned win establishes full-game benefit from the newer ranking beyond
the original-ranking alternative, strengthening the case for further proposal
learning. A clear control win favors that alternative in this matchup and
changes the immediate research priority. An interval excluding a useful learned
advantage weakens the case for this particular added component at32 worlds.
An inconclusive result remains inconclusive. None erases the verified win over155,
disproves proposal learning or establishes transitive dominance over other agents.

Do not automatically promote either player or start another training generation.
Review effect size, uncertainty and cost once, then choose a new question.

## Cost and verification

Estimate **30--40 hours including full replay**, plus implementation/checks,
subject to the probe. The immediately preceding comparable32-versus32 arena
took33.823h. Different choices, game lengths and machine load can change this.
Freeze an outcome-blind operational ceiling and disk reserve before launch.

Reuse the established eight hidden CPU workers, deterministic arithmetic and
durable one-shot launch discipline. Fully reconstruct every outer game and
every simulated continuation, independently rebuilding both proposal rules,
all worlds, private-information invariance, selected actions and statistics.
Keep invalid/failed evidence; do not replace games or silently reduce audit scope.

The prior arena binds existing src and analysis-Python inventories. Future
implementation should use a separate new location, such as scripts/experiments,
new tests/entry/protocol files and a new output directory. Verify completed
ancestors against their recorded inventories and preserve their artifacts;
do not edit frozen ancestor code to add the control. Exact paths, fresh seed
namespaces, schedules and verification bindings require authorized implementation.

## Why competing next steps wait

**Another learning generation is the strongest inexpensive alternative.** The
new arena supplies many already-paid teacher decisions on new proposal support.
However,161 learned from the original128-world teacher. The successful learned32
planner has not been shown stronger than that teacher: its64.43% against155 and
160's65.5% against155 cannot rank them. Updating from32-world saved labels may
change both noise and distribution; stronger next-generation targets are not
established.162's refresh and later generalization panel also caution against
assuming another training dose produces useful progress. This route merits a
separate controlled design, not an automatic refresh from the new records.

**Learned128 versus original128** cleanly tests combining proposal improvement
with the strongest validated sampling budget. Full replay would cost roughly
four times the latest arena, approximately5--6 days, before a timing probe.
A cheap conditional screen could precede that expense but would leave its
full-game transfer question open. Closing the cheaper32-world attribution
comparison is a better immediate use of compute.

**Learned32 versus original128** tests the strength/compute frontier, at about
2.5 times the latest full-arena cost, roughly3--4 days. Similar scores against
155 do not imply parity or a fourfold efficiency gain. This remains worthwhile
after the proposal mechanism's added value is clearer.

**Broader opposition and external validation** remain necessary for eventual
superhuman claims. The available fast local panel is weak relative to these
planners, while expensive or external opponents require separately budgeted
compatibility and evaluation work. No current result closes that gap.

**Batching is useful engineering**, especially after repeated long simulation
runs. Representative equivalence and speedup are unmeasured. It can be a
separately authorized parallel task, but should not alter this experiment's
teacher or verifier during execution. The positive halving screen also remains
separate; bundling it would obscure the comparison.

No new experiment, monitoring, promotion or training was started by this
recommendation.
