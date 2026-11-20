# Next after learned proposals: bounded full-game transfer test

2026-09-18. Recommendation only following the owner's request to think through
the best next step. Read AGENT_CURRENT and the completed proposal screen first,
then the current strategy and relevant 155/156/158/160/161/162 evidence. Two
independent read-only reviews challenged comparison, sample size and cost.
No implementation, inference, simulation, fitting or launch was performed.
Preserve all frozen files and intentionally dirty work. Confidence is moderate.

## Decision and rationale

Recommend **402 fresh full games of the learned fourth-proposal planner against
unchanged Experiment 155**, with 32 worlds on both sides and all three frozen
161 fits represented. A separate 12-game timing/integrity probe precedes main.

The latest screen found +0.8522 percentage points against the original random
fourth slot (95% interval +0.3540 to +1.4174) and +0.4209 points against another
035-ranked proposal (+0.0235 to +0.8409). All prescribed full replays passed.
Learning therefore has a supported role in proposing candidates under the
screen's conditions. That strengthens the programme's proposed route from
learned guidance back into search, while leaving its full-game link untested.

The screen chooses one move, then evaluates greedy 035 futures on positions
from 035 games without private reserved cards. Complete play changes all three:
both players repeatedly plan, their choices change the positions reached, and
private reserves occur naturally. This is the highest-value remaining transfer
question. Another conditional screen would still leave it unanswered.

## Proposed fixed design

- 201 fresh starting seeds with both candidate seats, 402 main games. Assign
  67 whole seed pairs to each of the three frozen 161 fits by an outcome-free
  identity rule. Use distinct seeds across fits. Keep that model fixed throughout
  both games of its pair. No best-fit selection, ensemble or model switching.
- Preserve exactly the screened proposal change: build the original 155 list,
  keep its first three candidates (035 top two plus the first random alternative),
  and replace only the fourth with the assigned fit's highest-scored remaining
  canonical group. Preserve all-legal input, group maxima, original concrete
  representatives, original-order ties and unchanged supports below four.
- Keep 32 shared observer-legal worlds per offered candidate, terminal WDL,
  greedy 035 continuations for both simulated players and original selection
  ties. Preserve lawful coupled random streams at matching observations. Only
  the assigned learned fit needs inference in the candidate's deployment path.
- The reference must match frozen 155 exactly. Every actual turn is played;
  there is no phase filter or exclusion of positions containing private reserves.
- Probe: six other starting seeds, two pairs per fit, 12 games. Inspect only
  timing, storage and integrity to decide operational feasibility. Probe wins
  cannot tune the intervention, sample size or model assignment.
- Main endpoint: candidate score (win 1, draw 0.5, loss 0), equally weighted
  across the three fits. Use 50,000 complete-pair bootstrap resamples within fit,
  retaining both seats together. Intervals condition on these fixed models.
  Per-fit, seat, margin and latency results are descriptive, not selection rules.

Predeclare evidence of superiority as a paired 95% lower bound above 50%.
Use observed score at least 55% plus that superiority criterion as a practical
progress screen, consistent with earlier planner comparisons. This does not
prove that the true score exceeds 55%. A statistically positive estimate below
the practical threshold remains positive evidence and must be reported as such.
An upper bound below 55% argues against a five-point advantage under these
conditions; an interval spanning parity and useful improvement is inconclusive.
No result authorizes automatic extension, tuning, promotion or a successor.

The exact namespaces, assignments, audit inventories and implementation need a
new frozen protocol after execution is authorized. This note is not that freeze.

## Why this size and these controls

402 games follow the balanced three-fit precedent of 158 and the approximately
400-game planner screens. Earlier pair variation suggests a confidence-interval
half-width around 4--5 percentage points, only a rough planning anchor. This is
a bounded test for sizeable benefit, not a powered confirmation of a small gain.
The +0.8522-point conditional effect supplies no valid full-game power calculation.

600 games would reduce typical uncertainty by only about 18% for roughly 50%
more work. Retaining both reference and raw-ranked control would require 804
games at the same per-comparison precision. Splitting 402 across both would
substantially weaken each comparison. The immediate priority is establishing
practical full-game value before paying for the second attribution comparison.

This choice has a real limitation: a win establishes the learned-proposal
planner's improvement over 155, not its full-game superiority to the cheaper
035-ranked replacement. The screen's attribution interval barely excludes zero;
that question must remain open. A later controlled comparison would be required
before claiming learning-specific full-game advantage over that alternative.

Use the 32-world reference to isolate the tested proposal intervention. The
128-world 160 planner remains a stronger retained comparator at four times the
sampling budget. Testing against it now would mix proposal benefit with a large
compute difference. A result at 32 worlds does not establish a new strongest
agent or transfer of the proposal gain to 128 worlds.

## Verification and provisional cost

Require crafted private-reserve fixtures for both acting seats, including own
known blind reserves and opposing unknown reserves. Verify that hidden deck
order and opposing hidden identities cannot change legal model input, scores,
proposals or RNG schedules. 155's legal world sampler already covers such
positions; the new learned adapter still needs its own explicit checks.

Preserve full outer-game reconstruction and **full replay of every collected
continuation**, with independently rebuilt proposals, worlds, means, choices
and final statistics. Run prescribed checks regardless of win rate. Keep all
failed evidence; invalid games stop the study rather than being replaced.
The shared engine is still a trusted dependency, not an independent rules proof.
New isolated files must preserve all existing frozen identities and evidence.

Budget roughly **30--45 hours including full replay**, plus implementation and
checks, pending the separate probe. This is an estimate: 155's 400 games took
13.5 hours with only one searching player; both sides search here. Recorded
32-world decisions in 155/160 average about 19--20 seconds. The recent screen's
51,520 selection continuations took about 20.7 minutes to collect and 19.1
minutes to replay, reinforcing the substantial audit cost. Game lengths,
hardware contention and storage can change the actual duration.

Use eight hidden CPU workers and the established windowless runtime. Check
projected disk use and retain a reserve before main; report an operational hold
if the probe forecast is unreasonable. Do not silently reduce audits or game
count to fit the estimate. No recurring monitoring is proposed.

## Alternatives and interpretation

Further training is less well grounded until this deployment role demonstrates
full-game benefit. 161 supports useful learning, but 162 and the broader panel
did not establish a productive additional refresh. That does not rule out other
data mixtures, objectives or training scales; it changes their immediate priority.

The saved halving result remains a candidate efficiency improvement. Its reused
data and different state population prevent treating its smaller numerical gain
as a direct head-to-head comparison with this proposal screen. Combining the two
now would obscure which intervention helped. Batching remains potentially useful
engineering, with equivalence and speedup unmeasured; it is not a prerequisite
for this serial scientific test and should not be bundled into it.

A positive arena would establish one useful learning-to-planning transfer for
these fixed models and budgets. Broader opposition, the raw-ranked control,
practical computation limits and further successful generations would still be
needed. A negative or inconclusive arena would narrow this exact deployment
recipe; it would not refute learned proposals or identify a unique failure cause.
The programme's eventual superhuman claim remains much further away.

Evidence: `docs/engineering/learned_proposal_screen_result.md`,
`docs/engineering/learned_proposal_screen.md`,
`docs/engineering/rollout_selector_gameplay_155_result.md`,
`docs/engineering/rollout_selector_champion_156_result.md`,
`docs/engineering/learned_continuation_158.md`,
`docs/engineering/world_budget_160.md`, `outputs/world_budget_160/result.json`,
and `docs/strategy/next_after_generalization_panel.md`.
