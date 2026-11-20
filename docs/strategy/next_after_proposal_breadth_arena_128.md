# Next experiment after the successful breadth arena

2026-10-02. Recommendation only. The completed breadth arena supports one
bounded attempt to teach the existing four-candidate planner from the broader
teacher's saved decisions. Test the learned proposals on fresh positions before
funding another full-game arena. Confidence in this allocation is moderate;
successful transfer is an empirical question.

AGENT_CURRENT was read first. The review considered the completed arena, the
breadth screen, previous learning and ranking failures, the wider strategy and
the earlier 134/137 evidence. Three independent reviewers examined learning,
data feasibility and competing investments. No fitting, inference, simulation,
statistical resampling, replay, experiment implementation or launch was performed.
All existing dirty work and frozen evidence remain intact.

## What the completed result establishes

The broader planner scored 56.2189% against the learned128 anchor: 223 wins,
6 draws and 173 losses in 402 games, comprising 201 reversed-seat pairs.
Its paired 95% interval was [51.6169%, 60.6965%]. Both fixed investment gates
passed. This establishes improvement in this matchup, not a true score above
55%, broad opponent dominance or superhuman performance.

The intervention retained the original four candidates and added two remaining
anchor-ranked and two identity-uniform alternatives, with 128 worlds per action.
The corresponding conditional breadth screen had already beaten equal-budget
extra precision. The arena now shows full-game transfer of this entire expansion
package. It does not isolate the ranked additions from the random additions.

The gain used more computation: candidate decisions averaged 127.866 seconds
versus 66.046 seconds for the reference, about 1.94 times as long. Main collection
took 80.866 hours and its audit 6.149 hours; total execution was 91.287 hours.
The valuable next question is whether some of the extra search can become a
better learned proposal at the original four-candidate search budget.

## Why this learning attempt is materially different

The arena saved 11,950 candidate-side decisions from all 201 pairs, with 67 pairs
per original 161 fit. Of these, 11,037 offered more than four actions and 9,819
offered all eight. Every offered action has 128 recorded outcomes. There are
88,384 recorded candidate action estimates, backed by 11,313,152 continuations.

Every broader root retains the exact original anchor prefix. Taking the first
maximum over that prefix reconstructs what the narrower search would choose
at the SAME reached state using the SAME worlds. This requires no new teacher
rollouts. Reference-side roots are different states and are not matched labels.

Earlier second-generation winner imitation scored 51.4925% against the anchor,
with interval [47.2637%, 55.7214%]. The later ranking pilot improved its training
criterion and changed proposals, but its independently assessed decision gain
was only +0.0434 percentage points, with interval [-0.3993, +0.5306]. Those
results weaken explanations based solely on an unchanged model, insufficient
optimization of the measured objective, or gross measured forgetting. They
constrain those recipes, not all learning or all small benefits.

The new opportunity is a gameplay-validated teacher with broader comparisons
at almost twelve thousand actual decisions. It is not another loss variation on
the same narrow comparisons. One earlier learned-proposal generation already
improved full-game search: learned128 scored 65.796% against original128.
Repeated useful transfer remains unproven.

## One controlled pilot

Train two families from each of the three original 161 initializers, giving six
new fits. Keep all fits; do not choose the best fit after testing.

| Family | New source states | Offered comparisons and target |
| --- | --- | --- |
| Broader supervision | All 11,950 breadth decisions | Up to eight recorded groups; imitate their first maximum mean outcome |
| Matched narrow supervision | Exactly the same decisions and worlds | Original up-to-four prefix; imitate its first maximum mean outcome |
| Unchanged anchor | No additional training | Original frozen 161 model |

Use the inherited offered-group winner cross-entropy recipe in both trained
families. The control's loss only compares its original four groups. Giving it
eight competing groups while forcing a four-group winner would actively demote
additional moves its narrow teacher never considered. The proposed contrast
therefore measures the BROADER SUPERVISION PACKAGE: more offered comparisons
and the corresponding winner. It is not a winner-label-only causal effect.

Both families receive identical full-legal model inputs, states, source order,
batch slots, initialization, optimizer settings and checkpoint opportunities.
Retain the previous 23,936-row corpus with its existing targets and split.
Add every new candidate-side row, including losses, agreements, ties, short
supports and private reserves. Exclude the operational probe and opponent-side
roots. Do not filter for added-action winners or high estimated margins.

Prospectively hash-assign 53 whole source pairs per source fit (161001/2/3) to
training and 14 to validation. Pool the same 159 training and 42 validation
pairs for all six new fits. Both seats and all turns of a source pair stay
together. Preserve every historical assignment.
Combine the training partitions without oversampling: four complete passes,
batch size 64 including the final short batch, fresh AdamW at learning rate
1e-5 and weight decay 1e-4, gradient clipping 1, float32 deterministic CPU with
one Torch thread, frozen value-head parameters. No loss, dose or seed sweep.

Use a common, prospectively fixed checkpoint schedule and validation criterion
for both families: equal weight per source pair, mean offered-group winner loss
within each pair. New validation rows use their broader recorded support and
winner for both families; historical validation retains its original support
and target. Include the initial checkpoint and select the earliest strict
minimum. Complete the fixed dose. This common development objective does not
itself establish improved play. Seal all six selected models before testing.

Do not simultaneously introduce an objective for the best alternative outside
the first three candidates. It may be worth investigating later, but would
change another mechanism now and gives only one observed outside action on
the old four-candidate rows. Keep the learning intervention interpretable.

## Test the deployed planner on fresh positions

The decisive endpoint is actual proposal utility inside four-candidate,
128-world search. Compare the broader learner, matched narrow learner and
unchanged anchor. Retain the original first three proposals and replace only
the fourth with each model's highest-ranked remaining canonical group.

Use 360 new source-game-disjoint positions, 120 per fit and 20 per
fit-by-acting-seat-by-phase cell, following the established fixed source-pool
construction. Preserve phases 0-19, 20-39 and 40+, short supports and private
reserves. Use new disjoint source seeds and world namespaces; do not reuse
previously inspected panels. Freeze eligibility, selection, shortage rules,
all model identities and the separate 18-root operational probe before outcomes.
The source population remains the original fast 161 students and must be named
as a limitation; it is not all positions reached by stronger planners.

Evaluate the deduplicated union of the three arms, at most six actions, on
128 common lawful selection worlds with unchanged greedy035 terminal
continuations. Each arm can choose only from its own up-to-four support.
Persist those choices before assessing only the distinct chosen actions on
128 disjoint assessment worlds. Assessment never selects another move or fit.
Every deployed arm retains the original budget even though research collection
shares outcomes between arms.

Average source games equally. Use the established 50,000 paired bootstrap
resamples within the 18 fixed cells, with a new frozen seed. Keep all arms
paired and independently reconstruct statistics. The proposed investment gate is:

- Broader learner minus unchanged anchor: mean gain at least +0.5 percentage
  points and paired 95% lower endpoint above zero.
- Broader learner minus matched narrow learner: paired 95% lower endpoint
  above zero.
- All prescribed integrity and operational conditions pass.

These are conditional terminal-score differences, not arena win rates. The
threshold is a compute-allocation rule, not a power guarantee or evidence that
smaller effects are worthless. Report the control-minus-anchor contrast and
per-fit summaries descriptively; do not select a new primary winner afterward.

A pass would justify proposing fresh full-game confirmation at the original
four-candidate budget. It would not show that the learner retains all of the
broader teacher's advantage. Improvement over the anchor without attribution
over the control would leave the source of progress unresolved. A miss closes
this exact broader-supervision recipe without panel extension or nearby sweeps;
it would not prove that the teacher has no learnable information.

## Main risks and alternatives

The strongest risk is that useful breadth cannot be compressed into this one
proposal through winner imitation. Common high-ranked moves may dominate labels;
the network extrapolates to legal actions absent from the recorded comparisons;
128-world winner labels are noisy. Saved maxima are training targets, not
independent evidence of learning quality. Never assign unoffered actions zero
value or treat them as measured negative outcomes. Splitting the saved worlds
afterward does not create a fully fresh test of the reached states and choices.

An independent strong-opponent benchmark is the strongest competing investment.
It matters before broad-strength claims or promotion. The existing related
fast-opponent panel is not an established informative benchmark for these
planners, while this pilot uses unusually direct comparisons already paid for.
The pilot takes priority; it does not discharge the generalization requirement.

An equal-budget eight-by-64 versus four-by-128 arena would address efficiency,
but dilutes per-action precision while increasing breadth. A failure could mix
useful coverage with added selection noise. The previous breadth-versus-precision
screen gives the present teacher a firmer basis for a cheap learning test.

Stronger continuations, value learning and a larger architecture remain open.
Earlier 140/158/159 and 142-153 do not provide a comparably localized, validated
replacement; 158's owner-stopped audit remains provisional. Likewise, 134's
31.875% full-game result and 139's favorable conditional diagnostics warn that
proxy improvement alone is insufficient. The 137 root-refill intervention did
not establish ranking gains; it did not rule out chance search in general.

## Cost and verification

The expensive teacher decisions already exist. New fitting needs no simulated
training games. Prior fits took minutes to tens of minutes each; the previous
three-proposer 360-position screen collected in about 78.5 minutes and audited
in 11.6 minutes. Broader export, more retained examples, independent fit
reproduction, source generation and preflight all add work.

Provisional execution expectation: roughly 4-8 hours, predominantly CPU work,
after implementation and preflight. This is an extrapolation, not a measured
ETA. Propose an eight-hour execution ceiling from pilot launch, a 5 GiB disk
reserve, and an outcome-blind probe forecast with time/storage allowances.
A forecast outside the cap produces a documented hold, not an automatic budget
increase. No automatic multi-day arena, promotion or successor follows.

The main fresh screen has at most 360*128*(6+3)=414,720 recorded continuations;
the 18-root probe adds at most 20,736. Deduplication should reduce actual counts.
Apply the accepted verification policy: verify all new source moves, exported
targets, splits, training and selection identities, proposals, budgets, choices
and statistics; independently reproduce new training. Inherit sealed prior
simulation evidence while checking source links and each root on use. Use
prespecified sampled replay on mature new continuation paths and full replay
of the bounded probe and stress roots. Do not re-run old simulations or all
new mature continuations by default.

Implementation must use new files outside frozen inventories. Existing export
adapters cap offered support at four; new exporters and their independent
checks must support eight while preserving lawful full-legal inputs and
canonical maximum scoring. No neural architecture change is required.

## Connection to the longer term goal

The desired loop is stronger search supplying useful examples, learning turning
those examples into better proposals, and better proposals improving affordable
search again. This is related to Expert Iteration's separation of planning and
generalization ([Anthony et al., 2017](https://arxiv.org/abs/1705.08439)) and the
study of policy improvement over sampled actions
([Hubert et al., 2021](https://proceedings.mlr.press/v139/hubert21a.html)). Those
papers provide context, not a guarantee for this imperfect-information planner.

The programme has demonstrated one useful learned-proposal cycle and now a
stronger, more expensive teacher. The next scientific milestone is another
useful transfer. Efficient repeated improvement, independent strong opposition
and declared-resource human evaluation remain necessary for the eventual
superhuman objective.

Local evidence: outputs/proposal_breadth_arena_128/{report.md,result.json,
final_audit.json,final_status.json}; outputs/proposal_breadth_128/report.md;
outputs/proposal_iteration_128/report.md; outputs/proposal_ranking_128/report.md
and recovery_01/closure.json; outputs/learned_proposal_128/report.md;
docs/strategy/next_after_proposal_breadth_128.md;
docs/strategy/next_after_proposal_ranking_128.md;
docs/strategy/next_after_generalization_panel.md;
docs/strategy/independent_superhuman_review_after_134_137.md;
docs/engineering/proposal_ranking_128.md;
docs/engineering/verification_policy.md.

This note proposes one bounded pilot. It does not authorize or start it.
