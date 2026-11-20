# Matched teacher precision experiment

The owner authorized this experiment on 2026-10-11. The question is whether
more precise guidance on the same training positions improves the existing
learned proposer. New implementation and results live in
`tools/teacher_precision_512` and `outputs/teacher_precision_512`; the entry
point is `run_teacher_precision_512.py`. All predecessors and dirty work remain
preserved. The complete protocol, passing checks, dependencies and input
artifacts are sealed before production.

## Scientific contrast

Extend the 720 already retired added TRAIN positions from the completed
independent-game coverage experiment from 128 to 512 worlds per offered action.
Retain the original candidate order, concrete representatives and all 128 old
outcomes. Collect 384 new independent lawful worlds using the unchanged
greedy035 terminal continuation policy for both simulated players. Nothing
from old assessment blocks enters training. No state, action, phase, margin,
fit, outcome or apparent confidence selects a training subset.

These 720 positions remain permanently retired from evaluation. The original
9,461 TRAIN rows and their 128-world targets remain unchanged. The 2,489 old
validation rows stay outside fitting; they are not a fresh success endpoint.

Use all three original161 initializers. Train the full existing policy while
freezing its value head. Preserve exactly the R1 BROAD batch streams and mixed
exposure: 57,022 old and 57,022 added row exposures, 1,784 updates, AdamW learning
rate 1e-4, weight decay 1e-4, gradient clipping 1, float32 deterministic CPU,
one Torch thread. Use soft score targets with temperature .05 on recorded
actions outside the actual first three. Empty and singleton outside supports
retain their batch slots and differentiable zero loss. Scores map WDL sums to
win1/draw.5/loss0 with the appropriate 128 or 512 denominator. Unrecorded legal
actions are never assigned fabricated values.

The control is the exact completed R1 BROAD128 learner for each initializer.
Authenticate its fixed final model and complete training/audit chain rather
than rerunning an already verified trajectory. Train only three new precision
models, always retaining the final update; no checkpoint or fit selection.
Save optimizer checkpoints every 64 updates, complete acquisition diagnostics
on the new targets and the common old targets, and independently reproduce all
three new optimizer trajectories. Poor acquisition limits a negative result;
it never triggers extra epochs, another learning rate or model selection.

## Training label collection and verification

Authenticate the complete inherited tensor export, source roots, original world
outcomes, source-game membership and completed audit chain. Rebuild the ordered
720-root label plan from those identities. New world seeds are
2310000000 + training_index*1024 + world_index, for world_index 128 through 511,
with a new fixed domain tag. Check disjointness from source and evaluation
streams and the inherited roots' original worlds before production.

Select 18 TRAIN roots by identity alone, balanced over original fit, acting
seat and phase, for the timing probe. Each receives all 384 new worlds and
full independent replay. These exact completed roots count toward the 720;
they are reused without recollection or repeated audit. After an outcome-blind
forecast, collect the remaining 702 roots. Save each new 128-world block and
its separate completion hash receipt before reuse, then seal the completed
root with both identities, so verified work can be recovered after interruption.
An existing block without its completion receipt is preserved and causes a hold.

Apply the accepted [verification policy](verification_policy.md): check every
new record's world law, candidate mapping, structure, terminal status and score
arithmetic. At every ordinary label root replay eight whole worlds from each
new 128-world block, retaining all candidates. Add full replay of a bounded
identity-selected metadata stress set (at most 39 roots across fit, seat/phase,
private-reserve categories and short supports). Fully inspect any unresolved
branch. The inherited exact repetition/equivalence logic is unchanged. A
training target that remains unresolved causes a documented hold; it is never
silently treated as a draw, filtered, replaced or given an invented score.

The four 128-world blocks also permit a prespecified descriptive cross-fit.
For each excluded block h, the precision choice uses the other three blocks
and the control choice uses block (h+1) modulo 4. Assess both choices on h.
Choose only among recorded actions outside the actual first three, retaining
the first maximum on ties. Empty outside supports contribute zero and remain
in the average. Average all four folds and all 720 roots equally. This
measures a 384-versus-128 conditional comparison on reused TRAIN positions; it
is not independent validation of the full512 winner, a fresh-state result,
or a criterion for choosing whether a valid learner receives its fresh test.

## Fresh planning evaluation

Seal all three precision models, original anchors and matched controls before
generating any fresh evaluation source games. Generate 1,080 main and 54 probe
games with the unchanged fast161 source policies. Main seeds begin 2305000000,
probe seeds 2306000000, and synthetic fixture seeds 2308000000. Replay every
actual source decision and check private-information invariance.

Use the inherited outcome-independent selection scheme: 360 main positions
from 360 different new games, with 20 in each of 18 source-fit/seat/phase cells,
plus 18 separate probe positions. Phases are turns 0-19, 20-39 and 40 onward.
Preserve short supports and private reserves. A fixed-source-pool shortage
holds the run without replacement games. This population represents fast161
play, not every distribution visited by the deployed planning champion.

Four arms are anchor, precision, control and teacher. Each learner supplies an
unrestricted lawful fourth proposal, preserving the same first three. The
teacher preserves anchor's four and adds two ranked and two identity-uniform
moves. Evaluate the union of at most ten candidates on 128 common selection
worlds. Seal choices before collecting 128 disjoint assessment worlds for the
at most four distinct selected actions. Evaluation seeds begin 2300000000,
with main offset 1000000, root offset index*512 and assessment offset 256.

Use the inherited bounded-outcome machinery for evaluation: unresolved outcomes
retain conservative intervals, shared chosen actions cancel exactly, and an
uncertified selection holds. Assessment never chooses a move or model. Fully
replay the 18 evaluation probe roots; apply eight-world-per-block sampled
replay plus bounded metadata stress and unresolved-branch checks on main.
Verify all newly changed proposals, budgets, choices, identities and statistics.
Main continuation replay is sampled, not exhaustive.

## Analysis and decision

Average the 360 source games equally. Use 50,000 paired whole-game bootstrap
resamples within the 18 fixed cells, keeping every arm together. Bootstrap seed
is 2303000000; the independent reconstruction must reproduce the statistics.
Report all three fits without selecting a winner. The joint investment gate is:

- precision minus anchor mean lower bound at least +.005 score (+.5 percentage
  points), with paired95% lower endpoint above zero;
- precision minus matched128 control paired95% lower endpoint above zero;
- all prescribed integrity and operational checks pass.

Both contrasts are required for the single treatment's success. Control versus
anchor, teacher versus anchor and teacher versus precision are descriptive.
Report conservative unknown-outcome bounds; never impute them. These are
conditional planning scores, not arena win rates or superhuman strength.
A pass supports a separately proposed game arena. A null constrains this
mixed-data precision intervention, not all precision, learning or data scaling.

## Resources and operation

New training collection is at most 720*8*384 = 2,211,840 continuations, before
verification and fresh evaluation. Expected CPU execution is roughly 15-24
hours after preparation, with substantial uncertainty. The active execution
safety ceiling is 48 hours and the disk reserve is 5 GiB. These are operational
limits, not an outcome-dependent sample extension or a promise of runtime.

Before main label collection, use only the fully audited probe's times, counts
and byte sizes to project the remaining label and replay work. Apply a 35%
allowance and reserve eight additional hours for later training, fresh source
collection, evaluation and verification. After the fresh evaluation probe,
reprice the remaining main evaluation using its actual timing and stress plan.
Both forecasts must fit the cumulative active ceiling and disk reserve. Extra
unresolved-branch replay is additional overhead covered by allowance and the
live boundary, not falsely claimed inside an exact replay-work count.

Use one detached hidden supervisor, at most eight one-thread simulation/source
workers and three fitting/audit workers. Use CPU only, inhibit system sleep and
allow display sleep. Test real windowless child creation before launch. Preserve
completed jobs, world blocks and optimizer state on any interruption or failure;
there is no automatic retry or silent overwrite. Report progress and provisional
completed-position scores when requested. No recurring monitor, automatic arena,
promotion, parameter sweep or successor belongs to this authorization.
