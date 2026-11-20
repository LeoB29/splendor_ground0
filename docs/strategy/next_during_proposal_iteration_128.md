# Thinking ahead while the controlled proposal iteration finishes

2026-09-25. The owner explicitly chose to finish the current experiment after
considering early abandonment, and requested next-step thinking in parallel.
Keep the frozen pipeline unchanged: complete402 Stage A games and verification;
Stage B remains authorized only after A passes its original gates. Do not alter
models, sample sizes, stopping rules, schedules or verification from interim
results. No successor, model inference, training, simulation or monitor was
started for this review. Three independent read-only reviews examined saved
validation reports, mechanism alternatives and prospective diagnostic design.

This is a new, unbound strategy note, not an amendment to the running protocol.
The current study is `outputs/proposal_iteration_128`, binding
`1a3c53142516a5d8f08376d0a7227ae9199952a9e077e1309d13fdcdf6d40ace`.
Its status at review start was192/402 collected games; that is not a final result.

## Recommended decision after verification

If Stage A does not establish the required practical progress, prioritize **one
bounded diagnosis of the actual fourth-proposal role**, using saved matched
positions first and, if needed, a small independent-world comparison across
representative positions. Do not immediately repeat training, change the loss,
increase model size or spend another full arena confirming a speculative fix.

If A passes, allow the already authorized B comparison to finish. If both pass,
the priority changes: repeated useful learning would justify a bounded
production-scale batching assessment before paying for further generations.
If A passes but B does not, retain the evidence of progress against the anchor,
while diagnosing the new corpus's contribution versus extra old-data training.
Neither absence of superiority nor similar point estimates establishes equivalence.

Confidence is moderate. The final estimates and intervals matter alongside the
gate labels. A small demonstrated benefit below55% is not a demonstrated failure
to learn; a wide interval is not proof of equivalence or harm. An integrity
failure takes priority over any scientific interpretation.

## What the existing validation reports already tell us

These are development diagnostics from the checkpoint-selection data, not new
held-out confirmation or estimates of playing strength. No new predictions were
generated. Validation contains4,889 rows from82 whole source pairs:2,408 rows/40
old pairs and2,481 rows/42 new pairs. The criterion weights pairs equally.

| Fit | Selected treatment update, of1192 | Treatment pair-mean CE | Replay pair-mean CE | Treatment teacher-agreement gain over anchor |
| --- | ---: | ---: | ---: | ---: |
|161001|1192|1.225181|1.232463|+0.51 percentage points|
|161002|1118|1.224744|1.231532|+0.63 percentage points|
|161003|1192|1.225406|1.232805|+0.37 percentage points|

All treatments improve the criterion against their anchor and selected replay
control on BOTH old and new partitions. Gross old-corpus forgetting, failure to
optimize this loss, and improvement confined to the new partition are therefore
poor leading explanations on these measured data. They remain weaker statements
than preserving arbitrary unseen behavior or strategic competence.

Two treatments select their final checkpoint. The third's selected-versus-final
loss difference is only0.000403. Unlike the earlier161/162 curves, these curves
do not justify saying that every fit has already turned upward. Continuing to
improve this criterion would still not establish improved planning.

Among2,481 new validation rows, teacher labels select slots0/1/2/3 on
908/581/447/545 rows. Thus the offered fourth is preferred on21.97% of rows.
The updates become BETTER at recognizing those cases:

| Fit | Fourth-slot predictions: anchor -> treatment | Correct fourth-slot predictions: anchor -> treatment |
| --- | ---: | ---: |
|161001|368 ->473|124 ->155|
|161002|349 ->498|116 ->163|
|161003|390 ->445|128 ->147|

The average loss improvement on new rows is concentrated in the teacher-picked
fourth cases; it is not merely improved confidence in the first three moves.
Overall offered-choice agreement rises only about0.24--0.60 percentage points
on new data, while roughly14--15% of offered choices change in either direction.
Consequently, neither "the network learned nothing" nor "training suppressed
the learned fourth move" is supported by these reports.

## The unresolved distinction

Training asks which of FOUR evaluated moves the teacher prefers. Deployment
asks which legal move to add AFTER the retained first three have been excluded.
Saved validation choices cannot establish that second ranking's quality.

Each recorded candidate set includes only one action outside its own fixed
first three. Simply renormalizing the loss over that remaining offered action
would give a one-option, zero-loss objective. This does not mean the corpus has
no exploration or alternative-ranking information: the first random alternative
is still compared with the learned fourth, and the retained old corpus has two
random alternatives beyond the original top two. These comparisons may generalize
to different streams and positions. Whether existing outcomes suffice for a
better target or more challenger coverage is needed remains an empirical question.

Rival explanations should remain distinct:

1. The update changes confidence but rarely changes the eligible fourth proposal.
2. Eligible proposals change, but the new ones are less useful or less complementary
   to the existing three.
3. Better candidates are supplied, but noisy selection or greedy035 continuation
   values fail to translate them into better full-game decisions.
4. Benefits differ across openings, later states, private information or the
   stronger opponents encountered in the new arena.
5. The full-game effect is modest or uncertain; the interim snapshot was misleading.

None is established by lower cross-entropy plus a weak provisional win rate.

## First use the matched states already saved

After Stage A is complete and its prescribed audit passes, pair the two seat
games from every initial seed. Both games start from the same state and search
stream, with treatment and anchor roles reversed. Their turn-zero records
therefore provide an exact same-state comparison. Some pairs retain a common
state/action prefix until their first action divergence.

Verify equality of full state snapshots, lawful inputs, streams, world seeds,
first-three proposals and the common candidates' outcomes before calling roots
matched. Compare all201 opening pairs, and report later common-prefix positions
separately. Do not select just disagreements or favorable games. Cluster by
whole initial pair and retain all three fits. Common-prefix coverage is biased
toward early/agreed trajectories and cannot represent every later game state.

Those records already contain both models' full-legal logits, actual fourth
proposals and their128-world outcomes. They can show how often fourth candidates
and final decisions change, and compare the fixed proposed candidates under
coupled worlds without new inference or simulation. Proposal identities are
chosen before the rollout outcomes, unlike a rollout-score maximum. A value for
a move selected by maximizing those same outcomes is optimistic and must not be
reported as independently assessed decision value. Split-world calculations,
if used, would assess reduced-budget selection, not the deployed128-world rule.

Never treat opposing roots at DIFFERENT reached states as counterfactual labels.
Never assign low values to unrecorded actions. This saved-evidence pass is a
short mechanism report, not a retrospective strength test or checkpoint search.

## If representative independent assessment is needed

Recommend one small, fixed, same-state study after the running pipeline completes,
with separate authorization and a frozen protocol. No new source arena is needed.
A concrete starting design is:

- Select36 completed Stage A seed pairs,12 per assigned fit, by a fixed identity
  hash unrelated to outcomes. Within each pair pool both seat games and take at
  most one pre-decision state per source-role(treatment/anchor) x phase cell.
  Phases are0--19,20--39,40+. This gives at most216 states. Keep private-reserve
  cases and short supports; report missing cells without chasing replacements.
- On each SAME state, use the corresponding frozen anchor, selected treatment
  and selected replay control to propose their actual highest-ranked move
  outside the common first three. Add one fixed identity-sampled random fourth
  as an exploration control. Deduplicate the union, at most seven actions.
- Evaluate the union on128 new selection worlds and128 DISJOINT assessment
  worlds, coupling all actions within each world. Preserve observer legality,
  greedy035 continuations, terminalWDL and tie rules. Each four-candidate arm
  selects using selection outcomes only; assessment scores that selected move.
- Primary diagnostic: treatment minus anchor, equally weighting the three fits.
  Secondary: treatment minus replay, random minus anchor, and selection over the
  larger union minus anchor. The larger union has more candidates and is only a
  conditional headroom diagnostic, not an equal-budget agent comparison.
- Report proposal agreement, selected-action agreement, independently assessed
  candidate/decision differences, phase coverage and split-world stability.
  Use whole-source-pair uncertainty within fit; do not treat216 states or their
  simulations as216 independent games. No claim of precise tiny effects.

The main collection ceiling is216*7*256 =387,072 continuations, with a12-root
probe ceiling of21,504. A previous103,104-continuation screen plus full replay
took about1h28m. That suggests an hours-scale diagnostic, but different states,
privacy and verification costs prevent a firm runtime promise. Propose an
eight-hour total execution ceiling, including probe and prescribed verification,
with an outcome-blind forecast before the main work. No automatic extension if
the result is uncertain.

Use the approved sampled verification policy for the mature continuation paths,
all-state/proposal/budget checks, and prespecified full stress/changed-path
fixtures. Do not automatically replay every new continuation merely because
the older screen did. Freeze the exact audit coverage before collection.

Fresh selection AND assessment blocks give the cleanest diagnostic. Existing
outcomes can alternatively be reused as selection data only if every missing
union action is evaluated on the same worlds, and assessment remains independent.
Choose that contract before measurements; never mix selectively favorable saved
branches with new ones. The completed arena and selected diagnostic positions
become development evidence for any resulting intervention.

No sweep over intermediate checkpoints is proposed by default. The initial,
selected treatment and selected replay models answer the immediate question
without opening a new selection search. Any subsequent change needs one fixed
learning intervention and fresh full-game evidence, rather than an indefinite
sequence of increasingly favorable proxy checks.

## How the diagnosis should change the next intervention

- Little deployed ranking change would motivate investigating what signal and
  coverage actually discriminate alternative proposals, not assuming more CE
  optimization solves it.
- Worse updated proposals would motivate one constrained update or alternative
  supervision test, chosen from demonstrated errors rather than a loss sweep.
- Useful proposals with weak independently assessed selection would shift focus
  toward selection noise, continuation valuation or decision allocation.
- A useful replay proposer would make the cheaper old-data route worth a fresh
  strength test; it would not become a promoted winner from this diagnostic.
- Small/uncertain diagnostic differences leave the mechanism unresolved. Report
  that, and select one next research change by its expected information value;
  do not keep expanding the diagnostic until a story appears.

These are contingent directions, not preauthorized experiments or causal claims.

## If both current gates succeed

Then this generation adds useful value beyond extra old-data practice. Prioritize
a newly specified, bounded128-world/eight-worker batching assessment before
scaling further generations. The old one-worker32-world benchmark measured1.245x
collection speedup and missed its1.25 gate; that remains a miss. Its old conditional
integration plan did not activate and is not renewed by this note.

A future assessment needs its own prospective contract, exact-trace equivalence
fixtures, production-relevant concurrent timing and actual sampled-audit cost.
Set a short engineering time cap and an adoption criterion based on amortized
time saved, before new timings. It must be safe to retain serial execution if
the cap or correctness checks fail. Do not round the historical result into a
pass, promise unmeasured GPU speedups, or optimize indefinitely before research.

Strong independent opposition remains necessary for eventual broad/superhuman
claims. Preparing compatible opponents is useful, but another large tournament
against the existing related local panel is not automatically the highest-value
next computation. Neither successful iteration nor this diagnostic establishes
strength against strong humans or an independent adversarial population.

Future long strength studies should prospectively consider a nonbinding futility
checkpoint using whole pairs and explicitly assessed power/compute tradeoffs.
That does not amend the current fixed run or retrospectively classify its early
snapshot as a failed stopping test.

## Primary local evidence

- `outputs/proposal_iteration_128/models_lock.json`, `learning_complete.json`
- `outputs/proposal_iteration_128/fits/*/result.json`, `validation_*.json`
- `docs/engineering/proposal_iteration_128.md`
- `docs/strategy/next_after_learned128.md`
- `docs/strategy/next_after_161.md`, `next_after_162.md`
- `docs/engineering/learned_proposal_screen_result.md`
- `docs/engineering/learned_proposal_arena_result.md`
- `docs/engineering/learned_proposal_attribution_result.md`
- `outputs/learned_proposal_128/report.md`
- `docs/engineering/batching_benchmark_result.md`
- `docs/engineering/batching_128_integration_plan.md`
- `docs/engineering/verification_policy.md`
