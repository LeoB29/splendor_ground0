# Learned fourth-proposal full-game experiment

Owner authorized implementation and execution on 2026-09-18: "ok let's do it".
This covers the separate 12-game timing/integrity probe and fixed 402-game main
study if operationally reasonable, including full replay. Recommendation:
`../strategy/next_after_learned_proposals.md`. All existing work is preserved.

## Fixed intervention and players

Candidate changes only the fourth root proposal of the original Experiment155
32-world planner. Keep its exact first three (035 top two distinct canonical
groups plus first uniform alternative). Use the assigned frozen161 model's
highest-scored group outside those three as the fourth. Score all lawful concrete
actions in one complete legal-action input; group score is maximum concrete
logit. Canonical group order breaks learned ties. Retain original035 concrete
representatives for the top two and first-legal representatives for all other
groups. Below four groups keep the entire original support unchanged.

Use all three selected161128-label models: seed161001 update563, seed161002
update488, seed161003 update563, with exact checkpoint hashes inherited from
the completed learned-proposal screen and verified through its model/fit audits.
The assigned model stays fixed throughout each pair and both games. Only that
model supplies the candidate's proposal inference; independent replay may score
all three to reconstruct the original screened implementation.

Reference calls unchanged155.choose_root directly, behind a new write-once
guard. Both players use32 common observer-legal worlds per offered action,
greedy035 continuations for both simulated actors, terminal WDL and first-offered
ties. No learned continuation, halving, batching, new training or extra world.
Same continuation count is not an equal-wall-time claim. Include every ordinary
game position, including own and opposing private reserved cards.

## Schedule and information boundaries

Main:201 fresh seeds, two candidate seats,402 games. Probe:six separate seeds,
both seats,12 games. Sort pair indices by digest
`["learned-proposal-arena-fit-v1", stage, pair]`; rank modulo three assigns the
models in ascending seed order. Main has67 pairs per fit, probe two per fit.
Index=2*pair+candidate_seat. Different fits use distinct initial seed panels.

Environment seeds:main2040000000+pair; probe2049000000+pair. Root stream is
digest(`["learned-proposal-arena-stream-v1", stage, pair]`), shared within each
pair and between players at matching legal observations. Preserve155's candidate
and world tags: nonce digest(`["155-candidate-v1", stream, legal_belief_hash]`)
and `155-world-v1|stream|legal_belief_hash|world_index:02d` for0..31. Hidden deck
order and opposing blind identities never enter the legal belief or model input.

Synthetic tests use a separate `test` stage,2049800000+pair, three pairs and
constructed near-terminal snapshots. Production refuses snapshot overrides;
test stages cannot write to production output or any descendant of it.
Names/seed namespaces were checked against existing source and protocols.

## Endpoint and decision rules

Main score is win1, draw0.5, loss0. Average the two seats per pair,67 pairs per
fit, then equally across all three fixed fits. No fit selection or ensemble.
Use50,000 stratified whole-pair bootstrap resamples with NumPy default_rng
seed2043000000. For each of50 chunks, draw1000x67 indices for each fit in
ascending seed order. Integer pair half-point totals0..4 sum over strata and
divide by804; linear empirical2.5/97.5% quantiles give the95% interval.

Report separately: superiority (lower bound>50%), practical progress (observed
score>=55% and superiority), and material target excluded (upper bound<55%).
Classification priority is `positive_practical_progress`, then
`positive_below_observed_target` when superiority alone passes, then
`below_material_target` when upper<55%, otherwise `inconclusive`.
Observed55% does not prove the true score exceeds55%. This bounded screen may
leave smaller benefits unresolved. No inference of expected match win rate from
the prior single-decision effect. Probe results never enter these statistics.

All main games and audits complete before summarizing strength. Fit, seat,
score-margin, private-reserve coverage and latency summaries are descriptive.
No post-hoc subgroup policy, sample extension, retry, promotion or successor.
A win supports this candidate package against155; raw-ranked-control attribution,
128-world strength, broad generalization and superhuman status remain untested.

## Complete verification

New modules `scripts/analysis/learned_proposal_arena.py` and
`scripts/analysis/verify_learned_proposal_arena.py` are isolated from frozen src.
The verifier independently rebuilds schedule, lawful inputs, group proposals,
model scores, every world, every trajectory, choices, outer games and statistics.
Reference/raw inputs and candidate learned inputs are invariant under hidden
deck reorderings and swapped opposing private cards at every real root.
Independent catalog-minus-known world construction preserves own known reserves.

Every collected continuation is replayed through149's separate routine, including
deterministic afterstate and refill composition verification. No sampled-inner
substitution. All outer games and traces are reconstructed through completion.
Trusted shared dependencies include engine, serialization, canonical grouping,
encoding and neural arithmetic; this is not a second Splendor rules engine.

Natural completion and the established two-forced-PASS score/tiebreak adjudication
are valid. Turn400, repetition4, no-progress60, stalled or illegal actions,
exceptions and nonfinite scores are failures, never targets. Preserve partial
evidence and stop; no exclusion/replacement or automatic recovery. Pending pool
jobs are cancelled on failure, while already-running game tasks may finish.

Main statistics are independently rebuilt from independently replayed game rows.
Finalization rechecks all game, root, audit and lock hashes as well as the frozen
identity. Every audit is required regardless of eventual win rate.

## Runtime, bounds and launch

Entry `run_learned_proposal_arena.py`; output `outputs/learned_proposal_arena`.
Commands: `check --checks <new-folder>`, `prepare --checks <checks.json>`,
`launch`; internal `run` is the durable one-shot pipeline. Current passing checks
and completed audited parents are required before freeze. Bind all existing src
Python files, analysis Python dependencies, tests, this protocol, recommendation,
models, selected-fit audit evidence, schedule and local runtime. Recorded ancestor
inventories are verified without adding new files to their historical identity.

Supervisor and both collection/audit pools use matching pythonw.exe, no console,
eight CPU workers and one deterministic Torch thread each. Worker health records
and append logs are saved. Launcher waits for a matching startup acknowledgement.
Each stage has a write-once launch and completion lock; duplicate launch is refused.

The probe completes collection and full audit before operational forecasting.
Estimate main runtime as(402/12)*(probe collection+audit elapsed)*1.25. Estimate
main storage as1.5*(402/12)*all probe artifact bytes+512MiB. Proceed only if the
forecast is at most60hours and projected remaining disk is at least5GiB. This is
a conservative operational ceiling around the provisional30--45hour estimate,
not a guarantee or permission to change scientific settings. Otherwise save an
operational hold and exit before main. Timing/storage alone determine this gate.

The pipeline then runs402 games, fully replays them, writes independently checked
results and exits. No scheduled monitoring, automatic extension, training,
model promotion or follow-on experiment. Implementation and checks are additional
to the estimated execution duration.
