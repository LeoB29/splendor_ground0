# Full-game learned versus original-ranking proposal comparison

Owner authorized implementation and execution on2026-09-19: "ok let's do it".
This covers the separate12-game timing/integrity probe and fixed402-game main
study when operationally reasonable, with every continuation replayed.
Recommendation: `../strategy/next_after_proposal_arena.md`. Preserve all prior
frozen and intentionally dirty work. No training, promotion or monitoring.

## Players and isolated change

Candidate is the exact successful learned32 player from the completed proposal
arena, directly delegating to its frozen `choose_root` with the assigned fit.
Models remain161001/update563,161002/update488,161003/update563; exact checkpoint
hashes and selection/full-fit audit evidence are inherited and reverified through
the completed arena/screen manifests. The assigned model stays fixed throughout
both games of its seed pair. No best-fit selection or model ensemble.

Both players construct the original155 list and preserve its first three
candidates in order:035's top two distinct canonical groups and first uniformly
sampled alternative. Only the fourth suggestion differs:

- Candidate: assigned161 model's highest group-max logit outside the common three.
- Control: highest035-ranked canonical group outside the common three. Sort
  concrete legal actions by descending035 score, original legal index on ties,
  and take the first occurrence of each group. The first remaining group wins.

The control exactly matches the completed screen's raw_control intervention.
Preserve top-two original035 concrete representatives and first-legal canonical
representatives for all other groups. Preserve original offered order, including
the common random third slot before the new fourth slot. For support below four,
both keep the entire original list unchanged. Coincident choices remain included.
All neural inference uses the complete lawful legal-action set. The control's
deployment path uses035 only; it never calls a161 model. Independent verification
may run all three161 models to reconstruct the frozen screen rules.

Both planners use32 common observer-legal worlds per offered candidate, terminal
WDL, greedy035 continuations for both simulated players and first-offered ties.
No learned continuations, extra worlds, adaptive allocation or batching. Include
all ordinary positions and own/opposing private reserves. Same sampling budget
does not itself prove equal wall time. Both035 and161 are learned policies; this
tests newer distilled ranking beyond the original ranking, not learning versus
no learning.

## Fresh schedule and randomness

Main:201 fresh seeds with both candidate seats,402 games. Probe:six different
seeds with both seats,12 games. Sort pair indices by digest
`["learned-proposal-attribution-fit-v1", stage, pair]`; rank modulo three assigns
models in ascending seed order. Main has67 pairs per model, probe two per model.
Index=2*pair+candidate_seat. Model strata use disjoint fresh starting seeds.

Environment seeds are2050000000+pair for main and2059000000+pair for probe.
Root stream is digest(`["learned-proposal-attribution-stream-v1", stage, pair]`),
shared within each seat pair and between players at matching lawful observations.
Retain155 nonce/world tags: nonce digest(`["155-candidate-v1", stream, belief]`)
and `155-world-v1|stream|belief|world_index:02d`, indices0..31. Belief is the hash
of the acting player's lawful information, never true hidden deck order or
opposing blind-card identity. Seed/tag namespaces were checked against the
repository before implementation.

Synthetic checks use an isolated test stage,2059800000+pair, three seat pairs,
and crafted near-terminal snapshots. Production forbids snapshot overrides.
Synthetic stages cannot write inside this experiment's production output.
No old experimental games or probe outcomes enter the new main endpoint.

## Statistics and decisions

Candidate score=win1/draw0.5/loss0. Average seats within each pair,67 pairs within
each fit, then equally across all three fixed fits. Use50,000 whole-pair bootstrap
samples within fit: numpy.default_rng seed2053000000;50 chunks; each chunk draws
1000x67 indices for each fit in ascending seed order. Sum integer pair half-point
counts0..4 over all sampled strata and divide by804. Use linear empirical
2.5/97.5% quantiles. Independent statistics rebuild the same endpoint separately.

Report superiority (lower bound>50%), practical advantage (observed score>=55%
and superiority), and material target excluded (upper bound<55%) separately.
Classification priority: `positive_practical_progress`, then
`positive_below_observed_target` for superiority alone, then
`below_material_target` for upper<55%, otherwise `inconclusive`.
Observed55% is not proof the true score exceeds55%; a null is not equivalence.
This bounded sample may leave a small effect unresolved. Intervals condition on
these fixed checkpoints and opponent. Per-fit/seat/margin/latency summaries are
descriptive and cannot select models, subsets, policies or extensions.

Do not summarize interim strength. Complete every prescribed main game and audit
before reporting. No repeat until significant, promotion or automatic successor.
A win establishes full-game benefit over this original-ranking control at32
worlds;128-world transfer, broader strength and repeated learning stay untested.

## Integrity and preserved dependencies

New core/verifier: `scripts/experiments/learned_proposal_attribution.py` and
`scripts/experiments/verify_learned_proposal_attribution.py`. Existing src and
analysis files are untouched. Verify completed ancestor identities and recorded
evidence; do not widen their historical inventories with these new files.

The independent verifier reconstructs fixed specs, raw/learned proposals,
complete legal inputs, original representatives, every determinized world,
every inner trajectory, every outer action/transition/result and final statistics.
Hidden-order/opposing-private-card changes must leave model inputs, logits,
proposals and schedules invariant at every actual root. World reconstruction
uses catalog minus legally known cards and preserves the actor's own reserves.
Every continuation uses149's separate replay including afterstate/refill
composition checks. The frozen rules engine, canonical primitives, encoders and
neural arithmetic remain shared trusted dependencies, not a second rules engine.

Only natural completion or the established two-forced-PASS adjudication is
valid. Turn400, repetition4, no-progress60, illegal/stalled moves, exceptions,
nonfinite values or altered artifacts are failures. Preserve partial/failed
evidence; no exclusion, replacement or automatic retry. Cancel pending tasks
on failure; already-running game tasks may finish and preserve evidence.
Every replay is required regardless of eventual strength. Finalization rechecks
game/root/audit/lock hashes and frozen code/model/runtime identities.

## Execution and cost gate

Entry `run_learned_proposal_attribution.py`; output
`outputs/learned_proposal_attribution`. Commands are `check --checks <new-folder>`,
`prepare --checks <checks.json>`, `launch`; `run` is the internal supervisor.
Current passing checks and the complete audited successful predecessor are
required before freeze. Bind src, analysis and new experiments Python inventories,
tests, protocol/recommendation, exact models/parents, schedule and runtime.

Use a durable detached pythonw supervisor, eight hidden pythonw CPU workers,
one deterministic Torch thread each, checked console absence, append logs and
write-once stage locks. Launcher waits for a matching acknowledgement. No GPU
dependency, duplicate run, watcher or recurring monitor.

Probe completes12 games and full replay before cost assessment. Forecast main
time=(402/12)*(probe collection+audit wall time)*1.25. Forecast main storage=
1.5*(402/12)*(all probe artifact bytes)+512MiB. Proceed automatically within the
owner's authorization only when forecast<=60h and projected remaining disk>=5GiB.
Otherwise write an operational hold and exit before main. The conservative
ceiling is separate from the provisional30--40h estimate. Timing, storage and
integrity alone determine this gate; no probe score is summarized or used.

Main runs402 games, fully replays every continuation, verifies final statistics,
writes the result and exits. No silent audit reduction, training, tuning,
promotion, extension, scheduled monitoring or further experiment follows.
