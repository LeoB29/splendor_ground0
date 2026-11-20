# Experiment152: training breadth versus target replication

2026-09-09. Owner authorized `docs/strategy/next_after_151.md` with "ok do it".
Freeze this protocol, configuration, implementation and input inventories before
any experiment source generation, timing continuation or fit. Preserve all dirty
work and frozen predecessors. Development evaluation only: no fresh performance
test, arena, promotion, self-play run or automatic scientific successor.

## Fixed comparison

| Arm | Independent source states | Worlds per state | Nominal four-action paths | Fits |
|---|---:|---:|---:|---|
| A |768|32|98,304|Three frozen151 selected models|
| B |Same768|First8 of the stored32|24,576|Three new models|
| C |B plus2,304 new =3,072|8|98,304|Three new models|

C-versus-A is a fixed nominal target-simulation-budget allocation comparison;
both coverage and replication change. C-versus-B isolates adding source states
at the same per-state replication, under this fixed training recipe. B-versus-A
reduces replication on the same states. Outcome-independent data selection and
all three seeds are retained. This is not a comparison of equal wall time or
epochs, and C-versus-A alone cannot identify coverage as the causal mechanism.

## Frozen learner and original data

Use the exact151 SharedCardModel,327297 parameters, legal455-float context and
90*22 card-token interface, with the same lossless1085-feature afterstate inputs.
No architecture or feature change, added lookahead, hidden identity or actual
future draw enters the model. Baseline advantage is exactly zero, scalar(action)
minus scalar(baseline). Full action distinctions and masks remain unchanged.
See the bound `shared_cards_151.md` protocol and code for the complete schema.

A is151's three frozen selected models, updates16/17/24 for149001/149002/149003.
Their exact validation predictions must reproduce151. No A refit or new selection.
B uses the same original768 feature rows/actions, with outcomes averaged over
stored149 worlds0..7 in original order. These worlds are not chosen by their
outcomes. C concatenates B first, followed by all2,304 new rows in source-index
order. Original train/validation sources remain disjoint from all new sources.
No protected143--151 panel or outcome is used for fitting or new evaluation.

The original192 validation states and all64 existing worlds/state remain fixed.
Independently aggregate reused labels, verify masks/features/lineage and seal
B/validation arrays during preparation. This is reused development data and
cannot provide confirmatory estimates or a whole-game strength result.

B and C each use seeds149001/149002/149003. Call151's unchanged fit directly:
CPU float32, one Torch thread/model, deterministic algorithms, AdamW lr.0003,
weight_decay.0001, remaining defaults, gradient norm cap1. Draw64 source states
uniformly with replacement per update, keeping all candidate actions. Batch
generator seed = model seed+149000. Same seeds/rule across arms; differently
sized datasets cannot share identical sampled state indices. Train all4096
updates. Validate updates0..128 inclusive then multiples128 through4096 (160
rows), select earliest strict minimum validation state-mean advantage MSE,
including initial update0. No early stop, restart, tuning or best-seed rescue.

Per-state loss averages real nonbaseline candidates only. A state with no
alternative contributes zero and stays in the state denominator; padding never
adds actions or inflates the nonbaseline denominator. Exact predicted ties
choose baseline first. Preserve full histories, selected/final tensors and
independently reproduced sampler exposure: draws, unique states and unique games
at every recorded update, selected checkpoint and end. One state/game makes
unique-state and unique-game exposure identical. No ensembling; average each
model's independently chosen outcome equally across all three seeds.

Run B then C, at most two concurrent fits to bound peak memory. Each arm still
has three seeds. Audit refits use two workers and the separate151 full training
implementation, preserving exact history/tensor comparisons. Worker concurrency
does not alter the scientific update schedule or add training data.

## New source games and panel

New source game seeds1520000000+index, index0..8191. Both seats use143's exact
CollectionBot and unchanged035 inference, checkpoint SHA256
4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd.
For the first20 turns, probability.2 of sampling among the top3 concrete actions
with softmax temperature.5; otherwise first greedy maximum. Behavior RNG
Random(source_seed+143003), shared across the two seats as143. Preserve complete
state hashes, actions, greedy/selected indices, turn/actor, final state, reason,
winner and action-trace hash for every attempted game. Independent replay uses
the original035 scorer and separately implements the behavior RNG/selection.

Generate in batches of8 on8 workers and process each completed batch in source
index order. Select the first feasible balanced prefix. The final dispatched
batch may contain at most7 unused suffix games; save and audit them, include
them in the8192 maximum, and stop dispatching further batches. A non-natural
attempted source, including such a suffix game, stops the experiment. No
outcome-based replacement or source-limit extension.

Select2,304 states from2,304 distinct games,384 per cell: turn<20,20..39,>=40
crossed with actual actor seat0/1. State must be nonterminal and neither player
may already hold a blind/unknown-provenance reservation. Root blind reservations
and later ordinary private play remain legal. This restricted population does
not establish a general private-information learning result.

Use149's deterministic augmenting-path matching: games capacity1, cells quota384,
visit eligible cells0..5, use free slot or relocate assigned games in insertion
order, each cell visited at most once per augmentation. Choose the eligible
position in an assigned cell by minimum digest(["152-position-v1",index,turn]).
Sort selected games by index. No source winner, target outcome or learned
disagreement guides inclusion. Preserve examined-game eligibility counts,
hashes, assignments and the unused prefetched suffix.

Use149's unchanged four-candidate rule: exact greedy035 concrete action first,
highest-scoring distinct canonical successor next, two uniform remaining classes
with engine-ordered representatives. If fewer than four exist, use all and pad
only for tensors. Score ties preserve engine order. RNG
Random("149-candidates-v1|152-new|source_index"). Save complete actions/keys/support.
The four-action budgets are maxima; report actual counts if a state has fewer
available candidates, without exclusions or additional states to spend a remainder.

Lock the complete panel, features and all attempted source hashes. A separate
process must replay ALL attempted source games, rebuild matching and every new
feature/mask, verify disjointness/quotas/privacy, and pass the bound panel audit
before any main label is generated. Exhausted source schedule stops before fitting.

## Targets, timing and termination

Each new state receives8 independent observer-legal worlds. World seed
1521000000+sample_index*1000+world_index, sample_index0..2303, world_index0..7.
Share each initial world across the fixed candidates. Take the candidate action,
then both players follow exact greedy035 to natural termination. Label is mean
terminal WDL(candidate)-WDL(baseline), from source actor perspective. WDL is
+1/0/-1; no score-margin or truncated heuristic targets. Save every world hash,
outcome, action-trace/final-state hash, reason and move count. Main cap73,728
new continuations. Every new continuation independently replayed once.

During preparation, after freezing identity, reuse149's six fixed probe states
with4 NEW worlds each; seeds1529100000+sample_index*1000+world_index, index0..5,
world_index0..3. At most96 timing paths, independently replayed once. Probe is
not part of A/B/C or a performance evaluation. Publish only runtime, integrity
and counts; no outcome gain tunes the experiment. Estimate label collection
seconds as probe_seconds/probe_paths*73728, explicitly accounting for small
six-state/startup bias. Source generation, fits and all audits are additional.
No hard wall-time deadline; work is bounded by source/world/update counts.

The source, main-world and probe namespaces are disjoint from predecessors and
each other. Existing training-world reuse is deliberate and locked. Natural
reasons are completed and two sole-PASS forced deadlock, retaining the established
score/card tie-break. Limits:400 absolute turns,4 repeated-state visits,60
consecutive no-progress moves. Conditional counters begin at the snapshot.
Any cutoff/engine failure invalidates progress; preserve all records, no silent
replacement, removal or extra replication. Eight collection/replay workers.

## Reporting and decision

Report all three arms, all three seeds and C-minus-A, C-minus-B, B-minus-A
contrasts for validation advantage MSE and selected WDL minus035. Use float64
for reporting, equal-state means and equal-seed averaging. No confidence interval
is interpreted as confirmation on this repeatedly reused validation panel.

All four primary development conditions must pass:

- C mean advantage MSE at least3% below A.
- C mean advantage MSE at least3% below zero.
- C mean selected WDL minus035 at least+.02.
- C mean selected WDL strictly greater than A.

Also report whether C improves BOTH mean MSE and selected WDL versus B. Without
that contrast, improvement cannot be attributed to additional coverage. This
mechanism diagnostic does not replace the primary screen or establish a causal
scaling law. If only B improves, report that without promoting B post hoc.

Run the complete audit regardless of statistical failure: initial features and
independent reused target arithmetic; source-panel audit binding; independent
new label means; exact frozen A predictions; every new continuation; all six
full refits, histories and exposure counts; independent predictions/statistics.
Result and final audit are hash-bound in final status. Even a development pass
stops here and only supports a later fresh-test proposal. No automatic successor.

## Freeze and execution

`run_data_allocation_152.py prepare` freezes five152 files (runner, entry point,
verifier, tests, this protocol), runtime/configuration and explicit151 inputs.
Verify the complete151 identity, anchored manifest/audit, selection/result/
predictions, development arrays and all three selected model files.151 in turn
checks explicit150/149/148 inventories. Never invoke149's dynamic inventory
after adding152 files and never modify any frozen predecessor code.

Preparation stores B/validation and the probe, completes independent initial-data
and timing audit, then seals preflight with hashes. `run` verifies preflight,
generates and audits the complete new source/feature panel, collects targets,
seals B/C/validation, fits B/C, locks models and predictions, reports results,
then completes the independent final audit. A process lock and save-new writes
prevent duplicate starts/overwrites; failure tracebacks are preserved. No
unrequested environment workarounds, NumPy BLAS/Torch mixing, or automatic retry.

Focused and full repository tests precede freeze. Tests cover first8 nesting,
complete array-prefix preservation, unchanged training/exposure reproduction,
source behavior and RNG, bounded prefetch/failure handling, namespace/budgets,
independent continuation/statistics checks, each primary gate, panel-audit failure
blocking labels, and full run closure without performance tests on either outcome.
