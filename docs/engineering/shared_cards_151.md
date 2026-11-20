# Experiment 151: shared-card afterstate advantage learner

2026-09-09. Owner authorized `docs/strategy/next_after_150.md` with "ok, let's
do it". Freeze this protocol, executable code, tests and all input inventories
before any actual fit. Preserve all dirty work and all frozen predecessors.
One recipe, three seeds, a development gate followed only on audited pass by
the conditional fresh test below. No arena, promotion or automatic successor.

## Question

Can shared semantic card processing improve useful move choices from the same
observable afterstate information and paired terminal-WDL targets as149/150?
Treatment is a fully trainable shared-card encoder. Control is the three frozen
150 dense-selected flat afterstate networks, selected at30/13/16 updates for
seeds149001/149002/149003. No control refitting or best-seed selection.

This is a bundled representation/architecture intervention, not an isolated
test of attention. Public card attributes are fixed game rules. The treatment
receives no new outcome information, hidden identity, actual future draw,
engineered lookahead or extra source states. Negative historical entity/value
experiments remain relevant; this is not the programme's first entity learner.

## Exact representation and architecture

Start exclusively from149's1085-float legal afterstate vector. Context retains
the original295 observable floats followed by all160 fields at offsets925:1085:
455 floats total. This preserves ordered visible-card/player/reserve/noble
information, resources, seat/terminal/pending-event fields and the complete
candidate action, including payments/returns, market/reserve index and targetID.

Each of90 catalog cards has22 floats: costs in five TOKEN_COLORS divided by7;
five bonus indicators; points/5; tier indicators1/2/3 (14 public attributes);
the six original observed-location indicators at295+6*k (unknown, market,
own reserve, opponent public reserve, own purchase, opponent purchase);
the original pre-action unknown-pool indicator835+k; and the complete-action
target indicator at984+k. Opponent private reservations stay in the unknown
complement; a pending private acquisition reveals no identity. Canonical rows
use the frozen149 CARDS order. The14 attribute tuples uniquely identify all90
cards. An exact inverse reconstructs all1085 original floats from canonical
tokens/context and checks duplicate target tags. No simulator is called by
the restructuring function.

Shared card MLP:22->32,GELU,LayerNorm32,32->32,GELU. Context MLP:
455->256,LayerNorm256,GELU,256->128,LayerNorm128,GELU. Attention query128->32,
key32->32,value32->64, each with bias. One attention pool across90 cards,
softmax(dot(key,query)/sqrt32); concatenate the64-float weighted value with
the32-float mean encoded card and128 context (224 total). Scalar head:
224->448,LayerNorm448,GELU,448->128,LayerNorm128,GELU,128->64,GELU,64->1,tanh.
All linears have bias; default PyTorch initialization and LayerNorm settings.
Exactly327297 trainable parameters versus320001 flat (+2.28%, within10%).

Prediction is scalar(candidate)-scalar(first/baseline), exactly zero at baseline
and bounded[-2,2]. Internal card-row permutation must preserve output within
float32 reduction tolerance. Context is held fixed during that permutation;
this does not impose color/market-slot/temporal game symmetries. The retained
flat context can still learn identity-specific rules; card processing is shared.

## Development data, fitting and gate

Reuse only149's frozen768 train/192 validation states, one per source game with
disjoint splits, their four fixed canonical candidates, masks, and existing
32/64-world paired terminal-WDL labels. No143-150 protected test panel/outcomes
are evaluated by this learner. Hash verification of existing inventories is
permitted and is not new test inference. No new development simulations.

All three seeds149001/149002/149003 run4096 updates. CPU float32, deterministic
Torch algorithms, one Torch thread/model, three fit workers. AdamW lr.0003,
weight_decay.0001, other defaults, gradient norm cap1. Each update draws64 states
uniformly with replacement using a separate Torch generator seeded seed+149000,
retaining all candidates/state. Same streams/settings as150; architecture
changes preclude identical weight initialization. No dropout or RNG-consuming
validation. Per-state MSE averages real nonbaseline candidates only; padding
excluded; a state with zero nonbaseline actions contributes zero and stays in
the state denominator. Average equally across states.

Validate on updates0..128 inclusive, then every128 through4096 (160 rows).
Retain the earliest strictly minimum validation MSE including initialization0.
Run the entire schedule, preserve selected/final tensors and both learning
histories. No requirement that a selected checkpoint occur early. No restarts,
architecture/capacity/optimizer grid, extra seeds, best-seed rescue or ensembling.
Each selected model chooses independently with first-candidate/baseline-first
ties and padded actions masked. Outcomes are averaged equally across all seeds.

Development pass requires ALL four, with metrics reported in float64:

- Equal-seed validation advantage MSE at least3% below the frozen flat control.
- That same MSE at least3% below the zero-advantage predictor.
- Equal-seed selected WDL minus035 at least+.02.
- Selected WDL strictly greater than the flat control's selected WDL.

No inferential claim from these reused validation outcomes/checkpoint choices.
Separate auditor rebuilds all960 features, independently aggregates stored
world outcomes, reproduces flat predictions exactly from150, independently
checks selected predictions/statistics and all three COMPLETE fits bitwise
(histories, selected/final weights). Audit occurs even if development fails.
Freeze model selection, data, predictions, result and audit bindings before
conditional work. Any development/integrity failure stops before new games or
continuations. Do not stop a running audit just because the statistical gate fails.

## Conditional stage, frozen before training

Only a development pass AND bound independent audit pass may enter this stage.
First an outcome-blind timing probe reuses149's six already fixed probe states,
four new worlds/action set per state (<=96 continuations). World seeds
1519100000+sample_index*1000+world_index, world_index0..3. Report elapsed time,
count and rough collection estimate seconds/count*196608, with worker-startup
and small-probe caveats; independent audit is additional. Probe outcomes never
alter choices, sample size, thresholds or model selection. No measured runtime
is claimed before this probe. No wall-time ceiling; explicit counts bound work.

Generate fresh exact greedy035 source games only after selection is locked.
Source seeds1510000000+index, indices0..2047, with no extensions. Greedy035 SHA:
4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd.
384 states from384 distinct games,64 in each phase(<20,20..39,>=40) crossed with
actual actor seat0/1. Nonterminal eligibility; neither player may already hold
a blind/unknown-provenance reservation at the source snapshot. New blind root
reservations and later ordinary private play remain legal. This restricted
conditional population is not general private-belief or whole-game evidence.

Retain149's deterministic augmenting-path matching: source games in index order,
capacity one state/game; cells0..5; fill free slots or relocate assigned games
in insertion order, each cell visited at most once per augmentation. Stop at
first feasible prefix filling all quotas. Within an assigned cell, choose
minimum digest(["151-position-v1",source_index,turn]); order samples by source
index. Record all examined sources, hashes, eligibility counts and rejected
games. Any non-natural source halts, never replaced based on outcome. Exhausted
source budget halts. No model disagreement or source winner guides selection.

Four candidates as149: exact concrete greedy035 first, highest-scoring distinct
canonical successor second, then two uniformly sampled remaining classes with
engine-ordered representatives; use all if fewer than four. Score ties preserve
engine order. RNG Random("149-candidates-v1|151-test|source_index"); full actions
and canonical keys recorded. No empirical target maximization at inference.
Lock panel, structured data and every fixed model prediction/choice BEFORE
collecting fresh test outcomes.

128 independent observer-legal worlds/state, coupled across candidates. Seeds
1511000000+sample_index*1000+world_index, index0..383, world_index0..127. These
are distinct from source/probe/prior namespaces and independent of model seed.
Each root action followed by both exact greedy035 players through terminal;
WDL+1/0/-1 from source actor view. At most196608 main continuations. Completed
and two sole-PASS forced deadlocks are natural, retaining established score/card
tie-break. Preserve149's400 absolute-turn,4 repeated-state,60 no-progress limits;
any invalid cutoff stops the experiment, no exclusions/replacements. Counters
start at the source snapshot. Save every world/outcome/trajectory digest/moves.
Eight collection workers and eight independent replay workers.

Report selected shared-minus035, shared-minus-flat WDL and shared-minus-flat
advantage MSE. Average the three paired model seeds within each state first.
Percentile bootstrap384 whole source games,50000 replicates, default_rng151999,
50 chunks of1000, common draws for all three contrasts, float64 arithmetic.
Seeds and continuations do not count as additional independent source games.
Fresh pass requires BOTH WDL means>=.02, BOTH95% CI lower bounds>0, BOTH CI
half-widths<=.02, MSE difference95% CI upper<0, and every integrity check.
.02 WDL equals1 percentage point expected score under these continuations.
Wider intervals fail the precision gate; do not extend samples or relax gates.

Independent audit rebuilds source matching with exact greedy action checks,
features/masks, locked choices, chronological locks, independently aggregates
outcomes/statistics, and replays every test AND probe continuation once with
the established independent149 rollout (root via afterstate decomposition).
Each primary fit was already independently reproduced before this stage.
Final status binds result/audit hashes. Any fresh failure stops this recipe;
even a pass only supports a later gameplay proposal. Champions084/035 unchanged.

## Identity and operation

151 has isolated entry point `run_shared_cards_151.py prepare|run`. Preparation
freezes six new files (model, runner, entry point, verifier, tests, this protocol),
runtime/config,150 manifest/audit/selection/result/predictions/control files and
149 probe panel.150's explicit frozen inventories verify all reused149/148
dependencies/artifacts safely without adding151 to their dynamic inventories.
Never invoke149's dynamic identity after adding151. Preparation stores separate
lossless structured arrays and their hash lock; no actual fits or continuations.
Execution is process-locked, save-new artifacts prevent silent overwriting,
failures preserve tracebacks. No frozen prior source modification.

Tests cover lossless all-legal-action projection in five representative states,
hidden-world invariance, card permutation invariance and gradients, independent
synthetic fit reproduction, earliest ties, separate statistics, every development
gate, fresh precision, namespace/budget constraints, independent terminal replay,
and run-sequence prevention of fresh work without both development/audit pass.
Run focused151/150/149 and full repository suites before freezing and launching.
