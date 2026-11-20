# Experiment 149: matched deterministic afterstate advantage learning

2026-09-08. Authorized by the owner's "ok do it" following
`docs/strategy/next_after_148.md`. This protocol and its executable configuration
are frozen before timing continuations or learning targets. Preserve all dirty
work and frozen predecessors. No arena, promotion or automatic successor.

Preparation attempt1 was archived at
`outputs/afterstate_advantage_149_preflight_attempt1` after a JSON integer/string
key comparison failed during probe-source reconstruction, before any timing
continuations, learning targets or fits. Its manifest, source games, failure and
code snapshots are preserved. Canonical-digest comparison repairs serialization
bookkeeping; all scientific settings and seeds remain identical. The corrected
preflight receives a new manifest before any targets. No main launch occurred.

Preparation attempt2 passed396 composition checks and96 natural timing paths.
A synthetic (non-game) statistics check then exposed float32/float64 averaging
differences against the stricter independent verifier. Preserve that attempt's
manifest, timing and code under `outputs/afterstate_advantage_149_preflight_attempt2`.
Both statistics implementations now aggregate in float64; no scientific settings
or learning/test outcomes informed the correction. Main collection remains unlaunched.

The first main launch then stopped at source selection before any learning targets,
fits or fresh test games: cyclic game-to-cell allocation could fill only104/113
middle training states per seat (128 required). Its frozen run, code, failure and
closeout are preserved in `outputs/afterstate_advantage_149_source_schedule_attempt`.
A read-only audit of all4608 development games found598/605 middle-eligible
training games and69/70 validation games; a disjoint balanced matching can fill
every required cell. Within the already authorized pilot, correct only this
pre-target engineering defect and freeze a separate run: assign games using
eligibility-based balanced matching, not a cell assigned before eligibility.
No target outcome, learner result, fresh test game or scientific gate informed
the repair. The failed launch remains closed and is not resumed. All quotas,
source ranges, model recipes, outcome budgets and advance gates are unchanged.
Build and hash-lock BOTH complete development panels/features during the corrected
preflight, before the main launch, so their actual feasibility is established
before collecting any main target. The main runner verifies and uses those locks.

## Question and intervention

Compare prestate plus complete action (control) against explicit deterministic
afterstate plus the same action. Both use fresh, identical fully trainable MLPs,
identical paired initialization, training examples, targets and optimization.
No frozen035 embedding, prior value checkpoint, learned rules or search target.
The reference and continuation policy are exact greedy035, SHA256
4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd.

Isolated `_DeferredEnv` overrides only card-draw operations in an unchanged
engine. It never reads the future draw to compute deterministic effects. A
distinct `Afterstate` wraps a private partial state; no policy receives that
partial state. An explicit card resolves the event. The engine applies payments,
returns, gold, known purchases/reserves, nobles, turn and final-round rules.
Private-reserve acquisition is deferred; its pending count and tier are encoded.
Resolution must match the original engine with that specified card first.
Tests cover every action type, alternative chance outcomes, exhaustion, private
ownership, noble/terminal handling and hidden-world feature invariance.

## Fixed representation and training

Both feature vectors contain the ordinary 295 observable floats; six observable
location bits per catalog card (unknown, market, own reserve, opponent public
reserve, own purchase, opponent purchase); the same pre-action unknown-pool
complement; starting-seat, terminal and deterministically known winner flags;
the same pending-event kind/tier descriptor derived from the root/action; and
the complete action. Action fields are type, tier, market/reserve index, all
take/return/payment token multiplicities, noble identity, target card identity
and its points/bonus/cost. Token multiplicities divide by10, card points by5,
costs by7; one-hot categories include absent fields. No action metadata allowed.
Catalog order is card_id/noble_id sorted. Pending private reserve increases the
observable reserve count without revealing an identity. Opponent private cards
remain in the unknown complement; deck order never enters either input.

The afterstate arm transforms only the board/status/known terminal fields. Both
receive the same action and pre-action legal pool/event information. MLP widths
are input ->256 (LayerNorm/GELU) ->128 (LayerNorm/GELU) ->64 (GELU) ->1 (tanh).
Predict candidate advantage as scalar(candidate)-scalar(baseline), exactly zero
for baseline and bounded [-2,2]. Both arms have exactly the same parameter count.
With the90-card/10-noble catalog, each input has1085 floats and each model has
320001 trainable parameters. No pretrained parameter enters either model.
This compact flat architecture is one controlled test, not an exhaustive test
of every possible afterstate architecture.

Three paired seeds:149001,149002,149003. AdamW lr=.0003, decay=.0001, gradient
norm cap1, 4096 updates, 64 uniformly sampled source states per update with all
their candidates. RNG for batches is training seed+149000. CPU float32, one Torch
thread, deterministic algorithms. Validate at0 and every128 updates; retain
earliest strictly minimum validation state-mean advantage MSE. Run full schedule;
no early stopping, restarts, tuning or best-seed selection. Each state's MSE
averages only real nonbaseline candidates. A forced-single-action state has loss
zero and still remains in the fixed state denominator. Padded entries are masked.
Report selected/final histories and per-seed decision results. No ensembling:
each model chooses independently, then average outcomes across paired seeds per
source state. Exact predicted ties choose the first candidate, always baseline.

## Fixed source and candidate selection

768 training/192 validation/192 fresh test states; one per source game, six equal
cells: turn<20,20<=turn<40,turn>=40 crossed with actual actor seat0/1. Initial
states must contain no already-held blind/unknown-provenance reserve for either
player. All root actions, including RESERVE_DECK, stay legal. Subsequent policy
play also retains ordinary private reserves. This does not validate general
private-belief learning. Terminal states are ineligible.

Train:143 development game indices0..4095. Validation:4096..4607. Verify records
against143's sealed development summary; never use its4608..5119 old test games.
Scan each source range in index order. Compute the cells with eligible positions
in that game. Use a deterministic augmenting-path matching, game capacity1 and
cell capacity quota: visit cells0..5, use a free slot, otherwise try relocating
existing assigned games in insertion order. Visit each cell at most once per
augmentation. Stop at the first source prefix for which every cell reaches quota.
Order selected games by index, and select within the assigned cell by minimum
SHA256 digest of ["149-position-v1",split,index,turn].
Exhausting the fixed schedule fails; do not extend it. No source winner, action
gain or model disagreement influences selection. Record every examined source,
eligible count and hash; preserve rejected sources. Non-natural source games
stop the pilot, rather than being replaced based on outcomes.

Fresh exact-greedy035 test source seeds1490000000+index, indices0..1023, same
balanced matching and stopping rule. Generate only after all six selected
models/data are hash-locked. Probe sources separately use1499000000+index,
indices0..59, one per cell. No test sources enter the probe.

Candidates: exact concrete greedy035 action first; highest-scoring concrete
action in a distinct canonical successor class second; two uniformly sampled
remaining canonical classes, using their engine-ordered representatives. All
classes if fewer than4. Score ties use original legal-action order. Uniform RNG
is Python Random("149-candidates-v1|split|source_index"). Preserve complete
actions and canonical keys. This is a four-candidate panel, not all-action
optimality or an oracle comparison. Lock panels/features before their targets.

## Outcomes and exposure limits

32 independent worlds/train state,64/validation,128/test. Worlds use observer-
legal determinization, seed=split_start+state_index*1000+world_index, starts
1490100000/1491100000/1492100000 respectively. Probe start1499100000. These
namespaces are disjoint. Couple the exact initial world across candidate actions;
do not reuse worlds across splits or condition them on action outcomes.

Take candidate action, then both players use exact greedy035 through natural
terminal. Z is +1/0/-1 from source actor view. Label each nonbaseline candidate
by mean(Z_candidate-Z_baseline), range[-2,2]. Stored per-world WDL, termination,
trajectory/final-state hashes and move counts allow full independent replay.
No score-margin or truncated heuristic outcomes enter training. Completed and
forced-two-sole-PASS deadlock are valid, using established score/card tie-break.
Limits:400 absolute turns,4 repeated state visits,60 no-progress moves; invalid
continuations are preserved and stop learning/advance, never silently replaced.
Counters start at the snapshot as this is a conditional continuation study.

Main collection <=245760 continuations, plus exactly4 worlds for each of6 probe
states (<=96 continuations). All main continuations replayed independently once
in the audit; all six full fits independently reproduced once. Eight collection/
audit workers; six fit/probe workers. No automatic restart after launch; failure
artifacts preserve the cause. No outcome-based extension. Probe publishes timing,
integrity and counts only, no outcome gain; estimate full cost before main launch.
No hard wall-time ceiling: continuation/source/update counts bound the work.

## Fresh evaluation and advance rule

Lock every model's predictions and chosen candidate on fresh test states BEFORE
collecting test continuations. Evaluate afterstate-selected minus control-selected
WDL and afterstate-selected minus baseline WDL, averaged over3 paired seeds per
state. Resample192 whole source games,50000 percentile bootstrap replicates,
NumPy default_rng149999, in1000-replicate chunks shared across contrasts. Report
all statistics in float64, including the within-state mean over model seeds.
Report
control minus baseline, advantage MSE difference and per-seed results too.
No model seed counts as an additional independent game.

Advance requires BOTH primary mean WDL gains>=.02, BOTH CI95 lower bounds>0,
BOTH CI half-widths<=.02, advantage-MSE afterstate-minus-control CI95 upper<0,
and all integrity/natural-termination audits passing. .02 WDL=1pp expected score
under these specific continuations; not whole-game strength. Wider intervals
are precision-limited. MSE alone cannot pass; do not use agreement with a noisy
empirical best action as optimal truth. No retuning/extra seeds/games on failure.

Separate-process verifier reconstructs all source selections and features,
checks chronological locks and hashes, replays every world/action with independent
rollout code (root via afterstate composition, later greedy score argmax), verifies
every result, refits all6 models with bitwise tensor/history checks, and separately
recomputes statistics/gates. Preserve148's explicit frozen inventory; never invoke
its dynamic identity after adding149 source. Final status binds result and audit.

Even a pass authorizes only a later integration/gameplay proposal. Negative or
inconclusive findings cannot rule out other encoders or optimal-play benefits.
Keep champions084/035 and the long-term improving self-play objective intact.
