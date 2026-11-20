# Experiment160: exact155 with128 versus32 worlds

Owner authorized implementation and execution on2026-09-15: "ok lets do it".
This protocol, all160 implementation/tests, the fixed schedule, successful full
repository test evidence and transitive scientific dependencies are frozen
before the off-panel timing/integrity run. Preserve all intentionally dirty work.

## Question and intervention

Does uniformly increasing hidden-world samples from32 to128 improve the proven
155 planner's complete-game strength? The128 candidate faces unchanged155 with32.
The reference delegates directly to frozen155.choose_root, after a160 one-shot
artifact guard. The128 path repeats exactly the155 calculation with96 additional
worlds. No global155 configuration is mutated.

Both players use greedy035 proposals, at most four canonical proposals (top two
distinct, then two uniformly sampled distinct alternatives), greedy035 simulated
play for both actors, terminal WDL objective and first-maximum ties. Smaller
supports remain smaller. No learned continuation, policy/value training, adaptive
allocation, additional proposals, phase restriction or modified objective.

At the same actor-legal public-belief state and exogenous stream, proposals match
and the32 reference world seeds are the first32 of the128 candidate world seeds.
Exact155 tags, candidate nonce, seed formatting and lawful determinization remain
unchanged. Search streams are independent of hidden information and outcomes.
This comparison spends unequal compute and cannot establish equal-time strength.

## Fixed games and randomness

Main:400 complete games,200 fresh initial seeds, each with seats reversed.
For index i, pair=i//2 and candidate_seat=i%2. Environment seed is1600000000+pair.
Search stream is digest(["160-root-stream-v1", "main", pair]). Both games in a
pair share environment seed and search stream. The RNG namespaces differ from
earlier experiments and from the probe. Freeze every spec in schedule.json.

Probe:8 games from4 other seat-swapped seeds,1609000000+pair; the same stream
formula with stage="probe". The probe is excluded from the scientific endpoint.
Fixtures use stage="test",1609800000+pair and external temporary output only.
Initial-state overrides are forbidden in production.

Every actual turn saves a complete root record, including32 reference decisions.
Actual and simulated games must end naturally: completed or the established
two-forced-PASS deadlock score adjudication. The unchanged bounds are400 turn
index, four repeated state signatures and60 consecutive nonprogress moves.
Any illegal action, invalid/nonfinite outcome, cutoff, exception or integrity
failure is fatal. Preserve its evidence; no exclusion, replacement, automatic
retry, reuse, resume, fallback player or automatic panel extension.

## Primary analysis and interpretation

Candidate win=1, draw=.5, loss=0. The resampling unit is the complete seat pair,
not an individual game, root, rollout or model seed. Sort by index. Represent each
pair by the sum of two game halfpoint counts (integer0..4). Generate50,000
bootstrap resamples of200 pairs with replacement using numpy.default_rng seed
1603000000, successive chunks of1000. Divide integer resample totals by800.
The percentile2.5/97.5 quantiles form the paired95% interval. Verify all primary
statistics against an independent implementation before scientific completion.

Frozen classification, in this order:

- **positive**: observed score>=55% and paired95% lower bound>50%. Use the exact
  integer point gate,10*total_halfpoints>=11*400. This is evidence of superiority
  with a useful observed effect; it does not prove the true score exceeds55%.
- **below_material_target**: otherwise the paired95% upper bound<55%. This argues
  against a five-point advantage in this test, without necessarily implying
  regression or ruling out a smaller benefit.
- **inconclusive**: all remaining outcomes. No automatic extension follows.

Report all400 games, wins/draws/losses, score, interval, descriptive candidate-seat
scores and mean score margins. Never choose a favorable seat, phase or subgroup.
This is a bounded screen for a sizeable benefit, not guaranteed power for a true
55% score. The provisional158 pair variation suggests roughly4.7pp CI halfwidth;
the point>=55% gate itself has about50% pass probability at a true55% score.
No interim win-rate inspection, informal futility or optional stopping rule.

## Prespecified integrity work

Use frozen155's fully verified scientific identity as baseline. Reuse frozen158
independent auditing infrastructure with its transitive code freeze checked;
158's owner-stopped full replay remains incomplete and is not a quality label or
scientific prerequisite for this160 gameplay conclusion.

For every probe and main game, independently reconstruct the actual initial state,
each legal move and public transition, all state/action/trace/root hashes, every
proposed action and raw035 score, hidden-information invariance, all determinized
world hashes, every recorded ending/WDL, all arithmetic and first-maximum choices,
and final natural termination and game result. Verify the fixed independent spec.

Replay exactly two world indices per root for every offered action. Rank each
wi in range(worlds) by digest(["160-audit-world-v1", stage, game_index, turn, wi]);
take the first two, then sort the selected indices for iteration. The hash rule
uses no score, action preference or outcome. In those worlds, independently
reconstruct each complete inner trajectory, raw035 information-invariant decision,
chance event and natural termination. All worlds still receive artifact,
determinization, ending and arithmetic checks. This is **sampled inner replay**,
not full replay of every simulated trajectory. Complete the same prescribed
verification on every statistical outcome. Probe uses the identical audit scope.

All root/game/audit records and stage locks are immutable, with hashes linked
through actual-game traces and stage seals. Verify all root hashes again before
analysis. There is no mutable progress file; logs append and completed roots,
games and stage locks supply read-only status. Source and protocol identity is
rechecked before each stage and before final scientific completion.

## Execution, timing and Windows behavior

One durable background pipeline: probe collection, probe audit, fixed operational
preflight, main collection, main audit, independent statistics and final report.
Eight CPU workers, one deterministic Torch thread per worker;035 only, no GPU.
Use the matching pythonw.exe supervisor and explicitly configure pythonw.exe as
the spawn executable before every pool. Each initializer verifies no console
window and no console attachment and saves process/executable/CPU-thread evidence.
Explicit append logs preserve failures even when pythonw has absent standard
streams. Tests exercise two successive eight-worker pools and audit fixture
spawning on actual Windows before launch. A detached supervisor alone is
insufficient, as159 demonstrated; do not modify frozen159 to address this.

The probe gate uses validity and storage only, never candidate win rate. Record
collection/audit wall times and estimate main duration as50 times their sum.
Record root-latency distributions for both budgets. This is a rough forecast
from a small off-panel batch, not a timing guarantee or a selection criterion.
There is no timing-based change to the400-game design.

Before main, require free disk space >=max(10GiB,1.5*(50*probe_artifact_bytes)+5GiB).
The safety factor accounts for differing game lengths; the reserve protects other
workspace work. Insufficient storage stops before main, preserving the probe and
reporting the external blocker. No automatic deletion or compressed/reduced
evidence fallback. Integrity or operational failure also stops the pipeline.

Budget estimate before probe:36-42hours collection, approximately two days with
sampled audit. Main duration will depend on actual game lengths and hardware.
Exclusive launch/run locks prevent duplicates. All stages belong to this one
authorized experiment. No training, promotion, successor, automatic recovery or
scheduled monitoring is authorized by this protocol; monitoring stays disabled.

## Outputs and launch evidence

Outputs live in outputs/world_budget_160: manifest/schedule/freeze, versioned
preparation test evidence, parent process/launch evidence, runtime worker records,
probe/main roots, games and audits, immutable per-stage locks, preflight.json,
failure.json if needed, result.json/report.md/final_status.json on completion.
Launch only after full repository tests pass under the identical frozen identity.
Old frozen evidence and all unrelated working-tree changes remain preserved.
