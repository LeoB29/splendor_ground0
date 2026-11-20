# Bounded PUCT32/64 timing and one-second strength comparison

24 October 2026. User authorized implementation and execution of timing checks
for 32 and 64 simulations, then a frozen 200-game comparison if one qualifies.
Larger-data work remains deferred. All earlier attempts and frozen evidence
remain unchanged. This is a separate version and one bounded attempt.

## Question and fixed players

Does a modest increase in search improve the verified Entity PUCT16 player
within the same one-second complete-move allowance? Both arms use the same
historically trained checkpoint, corrected rules, float32 CPU arithmetic,
three-world pool, depth64, cpuct0.4, dynamic FPU0.02965, no noise, batchone,
lawful observation seed, auxiliary decisions, safe support, and fresh agents
per phase. The only scientific change is simulations32 or64 versus16. Reference
profile and binary bytes remain exact. The challenger uses a byte-identical
copy of that binary at a separately authenticated path, for unambiguous process
ownership. One immutable model is shared by two persistent native transports;
no search tree, game history, mutable RNG or inference cache crosses decisions.
Both arms are searched players; prior greedy argmax-specific checks do not apply.

## Prospective calibration and selection

Before model calls, bind all sources, profiles, fixtures, tests, this contract,
runtime, schedules and prior-preservation inventories in freeze.json. Test
quotas32 then64, each in a fresh supervised worker. Each quota receives the same
23 saved roots (18core plus5edges), original and hidden-only variant, two passes,
both challenger and reference16:184timed complete requests per quota. Reverse
orders on pass2; preserve all observations, decisions, timings and work receipts.
Actual entry-point corrected catalogue/noble/private/payment/return/final-round
counterexamples, source guards, native/host parity, auxiliary equality, explicit
legacy rejection, repeat/privacy checks and historical reference16 reproduction
must pass. Late heuristic leaf counters remain distinct from neural callbacks.

A candidate qualifies only if all correctness checks pass and every completed
panel response from both arms arrives within0.700s. The true whole-move deadline
is1.000s and internal fail-closed guard0.900s; neither changes. Preserve slower
responses. A genuine player timing failure disqualifies that quota without a
retry; it does not authorize skipping correctness checks or replacing evidence.
The second predeclared quota may still be assessed. A non-timing correctness,
rules, source, supervisor or resource failure stops the entire attempt.

Choose64 if eligible, otherwise32 if eligible, otherwise stop with no comparison.
Selection uses timing/check status only, never matchup outcomes. Before any
game, save comparison_freeze.json binding the selected quota, source freeze,
complete calibration evidence and exact fresh schedule. This is an additional
selection seal inside the original clocks; it is not a new execution allowance.
There is no automatic fallback after this selection or any operational failure.

## Resource and clock envelope

Retain six hours total from one monotonic and UTC origin, including RAM waits,
qualification, selection sealing and cleanup. Stage caps: qualification1hour,
operational30minutes, main4hours, final30minutes. Collection cutoff is absolute
hour5.5. Initialization has at most60s per worker inside stage/global clocks.
One active decision on the same assigned performance-core logical CPU, one
numerical thread; own and verify all processes. Combined sampled worker/helper
RSS limit2GiB; require2GiB available RAM stable60s initially, then between-move
admission. At most10minutes cumulative waiting; never pause an active clock.

Evidence cap512MiB, with64MiB reserved for final checks and448MiB prefinal.
Require7GiB free disk initially and retain5GiB. Reuse the verified lossless gzip
log codec byte-for-byte, preserving logical16MiB stream limits, raw/compressed
hashes, bounded decompression and cleanup verification. Preserve failures; no
deletion to gain room. No sample reduction, timeout reset, retries, after-outcome
extension or cap increase. Source preparation is model-free; all new model calls
belong to the bounded execution.

## Operational gate and main games

After selection, run exactly three fresh swapped-seat pairs (six games), excluded
from strength statistics. All must finish naturally and replay with every move
within0.700s. Keep aggregate outcomes closed for the gate. Project200game cost
from complete six-game cost times(200/6)*1.35; apply the same factor to variable
evidence and independent replay cost. Add all fixed evidence and measured fixed
final overhead, including source rehash, independent arithmetic, up to48 complete
decision reproductions, log verification and30s cleanup/write reserve. Main
collection must fit its remaining4hours and absolute cutoff, prefinal448MiB,
and final verification30minutes. A failed forecast stops; no quota switch.

Main:100 fresh worlds, seats swapped,200games. Use canonical JSON [domain,stage,
pair_index] full SHA256 big-endian seed. Main domain
splendorust_puct_budget_1s_20261024_v1; operational domain
splendorust_puct_budget_1s_20261024_v1_operational. Stage indices0..99 and0..2,
global game_index2*pair+game, challenger seat(pair+game)%2. Check discovered
schedule/manifest/seedregistry files up to8MiB for collisions; disclose omissions.
Never reuse the completed greedy-comparison worlds as independent assessment.

During main games, the1s arrival deadline determines timeout loss. Attributable
player timeouts/illegal/malformed responses are losses, without replacement.
Infrastructure/rules/source/resource failures stop incomplete. Actual unresolved
deadlocks or2,000plies do not create invented draws/winners. All internal forced
pass/late heuristic conventions remain explicit and unchanged. Corrected end
of round rules alone determine natural outcomes and half-credit draws.

## Verification and analysis

All evaluator workers and parent authenticate verify_runtime_sources() once at
startup. Qualify the actual complete entry point, not only a helper. Independently
replay every move in all206games with the dictionary ledger, binding lawful
observations, actions, outcomes, original trusted clocks and exact quota. Record
simulation, leaf evaluation and host neural callback counts separately. Select
at most48complete decisions across arm,seat,phase and hidden-reserve category by
fixed hash domain splendorust_puct_budget_1s_20261024_audit_v1; reproduce exact
scientific outputs. Audit all worker logs and cleanup receipts, seal game records,
independently recompute statistics and reverify frozen inputs before acceptance.
Native internal hypothetical transitions remain an authenticated source/numerical
parity trust boundary, not an exhaustively replayed internal tree.

Score challenger wins1,draws0.5,losses0. Use100pair means,50,000paired bootstrap
replicates, NumPyPCG64 seed2026102402, linear percentiles. Retain the existing
nondegenerate and degenerate-interval interpretation rules. Useful supported
improvement requires score>=55%, paired95% lower bound>50%, each seat>=50%, and
all integrity gates. Smaller/unresolved differences do not imply equivalence.
No Elo, superhuman or automatic reference promotion claim follows. No new model,
training, larger-data study or other search change is authorized here.
