# Bounded complete-game collector comparison

5 November 2026. User authorized the recommended collector qualification and
compiled GPU16 versus CPU8 comparison. This separate version preserves all prior
players, source and evidence. No training or larger-data expansion is authorized.

## Fixed implementations and actual collector qualification

Use the original EF8A weights, float32, 512 simulations, three sampled worlds,
cpuct0.4, depth64 and corrected two-player rules. CPU uses the unchanged compiled
CPU network with eight concurrent clients. GPU uses the exactly qualified fixed-B8
TorchScript primitive network with sixteen clients and the unchanged row FIFO
service. No further inference optimization, timing observer or software-stack
change is introduced. Bind the accepted compiled graph fingerprint.

The prior timing transport does not expose root visits. Reuse the separately
qualified native capture binary and add its target-capture interface to both
routes, preserving the original search algorithm and scientific result. This is
collector plumbing, with its cost included equally in the timed comparison.
Require source/binary identity and exact original full scientific results on all
32 accepted saved positions per route before any games. Do not infer a policy
target from the selected move or invent a missing visit distribution.

Each new worker authenticates runtime rule sources once before loading a model.
Each route must pass 116 actual qualification decisions:67 corrected-entrypoint
checks,32 saved-position equivalence checks,16 collector/loader fixture calls and
one continuation of the noble final-round fixture. The latter yields one explicit
two-ply engineering fixture trajectory per route, using the already captured first
move, so the actual full-game/corpus loader can be qualified before collection.
These two fixture trajectories are separate from the64 fresh benchmark games.
Exercise all six corrected card faces, noble and final-round cases, gold/returns,
private reserves and identity/hash rejection through the actual chosen collector,
dataset decoder and batch collator. Check real root tensors and legal support.
Value labels are actor-oriented official terminal results. Forced, single-choice
or late fallback exclusions follow the frozen qualified capture contract.

## Predetermined complete-game workload

Two independent blocks of16 fresh setup seeds. Both routes use the same initial
states within each block:32 independent setups and64 generated trajectories total.
Run four separately owned workers in this order:CPU8 block1, compiledGPU16 block1,
compiledGPU16 block2, CPU8 block2. Both seats use the same route's unchanged player.

Each cell has a fixed16-game queue. CPU8 starts eight and assigns the next scheduled
game to a free worker without waiting for other games to finish; GPU16 starts all
sixteen. Record actual worker/slot assignments. Refill is only from this original
queue, never a replacement for a failed game. A natural terminal is required;
deadlock, ply cutoff and timeout cannot provide an official training label.

The primary collection clock starts at first release and ends after the final
durable corpus/target manifest and acceptance by the actual loader. Include all
game-ending tails, move checks, target extraction and required recording. Record
startup, five fixed warmups, selected model replays and saved audit costs separately.
Also report verified-data rates over the complete cell wall including those costs.
The final independent audit has its own reported duration and remains part of the
total study cost. Never sum overlapping concurrent per-game spans into wall time.

For every game independently replay every saved transition and target without a
model. Additionally replay the first neural decision in each of ply bands0..19,
20..39 and40+, if present, using the original complete-response clocks and a shifted
physical slot (original assigned slot+1)%8 with other clients inactive. Require
exact same-route full results and captures. Maximum192 selected model replays.
Do not substitute CPU-GPU bit equality for the established within-route checks.

## Bounds, failure policy and interpretation

Original60-second internal/75-second outer complete-response safeguards remain.
Responses beyond1/9/10seconds are descriptive and accepted if within those
safeguards. Preserve every failure; stop subsequent work at the first failure,
finish or preserve already released decisions under their original clocks, and
never retry, reset a clock, replace a game or discard a failed prefix. Incomplete
cohorts have no completed-cohort throughput claim and no synthetic outcome labels.

Total12hours, model cutoff10hours, qualification worker1hour, each game/replay
worker4hours. Actual worker-tree RSS8GiB, new evidence12GiB, minimum free disk1GiB.
No available-RAM reserve or forecast admission gate. One worker process at a time;
worker CPUs0..18, supervisorCPU19. Saved audit processes one cell at a time with a
bounded derived scratch index, retaining original journals as the authoritative
evidence. No scheduled monitor.

Report all paired and pooled rates. A descriptive practical GPU advantage requires
all four cells valid and at least5% higher verified completed-games/hour in both
paired blocks. Report usable-target rates and workload differences separately.
This criterion is interpreted only after completion; it does not halt an ongoing
comparison or imply statistical significance. No automatic promotion or training.

Require independent review and source-bound model-free tests before freeze, then
independent saved-evidence acceptance, actual controller exit and exact owned
process absence. Produce a concise report and at most two intuitive charts. The
original model remains the strength reference; external testing, six-world work,
training and larger-data work remain deferred.
