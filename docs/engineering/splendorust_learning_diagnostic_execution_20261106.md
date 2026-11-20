# Saved-corpus learning diagnostic: execution contract

6 November 2026. User authorization: **“ok do it”**, following the reviewed
[preparation](splendorust_learning_diagnostic_preparation_20261106.md). This
authorizes separate implementation, correctness qualification and the eight
specified fits if qualification passes. It authorizes no new source games,
search calls, strength assessment, architecture change or automatic promotion.

The preparation's scientific design and interpretation limits apply unchanged.
This document resolves implementation details before the first model call.
All original source, attempts, players and evidence remain immutable.

## Selection and fitting

Use only the accepted 256-game corrected corpus: 192 training games and the
already opened 32 validation plus 32 offline games. The engineering collector
comparison's trajectories are excluded. `selection.json` materializes every
game/root identity, input hash, split, batch index and probe batch before freeze.

Canonical hashing uses UTF-8 compact, sorted-key, ASCII-escaped JSON without NaN.
Game rank is SHA256([domain, "game", integer game ID]), with integer ID as a
tie-break; root rank adds "root", repetition, game ID and original ply. Original
source hashes bind evidence but do not influence sampling. The rotating 6/5/5
allocation uses position in that same 192-game rank order. Neither outcomes nor
confidence/loss values select records. Freeze binds the exact materialized orders.

Four cells use 64 games ×48 roots or 192 games ×16 roots, crossed with joint or
policy-head-only trainability, repeated twice. Each contains 3,072 unique roots
and equal numbers from the relative trajectory thirds. The sparse roots nest in
the dense pool. The two 64-game dense groups are disjoint.

The order is r0 dense/joint, broad/joint, dense/policy, broad/policy; then r1
broad/policy, dense/policy, broad/joint, dense/joint. Each starts from pristine
EF8A with the original architecture, float32 CPU execution and the fixed recipe.
There are 668 full batch-32 updates: six complete passes plus 92 batches of the
seventh predetermined permutation. The selected-corpus objective weights games
equally; the incomplete final pass introduces outcome-blind exposure variation.
Both scopes within a cell use identical batches. RNG seed is 202611012 + repetition.

Use four first scheduled batches for gradient probes at each fixed checkpoint.
Save checkpoints 0,334,667,668, including optimizer and RNG state. Reproduce only
the final update from667, preserving the original fit. Exact comparisons include
model/optimizer/RNG values and outputs/losses/gradients; elapsed time is excluded.
Two discarded engineering updates precede the eight fits. Total optimizer cap:
5,344 scientific +2 qualification +8 final-update reproductions =5,354.

All eight fits complete before opening the fixed diagnostic predictions. Eight
separate evaluator workers then examine checkpoints0/334/668 and the predetermined
gradient batches. There are exactly144 per-loss backward probes and no probe
optimizer calls. Shared/value tensors and fixed-input value-output bytes must
remain identical to the parent throughout policy-only work.

## Correctness and ownership

Every new fitter/evaluator calls `verify_runtime_sources()` once at startup,
including fit-only workers. Source rules, catalogue and atomic pre-reveal turn
contract are bound in plan, source metadata and checkpoints. Historical EF8A
initialization is explicitly distinguished from the corrected corpus.

Qualification exercises the actual selected loader and collator on the corrected
six card faces, nobles/final rounds, gold/returns, private reserves and canonical
value signs, including rejection of missing/wrong identities and hashes. It also
checks selection nesting, split exclusion, quotas and outcome invariance. The
qualified original collector evidence is reused; no new search/game is required.

Model work runs only after source-bound model-free tests, independent code
reviews and a separate immutable execution freeze. The retained controller starts
one original clock, supervises one fresh worker at a time and records actual exits.
Each worker is assigned to a Windows ownership job before its handshake permits
work. CPU0 hosts the worker; CPU2 hosts the supervisor. No implicit relaunch,
game replacement, checkpoint selection or clock reset is permitted.

Actual ceilings are 12 hours overall and an 8-hour model cutoff measured from the
same original clock; one hour qualification, one hour per fitter, two hours for
the entire evaluator sequence and four hours for saved audit, all clipped by the
global clocks. At most two model instances exist during parity/probes. Aggregate
owned RSS is capped at4GiB, new evidence at4GiB, with a1GiB actual free-disk floor.
There is no available-RAM reserve, stability wait or forecast admission gate.

Small mutable progress receipts are separate from immutable inputs/results.
The controller automatically starts saved audit and analysis after the final
evaluator. A technical failure stops and preserves the attempt. A valid negative
or mixed scientific result completes it. No scheduled monitor is created.

## Saved audit, analysis and completion

The saved audit reauthenticates inputs, exact selections, update schedules,
checkpoint archives and reproduction receipts. Predictions are bound back to
source root identities and targets. It checks the full planned fit/step/split
inventory, unchanged policy-only value bits, probe accounting and real worker
exit/cleanup receipts. No additional network evaluation occurs in this audit.

Use 20,000 paired whole-game bootstrap draws, seed20261106, sharing resampled
game indices across all cells/repetitions. Show both held-out splits, pooled
equal-game means, both repetitions and their fixed average. Nominal95% intervals
are exploratory. Report up to two charts and a recommendation; do not infer
stronger play, causal gradient interference or equivalence from null results.

Final acceptance requires an independent review of saved evidence, the retained
controller's observed exit and cleanup. Completion.json alone is insufficient.
Keep original EF8A as the reference; larger-data, external opponents, six-world
work and further speculative GPU tuning remain deferred.
