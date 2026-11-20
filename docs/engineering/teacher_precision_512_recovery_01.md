# Teacher precision timer recovery

The owner authorized repair and continuation on 2026-10-12. The original study
completed all 720 labels and their prescribed audits, then stopped during the
first training diagnostic before any optimizer update. A reused diagnostic
function checked its own eight-hour limit before the current model wrapper
could apply this study's 48-hour limit.

Preserve the frozen study code, all label records, independent checks, target
tensor, initializers, original launch, abort, hold and partial-stage receipts.
New recovery code lives in tools/teacher_precision_512_recovery_01. Recovery
freeze, logs, stage launches and progress live in outputs/teacher_precision_512_recovery_01.
New scientific training and evaluation artifacts continue in the original
outputs/teacher_precision_512 under its original scientific binding. Every new
stage lock points to a launch receipt binding the recovery freeze.

The repair makes private copies of the two diagnostic functions' global maps,
preserving their exact code, defaults and arithmetic, with only the inner
budget callback routed through the current study. It does not change ancestor
modules or configurations. Install the same repair in the supervisor and each
spawned hidden worker. One cumulative active-time boundary governs training,
independent reproduction, source games, planning evaluation and verification.
Retain the 48-hour active ceiling and five-GiB disk reserve. Count previous
runtime from the original launch to the recorded hold (about 13 hours 12
minutes), then add recovery runtime. Time stopped between processes does not
consume active time. Original launch timestamps are never reset.
The recovery launch receives its own completion hash seal before work begins;
changing its prior time, new epoch or recovery link invalidates the budget check.

Resolve only the exact hash-pinned original timer abort under this recovery
receipt. Any change to it, or any new recovery abort, still stops execution.
No blanket disabling of guards, automatic retry, monitor or successor is added.
Retain checkpoint and atomic evidence-writing behavior. Early worker errors
keep a traceback as well as the concise progress status.

Before launch, verify the complete saved label/block/completion/audit artifact
closure and the target seal without replaying trajectories or recollecting
labels. Require no existing optimizer updates or fresh evaluation artifacts.
The three fits start from their originally prescribed initializers and exact
batches/dose; controls, targets, seeds, gates and subsequent stages are unchanged.
Seal and independently reproduce fits before generating fresh evaluation games.

Regression tests reproduce the greater-than-eight-hour failure in ordinary and
independent-audit diagnostic paths; compare unchanged numerical reports and
full tiny fit/reproduction trajectories; enforce the true active ceiling,
abort and disk guards; preserve frozen source functions; exercise real hidden
workers at an aged launch; and validate completion ordering and receipt links.
The original scientific experiment is not extended or redesigned by this repair.
