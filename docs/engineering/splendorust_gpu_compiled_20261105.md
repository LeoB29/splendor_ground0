# Bounded compiled-GPU comparison

5 November 2026. User authorized implementation and execution in a separate version.
Previous source, models, settings, evidence and failures remain unchanged.

## Intervention

One candidate: trace the accepted full primitive GPU network with TorchScript on
an all-zero float32 input of shape8x31x48 on the selected RX7800XT, using
check_trace=False and strict=True; freeze with optimize_numerics=False. No
alternative compiler flags, operator rewrite, precision change, backend install,
driver change or fallback. Capture graphs, mathematical operator order, constants,
warnings and a stable graph identity. Mathematical operations must retain their
order; harmless shape/attribute constant propagation is not an arithmetic rewrite.

The control uses the unchanged eager GPU primitive implementation. Both use
physicalB8, GPU16 logical clients, original row FIFO scheduler with zero gathering
wait, EF8A weights, float32,512 simulations,3worlds,cpuct0.4,depth64, corrected
rules, original native transport, and all original verification journals/checks.
No timing observer is added to either player. Complete runtime preparation is
measured separately and excluded from the warmed search-panel throughput clock.
For the candidate it includes compilation and graph inspection; it does not
isolate the compiler's own execution time.

## Admission before speed comparison

Freeze source and prior evidence; require independent review and model-free tests.
Every new evaluator calls verify_runtime_sources once before loading models.

The support worker first builds eager and candidate runtimes. Capture5 physicalB8
zero warmup calls for each. Use36 predetermined physicalB8 fixtures: four full
root batches covering the accepted32 saved positions, then32 batches with one
saved root in its original physical row and zeros elsewhere. Repeat each fixture
twice on each implementation. Capture all154 outputs and original clocks. All
83 float32 output values in every row, including padded rows, must be bit-identical
between implementations and across repetitions. Occupied rows must exactly match
the accepted same-route saved root outputs. No numerical tolerance is substituted.

On success, run the full67 actual candidate corrected-entrypoint counterexamples.
Then a separately owned fresh candidate worker runs all32 saved positions through
the unchanged GPU16 service. Every completed full result, callback trace, root
float32 bytes and independently reconstructed corrected afterstate must exactly
match its accepted eager record. Only these two passed saved audits admit timing.
Later workers independently authenticate admission artifacts before model creation
and bind their compiled graph identity to the qualified graph.

Known unsupported compilation, changed mathematical graph, candidate unsupported
operations or exact-output mismatch are preserved scientific negatives. Stop before
timing without another compiler variant. During full decisions, preserve the actual
result, root, afterstate and service records before proving any exact mismatch.
Finish already released peers under their original clocks, then stop further waves.
Eager drift, checker/source/ownership errors, OOM/I/O faults and unrelated exceptions
are technical failures, never relabelled candidate negatives. Preserve all failures.

## Throughput comparison

Four fresh32-case cells: eager, compiled, compiled, eager. Two fixed paired blocks;
same32 states at plies0/12/24/40 of the prior eight accepted matched games. Prepare
all16 native adapters before panel release. Five fixed service warmups are outside
panel timing. Each cell executes two16-case waves; original complete search clocks
remain independent. Record first barrier release through final durable result
writes, including validation, required logging and inter-wave bookkeeping.

Report per-block and pooled completed-decision throughput, response distributions,
occupancy, complete runtime preparation costs, graph identity and exact reproduction. A useful
signal requires at least5% greater compiled throughput in each paired block, with
all four cells complete and exact. This is a descriptive engineering criterion,
not statistical significance or automatic promotion. Smaller or inconsistent gains
are reported honestly. No isolated-inference timing alone admits the candidate.

## Bounds and end conditions

Maximum160 full panel decisions (32qualification+128timing),67 entrypoint checks,
154 support calls plus the one declared compilation method's startup executions.
No fresh games, training, weight updates or larger-data collection.

Keep60-second internal and75-second outer complete-response safeguards. Completed
responses beyond1/9/10seconds remain accepted and descriptively reported. Never
retry or replace a decision, reset its clock or extend a bound. A deadline or
exact candidate drift stops further cells; a control deadline makes the comparison
incomplete and does not establish candidate inferiority.

One4hour total clock,3hour model cutoff,45minute support/entrypoint worker and
15minutes per panel worker. Actual worker-treeRSS8GiB, evidence4GiB, minimum free
disk1GiB. No available-RAM reserve or forecast admission gate. One study worker
at a time; worker CPUs0..18 and supervisorCPU19. No scheduled monitor.

Require independent saved-evidence acceptance, actual controller exit and exact
owned-process cleanup. A valid negative ends the study; retain its complete prefix
and all earlier failures. On a completed comparison, recommend the next step only.
Whole-game and collector qualification remain separate follow-ups before production
claims. Original model/CPU8 references remain; larger-data, training, external
opponents and six-world work stay deferred.
