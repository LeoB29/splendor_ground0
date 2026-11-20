# Experiment 134 Owner Runtime Amendment

Date recorded: 2026-08-31

## Timing and information state

This amendment was recorded while experiment `134` was running, after 21 of
200 paired seeds had completed. No interim wins, losses, draws, scores, margins,
seat results, or candidate-effect summaries were inspected. The decision was
based only on elapsed time, completed-pair count, process health, and the
measured throughput of the ten-worker run.

## Owner direction

The owner explicitly directed that experiment `134` continue without restart
and that its 72-active-wall-hour figure be treated as engineering telemetry,
not as a condition for accepting the gameplay-strength evidence.

No completed pair is discarded or migrated. The same frozen agents, seed
block, paired design, ten-worker configuration, outcome thresholds, bootstrap,
and integrity requirements continue unchanged.

## Interpretation

When the run completes, the automated runner may mark
`within_72_active_wall_hours` false and therefore emit its original formal
operational-invalid decision. Under this contemporaneous outcome-blind owner
amendment, that single timing condition is reported but excluded from the
scientific interpretation.

Every other gate remains mandatory, including:

- exactly 200 fixed pairs / 400 games;
- no seed replacement, addition, or outcome-dependent stopping;
- exact candidate and champion branch accounting;
- all continuations resolved and zero hard caps;
- bounded loop adjudication;
- expansion switches in every phase and at least 15% overall;
- candidate score share at least 55%;
- paired-bootstrap CI95 lower bound strictly above 50%;
- positive mean candidate margin;
- candidate score share at least 50% from each seat.

The partial owner-waived `087` panel remains outcome-sealed and is neither read
nor pooled. No training, promotion, installation, corpus generation, threshold
change, extra game, or automatic successor is authorized by this amendment.

## Operational instruction

Keep the current `134` process and its identity-matched progress file running.
Pause only on explicit owner request or genuine process/integrity failure.
