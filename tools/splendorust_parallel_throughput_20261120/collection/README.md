# Prepared collection route

This package is preparation only. The root launch gate requires explicit future
authorization and the completed exit and acceptance of collection4096. Importing
these modules never starts a model, game, benchmark or worker.

`preparation.build_plan(collection_specs, qualification_specs)` derives the
unchanged accepted CPU model/capture/fixtures from collection4096. Root entry
points are `launch.qualify(plan, jobs, boundary)` and
`launch.run(plan, jobs, boundary)`. The generic owned worker invokes
`worker.run(folder, plan, owner, boundary, request)`; every role calls the
corrected-rules source guard exactly once. The model supervisor additionally
polls `worker.active_registry_paths` under the original 60/75-second clocks.

Qualification retains all 116 actual entrypoint/capture/loader decisions, two
excluded complete engineering games, selected exact replays and independent
saved checks. Nothing from this pilot can be added to a training corpus.

The existing CPU8 route calls the qualified float32 graph separately for each
network callback with physical batch size **one**. It does not combine its eight
clients into a batch. Four new services with two clients therefore retain the
same graph and packing; numerical equality is required without tolerance.
The original control's CPU mask 0–18 and supervisor CPU19 remain unchanged.
Candidate services use four separately verified physical P cores, with a fifth
P core reserved for fresh replay services and audits. Their controller uses
separate E cores. Pinning the control's eight clients to one core would change
the baseline and is deliberately not the tested control.

All services finish initialization, five exact warmups and client construction
before the shared release. One SQLite transaction queue assigns the fixed sixteen
games in ascending order, only to idle clients. Every service has its own original
inference journal and ownership receipt. Cell assembly retains the actual
corrected target loader; the independent audit reconstructs every scientific
decision, target, queue assignment/outcome, clock, replay and callback attribution.

The order is control X, candidate X, candidate Y, control Y. Candidate X's checks
overlap candidate Y's play. Any check, identity or gameplay failure stops new
assignments, drains existing owned work under its original clocks, and quarantines
Y. There are no retries or replacement games. Candidate X is compared to control X
as soon as its audit passes; Y is compared when control Y is verified.

Cross-arm identity compares all response/capture fields, every action and visit
vector, root values and all floating-point bytes, final outcome and every target
field. The sole omitted target field is `row_sha256`: it hashes timing and routing
metadata that must differ between runs. Each value is authenticated against its
own complete source record before omission. No scientific value is rounded or
given a tolerance.

Rates include the actual collection pipeline, selected replays, audits and process
exits. Half the total shared cross-arm identity time is charged to each arm; this
is conservative for the candidate when an identity check overlaps Y play. With
only two cells, the reported completion cadence is one observed transition, not
an estimate of long-run steady state. The collection adoption bar remains 1.3×
verified games per hour, exact identity, zero failures and all checks passed.

The model-free tests cover actual CPU request packing, the shared queue, failure
draining, overlap/quarantine, exact identity including signed zero, source-digest
authentication, actual session dispatch/cleanup, once-startup source guards and
corruption of saved queue, timing and accounting records. Actual-model
qualification is pending the later authorized launch.
