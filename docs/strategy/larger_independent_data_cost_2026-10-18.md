# Cost of a larger independent data comparison

18 October 2026. Prepare **11,520 versus 46,080 independent labelled positions**
as the first substantial extension; price 184,320 as a later stage. Collection
and storage dominate. The current machine cannot accommodate either extension
without additional available storage. These are planning estimates, not a launch
freeze or a claim that data scarcity caused the previous failures.

## Scientific question

The completed [independent-game scaling study](../../outputs/independent_game_scaling_128/report.md)
already increased added positions from 720 to 2,880 to 11,520. Its large-minus-anchor
gain was +0.1145 percentage points, 95% interval [-0.2691,+0.5333]. The subsequent
[optimization extension](../../outputs/large_corpus_optimization_128/report.md)
reached approximately 98% training capture without establishing fresh improvement.
Another small increase or longer fit on the same rows would not settle whether
materially more independent data helps.

For a new comparison, retain the same model architecture, teacher task, source
population, 128-world label precision, old corpus, action representation and
initializer set across nested sizes. Keep whole source-game identities disjoint
from new evaluation. Use a common freshly collected evaluation for sealed models;
the already-opened panels are development evidence.

Two different questions need distinct training policies. Fixed updates measure
the return from more data at fixed fitting compute. Comparable added-row exposure
measures a package of more data and more training. Neither is a pure isolation
of data coverage at equal compute and equal acquisition simultaneously. Freeze
the choice, acquisition diagnostics, checkpoint rule and any bounded extension
before fitting; never fit until fresh results look favourable. Refit the 11,520
baseline under the same chosen recipe, rather than compare a new-dose model to
an unmatched historical checkpoint. Retain all three initializers.

## Measured collection costs

The completed scaling study generated 32,400 source games and 10,800 new labels,
adding to an inherited 720-position core. Eight simulation workers produced:

| Completed stage | Elapsed hours |
| --- | ---: |
| Source games and full source audits | 2.755 |
| 128-world label collection | 45.520 |
| Prescribed label audits | 2.448 |
| Total | 50.723 |

Evidence: `outputs/independent_game_scaling_128/` receipts
`source_train_{probe,remainder}_{collection,audits}_lock.json` and
`labels_{probe,main}_{collection,audits}_lock.json`. Worker configuration is in
that study's `manifest.json`. Times include those stages' measured work; they
are not the protocol's operational ceilings.

Assume the same three source games per selected position, phase eligibility,
action supports, eight-worker throughput and nested reuse of the existing 11,520
positions. The 184,320 target includes the smaller extension; do not sum them.

| Total added positions | New labels | New source games | Collection and audits | With 35% planning allowance |
| ---: | ---: | ---: | ---: | ---: |
| 46,080 | 34,560 | 103,680 | 162.31 h / 6.76 days | 9.13 days |
| 184,320 | 172,800 | 518,400 | 811.56 h / 33.82 days | 45.65 days |

The allowance is an analyst-selected contingency, not a measured confidence
interval. Runtime and source yield can change at scale. These figures exclude
fitting, fresh assessment and a whole-game confirmation.

The accepted compatible backend achieved approximately 26.66% savings in its
[controlled integration](../../outputs/simulation_backend_integration_pinned_128_01/final.json).
Applying a rounded 0.7334 multiplier to label collection only
would reduce the above reference totals to approximately 123.48 and 617.39 hours.
This is a sensitivity scenario, not validated eight-worker throughput or a
general arena speedup. Keep the unaccelerated budget as the funded case until a
future collector explicitly binds the accepted implementation and passes its
own integration checks. The completed speed campaign remains closed.

## Storage and fitting

The historical 10,800-label extension occupies 21,301,131,810 bytes, or 19.84 GiB,
in the measured source-game, source-audit, label-root, block, label-audit and
individual tensor files. Of this, source games account for 16.356 GB and label
blocks/completion receipts for 4.169 GB. Keeping that retention format gives:

| Target | Incremental raw artifacts | Measured historical set plus extension |
| ---: | ---: | ---: |
| 46,080 | 63.48 GiB | 83.32 GiB |
| 184,320 | 317.41 GiB | 337.25 GiB |

These are lower planning components, not total disk requirements. They exclude
the inherited 720 positions, original 9,461 rows, aggregate tensors, fitting
checkpoints, fresh evaluation, filesystem overhead and the free-space reserve.
At inspection C: had about 6.34 GiB free, only 1.34 GiB above the existing 5 GiB
reserve. Available RAM was about 3.22 GiB of 31.85 GiB on an i5-13600KF. These are
dated observations; recheck before any new work. No files were removed or moved.

The completed optimization extension took 10.079 hours for three models'
21,408 further updates and independent reproduction, with one fitting worker.
Using its approximate 79.2 added-row exposures and 50:50 old/added mixture yields:

| Added positions | Updates per fit | Three fits plus independent reproduction |
| ---: | ---: | ---: |
| 11,520 baseline | 28,544 | about 13.44 h |
| 46,080 | 114,176 | about 53.76 h |
| 184,320 | 456,704 | about 215.02 h |

Evidence: `outputs/large_corpus_optimization_128/learning_fits_lock.json` and
`learning_fit_audits_lock.json`. This extrapolates per-update cost, ignores
larger-corpus loading/diagnostic overhead and increases old-row exposure too.
It is a cost scenario, not the selected new training recipe. Conversely, keeping
7,136 updates would give only about 4.95 and 1.24 added-row exposures at the two
larger sizes; a negative result then would leave acquisition unresolved.

Historical retention saved 338 optimizer files per fit, averaging about 8 MB,
or 8.09 GB across three fits. Repeating a 64-update cadence would add roughly
40 GiB and 160 GiB of optimizer files at the larger sizes, before possible
growth in their history payload. Price a sparser restart-checkpoint schedule
prospectively; it must preserve the specified final/model-selection rule and
reproducibility. Keep every historical checkpoint intact.

## Evaluation and overall budget

Comparable 720-position fresh assessments, including sources, probe and audits,
took 7.50 hours with eight workers and six arms, and 9.42 hours with four workers
and four arms. A provisional 8–13 hour allowance is reasonable for a comparable
new assessment; collection design, number of fits and measured resources still
determine the actual budget. Recorded source/main/probe files used about 1.5–1.6
GiB, excluding additional metadata. The studies' `source_{probe,main}_*`,
`probe_*` and `main_*` lock receipts contain those stage measurements.

For the first 11,520-versus-46,080 comparison, the reference scenario is about
162.31 hours of new data, 13.44 + 53.76 hours of baseline/larger fitting and
reproduction, and 8–13 hours of fresh assessment: **237.5–242.5 hours**, roughly
**9.9–10.1 active days**, before whole-game confirmation and contingency. Applying
35% to that subtotal gives approximately **13.4–13.6 days**. This assumes stages
run serially at the observed resource configurations. Memory pressure, sparse
checkpoint engineering and retained-data access can increase it.

Reaching the 184,320 curve while also fitting the baseline and 46,080 level
would take approximately 1,102–1,107 reference hours, or 45.9–46.1 active days,
before whole-game confirmation; 35% contingency gives about 62 days. A future
costed plan must quote the whole curve, not only its final fit.

The previous 402-game learned/breadth arenas took roughly 78–91 hours in their
specific eight-worker environments. They are historical comparators only.
A new player's confirmation must be separately costed, including audits and
storage; the [external timed benchmark](external_strength_calibration_2026-10-18.md)
has a different execution contract. No cloud/GPU price or speed is assumed.

## Decision

Do not launch the larger curve on the present free disk. Prepare the 46,080
comparison first, with an explicit available-storage plan and a fully priced
checkpoint/evaluation budget. Keep 184,320 as an option if the first stage and
resource envelope justify it under a prespecified continuation rule. Neither a
positive training score nor an inconclusive fresh result automatically triggers
another size. External strength qualification remains the immediate priority.
