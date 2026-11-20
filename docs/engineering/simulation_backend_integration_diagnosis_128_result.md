# Diagnosis of the integration timing reversal

Completed 2026-10-17, about02:39 Europe/Paris. The main structured-proposer
experiment remained unchanged and running. All new work used completed historical
data, one hidden diagnostic worker at a time and preserved frozen sources.

## Conclusion

Uncontrolled processor placement is the strongest supported explanation for the
earlier timing reversal. The new encoder is faster on the sampled integration
workload when compared under matched conditions. A controlled change of CPU
placement can make it appear slower despite it winning on either processor.

The earlier integration did not record processor placement, so its precise cause
cannot be proved retrospectively. Core frequency, sibling/background contention
and other host effects are not fully separated. The original failed speed gate
remains failed; these diagnostics do not retroactively authorize adoption.

## Evidence

The original three-root integration passed exactness, full independent replay and
interruption recovery, but reference took522.947839s and fast547.281666s: fast
used4.6532% more elapsed time. Nine-model initialization took only0.11–0.20s per
worker and sampled available memory stayed above3.31GB. There is no persistent
prediction cache in either policy or in the prior guard wrapper.

A new fixed diagnostic replayed world indices0/42/85/127, all recorded candidates,
historical selection and integration selection/assessment from roots517/251/382.
Three variants (raw reference, guarded reference, fast) ran in one nine-model
process, with rotated order and a reversed second repeat. Every saved world,
trace, final state, termination and outcome matched exactly.

| Matched workload | Reference elapsed | Fast elapsed | Time saved |
| --- | ---: | ---: | ---: |
| Historical selection |25.8394s|17.4914s|32.31%|
| Integration selection |24.3707s|17.4297s|28.48%|
| Integration assessment |3.7512s|2.7495s|26.70%|

Fast won all18 case-by-repeat paired comparisons. The extra reference guard
changed pooled wall time by only+0.18%; pooled CPU time was effectively unchanged.
The guard-frequency asymmetry therefore does not explain the earlier reversal
on this diagnostic. This sample does not exclude workload variation in untested
worlds, but the slowdown was not reproduced on either recorded workload.

Eight separate profiles support reduced input-encoding cost, with model forward
and game-rule work broadly similar. Profiling adds unequal overhead to different
Python paths, so those timings are explanatory rather than speed estimates.
Legal-action histograms matched exactly within each paired profile.

## Controlled processor comparison

Local topology and hardware inventory identify an i5-13600KF, with performance
and efficiency cores. Windows reports logical processors0–11 in the higher
efficiency-class value and12–19 in the lower value. Normal diagnostic samples
observed migration between types; these sparse samples do not measure complete
residency and were not available for the earlier integration.

A second frozen diagnostic replayed the same integration selection work for
roots517/382, four worlds each, two repeats with reversed order. Only the new
worker's affinity changed, first to logical processor0 and then12; its original
mask was restored. Main affinity, priority, code and settings were untouched.

| Matched placement | Reference elapsed | Fast elapsed | Time saved | CPU time saved |
| --- | ---: | ---: | ---: | ---: |
| Same performance processor0 |20.3513s|14.2000s|30.23%|30.80%|
| Same efficiency processor12 |33.5527s|24.6071s|26.66%|25.36%|

Comparing fast on processor12 against reference on processor0 gives an apparent
20.91% elapsed-time regression and33.18% CPU-time regression. This directly
demonstrates that unequal placement can reverse both metrics, despite fast
winning under each matched placement. CPU time alone therefore does not control
hardware speed. One processor per class does not estimate a universal class
effect or prove which processor an earlier unobserved process used.

## Scope, validation and next step

The normal diagnostic completed54 timing blocks and8 profiles in166.1s; its
primary timings contain1368 repeated continuation executions from228 fixed
source/block/world/candidate cells. The pinned test completed16 blocks and576
repeated executions in98.1s. These are engineering measurements, not new playing
strength data or thousands of independent positions. Independent review checked
frozen dependencies, exact schedules, scientific hashes, recomputed totals,
result receipts and restored affinity. No unexpected failures occurred.

Recommend one new full-collector acceptance using the same historical panel,
with both versions assigned to the same logical processor and paired reversed
orders. Preserve the original exactness and independent-audit requirements and
report the new result separately. Only a passing controlled acceptance should
change the adoption decision. No such rerun or production switch was launched
as part of this investigation.

Evidence:
- Original failed gate: outputs/simulation_backend_integration_128_01/final.json.
- Normal diagnosis: outputs/simulation_backend_diagnosis_128_01, binding
  2608bcbe123875069c325eb7502ce795f0c6464b652f490206e0e3f9933edb57.
- Pinned diagnosis: outputs/simulation_backend_diagnosis_pinned_128_01, binding
  e1b72ba15332d2f197bab4f234525283e6093ebca1ba17e5d403dae3572ae8cf.
- New source: tools/simulation_backend_diagnosis_128. All previously frozen
  files are verified against their saved hashes; newly added diagnostic files
  are not treated as modifications to a historical dependency inventory.

Main status at02:39: supervisor54864/auditor1376 alive, stderr empty,1/6 fit
audits complete; second161001/relations at2560/28544. No fresh outcomes read.
