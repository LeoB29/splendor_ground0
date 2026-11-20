# Saved-world planner diagnostic

18 October 2026. The owner authorized implementation and execution after review
of the recommendation. This specification is bound before metric calculation
by the execution lock in the `ranker_planner_bridge_128` output namespace.
This is a post hoc development analysis of an
already-opened panel, with no confirmatory success criterion or automatic next
experiment. See the [recommendation](../strategy/next_after_structured_and_alternative_ranker_128.md).

## Fixed inputs and integrity

Use the completed structured study's 720 main roots, all from distinct source
games, balanced across 18 fit/seat/phase cells with 40 roots each. Fits are
161001, 161002 and 161003; seats 0 and 1; phases 0, 1 and 2. Preserve ascending
root index and lexicographic cell order. No probe roots enter the statistic.

Use both frozen alternative choices, `ridge` and `hgb`, from the existing
prediction seal. Do not rerun inference, fitting, prediction or the completed
one-shot evaluator. Main binding:
`57cd1786992c123e2e8fdedf6e4cf567196588efc8bd1ab9d2bbb24b7908f454`.
Alternative binding:
`c4cebd0e386787617404cf802eb3c4335801104449651f734dcd904ddf8e4a0a`.

Before calculation, a new execution lock must bind this specification, new
implementation and tests, runtime/library versions, and authenticated source
references. Revalidate the existing prediction seal, model/source dependencies,
main completion and audit chain, selection, and all 720 referenced root files
and audit receipts. Check their identities and prescribed audit status. This
reuses existing trajectory verification; it is not new replay or a claim of
exhaustive replay. Pin these known artifacts in the new lock:

| Artifact | SHA256 |
| --- | --- |
| Alternative predictions | `02ffa2375cce830236b649c85990b74f791b2f8a18f033d857a287693d854569` |
| Alternative prediction seal | `1c344116af9044d7183e1ef9ede9988e008b25f02ae1ff3f64e30e39b04a862f` |
| Alternative result | `ca16e70028b166a767ccea4ea2d36a9295b7d9be77ee365d566a71ffc80be821` |
| Main final status | `1c59c0f7b07bc4a37ca62ff519536932ca6c0afdcb90cfed729ad69fe5731e09` |
| Main final audit | `97ac6b861f81031ea92b5d3332c99a68c791fb3b3af991f4db26b4ea74394f2e` |

## Per-root calculation

1. Verify the sealed prediction and root agree on index, source game, fit, seat,
   phase and snapshot digest. Validate unique canonical union keys and their
   candidate indices, and the recorded outsider support using the original
   alternative evaluator's structural rules. Authenticate concrete
   representatives through the original root and audit references.
2. Take the ordered first three keys from the original anchor candidate set.
   The comparator is these keys followed by the sealed anchor outsider, when
   present. Require this to reproduce the original anchor's complete candidate
   list. Each alternative uses the same first three followed by its sealed
   outsider. Retain short supports and empty/singleton outsider cases; no
   duplicates or unrecorded actions are permitted.
3. Read exactly 128 saved **selection** worlds in their recorded order. Require
   complete candidate outcome coverage and resolved WDL values in {-1,0,1}.
   The completed panel has no unknown outcomes. Any missing, unknown, malformed
   or mismatched record stops this diagnostic; do not drop, impute or replace it.
   Original independent assessment worlds are not inputs to this calculation.
4. For direction A, select each planner's final action by maximum total score
   on indices 0–63, breaking ties by its offered candidate order. Assess the
   selected action on indices 64–127. Compute the alternative-minus-anchor
   paired mean score on those held-out worlds, with win 1, draw 0.5 and loss 0.
5. For direction B, select on 64–127 and assess on 0–63 with the identical rule.
   Average A and B within the root. If `dA` and `dB` are the sums of paired WDL
   differences on the respective held-out halves, the root score difference is
   exactly `(dA + dB) / 256`. Keep the integer numerator for independent checks.

The two halves give an honest separation of selection and assessment within
each direction, conditional on the fixed procedure. They are dependent uses of
one source root, not two independent observations. The analysis remains post
hoc because this panel's results informed the decision to perform it.

## Aggregation and reporting

Average root differences equally across all 720 roots. Report expected-score
percentage points by multiplying by 100. Primary descriptive summaries are both
models' means and nominal 97.5% percentile bootstrap intervals; no pass/fail or
model-selection rule is introduced. These intervals do not account for the
adaptive choice to perform this diagnostic and are not new confirmatory
evidence, even though they use the two-comparison Bonferroni convention.

Use 50,000 joint stratified root bootstrap replicates, NumPy `default_rng` seed
2813000000. In chunks of 1,000 replicates, iterate the 18 cells in the fixed order;
draw indices of shape `(1000,40)` uniformly with replacement within each cell.
Apply the same draws to both models, sum their integer root numerators across
cells, and divide by `720 * 256`. Use linear-interpolated percentiles 1.25 and
98.75. Retain the integer bootstrap totals or their reproducible digest for the
independent arithmetic check. The seed has no game-generation role.

Also report for each model, always with all roots in the denominator:

- Fraction of roots with a different proposed outsider from the anchor.
- Fraction of final actions differing from the anchor, averaging the two
  direction indicators within each root.
- Fraction selecting their offered fourth proposal, for the anchor and each
  alternative, again averaging the two directions within each root. No outsider
  means zero for this measure.
- Exact root count, cell counts, short-support counts and unknown count, plus
  both directional chosen keys and integer score differences in saved rows.

No favorable-half selection, additional split, subgroup claim, new ranking
recipe or full-legal extrapolation belongs to this analysis. No interval from
it replaces the completed studies' intervals or classifications.

## Implementation checks and resource envelope

Before analysis, test artificial cases that expose half leakage, ordered ties,
different fourth proposals losing to a common action, empty/singleton support,
duplicate/missing identities and rejection of unknown/missing outcomes. A second
implementation must reproduce all saved root choices and integer differences,
aggregate means and the specified bootstrap intervals. It should use an
independent arithmetic formulation rather than merely call the first function.

For the single execution: one CPU thread at below-normal priority;
20 minutes maximum including authentication, calculation and independent checks;
1 GiB maximum process RSS; at most 32 MiB of new files; at least 5 GiB free disk.
Reserve 64 KiB within that output allowance for completion or stop receipts.
Require at least 768 MiB available RAM and 2 GiB commit headroom at entry; stop
if available RAM falls below 128 MiB or commit headroom below 1 GiB. Poll bounds
between roots and bootstrap chunks, with a supervising timeout for a stalled
operation. Stream one root at a time and retain compact integer rows rather
than loading the full trace corpus. These are ceilings, not a measured
runtime forecast.

Read original artifacts only. Create new files exclusively within a new output
directory in this namespace; preserve every partial file on failure. Never
overwrite or restart an existing execution automatically. Do not change original
code, resources, models, manifests, receipts or conclusions. Recheck source
digests on completion. Record elapsed time, peak memory, inputs, outputs and
checks in a completion receipt. A bound or integrity failure produces a stop
receipt, not a scientific zero or an automatic relaxation.

Execution uses `python -B -m tools.ranker_planner_bridge_128.run prepare`, then
the same module's `execute` command. Preparation binds the synthetic checks to
the exact implementation and specification. A namespace-wide exclusive lease
prevents overlapping runs; the worker authenticates its supervisor and checks
that it remains alive. An existing scientific execution lock prevents another
calculation under a different directory name. All failure evidence is retained.

## Interpretation boundary

This diagnostic estimates conditional usefulness of a **64-world selector** on
the opened development panel. It does not estimate the deployed 128-world
selector, whole-game win rate, or performance over all legal outsiders. A
favorable result can inform a fresh study; a weak result can lower its priority.
Neither is an automatic arena trigger. Candidate selection, arena size,
practical target, operational probe and launch remain separate decisions.
