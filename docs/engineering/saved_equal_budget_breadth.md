# Saved equal-budget breadth screen

Owner authorized on 2026-10-03 after accepting
docs/strategy/next_after_proposal_specialist_128.md. This protocol implements
only the saved-data screen. It authorizes no new model inference, fitting,
source games, continuations, trajectory replay, fresh confirmation or arena.
Preserve all previous frozen dependencies and the intentional dirty worktree.

## Question and population

Does uniform allocation over the established broader candidate mixture improve
the four-candidate original161 planner at exactly the same continuation count?
At full support compare eight candidates with64 worlds each against four with128.
Both spend512 continuations. Equal counts are not an equal-time claim.

Use all360 main roots already sealed in outputs/proposal_breadth_128. There are
20 source-game-disjoint positions in each of18 fit x acting-seat x phase cells,
with the three original161 fits retained. The source is reused development
data from fast161 positions; the continuation policy is unchanged greedy035.
The original8x128 result and later gameplay result remain intact and contextual.
No old8x128 choice is assessed on outcomes it already used as a new control.

Pin source binding
231bdd640ac88af2470c24af38a04e28248a7edeba8e376173df5b356fd27a74,
result SHA256
2592d17b5ed5f1ca2a7f463ffa27b9e803db3600b0b3f62ea177aab26e2a2570,
and final-audit SHA256
f9273a1612636b4c13be6e47768a8ba05a0159f3b176aeddea5aca076faabf55.

## Exact decision and assessment rule

The stored outcome value is WDL=-1/0/1. Terminal utility is(WDL+1)/2; selection
can use integer twice-utility WDL+1. Let G be canonical support, m=min(G,4),
n=min(G,8), and B=128m. Candidate order/representatives remain the saved original
mixture: original four, up to two next anchor-ranked alternatives, then up to
two identity-uniform alternatives. Do not reconstruct or retune proposals.

For n>4, q=floor(512/n), r=512 mod n, L=q+(r>0), H=128-L. The common matrix
has128 complete n-candidate world vectors; the extension matrix has32(n-4)
world vectors for original m candidates. For each partition:

1. Uniform breadth uses the firstq common worlds for every candidate, plus
   common worldq for ther candidates in the remainder priority. Each candidate
   mean has its actual q or q+1 denominator. First original candidate wins ties.
2. The reference uses firstL common and firstH extension worlds for each of its
   original m candidates, exactly128 each. First original candidate wins ties.
3. Once choices are fixed from these cells, use the remainingH common worlds
   for both selected actions. This suffix is disjoint from both selection sets.

For n=5/6/7/8, (q,r,L,H) is respectively(102,2,103,25),(85,2,86,42),
(73,1,74,54),(64,0,64,64). All required extension prefixes exist. Compare means
as exact rationals or integer cross-products, avoiding float-created ties.

For G<=4, both arms use the complete identical support and all128 common
worlds. Reconstruct the identical choice, retain a zero paired difference, and
record no invented held-out scores. Do not exclude these roots.

There are exactly64 partitions, indexed0..63. For each domain common,
extension, and remainder, order the indices0..length-1 by ascending pair:

    (SHA256(compact_json([
      "saved-equal-budget-breadth-v1", 2140000000,
      source_spec.index, partition_index, domain, item_index
    ])).hexdigest(), item_index)

Compact JSON uses separators(',',':'), UTF-8 and the literal ASCII domain
strings above. All seeds and identities are outcome-independent. Permute whole
world vectors, never separate action columns. Separate domain tags distinguish
common, extension and remainder priorities. The same world's outcomes can play
different roles in different partitions; choices never pool partitions.

Persist every root's complete partition orders, allocation counts, selection
sums, choices, assessment sums and exact differences. Each partition's choice
calculation accesses only its allocated selection cells. Assessment values enter
only after those choices have been computed. Independently reconstruct all
access plans and choices; synthetic held-out perturbations must preserve choices.

## Aggregation and gate

Average64 paired differences within each root using the exact rational score.
Average the360 root means equally. Preserve18 cells sorted by(fit,seat,phase),
and sort each cell's20 roots by source index. The independent unit is the source
game/root, not the partition, world, candidate or fit.

Bootstrap50,000 paired samples, using numpy.default_rng(2143000000), int64 draws
from[0,20), generated in chunks of500 with shape(chunk,18,20). Index the root
means within each cell and average all360 selected means for each draw. The
95% interval uses numpy.quantile([.025,.975], method='linear'). The independent
implementation reconstructs the same declared draws with separately computed
root statistics and a different aggregation path. Numerical agreement tolerance
is1e-12 or tighter; exact fractions and discrete decisions must match exactly.
Never reuse the former fixed128-world /256 endpoint quantization.

The development gate requires mean gain>=0.005 (+0.5 percentage points),
paired95% lower endpoint>0, and every prescribed check passing. Classification
is positive_development_screen if these scientific thresholds pass, otherwise
practical_progress_not_established. Report the effect and both gates, not merely
the classification. Per-fit, phase, seat, support and choice changes are descriptive
only. No subgroup, partition, checkpoint, dose, schedule or threshold selection.

A positive result supports proposing fresh confirmation; it does not authorize
it automatically. A miss closes this fixed allocation screen. Prior use of these
data remains a limitation despite within-partition separation. These conditional
scores are not win rates between full-game players, and do not prove parity with
the more expensive8x128 planner or eventual superhuman performance.

## Source authentication and implementation checks

Authenticate source finalstatus→result/audit/report, original frozen manifest,
model receipt and four checkpoint identities, selection seal, all360 selected
specifications, main closure, root/audit inventories, every root and its receipt,
original selection artifacts, audit sample, and the360 selected source games
and source receipts. Check snapshot identity against each selected saved move,
fit/seat/phase, original candidate prefixes, complete WDL matrices, namespace
separation, original budgets and selection block seals. Check direct old breadth
engine/common/source/verify code identities, without importing game/model code.
Do not recursively reopen historical learning evidence or replay old trajectories.

The inherited main evidence records599,936 continuations and83,696 prescribed
replays. This study makes no new inner-trajectory replay claim. Every new
partition, budget, cell access, move choice, exact denominator and root endpoint
is independently reconstructed from the authenticated records. Check all50,000
bootstrap draws and final gates independently. Synthetic fixtures cover support
boundaries, exact ties, uneven allocations, invalid values, correlated world
vectors, selection/assessment leakage, reordered cells, duplicate roots,
tampered choices/means/budgets/partitions and artifact/path/hash failures.

Run focused preflight tests before binding code, protocol, strategy, runtime,
configuration, receipt and the selected input inventory. Store immutable per-root
analysis/audit artifacts, verify source and output hashes again before final
acceptance, and save a final result, audit, report and completion receipt. Existing
files are never overwritten. Only the new study's live status file is replaced.

## Execution and stopping

Implementation: new tools/saved_equal_budget_breadth package and
run_saved_equal_budget_breadth.py. Output: outputs/saved_equal_budget_breadth.
Commands are check, freeze, launch and internal execute. Official preflight uses
only synthetic scientific calculations; preparation may authenticate real inputs
but cannot calculate the new real-data comparison before the freeze is written.

Launch one hidden Python worker with one CPU numerical thread, durable stdout/
stderr/status and a one-hour deadline from launch. No visible console or recurring
monitor. Scoped sleep inhibition permits display sleep and is released on exit.
Require at least5GiB free disk throughout. Report implementation/preflight time
separately from the run. Expected checked execution is provisionally5-20minutes.

A discrepancy, missing record, changed identity, failed check, deadline or disk
limit stops acceptance and preserves evidence. Do not silently retry, extend,
change the algorithm, omit data or start a successor. If a preflight fixture
finds an implementation error before freeze, repair it and record a new passed
preflight receipt before launch. Post-freeze repairs require preserving the
failed version and explicit versioned recovery.

No automatic fresh collection, arena, learning, promotion or monitoring.
