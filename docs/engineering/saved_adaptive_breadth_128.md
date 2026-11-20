# Saved adaptive allocation development screen

Owner authorized on 2026-10-03 by accepting the recommendation in
docs/strategy/next_after_equal_budget_breadth_128.md. This protocol permits one
arithmetic-only screen and its independent verification. It authorizes no new
inference, training, simulation, replay, fresh confirmation, arena or successor.
Implementation and output are new under saved_adaptive_breadth_128; all frozen
predecessors remain unchanged.

## Fixed arms and evidence

Use all 360 original main roots from proposal_breadth_128, all three original
161 fits and all 18 fit/seat/phase cells. Preserve the exact candidate list and
the 64 per-root common-world, extension-world and remainder permutations of
saved_equal_budget_breadth. Their hash domain is saved-equal-budget-breadth-v1
and partition seed 2140000000. Verify exact agreement with that sealed study's
per-partition permutations and uniform/anchor choices. Do not copy its scores.

For n=min(G,8)=8, including G>8, the adaptive arm gives each candidate the
first 32 common worlds. Keep four highest sums, breaking ties by original
candidate index. Give those four the next 64 common worlds, then choose among
them by their combined 96 outcomes. Final ties again use original candidate
index, not survivor ranking. This spends 8x32+4x64=512 continuations.

Uniform uses the first 64 common worlds per candidate. Anchor uses the first
64 common worlds and first 64 independent extension worlds per original
candidate. Both also spend 512. Persist all choices and permitted accesses in
each partition record, and assess all choices on common positions 96 through
127 in that partition's permutation. No arm uses assessment to select a move.

For G=5/6/7, adaptive is exactly uniform: divide 512 by n with identity-assigned
remainders, giving two103/three102, two86/four85, or one74/six73 outcomes.
Anchor retains the same old prefix plus extension construction for 128 per
original candidate. Assessment retains the old suffix of 25/42/54 common
worlds respectively. For G<=4 all three rules are identical, with exact-zero
paired differences and no invented absolute assessment scores. Keep all roots.

Every mean comparison uses exact integer cross-products and original-index
ties. Assessment differences and within-root averages use reduced rational
numbers with the actual denominator. The old fixed128 outcome score grid is
inapplicable. No eight-by128 teacher control may consume this assessment tail.

## Analysis and fixed decision

Average the 64 partition differences within each root. Give roots equal weight.
The three reported contrasts are adaptive-minus-uniform, adaptive-minus-anchor
and uniform-minus-anchor. Recompute every contrast from its new common
assessment suffix; full-support U/A scores can differ from their old publication.

Use 50,000 paired bootstrap draws within the 18 fixed cells, 20 roots per cell,
seed 2163000000 and chunks of 500. Keep arms paired. Convert exact root fractions
to a common integer denominator before summation and apply linear percentile
quantiles at .025 and .975. Independent verification uses a different bootstrap
accumulation implementation with the same draws and endpoint convention.

Classify positive_development_screen only if adaptive-minus-uniform mean is
at least .0025 and its 95% lower endpoint is strictly positive, and
adaptive-minus-anchor mean is at least .005 and its lower endpoint is strictly
positive. All integrity checks must also pass. Otherwise classify
practical_progress_not_established, reporting each contrast and its interval.
These are conditional score units (win1/draw.5/loss0), not game win rates.

This corpus has been used before. Disjoint within-partition selection and
assessment, a new bootstrap seed and 64 overlapping partitions do not produce
new independent experiments. A pass supports separately proposing fresh S/U/A
confirmation with the same gates. A miss ends this fixed schedule, with no
neighboring schedule, candidate, subgroup, partition or threshold sweep.

## Verification and operational limits

Authenticate the original closed source panel, games, model identities, root
selection, source matrices and existing prescribed audit receipts. Also bind
the completed saved uniform screen's frozen identity, preflight, result, audit,
analysis inventory, all 360 analysis records and all 360 verification receipts.
Verify one-to-one links to original roots and exact retained U/A choices. This
inherits the original sampled trajectory verification; it performs no new
trajectory replay or historical model reproduction.

An independent implementation reconstructs every partition's permutations,
survivors, counts, allowed cell accesses, selections, scores and exact fractions.
It also reconstructs final statistics and gates. Focused synthetic tests cover
all supports, G>8, both tie stages, loss of an eliminated candidate, held-out
isolation, exact zero fallback, roots as bootstrap units, malformed/tampered
evidence, preserved prior choices, provenance and orchestration. No full
historical test suite is required for this arithmetic-only change.

Before any production calculation, pass the new focused suite and freeze its
receipt, all new code/tests, this protocol, the accepted recommendation,
verification policy, configuration/runtime identities and bounded source
inventory. Scientific files use exclusive creation; no overwrite or automatic
retry. Preflight failures retain their evidence in separately numbered folders.

Run a hidden detached supervisor and one arithmetic child, with numerical CPU
threads fixed to one. The supervisor stops that child at the original deadline.
The chat may close, display sleep is allowed, and system sleep is inhibited
only during execution. No visible terminals, GPU work or recurring monitor.
The original launch has a one-hour execution ceiling and five-GiB disk reserve;
check at each root/stage and final closure. A failed check retains partial
evidence and stops. No automatic restart or budget extension.

After independent verification, reauthenticate frozen identities, input and
output inventories and final links. Report recorded/inherited replay coverage
separately from this study's zero new simulations and zero new trajectory
replays. Save final result, audit, human-readable report and completion receipt,
then release the sleep inhibitor and exit. No next study launches automatically.
