# Second recovery: exact equivalence of unfinished continuations

The owner requested a valid restart on2026-10-05 after the first recovery held.
Preserve the original study and recovery_01 as immutable predecessors. Resume
the same402-game panel only after independently certifying its failed decision.
This is an explicitly disclosed second certificate amendment, not completion
under the unamended protocol. The provisional310-game tally was already known:
152 wins,3 draws,155 losses. No strength result selects the recovery rule, changes
the success criterion, discards a game or determines whether a root is accepted.

## Failure and reproduced evidence

Recovery_01 stopped at game310, turn58, U candidate side, fit161003/seat0.
All512 cells were attempted at8x64. World42/candidates0 and1 each reached the
original repetition safeguard after29 moves with null winner/WDL. Their known
WDL sums are both-59; each possible sum is[-60,-58]. Independent intervals cannot
establish the original first-maximum winner. Seven other workers stopped
cooperatively and saved partial games311-317; this was one ambiguous decision.

The two failed branches were reproduced exactly. The only final snapshot
difference is the order of player0's already-purchased T1_58 and T1_04 cards.
Sorting purchased-card IDs alone makes every other field identical, including
absolute turn, player to move, tokens, bonuses, scores, market and ordered hidden
decks. Canonical states coincide from continuation step3 onward. Ordered legal
actions and actual035 model inputs/scores agree. This evidence is saved separately
in recovery_diagnosis_02; preserve its documented unsuccessful postprocessing
attempt as well as the durable second reproduction. No rollout was extended.

## Narrow behavioral equivalence

Canonicalize only the order of each player's purchased_card_ids, and no other
field. The current035 inputs, environment legality/transitions, scoring and
tie-breaks depend on the owned cards' counts/bonuses, not acquisition order.
Market, deck, reserve and noble ordering remain exact. The fixed continuation is
deterministic, uses first-argmax035, and draws from the saved ordered decks.

For each proof, reproduce the exact original continuation, preserving its forced
first action, world, hidden state and every original stopping safeguard. Commit
the complete trace and final state against the previously recorded outcome
hashes. Reconstruct forced-pass and no-progress counters, plus the repetition
history including the final cutoff state's fourth visit. Keep only repetition
keys with the same per-player purchased-card sets as the final state: smaller
acquired sets cannot recur because purchases append unique cards and never
remove them. Within a fixed acquired-set epoch, sorting those IDs is an injective
relabeling of that branch's repetition keys. Preserve all remaining key fields.

A future-context record contains the canonical full final-state hash, sorted
live repetition counts, forced-pass count, no-progress count, fixed035 checkpoint
hash, and root scoring actor. Group unresolved cells only when those complete
contexts match exactly. This proves their subsequent behavior and any bounded
WDL completion agree. It does not identify an outcome or assign a draw. No
comparison based merely on similar evaluations, rollout lengths or signatures
is allowed. Unknowns without a matching proof remain independent variables.

## Exact grouped certificate and unchanged prior choices

Attempt the original recovery_01 interval certificate first. On success retain
its schema, original tie order and choice. Only an ambiguous root constructs
equivalence proofs for every unresolved repetition cell. Allocate no new worlds
and do not relax repetition, no-progress or maximum-turn safeguards.

For candidate i, let s_i be its known WDL sum, N_i its prescribed count and
a_ig the number of its unresolved cells in equivalence group g. The unknown
group value x_g lies in[-1,+1]. Candidate i's possible mean is
(s_i + sum_g a_ig*x_g)/N_i. For candidate c against competitor j, the exact
worst-case difference numerator is:

    s_c*N_j - s_j*N_c - sum_g abs(a_cg*N_j - a_jg*N_c)

Certify c only if this is strictly positive for every earlier j<c and nonnegative
for every later j>c. Use exact integers for collection and independent rational
arithmetic for verification. Save group memberships and pairwise lower bounds in
the coupled_repetition_original_order_v1 certificate. If no winner can be proved,
save a hold and abort cooperatively again. There is no heuristic fallback.

For the known failed root, the two unknowns share one context. Candidate0's
mean therefore always ties candidate1, while its worst possible sum-60 is at
least all later candidates' sums. The original first-maximum rule selects0 for
every common unknown value. This conclusion must be checked in production before
dispatching additional games; the diagnosis is not a substitute for that check.

## Evidence retention, execution and verification

Use a new recovery_02 output under the original arena. Hash-pin the original and
recovery_01 bindings, all effective records and historical failed ancestors,
diagnostic evidence, receipts and model registry. Retain completed indices0-309.
Reconstruct saved game prefixes310-317 using their exact actions and root hashes.
Complete only missing prescribed cells of the seven aborted roots; root310/58
already has512 cells. Keep predecessor records at their original paths/timestamps.
Each inherited record authenticates against its own generation's binding/launch.
Only the eight explicitly named incomplete roots may be replaced by new records.

The hidden run first certifies root310/58 and independently reconstructs its
proof/certificate. It then finishes games310-401, including the eight partials.
All402 completed games, including the310 retained games, undergo the prescribed
main audit. Preserve the original ordinary-world sample and171-root stress
selection. Replay every unresolved cell, deduplicating overlaps. Every saved
equivalence proof is independently reconstructed from its committed trace and
then checked against an independent035 replay. Verify original and new interval
certificates under their own explicit rules, all source lineage, budgets and
natural outer-game endings. No unfinished game enters a score tally.

Report allocated cells separately from additional proof/audit replay work.
Collection proof time and inherited computation remain included in cost reports;
same allocated-cell quotas do not imply identical total work or wall time.
Keep original402-game statistics,50,000 within-fit whole-seat-pair bootstrap,
observed score>=55% and paired95% lower>50% success gate. Final results disclose
both amendments, unresolved cells and coupled-certificate counts. No promotion,
training or automatic successor follows any result.

## Resource limits

Keep eight hidden CPU workers, one deterministic Torch thread per worker and
the original120-hour clock from epoch1791028925.3389308. Deadline remains
2026-10-08 14:02:05.339 Paris, including downtime, diagnosis and verification.
Keep5GiB disk reserve, sleep inhibition with display sleep allowed, bounded
eight-job dispatch and cooperative abort checks between worlds/roots.

Use the original outcome-blind probe's main gameplay forecast scaled by92/402,
retaining100% of its ordinary and stress audit forecast for all402 games, the
original25% time allowance and600-second close allowance. Add one hour forecast
allowance for extra proof work; do not deduct reusable partial-root work. Retain
the entire original9,729,707,417-byte storage projection conservatively. Require
the forecast to fit the original96-hour forecast cap and remaining absolute
deadline. The forecast is not an extension of either cap. Check actual resources
throughout the run. No recurring monitoring or automatic retry.
