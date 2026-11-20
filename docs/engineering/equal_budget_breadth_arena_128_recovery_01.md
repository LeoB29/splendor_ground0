# Recovery with certified decisions after an unfinished simulation

Authorized 2026-10-04 by the owner's "ok, do it" following the explanation of
the stopped equal-budget breadth arena. Preserve its frozen implementation,
protocol, all completed games and every failed artifact. This amendment resumes
the same 402-game schedule under one conservative extension of failure handling.
It is not completion under the unamended original protocol.

## Reproduced failure

Main game 13, turn 17, candidate side, world 114, candidate 3 reproduced its
saved repetition cutoff exactly: 18 continuation moves, including both actors
taking and returning the same blue token at their token limits. The diagnostic
saved trace and final-state hashes match the failed record. The board signature
repeats at continuation steps 12,14,16,18. The signature omits absolute turn,
which the network observes, so this proves the configured cutoff was reached,
not that the unchanged policy would cycle forever without the safeguard.

Only this branch was replayed for diagnosis. No main game scores were aggregated
or used to choose the amendment. The affected root has four candidates at 128
worlds each. Its failure is unrelated to uneven or eight-candidate allocation.

## Exact decision certificate

Keep 035, every seed, candidate, allocation, rollout safeguard and tie rule.
Run every planned cell. Only a repetition_cutoff with null winner and WDL can
remain unresolved. Completed and forced-pass-deadlock continuations retain their
original scores. Illegal execution, stalled trajectories, other cutoffs, corrupt
records and inference errors still fail. Outer games must terminate naturally.

For candidate i, let N_i be its full prescribed number of continuations, s_i the
sum of known WDL values and u_i the unresolved count. Its possible WDL mean is
bounded by [(s_i-u_i)/N_i,(s_i+u_i)/N_i]. Do not drop unresolved cells, alter the
denominator, assign a draw, add replacement worlds or impute a midpoint.

Certify candidate c only when its lower bound is strictly greater than every
earlier candidate's upper bound, and at least every later candidate's upper
bound. Use exact integer cross-products for collection and independent rational
arithmetic for verification. This preserves the original first-maximum tie rule
for every possible assignment of bounded scores to unresolved cells. When no
certificate exists, save the entire attempted root, report a hold and stop.
There is no second fallback or automatic extension of a simulation budget.

Natural-only roots retain the original arithmetic schema. At an affected root,
store null means for candidates with unknown outcomes, known sums, actual counts,
unresolved counts, bounds and a checkable certificate. Report the unresolved
work separately; never claim that all recorded continuations finished naturally.
The certificate establishes a move independently of the missing terminal result;
it does not supply that result or alter Splendor's draw rules.

## Preserved evidence and recovery order

Bind the original manifest 4afa5eb8b9d29cedae07c0c8eb00cb4cfbc34cf03fa1c11e97a7ae9bb6d30f7b,
original code identity, model registry, successful probe closure, failure,
diagnosis and all original game/root hashes in a new recovery inventory.
Retain completed games 0-12 and14-25. Reconstruct game13 from its original seed,
reuse its seventeen completed roots and preserve its failed root separately.
Resume that failed root's existing460 cells, including its one null result, and
execute only the52 missing prescribed cells. No replacement game or seed.

The hidden recovery first completes and certifies that exact failed root before
dispatching remaining games. A hold there incurs no further main gameplay.
If certified, finish game13 and games26-401, retaining all25 completed originals.
Resolve records through a hash-pinned original/recovery inventory. Never copy
and retimestamp old files to make them appear newly generated. New records carry
the original scientific binding and the additional recovery binding.

## Verification and final report

Check every actual game and decision independently, including all retained games.
Preserve the original identity-only ordinary-world sample and171-root stress
selection. Additionally replay every unresolved cell, even outside that sample;
count overlapping work once. Independently reproduce its repetition cutoff and
all saved fields. Verify every certificate, actual sparse budget, model/input
identity, recorded action and naturally completed game outcome. Check source and
recovery timestamps against their respective launch receipts and inventory links.

Keep prescribed replay counts separate from extra unresolved-cell verification,
and report terminal/unresolved continuations and certified-root counts. Never use
unresolved locations to replace or select ordinary/stress audit samples. Final
50,000 within-fit whole-pair bootstrap statistics and55%/lower-bound>50% gates
are unchanged and independently reconstructed from all402 completed games.

Reuse accepted probe evidence without repeating its simulations. Preflight covers
exact bounds/ties, mixed original/recovery provenance, failed-prefix preservation,
independent cutoff replay, sparse budgets, hidden workers and failure cancellation.
Do not repeat historical training or exhaustive mature continuation verification.
Preserve failed preflight attempts. Freeze recovery code/tests/protocol before
production writes. Final results explicitly disclose this amended protocol.

## Resources and execution

Retain eight hidden CPU workers with one deterministic Torch thread each, no GPU,
display sleep allowed and system sleep inhibited. Keep the original120-hour clock
from epoch1791028925.3389308, including downtime and diagnosis. Deadline remains
2026-10-08 14:02:05.339 Paris. A recovery launch time never resets that clock.

Use the original passed probe's full remaining forecast57.495651h conservatively,
without deducting completed work. Require it to fit within both the original96h
forecast ceiling and remaining original deadline, and retain5GiB free after its
full projected main storage9,729,707,417 bytes. Actual resource checks continue
between worlds/roots and at stage boundaries. No additional continuation worlds
are allocated by certification. Extra audit work remains within the same cap.

Submit at most eight jobs concurrently. On the first error or ambiguous decision,
write an abort marker so other workers stop at their next cooperative boundary,
preserving partial records. Do not drain a long queue of additional games.
No automatic retry, monitoring, training, promotion or successor. The reservation
strategy anecdote remains a separate note for possible later exploration.
