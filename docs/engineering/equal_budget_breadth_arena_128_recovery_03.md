# Verification-only recovery after checkpoint representation failure

The owner requested a fix and resumption on 2026-10-06. All 402 games are complete:
198 wins, 4 draws, 200 losses (49.7512% score). These known provisional outcomes
do not select the repair, audit sample, or stopping criteria. Preserve original,
recovery_01 and recovery_02 source, bindings, logs, failed records and artifacts.

## Cause and narrow correction

The independent audit stopped when a live inherited-prefix snapshot was compared
directly with its JSON-loaded checkpoint. Live deck_counts uses integer keys;
JSON object keys are strings. Game 13 at turn 17 and all eight R2 resumed game
prefixes have this representation difference. A no-inference diagnostic replayed
all 402 games / 23,946 legal moves, authenticated 319 historical artifacts and
confirmed all per-move hashes, trace hashes, final states and outcomes. Only
deck_counts key types differ at the nine inherited boundaries. The diagnostic
does not replace the prescribed policy and continuation audits.

Copy the R2 independent verifier, replacing only its inherited snapshot equality
with the existing exact canonical JSON digest comparison. Preserve every state
field, ordered list, purchased-card order and value. This is the same comparison
already used for initial states, root snapshots and final states. Add game/turn
context to assertion failures. Tests enforce complete AST equivalence apart from
this boundary helper and the audit-control import; reject substantive mutations.
The missing regression was audit_game crossing an actual JSON checkpoint.

## Evidence and execution

Keep R2 as the immutable data namespace. A facade forwards unchanged R2 source
provenance and resolution, while enforcing the new audit abort flag and original
absolute deadline and disk reserve. The historical R2 abort is retained as evidence.
Do not copy, regenerate or reseal games or the ordinary/stress audit sample.

Freeze an R3 manifest pinning code, tests, protocol, passing check logs, diagnosis,
R2 frozen inputs, full 402-game lock, existing sample/seal and eleven passed audit
receipts (0-7, 9, 11, 12). Reuse these eleven unchanged: they have no inherited
boundary affected by the correction. Audit exactly the remaining 391 games.
First audit games 13 and 310 fully using the existing sample, exercising both
recovery generations and the coupled certificate. Both must pass before dispatch
of the remaining 389 games. These two audits count toward the final 402 receipts.

Run a detached hidden supervisor and at most eight CPU workers, one deterministic
Torch thread each. Initial two-game canary phase uses two workers. Save each new
receipt immediately; bounded dispatch and cooperative abort preserve results if
another real failure occurs. Failure records identify the game and traceback.
No gameplay, training, promotion, recurring monitoring, automatic retry or successor.

New receipts retain original scientific and R2 source provenance and add their
R3 audit binding. Finalization authenticates exactly one passed receipt per game,
all raw endpoints and root hashes, the unchanged complete sample/coverage, every
unresolved-cell/proof addition, and independently reproduced original statistics.
The 50,000 within-fit paired bootstrap, observed score >=55% and paired 95% lower
>50% gate remain unchanged. Final reports disclose both decision-certificate
amendments and this representation-only verifier correction.

## Resource limits and reporting

Keep the original deadline of 2026-10-08 14:02:05.339 Paris and 5-GiB reserve.
Forecast uses the greater of the original full-panel audit estimate and observed
passed-audit wall time extrapolated to 391 games / eight workers, with original
25% allowance, 600 seconds finalization and one hour contingency. Keep the entire
original projected storage as a conservative reserve. The forecast must fit the
remaining original deadline and original forecast ceiling; it does not reset them.
Chat closure must not interrupt the hidden process. Inhibit system sleep while
allowing display sleep. Results remain provisional until final verification passes.
