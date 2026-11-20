# Fixed training effort with more independent games

7 November 2026. User authorization: implement and qualify the study with
1,024 training games and128 held-out games, then run if checks pass. Keep
training effort fixed and original EF8A as reference. This separate execution
implements the accepted6November preparation; it does not edit that preparation
or any completed study. No cloud rental, external opponent or six-world work.

## Scientific contract

Collect1,152 fresh independent corrected-rules games with the original EF8A
teacher at512 simulations, three hidden worlds, cpuct0.4, depth64 and float32.
CPU8 uses the accepted16-game queue/refill mechanics. There are72 cells. Each
setup/randomness seed appears in one collection trajectory. Predetermine the
1,024training/64validation/64test partitions and balance first seat in each.
Retain the full genuine root-visit policy targets and official terminal outcomes;
never invent a label for an unfinished, deadlocked or failed game.

The single predetermined192-game narrow subset is nested in the1,024-game pool.
Fit narrow192×16roots and broad1,024×3roots, two fixed repetitions each. Each fit
has3,072distinct examples,96updates of batch32 and1,024roots from each relative
trajectory third. The narrow arm rotates its extra sixth root across the thirds
for64games each; the broad arm uses one root per third per game. Selection hashes,
subset and optimizer seeds are fixed before collection. Insufficient usable
roots, a missing row or failed game stops incomplete, with no substitution.

All four fits begin from pristine historical EF8A. Architecture, float32, shared
layers, both policy/value heads and the accepted joint loss coefficients are
unchanged. AdamW LR1e-5, weight decay1e-4, clip1. Save checkpoints0,95,96, with
exact95→96 reproduction of model, optimizer, RNG, outputs, losses and gradients.
The old historical-data prefix equality is inapplicable to fresh roots; the
learning math remains source-bound to the accepted one-pass implementation.
There are384scientific updates, two discarded qualification updates and four
last-update reproductions:390maximum optimizer calls. No sweep or early stopping.

Commit all four final checkpoints before held-out model predictions. Correctness
auditors may inspect raw held-out trajectories and targets earlier. Evaluate
original EF8A and all four fits on all usable positions of the128held-out games,
and selected training roots separately. Primary offline contrast: equal-repetition
mean broad-minus-narrow policy cross-entropy, first averaging within each game.
Value MSE and each model versus original are secondary. Report the two64-game
partitions and the prospectively pooled128. Use20,000 paired whole-game bootstrap
draws, stratified by partition, with common indices across models. Two training
repetitions do not double the128independent assessment games.

Both broad fits are prospectively nominated for live play regardless of offline
loss signs. Each receives correctness qualification and six excluded operational
games, then200fresh main games against original EF8A at matched128simulations.
The total is400main games in200disjoint swapped-seat pairs, alternating ten-game
blocks between candidates. Three worlds, cpuct0.4, depth64 and unchanged reference.
The equal-average primary and50,000stratified paired bootstrap remain fixed.
Official AND conservative useful-gain criteria require score≥55%, two-sided95%
lower bound>50%, each pooled seat≥50%, and the existing degenerate safeguard.
Administrative games are candidate losses in conservative analysis. Classified
hard-guard expiry is an official loss for the responsible player. Infrastructure
or correctness failures stop incomplete. Narrow models have offline assessment
only: this study cannot claim broad beats narrow in live games. No automatic
checkpoint promotion, winner selection, sample extension, retry or replacement.

## Rules, provenance and entry-point qualification

Ruleset `splendor_base_qualified_rules_20261018_v1`, catalogue SHA256
`db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312`, turn contract
`atomic_prereveal_completed_turn_v1`. Use approved corrected decoders, explicit
legal lists, independent transition and outcome checks, and all90corrected card
objects in hidden-world reconstruction. EF8A SHA256
`ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb` remains an
explicitly historical warm start, never relabelled corrected training data.
Atomic pre-reveal and internal forced-pass/deadlock conventions remain disclosed.

Every collector, trainer, evaluator and fresh verifier process verifies runtime
rule sources exactly once at startup and preserves that receipt. The actual
collector→target capture→new loader→encoded batches→trainer chain must pass the
six corrected card faces, non-purchase/multiple/final-round nobles, gold payments
and returns, private reserves, value orientation and bad identity/hash rejection.
Keep accepted native graph/numerical/full-decision/legal-support checks. A source
manifest or outer corrected environment alone is insufficient qualification.

Two fresh predetermined engineering games, excluded from all1,152scientific games
and all400strength games, additionally qualify the new lossless recording route
end to end and supply genuine terminal-labelled rows for the two discarded
trainer updates. These run inside the original four-hour qualification stage.
No scientific collection is released before the complete qualification gate.

Each collection game and target receives independent saved checks; each cell
is accepted before the next starts. Replay the first neural decision in each
available ply band0–19/20–39/40+ per game at the prescribed shifted physical slot,
at most3,456collection reproductions. Entire corpus acceptance precedes training.
Strength retains all full saved checks and at most96selected exact reproductions
per candidate; overlaps are deduplicated with all coverage cells accounted for.

## Recording and storage

Use a separate qualified lossless format for high-volume immutable JSON evidence.
Compression must preserve every logical record and all prior verification checks.
Physical file descriptors hash actual stored bytes; decoding must reject malformed,
truncated, corrupted or mismatched data. Tests include saved-record round trips and
fresh complete games before production admission. Mutable clocks/status and live
communication retain their separately authenticated format. No historical evidence
is removed. Compression ratios and duration forecasts are advisory, not admission
gates; actual disk/output ceilings are enforced.

## Ownership, deadlines and resource bounds

One retained root owner creates one exclusive claim. All children inherit its
original wall and monotonic clocks and exact frozen source/runtime identity.
Owned Windows jobs, exact PID plus creation-time checks, observed actual exits
and cleanup precede success. A completion JSON alone is not final acceptance.
All work runs automatically through qualification, collection, corpus acceptance,
four fits, held-out evaluation, strength qualification/operations/main, reproductions
and saved-only independent final acceptance. No scheduled chat monitor is created.

Overall120hours; model cutoff96hours. Stage ceilings, always clipped to those
original clocks: qualification4h, collection60h including cell checks, corpus
acceptance8h, all fits4h, offline6h, all strength qualification/operations/main40h,
strength reproduction8h, final24h including a12h independent reader. Each16-game
cell including its saved check has4h. These maxima do not add time or reset clocks.

Collection decisions retain60seconds internal/75seconds full response. Strength
retains9/10seconds. One second and700ms are descriptive, with no admission gate.
No active move can receive a reset clock. Preserve the first failure and all
released work; stop later releases, with no game replacement or synthetic outcome.

Actual owned RSS ceilings:8GiB collection/audit,4GiB fit/offline,2GiB strength
inference. At most8collection search clients plus one service, and one phase active.
Actual new-evidence ceiling128GiB, with16GiB reserved for finalization; actual free
disk floor5GiB. No available-RAM reserve, stability wait or forecast stop. The
48–60hour forecast is advisory. Original EF8A remains reference after any result.

## Acceptance and interpretation

Model-free tests and three independent source-bound reviews precede execution
freeze. Dynamic corpus/selection/checkpoint manifests are authenticated along the
fixed recipe; no future checkpoint is falsely claimed to be hashed prelaunch.
Final acceptance independently checks saved science, exact planned counts, source
preservation, all actual stage exits, complete evidence inventories and process
cleanup. Report remaining uncertainty and one or two intuitive charts.

The offline experiment tests whether wider game coverage helps generalization at
fixed training effort. Live games test whether the two predetermined broad fits
improve on original EF8A. A negative or inconclusive result is valid science and
does not authorize automatic additional training. Neither outcome by itself proves
superhuman play or establishes the optimal amount of self-play data.
