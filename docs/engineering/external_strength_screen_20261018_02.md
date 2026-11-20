# SplendoRust qualification and external competitiveness screen

This prospective protocol implements the request to find a stronger external
opponent on the corrected rules, qualify it, and conditionally run the same
60-game development screen. The larger independent-data study remains deferred.
No model is trained and no historical engine, checkpoint, result or completed
screen is changed.

## Candidate and evidence

The selected source is [SplendoRust](https://github.com/paytonjjones/splendorust/tree/1828507b8d2f0e02bc03d102b9c6d29a54fa8d57),
MIT licensed, pinned to commit `1828507b8d2f0e02bc03d102b9c6d29a54fa8d57`.
Its public [champion record](https://github.com/paytonjjones/splendorust/blob/1828507b8d2f0e02bc03d102b9c6d29a54fa8d57/docs/CHAMPION.md)
binds the 4,780,883-parameter Entity model and reports a substantial comparison
with another open-weight agent. The
[matched-simulation comparison](https://github.com/paytonjjones/splendorust/blob/1828507b8d2f0e02bc03d102b9c6d29a54fa8d57/research/sprint48/opponent_search/RESULT-6400.md)
reports 58.85% over 1,000 games at 6,400 simulations per move. These are author
results on AlphaZero's rules and different hardware; they motivate selection,
but do not establish the strength of the configuration below.

The checkpoint SHA256 is
`ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb`;
the published portable model SHA256 is
`57f6e227f8ac0382b7fa67dba6b58ec6663d1f635c7b9f583937867cd6692deb`.
Use safe weights-only loading and the authenticated upstream Entity definition.
The Rust search, canonical engine, public encoder and auxiliary heuristic
sources are copied without edits into an isolated build. The local transport
uses the upstream host-inference callback with single-thread CPU PyTorch.
Check host outputs against the published Rust model before acceptance.

## Explicit local profile

This is a separately named adaptation, not the published PUCT6400 entrant.
Preserve PUCT, depth64, world pool3, chance universes3, cpuct0.4, dynamic FPU,
FPU reduction0.02965, zero uniform prior, no noise and batch size1. The
chance-universes setting is inactive in the upstream canonical search path;
its native-champion chance scheduling is not used here. Select only
the simulation quota by the timing procedure below. Start a fresh search agent
for each whole turn, seeded solely from the public-observation SHA256; no tree
or private history persists. This also removes upstream cross-turn cycle memory.
Retain and record upstream heuristic selection for auxiliary phases, single
choices and turns at or beyond124; do not misdescribe those as neural search.

Translate all90 cards and10 nobles by complete semantic identity. Rotate seats
by the recorded starting player so that the native turn/parity convention holds.
Never supply actual deck order, referee seed or an opponent's blind-reserve face.
At each Main phase, require the native complete legal set to equal the mapped
referee Main choices. Preserve the native Main enumeration and tie order.

The referee requires one atomic action. Payment, return and noble choices use
the upstream auxiliary heuristic restricted to the referee's complete atomic
choices. Construct intermediate observations algebraically before unknown
refills or blind-reserve faces are revealed. This timing adaptation can change
auxiliary choices relative to native multi-phase play. Deterministic auxiliary
ordering is part of the adapter. A referee forced-pass with no native Main
choice is an explicit adapter action; it is never scored as a deadlock draw.
Search retains the native canonical deadlock evaluation. These differences are
recorded rather than silently equated with an unchanged published player.

## Compatibility and timing selection

Use the same predetermined18 saved core positions and five edge cases, converted
to labelled corrected-rules counterfactual fixtures. Check the90/10 catalogue,
observer masking, full Main support, payment/return/noble support, start-seat
rotation, forced pass and pre-reveal auxiliary observations. Require maximum
absolute native-versus-host policy/value error at most0.0005 on encodable
fixtures. Preserve unsupported fixtures and every failure; a prerequisite
failure prevents a match launch.

Before any match outcome, try fixed quotas128,64,32,16,8,4,1 in descending order.
For each quota, time two passes over the23 positions with the external one-second
clock on the assigned performance core. Stop an unsuccessful quota on its first
failure or response above0.70 seconds, retain that evidence, and try the next.
Select the first quota whose46 requests are legal, deterministic and at most
0.70 seconds. There is no selection by playing strength. The adapter also has
a0.90-second internal fail-closed allowance; the complete request retains the
same externally enforced one-second deadline. If quota1 fails, do not screen.

After source qualification, freeze the selected profile, code, model, binary,
receipts, schedule and this protocol. Repeat the full92-request dynamic gate:
both players on23 fixtures, each original and hidden-state variant. Require
46 identical privacy pairs and every response legal and within one second.
No screen starts if any prerequisite fails. A necessary implementation repair
requires a new preserved qualification attempt before a fresh freeze.

## Fixed screen and acceptance

Local player remains raw035, checkpoint
`4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`,
without retraining. The corrected referee remains
`splendor_base_qualified_rules_20261018_v1`, catalogue
`db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312`.
The historical four/eight-candidate planner results do not establish strength
on these corrected rules.

Run exactly30 fresh initial worlds in both seats,60 games. Derive full SHA256
seeds from `splendor_external_corrected_rules_20261018_stronger02_development_v1:pair:<index>`.
Explicitly exclude the first screen's30 seeds and check discovered historical
seed manifests up to8MiB, disclosing larger omissions. Alternate first-run seat.
Freeze50,000 paired bootstrap resamples, standard-library Random seed
2026101803510, linear2.5/97.5 percentiles, and opponent win/draw/loss scores
1/0.5/0. Report results without adding a post-outcome success threshold or games.

Use one supervised worker with a persistent Rust child, one assigned performance
core and one numerical thread. Charge initialization separately with a60-second
cap; include Python, Rust and authenticated Windows console helpers in sampled
memory and CPU accounting. Pin a console helper once during initialization if
Windows created it before its parent's affinity was set; record that adjustment.
Any later affinity drift fails supervision. The Windows job owns the process
tree. Each whole turn has one real one-second clock, including
serialization, model inference, search, auxiliary choices and final receipt.

The qualification-plus-screen execution cap is four hours, evidence cap256MiB
with8MiB finalization reserve, combined worker RSS cap2GiB, and free-disk reserve
5GiB. Require1GiB available physical memory and3GiB commit headroom before each
worker. Compiler/build caches are separately inventoried preparation artifacts;
they are not independent-data study output. Do not delete frozen evidence.

Timeout, player crash or illegal response is a counted loss with no replacement.
Referee defects, adapter prerequisite failure, initialization failure and resource
limits block acceptance. Stop unresolved games at2,000 plies without inventing
a repetition draw. Independently replay every move and bind all clock, action,
observation, seed, score and profile identities. Verify all840 prior frozen inputs
and the entire previous completed screen evidence tree, plus all new frozen
inputs, before accepting the result. This development screen can identify a
useful opponent; it does not establish elite strength or authorize a larger study.
