# Learned fourth-proposal arena: full-game improvement established

Completed 2026-09-19 at10:59:53 Paris after33h49m including the separate probe,
main collection and all prescribed full replays. The supervisor exited. No
failure, nonempty error log, automatic extension or successor was recorded.

## Result

The learned fourth-proposal planner scored **259 wins, zero draws and143 losses
in402 fresh games** against unchanged Experiment155: **64.4279%**, with a
stratified paired95% confidence interval of **59.7015% to69.1542%**.

Both predeclared superiority and observed55% practical-progress criteria passed.
Classification: `positive_practical_progress`. The estimate and interval are
conditional on the three fixed161 checkpoints and this opponent/budget.

| Fixed fit | Games | Score |
| --- | ---: | ---: |
|161001|134|66.4179%|
|161002|134|61.1940%|
|161003|134|65.6716%|

Each fit used67 distinct starting seeds, with seats reversed. Candidate seat0
scored66.1692% over201 games; seat1 scored62.6866% over201 games. Fit and seat
figures are descriptive, not separate significance claims or selection gates.

## What this establishes

Replacing the fourth random candidate with a frozen learned suggestion improved
complete-game play against155. Both players repeatedly used the32-world planner,
and the new candidate retained the original first three proposals, concrete
representatives, greedy035 continuations, terminal objective and tie rules.
Private reserves were included:15,682 of24,266 real decision roots contained an
own or opposing private reserve. No additional training occurred.

This supplies a successful full-game transfer from the screened learned-proposal
intervention into planning. The earlier screen also favored learned proposals
over an extra035-ranked candidate under its conditional evaluation; that
attribution remains limited to those conditions. This arena did not test the
raw-ranked replacement opponent, so it does not establish full-game superiority
to that cheaper alternative. It also did not test the stronger128-world planner,
broader opponents, a repeated improving learning loop or superhuman performance.

The same per-candidate simulation budget was used. Measured mean decision wall
times were18.527s for the candidate and18.569s for reference, with p95 values
37.128s and37.059s. These are descriptive instrumented match timings, not a
controlled equal-wall-time or speedup claim.

## Verification and resource use

- All402 main outer games,24,266 roots and **3,050,016 continuations** received
  full independent reconstruction/replay; independent final statistics agree.
- The separate12-game probe and its94,112 continuations also passed full replay.
  Probe outcomes were excluded from the scientific endpoint.
- Main collection took15.699h; main full replay took16.753h. Probe collection
  and replay took0.670h and0.677h. Total pipeline duration was33.823h.
- Completion status verification rechecked the frozen identity, all four
  probe/main collection/audit locks and linked artifacts, and final result,
  report and audit hashes. Finalization had also rechecked every main/probe root
  hash. This status request did not run new inference, simulations or statistics.
- The verifier shares the frozen engine and stated canonical/encoding/neural
  primitives; it is not a second implementation of Splendor's rules.

Frozen binding:
`2154ba0b27284ae6077300376e6f4694254ab84f8826be8ce992d284add59a81`.

Result SHA256:
`a470f2ac783228c2dd9b1db83368f6dd49d1c68791df6aef27834109542f466e`.

Audit SHA256:
`92f8ff0a13ad27f681a348c1bc9c36603b78fba926b375a6a67040776afafa23`.

Authoritative evidence: `outputs/learned_proposal_arena/result.json`,
`final_audit.json`, `final_status.json`, `report.md`, completion records and
stage locks. Frozen protocol: `docs/engineering/learned_proposal_arena.md`.
All existing work remains preserved. No promotion, successor or monitoring
was started by completion or this status request.
