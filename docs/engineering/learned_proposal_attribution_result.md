# Learned proposals beat the original-ranking control at32 worlds

Completed2026-09-20 at23:26:33 Paris, after34h10m including the operational probe,
main collection, full independent replay and final checks. The recorded
supervisor and audit workers exited. No failure, operational hold or nonempty
error log was present at closure. No successor or benchmark started.

## Final result

Across402 fresh games, the learned fourth-proposal planner scored **229 wins,
zero draws and173 losses:56.9652%**, paired95% confidence interval
**52.4876--61.4428%**, against the planner using the next035-ranked proposal.

Both frozen numerical criteria pass: the lower bound exceeds50%, and the
observed score exceeds55%. Final classification is `positive_practical_progress`.
The interval does not establish that the true score exceeds55%; the practical
criterion concerns the observed score. The earlier owner-requested provisional
readout agrees with the final independently verified statistics.

Both planners kept the same original first three candidates, concrete action
representatives, tie ordering,32 shared observer-legal worlds and greedy035
terminal continuations. Only the fourth suggestion differed: the assigned161
model's highest remaining group versus the original035 ranking's next group.
All three fixed161 models and both seats were retained.

| Fixed fit | Wins / games | Score |
| --- | --- | --- |
|161001|81 /134|60.4478%|
|161002|79 /134|58.9552%|
|161003|69 /134|51.4925%|

Candidate seat0 scored120/201=59.7015%; seat1 scored109/201=54.2289%.
These are descriptive subgroups, not independent significance claims or model
selection criteria. In particular,161003's point estimate is close to parity.
The primary interval conditions on these three fixed models and this matchup.

## Meaning and limits

The newer distilled ranking adds full-game value beyond using more of the
original learned035 policy's ranking, at the tested32-world budget. Alongside
the completed64.4279% result against the original random-fourth-slot planner,
this supports learned move proposals as a useful deployment role.

This compares two learned rankings, not learning versus no learning. It does not
establish superiority over the original128-world teacher, transfer of the gain
to128 worlds, equal-wall-time efficiency, broad opponent strength, a repeatedly
improving learning loop or superhuman performance. The next128-world comparison
remains a recommendation, not an authorized or started study.

## Verification and cost

- All402 main games,24,136 decision roots and **3,040,256 continuations** fully
  replayed. Independent final statistics matched exactly.
- The separate12-game probe also passed full replay:706 roots and88,672
  continuations. Probe results were excluded from the scientific endpoint.
-18,079 main roots involved own or opposing private reserves. The verifier
  checked lawful model inputs, hidden-information invariance, independently
  reconstructed worlds/proposals, all trajectories, outcomes and choices.
- Main collection took15.8155h; full main replay16.9875h. Probe collection and
  replay took0.6414h and0.6836h. Total launch-to-final-status duration34.1635h.
- Mean candidate/reference decision times were18.7875/18.6526 seconds; p95
 37.6758/37.5178 seconds. These are descriptive instrumented timings, not an
  independently controlled equal-time experiment.
- Closure rechecked all365 frozen dependencies,429 parent files, final result/
  report/audit hashes, all four stage locks and their linked games/audit receipts,
  per-game root linkage and exact replay totals. The pipeline's finalization
  separately rehashed every original root. No inference, simulation or bootstrap
  was rerun for closure.
- The rules engine and stated encoding/neural primitives remain shared trusted
  dependencies; this is not an independent implementation of Splendor's rules.

Frozen binding:
`0f2772aebd99e2fef939dc1a9464790daa1f6c3b1a8e9fe1326a2361a85e44e2`.

Result SHA256:
`7a1aa0d4fb2da8b7f6159f8bc830efaefa384aaf3816bea6e52af024dca87461`.

Final audit SHA256:
`d658a87d1d34bb29ab3371ee43d7845acf3815df3fb8e6e27115a4c16ad1bb3f`.

Authoritative artifacts: `outputs/learned_proposal_attribution/final_status.json`,
`result.json`, `final_audit.json`, `report.md`, stage locks and completion records.
Preserve them and all intentionally dirty work. Future ordinary experiments use
the owner-approved [verification policy](verification_policy.md); this study
completed its original exhaustive requirement. The prepared batching benchmark
and [recommended128-world study](../strategy/next_after_proposal_attribution.md)
remain unstarted. No training, promotion, extension or monitoring was added.
