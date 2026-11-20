# Learned fourth-proposal screen: completed result

Completed 2026-09-18 at 00:06:13 Paris, approximately 1h35m after the
22:30:43 launch. The main study took 1h28m, including its full replay audits.
No new training was performed.

## Results

All 288 main positions from distinct fresh 035 games, all three frozen 161 fits
and both controls were retained. Each arm offered four candidates when legal
support allowed, with 32 selection worlds per candidate and 128 separate
evaluation worlds per position. Only the fourth proposal varied.

| Prespecified contrast | Score gain | Paired 95% confidence interval |
| --- | ---: | ---: |
| Average learned proposal minus original random-slot reference | +0.8522 percentage points | +0.3540 to +1.4174 |
| Average learned proposal minus extra 035-ranked proposal | +0.4209 percentage points | +0.0235 to +0.8409 |

Both lower confidence bounds exceeded zero, meeting the frozen classification
`positive_conditional_learning_contribution`. The second contrast's lower bound
is close to zero, so the strength of the attribution evidence should not be
overstated. The intervals are conditional on the three fixed trained models.

The original reference's held-out score was 53.8425%; the 035-ranked control's
was 54.2738%. The three learned variants scored 54.6197%, 54.7160% and 54.7485%.
These are continuation outcome scores after a selected move, not head-to-head
win rates between complete planners. Learned variants changed the selected move
in 26.4% to 27.1% of positions relative to the reference. Their fourth proposals
averaged ranks 6.57 to 6.77 under the original policy, versus 3.05 for the ranked
control and 14.32 for the original random fourth proposal. These are descriptive
measurements, not additional significance claims.

## Interpretation and limits

Learned suggestions improved the set of moves available to the planner. The
gain also exceeded simply offering another high-ranked move from the original
policy, supporting a useful contribution from the learned ranking.

This is fresh-position evidence for a single decision followed by greedy 035
play. Selected positions exclude private reserved cards, and the continuation
policy remains fixed. It does not establish full-game improvement, equal-time
strength, superhuman performance or a productive repeated learning loop.
The same selection simulation budget was used per offered candidate; learned
inference adds a small measured cost. No successor or promotion was started.

## Integrity and retained evidence

- Main: 51,520 selection and 51,584 evaluation continuations (103,104 total).
- Separate timing probe: 3,264 selection and 3,200 evaluation continuations
  (6,464 total); probe outcomes were excluded from the main result.
- Every collected continuation and source game was fully replayed. All
  prescribed audits and independent final statistics passed.
- Read-only completion verification confirmed the frozen code/runtime/evidence
  identity, all 16 stage locks, final result/report/audit hashes and state-results
  hash. No failure record or stderr output; supervisor process had exited.
- Frozen binding:
  `2a824e8801bcc594199567868e75548607697618c1d33d0592ee1154e735db71`.
- Result SHA256:
  `6766abfc5ea6992cf1b26ee2686c4bb8a62ad13d433c9d5715332f58dbcdf684`.
- Audit SHA256:
  `68b2556a75255b24cacecfd05450d6b03cd4f423c5563113a4df1a2fe69761ad`.

Authoritative files are in `outputs/learned_proposal_screen`: `result.json`,
`final_audit.json`, `final_status.json`, `report.md`, `state_results.json`,
and the probe/main stage locks. The frozen design remains in
`docs/engineering/learned_proposal_screen.md`.
