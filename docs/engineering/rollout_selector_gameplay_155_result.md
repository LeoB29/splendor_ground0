# Experiment 155: complete-game rollout selector result

Completed 2026-09-11 at07:37:29 Paris, after13h32m30s including preflight and
full independent replay. Process exited0. All statistical and integrity gates
passed. No training, promotion,084 comparison or successor was launched.

The fixed32-world selector won382 of400 complete games against exact greedy035:
**95.5% score share**, paired95% confidence interval **[93.5%,97.25%]**. There
were18 losses and no draws. Both seats scored191 wins and9 losses. This is
strong evidence that repeated application of this selector improves complete
gameplay against035, beyond154's conditional one-move result.

## Frozen design and endpoints

200 fresh starting seeds, two candidate seats per seed; paired initial decks
and nobles, independent search streams. Candidate acts on every own turn using
greedy035, the highest-ranked distinct canonical alternative and two uniformly
sampled remaining groups (retain smaller support). It shares32 observer-legal
worlds across candidates and chooses the greatest mean terminal WDL; baseline
first in exact ties. Both players use exact greedy035 inside continuations.
No recursive selection, phase filter, extra world or subsequent tuning.

Bootstrap:50,000 complete-pair resamples, frozen seed1553000000. Both gates pass:
score share>=55% and lowerCI>50%. Full integrity also passes. Mean score margin
was+6.1875 points; mean game length60.925 turns (p95=68, maximum74).

## Coverage and cost

- 12,185 candidate decisions; 6,228 changed the baseline action (51.112%).
- Candidate support1/2/3/4 occurred13/23/26/12,123 times respectively.
- 6,563 roots held own private reserves; 5,714 held opposing private reserves.
  At least one player held a private reserve at8,564 roots. Every actual root
  passed hidden-order/opposing-private-card invariance checks.
- Early/middle/late action-change rates were63.4%/55.3%/35.364%. These are
  descriptive counts, not evidence for a new phase policy.
- Main collection generated1,556,128 continuations and51,321,818 simulated
  moves. There were23 valid forced-two-PASS inner deadlocks; all other paths
  completed normally. All400 outer games completed normally. No invalid
  termination, exclusion or replacement occurred.

| Decision measurement | Mean | p95 | Maximum |
|---|---:|---:|---:|
| Selector wall seconds |19.485|39.122|62.907|
| Selector CPU seconds |17.894|35.325|58.047|
| Greedy035 wall seconds |0.00427|0.00684|0.01844|

Times were measured with eight game workers, one Torch thread each. Mean game
wall time was594.83s. Main collection took8h18m04s; the complete replay stage
took4h47m32s. Preflight collection+audit took25m47s. The original16-24h total
estimate was conservative; neither game count nor search budget was reduced.
Raw-policy CPU measurements are also saved but have coarse timer resolution.

## Integrity and interpretation

All1,243 repository tests passed before freeze. The off-panel preflight's8 games,
250 roots and31,904 continuations were fully audited before main launch. Main
replay reconstructed all400 games/24,370 real moves, every candidate set and
selected action, and every inner continuation. World reconstruction used an
independent catalog-minus-legally-known construction. Paired statistics and
operational summaries were independently reproduced. The auditor shares the
frozen engine/scorer, so this is not an independently implemented game-rules test.

Post-run verification checked frozen155 and parent154 identities, test evidence,
all408 game and replay-certificate hashes, all12,435 root hashes and linked
result/audit/process-exit artifacts. No failure artifacts or experiment processes
remain. Full evidence is in `outputs/rollout_selector_gameplay_155`.

This compares a computationally expensive selector with fast035. It does not
establish equal-time efficiency, superiority to084, superhuman performance,
learnability or an improving self-play loop. The world sampler remains uniform
over legally unknown cards, without an opponent-history strategic posterior.
Champions084/035 are unchanged. The experiment is closed.

Binding: `b287ea26424623b3825e49404467329162fcefb3e114b0064c49bc11b867e8a0`.
Result SHA: `1fbe33e2582c64f251eb80b25d5bc619f89425f95a0164ca12dceebe65097f98`.
Main-audit SHA: `c4e943e32678501e4952233db8a480155182104e1c42343bc87d5198d3469f7d`.

Sources: frozen155 protocol, `result.json`, `main_audit.json`, `probe_audit.json`,
`main_games_lock.json`, `final_status.json`, `preparation/exit.json` and
`post_run_verification.json` in the experiment output directory.
