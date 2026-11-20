# AlphaZero Corrective Loop `091`

Status: frozen engineering protocol  
Date frozen: 2026-08-09  
Scope: architecture/debugging evidence only; no champion promotion claim

## Purpose

`090` proved that the iterative policy/value loop runs, but generation 3
regressed against generation 2. `091` tests the specific correction chosen in
the `090` report:

1. a child is never installed automatically;
2. held-out value stability and paired parent arenas are explicit acceptance
   gates;
3. search must beat the accepted raw network before its visits generate a new
   training corpus;
4. replay recency and safe runtime improvements are applied without using
   `035` as a teacher.

## Locked inputs

- `090` result SHA-256:
  `41a1d89f68e9c15b47679bae376e58528bf505799cc91dc984adec70e9e90621`.
- Incumbent: `090` generation 2, not regressed generation 3.
- Incumbent checkpoint SHA-256:
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`.
- Replay window contains only `090` generations 1 and 2:
  - generation 1 replay SHA-256
    `500b9e6a68e8e2332767c03eaecede595a63ee30b38b52a5efd001f6619d925a`;
  - generation 2 replay SHA-256
    `33f8bc093d659a41618932d9a8ec0f48451c64bdb76979b42c3de81b88a1cf7b`.
- Generation 0 is excluded as the random-policy distribution. Generation 3 is
  excluded because it is the rejected child, not a replay producer.
- Validation membership is a stable whole-game split using salt `910091`.

## Candidate training screen

Both candidates start from the exact incumbent and use AdamW, batch size 256,
weight decay `1e-4`, at most six epochs, early-stopping patience two, and
minimum objective improvement `0.002`.

| Candidate | Learning rate | Policy weight | Value weight |
|---|---:|---:|---:|
| `balanced_half` | `1e-4` | 1.0 | 0.5 |
| `value_quarter` | `1e-4` | 1.0 | 0.25 |

Loss components are always reported unweighted as well as through the weighted
training objective. A candidate is offline-eligible only if, on the common
held-out games:

- weighted objective delta child-minus-parent is `< 0`;
- policy-loss delta is `<= 0`;
- value-loss delta is `<= +0.025`;
- policy target-top agreement is no more than `0.01` below the parent.

Among eligible candidates, select lowest child validation objective, then
lowest policy loss, then the table order above. Candidate arenas are never
used to choose between hyperparameters.

## Candidate acceptance/rollback gate

The offline-selected candidate plays eight paired seeds (16 games) in each of:

- child raw versus incumbent raw;
- child search versus incumbent search, with 64 simulations, `c_puct=1.5`, no
  root noise, and otherwise the `090` search contract.

Seeds start at `1,610,000` and `1,610,100`. Acceptance requires in **both**
arenas:

- score share `>= 0.50`;
- mean candidate margin `>= 0`;
- every game terminates normally.

If any condition fails, the candidate is rejected, the installed checkpoint
remains generation 2, and no fresh self-play is generated. This is the intended
rollback behavior, not a failed run.

## Search-improvement gate

Only after candidate acceptance, compare these fixed search configurations
against the accepted raw network on four paired discovery seeds each, starting
at `1,611,000`, `1,611,100`, and `1,611,200`:

| Search | Simulations | `c_puct` |
|---|---:|---:|
| `baseline_64` | 64 | 1.5 |
| `focused_96` | 96 | 1.0 |
| `exploratory_96` | 96 | 2.0 |

All use real observer-relative search, zero evaluation noise, and the remaining
`090` defaults. A discovery configuration is eligible at score share `>=0.625`,
positive mean margin, and all normal terminations. Select by score share, mean
margin, fewer simulations, then table order. Confirm only that configuration
on eight new paired seeds starting at `1,612,000`.

The search-improvement gate passes only if confirmation score share is
`>=0.5625` (at least nine wins' worth in 16 games), mean margin is positive,
and all games terminate normally. Otherwise keep the accepted raw child but
generate no new replay.

## Conditional continuation

If both gates pass:

1. Generate exactly 32 self-play games from the accepted child, using the
   confirmed search, seeds `930000--930031`, the `090` noise/temperature
   contract, and durable per-game replay/progress.
2. Train one successor with the selected candidate training configuration on a
   two-generation window: `090` generation 2 plus the new replay.
3. Apply the same offline screen and the same two 16-game acceptance arenas on
   seeds `1,613,000` and `1,613,100`.
4. Install the successor only if it passes. Otherwise roll back to the first
   accepted child.

This continuation is still engineering evidence. `035` may be measured once
for eight paired games against the finally installed raw checkpoint, starting
at `1,614,000`; it cannot affect selection or acceptance.

## Runtime changes allowed

- Reuse immutable loaded inference models across games in an arena while
  creating fresh bots/search trees per game.
- Bucket training rows by action/entity shape to reduce padding.
- Use the two-generation replay window above.
- Record early stopping and restore the best validation checkpoint.
- Add deterministic parallel self-play only if seed order, replay bytes, and
  summaries reproduce the single-worker path. Under concurrent `087` load the
  default remains one worker.

No action pruning, hidden-information relaxation, `RESERVE_DECK` penalty,
heuristic rollout, `035` target, or outcome-dependent dose extension is allowed.

## Reporting

Report all offline candidates, gates, rollbacks, termination reasons, realized
dose, replay/action diagnostics, wall time, hashes, and tests. `087` must remain
untouched in the background; report only its parent/worker health, completed
pair count, target count, and stderr size until its sealed run completes.
