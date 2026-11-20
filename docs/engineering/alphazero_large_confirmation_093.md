# AlphaZero Exact-Candidate Large Confirmation `093`

Status: frozen engineering protocol  
Date frozen: 2026-08-10  
Scope: engineering-incumbent decision; no project champion promotion claim

## Purpose

`092` selected `value_quarter` and obtained an independent 21-11 searched
result, but its 16-paired-seed bootstrap CI95 lower bound was only 46.875%.
`093` addresses only that precision failure. It freezes the exact candidate,
incumbent, search, and gates, removes all candidate selection, and increases the
confirmation dose to 50 fresh paired seeds / 100 games.

## Locked inputs

- `092` result SHA-256:
  `b3a5c2b85174bf73c119b4b10b5b821b6fb0391173c04b920fa523460061c3e1`.
- Candidate: exact `value_quarter` generation 3, SHA-256
  `87e7d591bb779df5a378b456abebd6cae94ec32f88420ce7830e5f5a41d0ea2f`.
- Incumbent: exact `090` generation 2, SHA-256
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`.
- Locked generation-2 replay for conditional training, SHA-256
  `33f8bc093d659a41618932d9a8ec0f48451c64bdb76979b42c3de81b88a1cf7b`.
- Candidate training configuration is read from the hash-locked `092` result
  and must be exact `value_quarter`: learning rate `1e-4`, policy weight 1.0,
  value weight 0.25, batch 256, six-epoch ceiling, two-epoch early stopping,
  stable split salt `910091`, and bucketed batches.

## Large searched confirmation

Candidate and incumbent both use real observer-relative search with 64
simulations, `c_puct=1.5`, zero evaluation noise, and all remaining `090`
defaults. Play paired seeds `1,630,000--1,630,049`, both seats, exactly 100
games. Persist every game through the resumable exact-prefix arena contract.

Use paired seed as the bootstrap cluster. Compute 50,000 deterministic
bootstrap resamples with replacement, seed `930093`, and linearly interpolated
2.5th/97.5th score-share percentiles.

Acceptance requires all of:

- score share `>=55%`;
- paired-bootstrap CI95 lower bound `>50%`;
- positive mean candidate margin;
- candidate score share `>=50%` from each seat;
- all 100 games terminate normally.

On the same 50 seeds, also run a 100-game raw-policy diagnostic. It is reported
but cannot change searched acceptance.

If confirmation fails, retain generation 2 and stop. Do not extend, pool prior
panels, change search, or generate replay.

## Conditional search-improvement gate

Only after searched confirmation passes, install generation 3 as the
**engineering incumbent** and compare fixed search configurations against its
own raw policy:

| Search | Simulations | `c_puct` | Discovery seeds |
|---|---:|---:|---:|
| `baseline_64` | 64 | 1.5 | `1,631,000--1,631,003` |
| `focused_96` | 96 | 1.0 | `1,631,100--1,631,103` |
| `exploratory_96` | 96 | 2.0 | `1,631,200--1,631,203` |

Discovery eligibility requires score share `>=62.5%`, positive margin, and all
normal terminations. Select by score share, margin, fewer simulations, then
table order. Confirm only the selected configuration on paired seeds
`1,632,000--1,632,007`. Confirmation requires score share `>=56.25%`, positive
margin, and all normal terminations.

If search confirmation fails, keep generation 3 as engineering incumbent but
generate no self-play.

## Conditional self-play and generation 4

Only after both confirmations pass:

1. Generate 32 self-play games from generation 3, seeds `950000--950031`, with
   the confirmed search and exact `090` root-noise, temperature, action-legality,
   and hidden-information contracts.
2. Train generation 4 on a two-generation replay window: locked generation-2
   replay plus the new generation-3 replay, using the exact candidate training
   configuration above.
3. Apply the unchanged `091` offline non-regression screen.
4. If offline-eligible, compare generation 4 versus generation 3 under the
   confirmed search on paired seeds `1,633,000--1,633,007`; acceptance requires
   score share `>=56.25%`, positive margin, and all normal terminations. Raw play
   on those seeds is a non-veto diagnostic.
5. Install generation 4 only if it passes; otherwise roll back to confirmed
   generation 3.
6. Measure the finally installed raw checkpoint versus `035` once on paired
   seeds `1,634,000--1,634,007`. This is an external thermometer only.

## Boundaries and reporting

- `035` is never a teacher, target, prior, rollout, initialization, or gate.
- No action pruning or penalty, including for `RESERVE_DECK`.
- No outcome-dependent extension, seed replacement, pooling, or retroactive
  waiver.
- `087` remains untouched and sealed. Report only parent/worker health,
  completed pair count, target count, active wall time, last progress time, and
  stderr size.
- Report confirmation/seat/bootstrap anatomy, every conditional gate and dose,
  raw diagnostics, terminations, hashes, tests, installed checkpoint, and
  whether self-play actually opened.
