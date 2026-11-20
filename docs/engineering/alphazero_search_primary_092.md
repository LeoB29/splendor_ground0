# AlphaZero Search-Primary Confirmation `092`

Status: frozen engineering protocol  
Date frozen: 2026-08-10  
Scope: architecture/development evidence only; no champion promotion claim

## Purpose

`091` produced a criterion conflict: its selected child lost raw-policy play
6-1-9 but won equal-dose searched play 11-0-5. The frozen conjunction correctly
rolled back. `092` prospectively tests the programme decision that the deployed
AlphaZero-style agent is the searched agent, while raw play remains an important
mechanism diagnostic rather than an automatic veto.

No `091` gate is waived retroactively. `092` uses only fresh gameplay seeds.

## Locked inputs

- `091` result SHA-256:
  `b3dc44707e8d5948c3f18da5799bfc27c6bd99411e0e9b799ec84cbaeaa3335b`.
- Incumbent: exact `090` generation 2, SHA-256
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`.
- Candidate `balanced_half`, SHA-256
  `cf930d481696f11aecb0cb556f98ec51ef044bab8c2c24838a9e12bdb8300b31`.
- Candidate `value_quarter`, SHA-256
  `87e7d591bb779df5a378b456abebd6cae94ec32f88420ce7830e5f5a41d0ea2f`.
- Both candidates already passed the frozen `091` offline non-regression screen.
  Cross-weighted objective levels are not used in `092` selection.
- Search for candidate comparison is fixed at 64 simulations, `c_puct=1.5`,
  real observer-relative mode, zero evaluation noise, and all remaining `090`
  defaults.

## Durable arena contract

Before gameplay, the arena runner must persist one row after every game. Its
identity contains arena name, seed start, pair count, and maximum turns. Resume
is allowed only when stored rows are the exact prefix of the frozen
`(seed, candidate seat)` schedule. Resumption recreates fresh bots/search trees
for unfinished games and cannot change completed rows.

## Candidate discovery and selection

Each candidate plays the incumbent under equal-dose search on the **same** eight
paired seeds `1,620,000--1,620,007` (16 games each). Reusing the same seeds is
intentional common-random-number comparison.

A candidate is discovery-eligible only if:

- score share `>= 0.50`;
- mean candidate margin `>= 0`;
- every game terminates normally.

Among eligible candidates select, in order:

1. higher score share;
2. higher mean margin;
3. lower comparable unweighted held-out total loss from `091`;
4. fixed order `balanced_half`, then `value_quarter`.

Discovery outcomes select only one candidate. They are not pooled with
confirmation.

## Fresh searched confirmation

The selected candidate plays 16 new paired seeds
`1,621,000--1,621,015` (32 games) against the incumbent under the same search.
The paired seed is the bootstrap cluster. Use 20,000 deterministic bootstrap
resamples with replacement, seed `920092`, and the linearly interpolated
2.5th/97.5th percentiles of score share.

Search-primary acceptance requires all of:

- overall score share `>= 0.5625`;
- paired-bootstrap CI95 lower bound `> 0.50`;
- positive mean candidate margin;
- candidate score share `>= 0.50` from each seat separately;
- all 32 games terminate normally.

The same selected candidate also plays a raw-policy diagnostic against the raw
incumbent on those 16 paired seeds. Raw results are reported but cannot change
the prospectively search-primary acceptance decision.

If searched confirmation fails, retain generation 2 and stop. Do not extend the
panel, pool discovery, switch candidates, open search configuration discovery,
or generate replay.

## Conditional search-improvement gate

Only an accepted candidate may enter the unchanged `091` search screen against
its own raw policy:

| Search | Simulations | `c_puct` | Discovery seeds |
|---|---:|---:|---:|
| `baseline_64` | 64 | 1.5 | `1,622,000--1,622,003` |
| `focused_96` | 96 | 1.0 | `1,622,100--1,622,103` |
| `exploratory_96` | 96 | 2.0 | `1,622,200--1,622,203` |

Each discovery block is four paired seeds. Eligibility requires score share
`>=0.625`, positive mean margin, and all normal terminations. Select by score
share, margin, fewer simulations, then table order. Confirm only the selected
configuration on eight paired seeds `1,623,000--1,623,007`.

The search gate passes at confirmation score share `>=0.5625`, positive mean
margin, and all normal terminations. Failure retains the accepted searched
candidate as the engineering incumbent but generates no replay.

## Conditional self-play continuation

If searched candidate confirmation and search-improvement confirmation both
pass:

1. Generate exactly 32 self-play games from the accepted generation-3 child,
   seeds `940000--940031`, using the confirmed search and the `090`
   noise/temperature/privacy contract.
2. Train generation 4 on a two-generation window: locked `090` generation-2
   replay plus new generation-3 replay. Use the selected candidate's exact
   `091` training configuration, including stable validation and early stopping.
3. Require the unchanged `091` offline non-regression screen.
4. If offline-eligible, play generation 4 versus generation 3 under the
   confirmed search on eight paired seeds `1,624,000--1,624,007`. Acceptance
   requires score share `>=0.5625`, positive mean margin, and all normal
   terminations. Raw play on the same seeds is a non-veto diagnostic.
5. Install generation 4 only if it passes; otherwise roll back to the accepted
   generation-3 searched child.

Finally, the installed raw checkpoint may play `035` once on eight paired seeds
`1,625,000--1,625,007`. This is an external thermometer and cannot affect any
selection, acceptance, or champion status.

## Boundaries and reporting

- No `035` prior, teacher, target, initialization, rollout, or candidate filter.
- No action pruning or penalty, including for `RESERVE_DECK`.
- No seed replacement, outcome-dependent extension, gate relaxation, or
  discovery/confirmation pooling.
- `087` remains untouched and sealed; report only process health, completed
  pair count, target count, active wall time, and stderr size.
- Report every discovery candidate, paired bootstrap identity, seat split,
  raw diagnostic, conditional dose, rollback, termination, hash, and test.
