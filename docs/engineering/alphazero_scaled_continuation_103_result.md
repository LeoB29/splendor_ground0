# AlphaZero Scaled Continuation 103 — Result

Date completed: 2026-08-16

## Decision

`generation_8_compounded_035_benchmark_complete_no_install`

The unchanged AlphaZero recipe compounded searched playing strength for a
second consecutive three-generation block. Generation 8 remains far below raw
legal `035`, but the now-measured gap is compatible with the pre-experiment
20--30% searched-score estimate. No checkpoint was installed or promoted.

## Continuation integrity

- Gen5, gen6, and gen7 self-play each produced 256/256 normally completed
  games: 768 clean games total, with no exclusions or replacement attempts.
- Gen6, gen7, and gen8 checkpoints passed lineage, finite-metric, validation,
  reload, and hash contracts.
- Exact gen8 SHA-256:
  `beea26beddf91c7b0ab8f531ba6bc2bfa104d4e635e12af8002d04f7fb9c9281`.
- stderr remained empty.

## Powered internal endpoint

All panels used 100 fresh paired seeds / 200 games.

| Panel | Wins-draws-losses | Share | Margin | Seat 0 | Seat 1 | Paired CI95 | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| searched gen8 vs searched gen5 | 118-0-82 | 59.0% | +1.46 | 64.0% | 54.0% | [52.0%, 66.0%] | pass |
| raw gen8 vs raw gen5 | 110-2-88 | 55.5% | +0.32 | 61.5% | 49.5% | [48.25%, 62.75%] | mixed: seat 1 missed by 0.5 pp |
| gen8 search vs gen8 raw | 141-3-56 | 71.25% | +2.90 | 76.0% | 66.5% | [65.0%, 77.25%] | pass |

Every game completed normally. The controlling internal decision is
`endpoint_compounding_with_incomplete_raw_transfer`. Search strength compounded
clearly and search remains highly effective; raw transfer was positive overall
but did not clear both seat safeguards.

## Absolute `035` benchmark

Because the primary internal endpoint passed, both prospectively conditional
panels opened on 100 new paired seeds / 200 games.

| Panel | Wins-draws-losses | Share | Margin | Seat 0 | Seat 1 | Paired CI95 |
|---|---:|---:|---:|---:|---:|---:|
| searched gen8 vs raw `035` | 53-0-147 | 26.5% | -4.695 | 32.0% | 21.0% | [20.5%, 33.0%] |
| raw gen8 vs raw `035` | 24-1-175 | 12.25% | -8.175 | 12.5% | 12.0% | [8.0%, 17.0%] |

All 400 games completed normally. Generation-8 search more than doubled the
raw network's score share against `035` and reduced the mean deficit by 3.48
points, but the remaining external gap is decisive. This does not contradict
the internal result: successful self-improvement can begin far below a mature
external policy.

## Interpretation

- Two separate powered endpoint experiments now show three-generation searched
  compounding: gen5 over gen2 at 58.25%, then gen8 over gen5 at 59.0%; both
  paired intervals exclude 50%.
- The search operator is not collapsing. Its lift at gen8 is larger than at
  gen5, and it materially improves performance against `035`.
- Network-only transfer is the weaker link. Raw gen8 moved positively over raw
  gen5 but remained only 12.25% against `035`.
- The previous indirect 20--30% estimate for searched gen8/gen5-level play
  against `035` was well calibrated; direct gen8 evidence is now controlling.
- Do not promote gen8. Exact generation 2 remains installed under the frozen
  contracts, while gen8 is the latest successful sandbox endpoint.

## Runtime and identities

- Runtime: 108,651.27 seconds (30.18 hours).
- Final result SHA-256:
  `0152c113dee56bd156483b0ab6d3a55d48e35b70cd3bcc3f5f9c0b5c2c0f7f96`.
- Installed checkpoint remained exact generation 2:
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`.

