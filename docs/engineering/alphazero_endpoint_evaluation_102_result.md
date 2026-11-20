# AlphaZero Rolling-Lineage Endpoint Evaluation 102 — Result

Date completed: 2026-08-15

## Decision

`endpoint_compounding_with_raw_transfer_and_healthy_search`

The exact generation-2 to generation-5 rolling lineage produced a meaningful
searched endpoint improvement. The raw network also crossed the prospectively
frozen practical threshold, and generation-5 search remained strongly useful
relative to generation-5 raw play. All three practical gates passed.

This is an internal lineage-strength result, not a promotion. Generation 2
remains installed and no checkpoint was copied or substituted.

## Fixed panels

All panels used the same 100 fresh paired seeds (`1800000..1800099`), 200 games,
and normal seat swapping. Search used 256 simulations and `c_puct=2.25`.

| Panel | Wins-draws-losses | Share | Margin | Seat 0 | Seat 1 | Paired CI95 | Assessment |
|---|---:|---:|---:|---:|---:|---:|---|
| searched gen5 vs searched gen2 | 116-1-83 | 58.25% | +1.385 | 62.0% | 54.5% | [51.5%, 64.75%] | pass |
| raw gen5 vs raw gen2 | 110-1-89 | 55.25% | +0.995 | 55.5% | 55.0% | [48.0%, 62.25%] | pass |
| gen5 search vs gen5 raw | 129-0-71 | 64.5% | +1.865 | 67.0% | 62.0% | [58.0%, 71.0%] | pass |

Every game ended with termination reason `completed`.

## Interpretation

- The primary searched endpoint cleared 55%, had positive margin, passed both
  seat safeguards, and its paired interval also excluded 50%. This is the
  clearest evidence so far that the AlphaZero-style loop compounded strength
  across several generations.
- Raw gen5 barely but symmetrically cleared the practical gate. Its interval
  overlaps 50%, so raw transfer is promising rather than independently
  conclusive at strict confidence.
- Search added a large, seat-balanced lift to generation 5, with its interval
  excluding 50%. There is no endpoint evidence of search collapse.
- The small 16-game intermediate panels in 099/101 were too noisy to describe
  the final trajectory reliably; the prospectively fixed endpoint test was the
  appropriate arbiter.

## Integrity

- Status: `complete`.
- Runtime: 25,455.81 seconds (7.07 hours).
- All 600 scheduled seed-seat keys completed in exact order.
- stderr remained empty.
- Final result SHA-256:
  `82e515abb4f1edbf2333f0cae93b02388db96b58d10defb7b92195810ce6b3be`.
- Exact generation 2 remains installed:
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`.

