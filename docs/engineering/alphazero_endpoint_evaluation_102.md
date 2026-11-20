# AlphaZero Rolling-Lineage Endpoint Evaluation 102

Date frozen: 2026-08-15

## Purpose

Experiment 102 measures whether the completed generation-2 to generation-5
rolling AlphaZero lineage compounded playing strength. It is an endpoint test,
not another training generation and not a promotion arena.

## Frozen inputs

- Generation 2: `outputs/alphazero_pivot_090/pilot/generation_002.pt`
  (`60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`).
- Generation 5: `outputs/alphazero_clean_continuation_101/checkpoints/generation_005.pt`
  (`201e34128c9588722ce78edc85e8672675b1075dc0696651484718ddd7fa2444`).
- Completed 101 result: `outputs/alphazero_clean_continuation_101/result.json`
  (`4e29b92f13232f31f082fd0a3c5eb00bf2bd1068a18956832b048cfb7bb04a05`).

The runner must verify all three hashes and the completed 101 decision before
playing a game.

## Fixed evaluation operator and schedule

- Search: 256 simulations, `c_puct=2.25`, real mode, no root noise.
- Maximum game length: 200 turns.
- Fresh environmental seed block: 1,800,000 through 1,800,099.
- Every panel uses all 100 seeds twice, swapping the candidate seat: 200 games
  per panel and 600 games total.
- The same seed block is deliberately shared across panels to improve matched
  mechanism interpretation. It is untouched by training and prior evaluation.
- Each game is written atomically to a panel progress artifact. Resumption must
  accept only an exact prefix of the frozen seed-seat schedule.

## Panels

The panels run in this fixed order and all run regardless of earlier results:

1. `searched_generation_5_vs_searched_generation_2` (primary endpoint test).
2. `raw_generation_5_vs_raw_generation_2` (network-transfer diagnostic).
3. `generation_5_search_vs_generation_5_raw` (search-health diagnostic).

For each panel, report wins, draws, losses, score share, mean score margin,
seat-specific score shares, termination reasons, and a 50,000-sample paired
bootstrap 95% interval clustered by environmental seed.

## Prospectively frozen practical assessment

A panel passes its practical-strength gate only when all of the following hold:

- score share is at least 55%;
- mean candidate score margin is positive;
- score share is at least 50% from each candidate seat; and
- every game terminates normally (`completed`).

The bootstrap interval is reported as evidence strength but is not a binary
mechanism gate. Formal installation, if later desired, requires a separate
untouched promotion protocol.

Primary interpretation:

- practical gate pass: meaningful endpoint compounding;
- 50% to below 55% with normal games and nonnegative margin: practically flat;
- below 50%, negative margin, abnormal termination, or a failed seat safeguard:
  endpoint failure or mixed evidence requiring diagnosis.

The raw and search-health panels classify the likely mechanism. They do not
retroactively veto a primary pass.

## Safety and scope

- No self-play corpus is generated.
- No network is trained.
- No checkpoint is copied, installed, promoted, or substituted.
- Generation 2 remains the installed incumbent throughout.
- No 035 or human arena is run.
- Partial results are not used to stop early or alter the schedule.

