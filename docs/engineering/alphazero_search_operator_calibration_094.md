# AlphaZero Search-Operator Calibration 094

## Purpose

Programme 093 rejected the last generation-3 candidate on a 100-game searched
confirmation. Generation 2 remains the accepted network. Experiment 094 asks
the more basic question that must be answered before further self-play:
**does the current PUCT implementation reliably improve generation 2's raw
policy at any reasonable search dose?**

This is an engineering calibration, not a champion-promotion experiment. The
generation-2 checkpoint, its locked generation-2 replay, and 093's result are
immutable inputs. Experiment 087 continues independently and its outcomes must
remain sealed.

## Frozen inputs

- Generation-2 checkpoint:
  `outputs/alphazero_pivot_090/pilot/generation_002.pt`
- Generation-2 self-play replay:
  `outputs/alphazero_pivot_090/pilot/generation_002_selfplay.jsonl`
- Controlling 093 result:
  `outputs/alphazero_large_confirmation_093/result.json`

The implementation must hash-check all three inputs. No generation-3
checkpoint, `035` policy, or `087` artifact may influence selection.

## Stage A: locked value audit

Using every complete row in the generation-2 replay, report raw-network and
stored search-root value error against the final WDL target:

- count, mean prediction, mean target, bias, MAE, MSE, sign accuracy, and
  binary-probability Brier score;
- the same summaries by phase (`early`: turns 0--19, `mid`: 20--39, `late`:
  40+) and by outcome (`loss`, `draw`, `win`);
- ten equal-width calibration bins over mapped win probability `(value+1)/2`.

This audit is diagnostic only and cannot select a search configuration.

## Stage B: bounded discovery

Compare nine fixed PUCT configurations against the exact same generation-2
raw policy. Search mode is `real`, root noise is zero, maximum depth remains
96, and all other search behavior remains unchanged.

| simulations | c_puct values |
| ---: | --- |
| 64 | 0.75, 1.50, 2.25 |
| 128 | 0.75, 1.50, 2.25 |
| 256 | 0.75, 1.50, 2.25 |

Each configuration receives 10 fresh paired seeds (20 games), using disjoint
seed blocks beginning at 1,640,000. Discovery eligibility requires:

- score share at least 55%;
- positive mean score margin;
- both seat score shares at least 45%;
- all games terminate normally.

At most one eligible configuration is selected by score share, then mean
margin, then fewer simulations, then distance of `c_puct` from 1.5, then
frozen grid order. Discovery panels are not pooled.

## Stage C: independent confirmation

If discovery selects a configuration, test it on 50 new paired seeds / 100
games beginning at seed 1,650,000. Acceptance requires all of:

- score share at least 55%;
- paired-seed bootstrap 95% lower bound strictly above 50%, using 50,000
  resamples and seed 940094;
- positive mean score margin;
- both seat score shares at least 50%;
- all games terminate normally.

Discovery and confirmation are never pooled. If no discovery setting is
eligible, confirmation is not opened.

## Decisions and consequences

- `search_operator_confirmed`: the selected search specification becomes the
  approved improvement operator for the next self-play experiment. 094 itself
  still creates no replay, child checkpoint, or champion change.
- `no_search_configuration_discovered`: stop search scaling and improve value
  learning, data, or model capacity before generating new replay.
- `search_operator_not_confirmed`: treat discovery as sampling noise and take
  the same value/model corrective path.

Generation 2 remains installed in every case. There is no `035` arena and no
champion promotion in 094.

## Integrity and execution

- Arena schedules must be durable exact prefixes and resumable.
- Every game must use the frozen checkpoint for both agents.
- Tests and this protocol are completed before opening gameplay evidence.
- Results, input hashes, elapsed time, selection, gates, and decision are
  written atomically beneath
  `outputs/alphazero_search_operator_calibration_094/`.
