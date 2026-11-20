# AlphaZero Search-Operator Calibration 094 — Result

## Outcome

Experiment 094 **confirmed a stronger search operator for generation 2**.
The selected configuration, 256 simulations with `c_puct = 2.25`, beat the
exact same generation-2 raw policy by **64-0-36** on the independent 100-game
confirmation panel. Score share was **64.0%**, mean score margin was **+1.66**,
and the paired-seed bootstrap CI95 was **[54.0%, 73.0%]**. Candidate seat
shares were 72.0% from seat 0 and 56.0% from seat 1. All frozen acceptance
gates passed.

The formal decision is `search_operator_confirmed`. This is an engineering
mechanism result, not a champion promotion. Exact generation 2 remains
installed; 094 created no self-play replay, child checkpoint, or `035` arena.

## Frozen provenance

- Protocol:
  `docs/engineering/alphazero_search_operator_calibration_094.md`
- Protocol SHA-256:
  `78ee76c12a466de9b0793bed14808b21d9e8cc8692a4ad28e63192f0787c749e`
- Result:
  `outputs/alphazero_search_operator_calibration_094/result.json`
- Result SHA-256:
  `fd386489640d0b6bdff806c47effedd7f388d1ae678219d97c91957ec9ad2e79`
- Generation-2 checkpoint SHA-256:
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`
- Generation-2 replay SHA-256:
  `33f8bc093d659a41618932d9a8ec0f48451c64bdb76979b42c3de81b88a1cf7b`
- Controlling 093 result SHA-256:
  `3fcba198fbcdcf48b4909c84eae6e8402dfb8fee941231791efa98fe0423f54d`

All input hashes reproduced at final audit.

## Discovery grid

Each configuration played 10 fresh paired seeds / 20 games against raw
generation 2. The panels were not pooled.

| simulations | c_puct | W-D-L | share | mean margin | eligible |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 64 | 0.75 | 11-0-9 | 55% | +0.60 | yes |
| 64 | 1.50 | 12-0-8 | 60% | +2.95 | yes |
| 64 | 2.25 | 12-0-8 | 60% | +1.50 | yes |
| 128 | 0.75 | 11-0-9 | 55% | +0.10 | yes |
| 128 | 1.50 | 13-0-7 | 65% | +2.80 | yes |
| 128 | 2.25 | 11-0-9 | 55% | +1.00 | yes |
| 256 | 0.75 | 12-0-8 | 60% | +2.80 | no (seat gate) |
| 256 | 1.50 | 13-0-7 | 65% | +2.40 | yes |
| 256 | 2.25 | 15-0-5 | 75% | +4.10 | yes |

The frozen selector chose `puct_256_c2_25`. The broad positive pattern is
useful context, but only the selected configuration was allowed to open
confirmation.

## Independent confirmation

- Schedule: exact paired seeds `1650000--1650049`, both candidate seats.
- Games: 100, with 100 unique seed-seat keys and the exact schedule order.
- Result: 64 wins, 0 draws, 36 losses.
- Score share: 0.64.
- Mean margin: +1.66.
- Seat-0 share: 0.72.
- Seat-1 share: 0.56.
- Paired bootstrap: 50,000 resamples, seed 940094, mean 0.64, CI95
  `[0.54, 0.73]`.
- Termination: 100/100 `completed`.

The bootstrap was independently recomputed from the final arena rows and
matched exactly. Arena point share and mean margin also reproduced exactly.

## Value audit

The locked generation-2 replay contained 4,208 positions from 64 games.

| prediction | MAE | MSE | outcome-sign accuracy |
| --- | ---: | ---: | ---: |
| raw network value | 0.9041 | 1.0314 | 57.72% |
| stored 64-simulation search-root value | 0.8924 | 1.0057 | 58.32% |

The stored search root only marginally improved pointwise WDL prediction.
Both estimators were poorly calibrated at the probability extremes, and the
midgame was the weakest phase. These are diagnostics on correlated on-policy
positions with high-variance final-game targets, not an independent value
benchmark. They nevertheless identify value learning as an important remaining
weakness.

The gameplay result is not contradictory: a search can choose better actions
through relative priors and leaf comparisons even when its scalar root-value
estimate is only modestly better calibrated. 094 demonstrates that enough
search at the newly selected exploration setting turns the current network
into a substantially stronger decision policy.

## Integrity and verification

- Complete pre-evidence suite: 699/699 passed in 338.23 seconds.
- Focused 090--094 contracts: 29/29 passed before evidence and 29/29 after
  evidence.
- Runtime: 240.14 minutes under concurrent 087 load.
- Runner stderr: zero bytes.
- All 180 discovery games and 100 confirmation games completed normally.
- Exact checkpoint remained installed and hash-verified.
- No self-play, training, child checkpoint, `035` arena, or champion change.
- The owner-waived 087 process and artifacts were independent; its outcomes
  were not inspected.

## Interpretation and next step

093 showed that the `value_quarter` child was not a durable improvement under
the old 64-simulation operator. 094 now localizes the problem more precisely:
the AlphaZero direction is viable, but 64 simulations at `c_puct = 1.5` was an
underpowered improvement operator for generation 2. Increasing both search
dose and exploration produced a confirmed improvement over the raw network.

The recommended 095 is therefore a controlled return to the loop:

1. freeze exact generation 2 and the confirmed 256 / 2.25 operator;
2. generate a fresh, sufficiently sized self-play replay with that operator;
3. train one stabilized successor, retaining generation-2 replay and the 091
   value/validation safeguards;
4. make searched child-versus-parent performance under the same confirmed
   operator primary, with raw play diagnostic only;
5. independently confirm any child before installing it or continuing another
   generation.

Do not reopen `value_quarter`, tune on the 094 confirmation seeds, or treat
094 itself as evidence that a newly trained child will necessarily improve.
