# AlphaZero Confirmed-Operator Successor 095

Status: frozen engineering protocol  
Date frozen: 2026-08-11  
Scope: controlled loop resumption; no champion-promotion claim

## Purpose

Experiment 094 established that generation 2 searched with 256 simulations and
`c_puct = 2.25` is reliably stronger than the same network's raw policy. It did
not establish that a successor can learn that improvement. Experiment 095
tests exactly the next AlphaZero link:

`accepted generation 2 -> confirmed search -> fresh self-play -> one child -> searched parent arena`.

There is one training configuration and no hyperparameter selection. Raw play
is diagnostic only. The owner-waived 087 run remains independent and
outcome-sealed.

## Frozen inputs

- Generation-2 checkpoint:
  `outputs/alphazero_pivot_090/pilot/generation_002.pt`, SHA-256
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`.
- Retained generation-2 replay:
  `outputs/alphazero_pivot_090/pilot/generation_002_selfplay.jsonl`, SHA-256
  `33f8bc093d659a41618932d9a8ec0f48451c64bdb76979b42c3de81b88a1cf7b`.
- Controlling 094 result:
  `outputs/alphazero_search_operator_calibration_094/result.json`, SHA-256
  `fd386489640d0b6bdff806c47effedd7f388d1ae678219d97c91957ec9ad2e79`.

The implementation must verify these hashes and verify that 094's decision is
`search_operator_confirmed` with the selected specification exactly
`(simulations=256, c_puct=2.25)`.

## Stage A: fresh self-play

Generate exactly 64 games from exact generation 2 on seeds `970000--970063`.
Every move uses real information-legal PUCT with:

- 256 simulations;
- `c_puct = 2.25`;
- root Dirichlet-noise fraction 0.25;
- the existing total concentration, temperature-through-turn-19, depth,
  repetition, no-progress, and maximum-turn contracts;
- search seed 950095.

The replay records all legal actions, visit targets, raw priors/values, root
values, selected actions, outcomes, and search telemetry. `RESERVE_DECK`
remains legal and unpenalized. Self-play is durably resumable at exact completed
game boundaries; resumption must verify the identity and exact seed prefix
before appending.

All 64 games must produce complete outcome-labelled rows. Abnormal termination
is reported and makes the corpus ineligible for training; it cannot be silently
replaced or extended.

## Stage B: one stabilized successor

Only if the corpus is eligible, train one generation-3 child initialized from
exact generation 2 on this fixed two-corpus window, in this order:

1. retained generation-2 replay;
2. fresh 095 high-dose replay.

The single frozen training configuration is:

- AdamW; learning rate `1e-4`; weight decay `1e-4`;
- batch size 256; at most six epochs;
- policy-loss weight 1.0; value-loss weight 0.5;
- whole-game validation fraction 0.2 with seed 950095;
- shape-bucketed batches;
- early-stopping patience two and minimum objective improvement 0.002;
- restore the best validation checkpoint.

Offline eligibility requires all of:

- child-minus-parent validation objective `< 0`;
- validation policy-loss delta `<= 0`;
- validation value-loss delta `<= 0`;
- target-top agreement no more than 0.01 below the parent;
- every populated early (`<20`), middle (`20--49`), and late (`>=50`)
  validation slice has value-loss delta `<= +0.025`.

If any check fails, stop with generation 2 installed and open no gameplay
arena.

## Stage C: searched development gate

The one offline-eligible child plays exact generation 2 under matched
evaluation search: 256 simulations, `c_puct=2.25`, no root noise, otherwise the
same real-search contract. Use 16 fresh paired seeds / 32 games beginning at
seed 1,660,000.

The development gate requires:

- score share at least 56.25%;
- positive mean margin;
- both candidate seat score shares at least 50%;
- all 32 games terminate normally.

This panel is only a gate and is never pooled with confirmation. If it fails,
stop and retain generation 2.

## Stage D: independent searched confirmation

If development passes, test the same fixed child and search configuration on
50 new paired seeds / 100 games beginning at seed 1,661,000. Acceptance
requires all of:

- score share at least 55%;
- paired-seed bootstrap CI95 lower bound strictly above 50%, using 50,000
  resamples and seed 950095;
- positive mean score margin;
- both seat score shares at least 50%;
- all games terminate normally.

After the searched panel is complete, run a same-seed 100-game child-raw versus
parent-raw diagnostic. It cannot veto or rescue the searched decision.

## Decisions

- `generation_3_confirmed_and_installed`: all gates pass. Install the child as
  the accepted AlphaZero-loop checkpoint, but do not change either programme
  champion and do not automatically start another generation.
- `selfplay_corpus_ineligible_keep_generation_2`.
- `offline_successor_ineligible_keep_generation_2`.
- `searched_development_failed_keep_generation_2`.
- `searched_confirmation_failed_keep_generation_2`.

Discovery, confirmation, raw diagnostic, or earlier 091--094 panels are never
pooled. `035` is not opened. No training retry, seed replacement, threshold
waiver, or outcome-dependent extension is allowed.

## Verification and reporting

- Freeze this protocol and tests before opening 095 gameplay evidence.
- Hash-check inputs, replay, child, installed checkpoint, and final result.
- Audit the exact self-play seed prefix and every arena seed-seat schedule.
- Report offline aggregate/slice metrics, replay/search diagnostics, both
  searched panels, raw diagnostic, bootstrap, terminations, dose, elapsed time,
  stderr, tests, and decision.
