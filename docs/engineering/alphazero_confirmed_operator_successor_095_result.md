# AlphaZero Confirmed-Operator Successor 095 — Result

## Outcome

Experiment 095 completed the controlled learning cycle but **did not confirm
the generation-3 child**. The frozen decision is:

`searched_confirmation_failed_keep_generation_2`.

The child passed strong offline diagnostics and the 32-game searched
development gate, then scored essentially at parity on the independent
100-game searched confirmation:

- 50 wins, 1 draw, 49 losses;
- score share 50.5%;
- mean margin +0.60;
- paired-bootstrap CI95 `[41.5%, 59.0%]`;
- seat shares 56% and 45%.

It missed the score-share, strict bootstrap, and second-seat gates. Exact
generation 2 remains installed. There is no champion change.

## Frozen provenance

- Protocol:
  `docs/engineering/alphazero_confirmed_operator_successor_095.md`.
- Protocol SHA-256:
  `9ac6d35f5f8e53bcf7efdfd2abb33527fa461480d4438b818d9fb39e09f884bc`.
- Result:
  `outputs/alphazero_confirmed_operator_successor_095/result.json`.
- Result SHA-256:
  `6d2ec319982ced65855d526e1fe2c692c1f4bda49d8e3f60c473b693b4714655`.
- Parent generation-2 SHA-256:
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`.
- Retained replay SHA-256:
  `33f8bc093d659a41618932d9a8ec0f48451c64bdb76979b42c3de81b88a1cf7b`.
- Controlling 094 result SHA-256:
  `fd386489640d0b6bdff806c47effedd7f388d1ae678219d97c91957ec9ad2e79`.

All frozen inputs reproduced.

## High-dose self-play

Exact generation 2 generated 64/64 normal games on exact seeds
`970000--970063` using 256 simulations, `c_puct=2.25`, and frozen self-play
noise/temperature:

- 4,106 replay positions;
- 1,051,136 total simulations and 1,030,943 expansions;
- 2.396 recorded search hours;
- mean 64.16 turns per game;
- exact 256 root visits on every row;
- all terminal outcomes and final scores present;
- fresh replay SHA-256
  `c58f9955362d1bb82b47ed34dd4bc0358cdf9a00116fca733b0171cb6d2cae47`.

Search's sampled action differed from the raw-policy top action on 52.02% of
positions; visit argmax differed on 47.25%. `RESERVE_DECK` was selected 173
times, 4.21% of positions. It remained unpenalized and fully legal.

The resumable-game audit reproduced the exact seed order, complete rows,
search configuration, and visit accounting.

## Training and offline gate

The one frozen balanced child trained for all six epochs on retained
generation-2 replay plus fresh 095 replay. Child checkpoint SHA-256 is
`36476fa8333a208c6d14559894cc491f9aa8f75a0a3c0c1b2972090d073090a4`.

Held-out child-minus-parent changes were strongly positive offline:

| metric | delta |
| --- | ---: |
| weighted objective | -0.235601 |
| policy loss | -0.118158 |
| value loss | -0.234886 |

Policy target-top agreement rose from 53.95% to 59.42%. Policy and value loss
improved in early, middle, and late validation slices, so every offline gate
passed. This is a clean demonstration that better supervised replay fit is not
sufficient evidence of stronger searched gameplay.

## Gameplay gates

### Searched development

On 16 paired seeds / 32 games, the child passed exactly at the point boundary:

- 18-0-14;
- score share 56.25%;
- mean margin +1.09375;
- seat shares 50% and 62.5%;
- 32/32 normal completions.

### Independent searched confirmation

On exact fresh seeds `1661000--1661049`, both seats:

- 50-1-49;
- score share 50.5%;
- mean margin +0.60;
- seat shares 56% and 45%;
- paired bootstrap: 50,000 resamples, seed 950095, mean 50.5%, CI95
  `[41.5%, 59.0%]`;
- 100/100 normal completions.

The bootstrap was independently recomputed and matched. The development and
confirmation panels were not pooled. The development result was small-panel
noise for acceptance purposes.

### Raw diagnostic

On the same 100-game schedule, child raw versus parent raw scored 53-1-46:
53.5% share and +0.82 mean margin, with 100/100 normal completions. This is
slightly positive but still near parity and cannot rescue the failed searched
gate.

## Integrity and execution

- Complete pre-evidence suite: 703/703 passed in 372.87 seconds.
- Focused 090--095 contracts: 33/33 passed before evidence and 33/33 after.
- Exact arena schedules and all seed-seat keys reproduced for development,
  searched confirmation, and raw diagnostic.
- Runtime: 426.69 minutes under concurrent 087 load for most of the run.
- Runner stderr: zero bytes.
- Generation 2 remains installed with its exact hash.
- No `035` arena and no champion change.
- The owner-authorized strategic stop of 087 did not touch 095; 095 finished
  its unchanged frozen schedules.

## Interpretation

094 remains valid: 256 / 2.25 search substantially improves generation 2 over
its raw policy. 095 adds the next result: fitting one child to those search
visits and outcomes did **not** transfer that improvement into a reliably
stronger searched agent.

This does not invalidate the AlphaZero direction, but it localizes the current
bottleneck to the learning/representation step rather than search dose. The
network improved replay losses and raw play slightly, yet equal-dose search
with the child was not stronger. More unchanged self-play generations or a
training retry would therefore be poorly justified.

## Recommended next step

Use 096 as a mechanism audit of the already-created parent, child, and replay,
without generating another expensive corpus first:

1. compare parent and child policy/value calibration by phase, action family,
   confidence, and search-disagreement strata;
2. run fixed hybrid-search ablations—child policy with parent value, and parent
   policy with child value—to identify which learned head prevents transfer;
3. measure whether the child reproduces search rankings on the decisive states
   where 094 search changes the raw action, rather than only average replay
   cross-entropy;
4. change the value target, policy target, or representation only after that
   attribution.

Do not promote this child, pool the 18-14 development result with confirmation,
or repeat the same training recipe on another random seed.
