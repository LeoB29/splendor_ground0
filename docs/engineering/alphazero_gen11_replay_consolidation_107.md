# AlphaZero Generation-11 Replay Consolidation 107

Date frozen: 2026-08-18

## Purpose

Experiments 103--105 established real but narrow internal compounding, while
105 found a recency pattern in which gen11 improved on the newest replay and
regressed on the oldest. Experiment 106 then found no external progress over
gen8: searched gen11 and searched gen8 both scored 26.5% against raw legal
`035`. Experiment 107 is therefore a single-variable test of replay retention,
not another ordinary self-play generation.

Starting from exact gen11, fit one same-architecture successor on all four
existing clean gen7--gen10 replay corpora. Test whether this recovers older
policy/value targets without sacrificing the newest corpus, then test whether
that retention produces gameplay strength.

## Frozen inputs

- Completed 105 result SHA-256:
  `71c2cb3564468059b7d82906d5e18f6fd8f4c144475c4e353dcc66156616a401`.
- Completed 106 result SHA-256:
  `7e4bb4c7d53f13bcf29258f5856ab0b5d6fbe56844c6c1d9d742426e780c0ca1`.
- Exact gen11 checkpoint SHA-256:
  `c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0`.
- Clean replay SHA-256 values:
  - gen7: `06ee156543da64de3126eed00cb363ed5f2d4c0e49e093550024fffc2ace5a1c`;
  - gen8: `96f0d346743f30bd29e329f4bf3e2a7a361e778391c6d6e5833ca2a06af9be10`;
  - gen9: `1b13ef038ccc6b3f695d7c7860a2284272ae18c8e75d44028e4cbb6f69914f0c`;
  - gen10: `09279d9593acfd944a32571716a6527c0be40e41055b6fa1f5bc25c88f43a60c`.
- Raw legal `035` checkpoint SHA-256:
  `4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`.
- Retained installed gen2 SHA-256:
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`.

The runner verifies all hashes, the exact 105 and 106 controlling results,
gen11 lineage, clean replay identity, and retained installation before work.

## Stage 1: consolidated fit and held-out retention gate

Train exact gen11 once on the complete four-corpus window `[gen7, gen8, gen9,
gen10]`; do not generate or resample self-play. The child is labelled gen12
only because the checkpoint format requires monotone lineage. It is the 107
consolidation candidate, not evidence for an ordinary gen11-to-gen12 loop.

Keep the proven training recipe unchanged: AdamW, 12 epoch ceiling, batch 256,
learning rate `1e-4`, weight decay `1e-4`, whole-game 20% validation split,
policy/value weights `1.0/0.5`, patience 3, minimum improvement `0.001`, and
bucketed batches. Only the replay window changes. The split seed is `1070107`.

The validation split is made across all 1,024 games before fitting. Using the
exact held-out rows never shown to the optimizer, compare candidate and parent
on identical states and targets. Retention passes only if all are true:

1. aggregate held-out objective delta, candidate minus gen11, is below zero;
2. gen7 held-out objective delta is below zero;
3. gen8 held-out objective delta is below zero;
4. gen10 held-out objective delta is at most `+0.005`.

Every corpus must contribute at least one held-out game and row. Gen9 is
reported but is not an independent binary constraint. A failed retention gate
stops 107 before gameplay.

## Stage 2: fresh internal gameplay

If retention passes, use fresh seeds `1950000..1950099`, both seats, for:

1. raw consolidated candidate versus raw exact gen11, 200 games;
2. searched consolidated candidate versus searched exact gen11, 200 games.

Search uses 256 simulations, `c_puct=2.25`, real mode, and no root noise. Both
panels run. Each reports W-D-L, share, margin, seats, terminations, and a
50,000-sample paired bootstrap. The searched panel controls continuation and
passes only at >=55% share, positive margin, both seats >=50%, and all normal
terminations.

## Stage 3: conditional external thermometer

Only an internal searched pass opens fresh seeds `1960000..1960099`, both
seats, for:

1. searched consolidated candidate versus raw legal `035`, 200 games;
2. raw consolidated candidate versus raw legal `035`, 200 games.

Compare each result with the corresponding exact gen11 106 anchor by an
independent seed-pair bootstrap. Also report candidate search lift using the
same-seed external panels. `035` supplies no training target, prior, action,
replay row, selection signal, or continuation policy.

## Scope and interpretation

- No new self-play and no architecture, search, loss, or optimizer change.
- Training and candidate selection use no `035` information.
- Every arena commits after each game and resumes only an exact schedule
  prefix. Training is hash-sealed after a valid checkpoint/metrics round trip.
- No checkpoint installation or promotion is authorized under any outcome.
- A pass would support broader replay retention as the next loop change. A
  retention pass without internal gameplay gain would show that offline target
  recovery is insufficient. A retention failure would reject this exact
  consolidation dose before spending on arenas.
