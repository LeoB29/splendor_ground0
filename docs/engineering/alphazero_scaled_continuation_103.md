# AlphaZero Scaled Continuation 103: Generation 5 to Generation 8

Date frozen: 2026-08-15

## Decision being tested

Experiment 102 established that the clean generation-2 to generation-5
AlphaZero lineage compounded strength. The owner chose to spend the next
compute budget advancing that lineage rather than benchmarking generation 5
against `035` early. Experiment 103 holds the successful recipe fixed for
three more links, evaluates generation 8 against generation 5, and opens a
generation-8-versus-`035` benchmark only if the internal endpoint compounds.

## Frozen inputs

- Installed generation 2 checkpoint:
  `outputs/alphazero_pivot_090/pilot/generation_002.pt`, SHA-256
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`.
- Sandbox generation 5 checkpoint:
  `outputs/alphazero_clean_continuation_101/checkpoints/generation_005.pt`,
  SHA-256
  `201e34128c9588722ce78edc85e8672675b1075dc0696651484718ddd7fa2444`.
- Clean generation 4 replay:
  `outputs/alphazero_clean_continuation_101/selfplay/generation_004_sanitized.jsonl`,
  SHA-256
  `7e15847d2cdf51ecd467a2bb6f38f13e1901bd7337971bbe8f4817ace1dec244`.
- Completed 101 result SHA-256:
  `4e29b92f13232f31f082fd0a3c5eb00bf2bd1068a18956832b048cfb7bb04a05`.
- Completed 102 result SHA-256:
  `82e515abb4f1edbf2333f0cae93b02388db96b58d10defb7b92195810ce6b3be`.
- Raw legal `035` checkpoint SHA-256:
  `4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`.

All hashes and the controlling 101/102 decisions must pass before computation.

## Fixed continuation

Run exactly three links: gen5 -> gen6 -> gen7 -> gen8.

For each parent generation:

- generate 256 fresh self-play games using 256 simulations, `c_puct=2.25`,
  real mode, and 25% root noise;
- retain only normally completed games and replace abnormal games outcome-blind
  with consecutive post-schedule seeds, up to 16 attempts;
- train the child on the two most recent clean parent-generated corpora;
- use the established 12-epoch maximum, batch size 256, learning rate `1e-4`,
  weight decay `1e-4`, policy/value weights `1.0/0.5`, whole-game 20%
  validation, bucketed batches, patience 3, and minimum delta `0.001`;
- require finite metrics, the correct parent/child checkpoint lineage, a
  nonempty validation set, and at least one epoch;
- run the unchanged three 8-pair diagnostics (raw child/parent, searched
  child/parent, and child search/raw). These are health observations only and
  never gate advancement.

Self-play seed bands are `1820000..1820255`, `1830000..1830255`, and
`1840000..1840255` for generations 5, 6, and 7. Replacement seeds begin at
the corresponding band plus 256. The first replay window is clean gen4 plus
clean gen5; it then rolls normally.

## Powered generation-8 endpoint

After valid generation-8 training, run three 100-pair / 200-game panels on the
same fresh seeds `1850000..1850099`:

1. searched gen8 versus searched gen5 (primary);
2. raw gen8 versus raw gen5;
3. gen8 search versus gen8 raw.

Search uses 256 simulations, `c_puct=2.25`, real mode, and no root noise. Each
panel reports score share, mean margin, seat shares, normal terminations, and a
50,000-sample paired bootstrap CI95. The practical gate is >=55% share,
positive margin, both seats >=50%, and all games normally completed. The CI is
reported but is not a binary engineering gate.

## Conditional `035` benchmark

Only if the primary searched gen8/gen5 endpoint passes its practical gate, run
on fresh seeds `1860000..1860099`:

1. searched gen8 versus raw legal `035`, 100 pairs / 200 games;
2. raw gen8 versus raw legal `035`, the same 100 pairs / 200 games.

This estimates absolute strength. It does not promote or install generation 8.
If the internal endpoint does not pass, both `035` panels remain unopened.

## Resume and integrity

- Self-play replay/progress is appended once per completed game and accepts
  only the exact frozen seed prefix on resume.
- Arena progress is atomic once per game and accepts only the exact seed-seat
  prefix on resume.
- Each completed sanitation, training, and generation stage receives a
  hash-locked manifest. A restart verifies and skips those stages.
- A crash inside training may repeat only that incomplete training stage; it
  cannot change the frozen inputs or advance from an unsealed child.
- No partial outcome changes the schedule, training recipe, or conditional
  rule.

## Scope

- No checkpoint is installed, promoted, or substituted automatically.
- Exact generation 2 remains installed throughout.
- `035` is never a teacher, prior, encoder, target, candidate generator, or
  continuation policy.
- The research-champion resolver is not opened in this experiment.

