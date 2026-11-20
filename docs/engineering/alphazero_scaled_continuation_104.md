# AlphaZero Scaled Continuation 104: Generation 8 to Generation 11

Date frozen: 2026-08-16

## Purpose

Experiments 102 and 103 produced two consecutive powered three-generation
searched endpoint gains: gen5/gen2 at 58.25% and gen8/gen5 at 59.0%. Experiment
104 tests whether that approximately stable compounding slope persists for a
third unchanged block and whether the gain transfers to the measured external
`035` matchup.

## Frozen inputs

- Installed generation 2 checkpoint SHA-256:
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`.
- Sandbox generation 8 checkpoint:
  `outputs/alphazero_scaled_continuation_103/checkpoints/generation_008.pt`,
  SHA-256
  `beea26beddf91c7b0ab8f531ba6bc2bfa104d4e635e12af8002d04f7fb9c9281`.
- Clean generation 7 replay:
  `outputs/alphazero_scaled_continuation_103/selfplay/generation_007_sanitized.jsonl`,
  SHA-256
  `06ee156543da64de3126eed00cb363ed5f2d4c0e49e093550024fffc2ace5a1c`.
- Completed 103 result SHA-256:
  `0152c113dee56bd156483b0ab6d3a55d48e35b70cd3bcc3f5f9c0b5c2c0f7f96`.
- Raw legal `035` checkpoint SHA-256:
  `4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`.

The runner must verify every hash, the final gen8 identity, the successful 103
internal endpoint, and retained generation-2 installation before computation.

## Fixed continuation

Run exactly gen8 -> gen9 -> gen10 -> gen11. Preserve the successful recipe:

- 256 fresh self-play games per parent;
- 256 simulations, `c_puct=2.25`, real mode, 25% self-play root noise;
- outcome-blind exclusion and replacement of abnormal games, maximum 16
  consecutive post-schedule attempts;
- rolling two-clean-corpus replay window, beginning with clean gen7 + gen8;
- 12 epochs maximum, batch 256, AdamW learning rate `1e-4`, weight decay
  `1e-4`, whole-game 20% validation, policy/value weights `1.0/0.5`, bucketed
  batches, patience 3, and minimum delta `0.001`;
- finite metrics, correct checkpoint lineage, nonempty validation, and at least
  one epoch;
- unchanged 8-pair raw transfer, searched transfer, and search-health panels as
  non-gating health observations.

Self-play seed bands are `1880000..1880255`, `1890000..1890255`, and
`1900000..1900255`; replacements begin at each band plus 256. Training/search
seeds continue the established sequence.

## Powered gen11 endpoint

On fresh seeds `1910000..1910099`, run three 100-pair / 200-game panels:

1. searched gen11 versus searched gen8 (primary);
2. raw gen11 versus raw gen8;
3. gen11 search versus gen11 raw.

Evaluation search uses 256 simulations, `c_puct=2.25`, real mode, and no root
noise. Report score share, mean margin, both seats, all termination reasons,
and a 50,000-sample paired bootstrap CI95. The practical gate remains >=55%,
positive margin, both seats >=50%, and all games normally completed. The CI is
reported but is not a binary engineering gate.

## Conditional external benchmark

Only if the primary searched gen11/gen8 panel passes, run on fresh seeds
`1920000..1920099`:

1. searched gen11 versus raw legal `035`, 200 games;
2. raw gen11 versus raw legal `035`, 200 games on the same seeds.

The benchmark measures whether internal compounding transfers externally. It
does not install or promote gen11. A mid-30% searched share is a rough planning
expectation, not a selection threshold.

## Resume, safety, and scope

- Self-play and arenas commit atomically after every completed game and resume
  only exact frozen schedule prefixes.
- Sanitation, training, and complete generations use hash-locked stage
  manifests; verified stages are skipped after restart.
- A crash within an unsealed training stage may repeat only that stage.
- Small diagnostic outcomes never stop or alter the lineage.
- The `035` condition depends only on the completed primary internal gate.
- No checkpoint is installed or promoted automatically; exact gen2 remains
  installed and the research champion remains unopened.
- Review 104 before any further block or architecture/capacity change.

