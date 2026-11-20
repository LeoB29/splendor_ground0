# AlphaZero Generation-11 External Benchmark 106

Date frozen: 2026-08-18

## Purpose

The prospectively pooled 400-game result in 105 confirmed a narrow searched
gen11/gen8 practical gain. Experiment 106 measures how much that internal gain
transfers to the raw legal `035` external baseline and compares it with the
exact gen8 anchor measured in 103.

## Frozen inputs

- Completed 105 result SHA-256:
  `71c2cb3564468059b7d82906d5e18f6fd8f4c144475c4e353dcc66156616a401`.
- Completed 103 result SHA-256:
  `0152c113dee56bd156483b0ab6d3a55d48e35b70cd3bcc3f5f9c0b5c2c0f7f96`.
- Exact gen11 checkpoint SHA-256:
  `c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0`.
- Raw legal `035` checkpoint SHA-256:
  `4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`.
- Retained installed gen2 SHA-256:
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`.

The runner verifies all hashes, exact gen11 identity, the exact 105 pooled pass,
the exact 103 gen8/`035` searched and raw anchors, and retained gen2
installation before any game.

## Fixed panels

On fresh seeds `1940000..1940099`, both seats, run:

1. searched gen11 versus raw legal `035`, 200 games;
2. raw gen11 versus raw legal `035`, 200 games on the same seeds.

Gen11 search uses 256 simulations, `c_puct=2.25`, real mode, and no root noise.
Each arena commits atomically after every game and resumes only an exact frozen
schedule prefix. Report W-D-L, score share, mean margin, both seats, all
termination reasons, and a 50,000-sample paired bootstrap CI95.

## Predeclared comparisons

- Compare searched gen11 with searched gen8's exact 53-0-147 / 26.5% `035`
  anchor using a 50,000-sample independent seed-pair bootstrap difference.
- Compare raw gen11 with raw gen8's exact 24-1-175 / 12.25% anchor the same way.
- Because the new searched and raw panels share seeds, report a paired
  seed-cluster bootstrap of gen11 search lift against `035`.

These comparisons report transfer and mechanism. They do not select a model.
Parity with `035` requires at least 50% share, positive margin, both seats at
least 50%, and all normal games. Below parity, a difference interval entirely
above zero confirms external progress; otherwise a positive point difference
is directional evidence only.

## Scope

- No self-play, training, checkpoint mutation, installation, or promotion.
- `035` is an external thermometer only; it supplies no prior, target, action,
  value, replay row, or continuation policy.
- Both panels run regardless of the searched outcome.
- Review the benchmark together with 105's replay-forgetting evidence before
  choosing broader replay retention, another scaling block, or capacity work.

