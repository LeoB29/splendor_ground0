# AlphaZero Hybrid Transfer Test 109

Date frozen: 2026-08-20

## Purpose

Experiment 108 found a parent-safe coherent interpolation that improved every
held-out replay objective but remained flat against gen11 in both raw and
searched gameplay. That test used only 25% of the 107 update. The final
unresolved consolidation hypothesis is that the full 107 policy update is
useful when its demonstrably harmful value update is removed.

Experiment 109 is the decisive transfer test for that hypothesis. It uses the
full 107 network for policy and exact gen11 for value during search. It does not
train, generate replay, consult `035`, install, or promote anything.

## Frozen inputs

- Completed 108 result SHA-256:
  `57392ef95e50b54ad239b870213240e37935a1a5a037cacb0442cf42f344aca4`.
- Exact gen11 checkpoint SHA-256:
  `c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0`.
- Exact full 107 consolidation checkpoint SHA-256:
  `c21c0ed3765e09820eb415c7369d0be0733913915e29d288db825cbcf9461c89`.
- Retained installed gen2 SHA-256:
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`.

Before any game, verify all hashes; exact gen11/gen12 compatible lineage; the
exact 108 lambda-0.25 selection and 98-1-101 / 98-5-97 gameplay failure; the
hybrid's exact offline retention pass; and retained gen2 installation.

## Fixed fresh panels

On fresh seeds `1980000..1980099`, both seats, run both panels regardless of
the raw outcome:

1. raw full-107 policy versus raw exact gen11, 200 games;
2. searched hybrid (full-107 policy, exact gen11 value) versus searched exact
   gen11, 200 games.

The hybrid uses separate immutable inference networks: all policy logits come
from the full 107 checkpoint and all leaf/root values come from exact gen11.
Search is fixed at 256 simulations, `c_puct=2.25`, real mode, and no root noise.
Each panel reports W-D-L, score share, mean margin, seat shares, termination
reasons, and a 50,000-sample paired-seed bootstrap. Arenas commit after every
game and resume only an exact schedule prefix.

## Comparisons and decision

Using disjoint-seed independent paired bootstraps, report:

- raw full-107 policy minus raw lambda-0.25 versus gen11;
- searched hybrid minus searched lambda-0.25 versus gen11.

These differences are mechanism descriptions, not extra candidate-selection
gates. The searched hybrid is the sole primary endpoint and passes only at
>=55% share, positive margin, both seats >=50%, and all normal terminations.

- A pass supports the value-blocker hypothesis and authorizes design of a
  coherent parent-value-anchored training experiment.
- A failure closes the replay-consolidation/interpolation branch. Do not test
  more interpolation weights or loss-only variants; the next substantive
  direction is a higher dose of fresh gen11 self-play.

No outcome authorizes installation, promotion, `035`, or automatic training.
