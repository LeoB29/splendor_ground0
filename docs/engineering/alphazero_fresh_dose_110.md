# AlphaZero Fresh Gen11 Dose Experiment 110

Date frozen: 2026-08-21

## Purpose

Experiments 107--109 closed the stale replay-consolidation branch. Offline
policy/value improvements from old replay did not produce meaningful gameplay
strength. Experiment 110 tests the next distinct hypothesis: exact gen11's
healthy search teacher needs substantially more fresh, on-policy games for its
policy and sparse terminal value targets to generalize.

This is one new successor experiment, not another three-generation block.

## Frozen inputs

- Completed 109 result SHA-256:
  `ba4318190b84775a097fbc3a656b823447f05edd45618fde5be6800e3d0e81a1`.
- Exact gen11 checkpoint SHA-256:
  `c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0`.
- Retained installed gen2 SHA-256:
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`.

Verify all hashes, the exact 109 branch-closing result, exact gen11 lineage,
and retained gen2 before generating a game.

## Stage 1: 1,024 fresh games in four deterministic shards

Run four independent 256-game shards from exact gen11. Shard seed starts are
`2010000`, `2020000`, `2030000`, and `2040000`; the gaps reserve collision-free
replacement bands. Every game uses the confirmed 256-simulation,
`c_puct=2.25`, real-mode search with 25% root noise and search seed `1100110`.
Maximum turns remain 200.

Run at most four shard workers concurrently. Each worker and every inherited
BLAS/OpenMP runtime is fixed to one thread. Games are separately sealed, so a
crash loses at most the active game and completed seed files resume exactly.

Sanitation is outcome-blind. Keep only normal `completed` games. Within each
shard, use consecutive seeds beginning at original seed start + 256 until
exactly 256 clean games exist, with at most 32 replacement attempts. Hash-seal
each clean shard and verify unique seeds, contiguous turns, exact gen11 replay
identity, complete targets, and exact search dose.

## Stage 2: prospective data partition

Within every clean shard, order the accepted games by their original/replacement
schedule:

- the last 32 become a common untouched holdout: 128 total games;
- the first 64 form the low-dose pool: 256 total games;
- all except the holdout form the high-dose pool: 896 total games.

The low-dose pool is therefore a strict subset of the high-dose pool. Hash-seal
all three JSONL files. Neither candidate may train on the common holdout. No
gen7--gen10 replay is used.

## Stage 3: controlled parent-safe fitting

Train two generation-12 sandbox candidates from exact gen11:

1. low dose on the 256-game pool;
2. high dose on the 896-game pool.

Use the proven architecture and objective unchanged: AdamW, 12-epoch ceiling,
batch 256, learning rate `1e-4`, weight decay `1e-4`, whole-game 20% internal
validation, policy/value weights `1.0/0.5`, patience 3, minimum improvement
`0.001`, bucketed batches, seed `1100110`, and no corrective auxiliary loss.

The only training safeguard added is parent-safe selection: exact epoch-zero
gen11 is eligible to win validation selection. If no epoch improves its
internal validation objective by at least `0.001`, save the unchanged parent
weights and mark that arm unqualified. Indexed JSONL loading may be used to
avoid materializing the full corpus; it must preserve the existing deterministic
whole-game split and bucket/batch order.

## Stage 4: untouched common-holdout gate

Evaluate exact gen11, low-dose, and high-dose checkpoints on identical rows
from the 128 common holdout games. Report policy cross-entropy, value MSE,
objective `policy + 0.5*value`, top-action agreement, and policy drift.

An arm qualifies for gameplay only if:

- a nonzero training epoch beat epoch-zero selection;
- common-holdout objective is strictly lower than gen11;
- all checkpoint/training metrics are finite and lineage-valid.

The gate is an engineering filter, not a strength conclusion.

## Stage 5: fresh internal gameplay

For every qualified arm, run on seeds `2050000..2050099`, both seats:

- raw candidate versus raw exact gen11, 200 games;
- searched candidate versus searched exact gen11, 200 games.

Search uses 256 simulations, `c_puct=2.25`, real mode, and no root noise. Raw
panels are cheap and run first. Qualified searched panels may run concurrently,
one thread each, and resume by exact game prefix. Each panel reports W-D-L,
share, margin, both seats, terminations, and a 50,000-sample paired bootstrap.

If both arms qualify, report the same-seed paired difference between their
raw panels and between their searched panels. The high-dose primary passes only
at >=55% share, positive margin, both seats >=50%, and all normal terminations.

## Decisions and scope

- High-dose pass with a higher point score than low-dose supports continuing
  the AlphaZero loop at the higher fresh-data dose.
- High-dose pass without outperforming low-dose establishes fresh-data
  progress but not a dose-specific advantage.
- High-dose failure closes data-dose escalation at this scale and makes model/
  value capacity the next justified direction.

No outcome authorizes `035`, installation, promotion, or another automatic
generation. Exact gen2 remains installed.
