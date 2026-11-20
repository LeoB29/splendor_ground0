# AlphaZero Direct Value-Path Duel 112

Date frozen: 2026-08-25

## Purpose

Experiment 111 did not meet its strict harmful-head attribution rule, but it
showed an unexpectedly large directional effect: restoring generation-11 value
while retaining the high-dose policy reduced paired score share by 6.875
percentage points. Experiment 112 tests that clue directly and causally.

It compares two searched agents whose complete policy computation is identical.
The candidate uses the high-dose checkpoint for both policy and value; the
baseline uses the same high-dose policy computation but the exact generation-11
value computation. Each functional path includes its checkpoint's own trunk, so
this is a full value-path intervention, not a literal final-layer tensor swap.

## Frozen inputs

- completed 111 result SHA-256:
  `02fbd6cbf2a971c06adc435f1b537cc7aa24ac5d4251b7783efe8de943a7e2aa`;
- exact high-dose generation-12 checkpoint SHA-256:
  `a9553311c5e49df4c809ad9238e9f769aa2307f562816c78e0efcadeb282fa74`;
- exact generation-11 checkpoint SHA-256:
  `c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0`;
- retained installed generation-2 checkpoint SHA-256:
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`.

Before evidence, verify all hashes; the complete 111 decision and all three
frozen 111 gameplay panels; generation-11/high-dose lineage; selected high-dose
epoch 1 and parent-safe status; and the retained installation.

## Frozen arena

Use fresh game seeds `2070000..2070199`, both candidate seats, for 400 total
games. Both agents use 256 simulations, `c_puct=2.25`, real mode, zero root
noise, and deterministic operative search seed `2 * game_seed + seat`.

Run four concurrent one-thread shards of 50 paired seeds each:

1. `2070000..2070049`;
2. `2070050..2070099`;
3. `2070100..2070149`;
4. `2070150..2070199`.

Each shard writes a sidecar identity before its first game, binding the protocol,
candidate and baseline policy/value hashes, search configuration, exact seed
range, and maximum turns. Each shard resumes only by an exact completed-game
prefix. The sealed combined arena must validate all 400 unique scheduled rows,
then report W-D-L, score share, mean margin, both seats, termination reasons,
and a 50,000-sample paired-seed bootstrap.

The intervention changes the value computation on every nonterminal leaf
inference while holding the policy computation and all search settings fixed.
No action-level disagreement rate is a primary endpoint.

## Primary endpoint and decision

The primary endpoint is the full-high-dose candidate's combined score share.
It passes only if all of the established practical checks pass:

- score share at least 55%;
- positive mean score margin;
- score share at least 50% from each candidate seat;
- every game terminates normally.

The paired bootstrap is reported as uncertainty and does not replace the frozen
gate. Four hundred games give roughly 80% power for a true score share near 57%
under a simple independent-game approximation; the paired and conjunctive seat
checks make exact power lower.

If the gate passes, record causal evidence that the newer value path improves
search with policy held fixed, and review a separated or more conservatively
updated policy/value architecture. If it fails, do not authorize that redesign
from this evidence and move to broader representation/capacity or teacher
quality work.

This experiment performs no training or self-play, runs no `035` benchmark, and
cannot install or promote a checkpoint. Exact generation 2 remains installed.
