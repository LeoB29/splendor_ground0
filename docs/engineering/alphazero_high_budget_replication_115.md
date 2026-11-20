# AlphaZero High-Budget Distillation Replication 115

Date frozen: 2026-08-26

## Purpose

Experiment 114's same-architecture candidate passed untouched offline gates and
was flat at 256 simulations, but achieved a promising 55.75% practical result
against exact gen11 when both used 1,024 simulations. Its 200-game paired
interval still included 50%. Experiment 115 independently replicates that exact
high-budget contrast and prospectively pools the two panels.

This experiment performs no self-play or training. It does not alter either
checkpoint or any search setting.

## Frozen inputs

- completed 114 result SHA-256:
  `dd8bbaa7876b881db69d44fb4242624e1f227981b4628da392b698491c231ad5`;
- exact 114 candidate SHA-256:
  `93e548a35f2a42194c70602928fa1ae9398fe042dbad2b2c876694fa97a9c4bd`;
- exact generation-11 checkpoint SHA-256:
  `c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0`;
- retained installed generation-2 checkpoint SHA-256:
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`.

Verify all hashes; the exact completed 114 decision, offline pass, training
lineage, candidate hash, both gameplay panels, and retained installation; and
the full exact 114 diagnostic schedule on seeds `2140000..2140099` before new
evidence.

## Fresh replication

Run the exact 114 candidate against exact gen11 with both agents using 1,024
simulations, `c_puct=2.25`, real mode, zero root noise, and operative search seed
`2 * game_seed + seat`. Use fresh seeds `2150000..2150099`, both candidate
seats, for 200 games.

Run four concurrent one-thread shards of 25 paired seeds. Each shard writes a
sidecar before its first game binding protocol, both checkpoint hashes, both
search configurations, exact schedule, and maximum turns. Each resumes only by
an exact completed-game prefix. Combine only all 200 scheduled rows and report
W-D-L, share, margin, seats, terminations, and a 50,000-sample paired bootstrap.

## Prospectively pooled primary

After the fresh replication seals, combine its 100 paired seeds with the exact
100 paired seeds from 114's diagnostic, preserving every row and verifying all
400 unique scheduled games. Report the same metrics and a fresh 50,000-sample
paired bootstrap over all 200 seed clusters.

The pooled result confirms high-budget absorption only if score share is at
least 55%, mean margin is positive, both candidate seats are at least 50%, all
400 games terminate normally, and paired CI95 is strictly above 50%.

If it passes, review a short two-generation 1,024-simulation continuation block
using the 114 candidate as parent. If it fails, do not continue the small
network at high budget; proceed to the medium representation/capacity pilot.

No result authorizes automatic continuation, `035`, installation, or promotion.
Exact generation 2 remains installed. Expected wall time is 3.5--5 hours with
four workers.
