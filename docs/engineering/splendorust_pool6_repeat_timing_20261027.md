# Fixed repeat-timing diagnostic — 27 October 2026

Authorized scope: repeat the three saved slow positions and fixed controls with
both unchanged players. No new games, training, external opponents, promotion or
large-data work. This separate attempt preserves the verified negative live
admission and every earlier failure. Source is
`tools/splendorust_pool6_repeat_timing_20261027`; evidence is
`outputs/splendorust_pool6_repeat_timing_20261027_01`.

## Question and players

Do the observed delays recur on the same inputs, and how much of their duration
falls within the existing handler and response-emission boundaries? This is a
small diagnostic, not a live reliability or playing-strength assessment.

The incumbent is unchanged optimized64: original three-world pool, canonical
observer seed A, 64 total simulations. The candidate is the unchanged six-world
live player from the negative admission. Preserve profiles, weights, numerical
precision, graph backend, search settings, compound choices, late heuristics,
complete response payloads, callback evidence and the minimal timing observer.
The new worker facade changes only output/freeze ownership. Exact raw player
identities remain the original identities. No native recompilation.

## Fixed positions and order

Targets are `reserved_06`, `gold_returns`, and `reserved_00`, selected because of
the previous 737, 772 and 714 ms observations. Both players run all three.
Controls are the lexicographically first eligible reserved position in each
existing early (12/13), middle (36/37) and late (60/61) turn stratum, excluding
target IDs: `reserved_01`, `reserved_07`, `reserved_12`. Selection uses no latency,
action quality or outcome filter. Frozen observer, corrected snapshot and source
artifacts bind every request. No newly simulated trajectory is needed.

Exactly 12 rounds × 6 positions × 2 arms = 144 ordinary requests. Position order
rotates through six cyclic orders, followed by their reversed orders. Both arms
are consecutive within each position; frozen goes first when
`(round + position_index)` is even. Each root has each arm first six times. All
rounds count, including the first. No discarded warmups, replacement, extension
or retry until success.

Two resident workers run one request at a time on the original player CPU 0;
supervisors remain isolated on CPUs 2–19, excluding sibling CPU 1. Both are
started, graph-checked and corrected-entrypoint-qualified before the timed
matrix. Each worker builds its graph once. The inherited 52-decision suite and
11 graph cases per arm give 248 scientific decisions, 22 graph cases and 148
parent requests in total. Startup order is frozen then candidate. Resource
stage deadlines may be rebound only between stages with no request active;
individual decision clocks never reset.

## Clocks, checks and stopping

The hard complete-response deadline is 1.000 seconds; the internal guard stays
0.900 seconds. The prior 700 ms admission margin is descriptive in this diagnostic.
It remains the unchanged failed criterion of the previous attempt. Zero crossings
here would not retroactively pass admission or authorize games.

Stop on the first actual timeout, unexpected rejection, illegal/mismatching
result, integrity violation or resource failure. Preserve the failure and the
verified prefix; never restart a failed request or replace a worker. Report
partial coverage honestly. A timing failure here is not a game loss because
there are no games; the existing future benchmark timeout-as-loss rule remains.

All completed decisions must match the saved source and repeat exactly within
their player/position cell. Eleven cells have saved ordinary responses. Frozen
`gold_returns` uses the accepted actual-entrypoint fixture; that fixture alone
contains extra inference arrays absent from ordinary incumbent replies, so only
those capture-only arrays are removed from that historical comparison. Actions,
callback hashes, work counts and every common numerical field remain exact.
Across the new repeats, full scientific responses must match exactly, including
all candidate callback float32 bits. Retain raw responses before any comparison.

Call corrected runtime source verification once at each worker startup before
model loading. Run actual entrypoint counterexamples for corrected card costs,
all 90 native cards, nobles, terminal/compound choices, gold/returns, private
reserves, encoded legal support and invalid identity/legacy rejection. Preserve
the corrected catalogue, atomic prereveal turn contract and existing deadlock
conventions. Check independent observer and transition validity, raw transport,
trusted clocks, sidecars, graph parity, source hashes, placement, log compression,
worker closure, resource coverage, evidence seals and actual owner exit statuses.
Final independent review must accept the exact evidence before reporting success.

## Resource envelope

One non-resettable 3,600-second wall-clock cap: collection ends by 1,800 seconds;
qualification and diagnostic each have a 900-second stage cap; final verification
has up to 1,800 seconds. Allow 2 GiB total evidence, reserving 512 MiB for final
proofs. Require 12 GiB initial free disk and 5 GiB maintained. Require 2 GiB
available RAM stable for 60 seconds initially and between moves, with at most
600 cumulative seconds waiting; worker-tree aggregate RSS is capped at 2 GiB.
The waiting policy and clock accounting remain those of the accepted guardian.
At most two worker trees and two graph builds. Preserve original per-route raw
request, response and log caps, including startup suite exception. All three
owners must exit, and recorded owned process identities must be absent.

## Analysis and interpretation

For every position/player show all twelve receipt, handler and emission times,
median, nearest-rank p90, maximum, 700 ms crossings, actual failures, work counts
and response sizes. Also show same-round candidate-minus-incumbent differences.
No confidence interval treats repeated dependent requests as independent trials.
No outcome-based trimming, new model calls or hidden replay inference is allowed.

Repeated slow handlers would direct the next investigation toward shared
compute/waiting costs; a persistent candidate emission gap would identify a
separate transport/evidence cost. Non-recurrence would support variability in
this saved panel, not prove a host-load cause. Two resident/interleaved workers
differ from the earlier serial admission lifecycle. A clean small panel cannot
establish live-game tail reliability or stronger play. Recommend a separate next
action after full verification; do not launch it automatically.
