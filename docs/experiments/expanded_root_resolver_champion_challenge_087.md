# Experiment 087: Expanded-Root Resolver Champion Challenge

**Preregistered:** 2026-08-08, before implementing `087`, running its
deterministic smoke, or observing any `087` gameplay outcome.

## Question and authorization

Fresh offline replication `086` passed every primary, argmax-quality, phase,
privacy, determinism, and operational gate. The exact four-discovery-
continuation expanded selector improved over the current four-root resolver by
+0.051199 on 300 untouched states, CI95 [+0.042185,+0.060546], and changed the
selected action in 67.33% of states. Its result authorizes one separately
preregistered 200-pair gameplay panel against the current research champion.

`087` asks one question:

> Does the exact expanded-root resolver validated by `086` defeat the frozen
> `050`/`051` four-root public-belief resolver in fresh paired gameplay strongly
> enough to become the research champion?

This is a gameplay promotion experiment, not further search design. No root,
budget, utility, tie, fallback, or threshold may be tuned from `085`, `086`,
the smoke, progress, or interim outcomes.

## Frozen candidate and comparator

The candidate wraps frozen raw legal checkpoint `035` and is exactly the
selector validated by `086`:

- rank distinct canonical legal roots by `035`;
- retain the first four roots required by the champion contract;
- additionally retain every lower-ranked, individually structurally stable
  non-`RESERVE_DECK` public root;
- use three observer-relative determinizations, the top four distinct canonical
  `035` opponent responses, and four stochastic-`035` discovery continuations
  for every retained leaf;
- resolve with the same two-ply information-set aggregation, utility, and
  canonical tie rule as `050`/`051`; and
- fall back to raw `035` for the turn under the exact champion structural
  failures. Instability of a lower-ranked optional root excludes only that
  root, as in `085`/`086`.

Lower-ranked `RESERVE_DECK` roots remain excluded because their hidden draw is
not a stable public target. A `RESERVE_DECK` root already present in the frozen
first four remains part of the champion prefix. Candidate selection sees only
the four discovery continuations; no `086` evaluation continuation or outcome
is available in gameplay.

The comparator is the exact serial `050`/`051` research champion: the first
four distinct canonical `035` roots, three determinizations, four opponent
responses, and four discovery continuations per retained leaf, for exactly 192
continuations per completed search. Both bots are constructed fresh for every
game and use the same frozen checkpoint:

`outputs/legal_action_scorer_035_legal_observation/`
`legal_action_attention_scorer_best.pt`, SHA-256
`4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`.

Required provenance:

- research-champion confirmation `084` result SHA-256
  `1ed98024072b92baf3c30bc896b9dae2484f7d80b727959307dfd83bbfe2cd0f`;
- authorizing offline replication `086` result SHA-256
  `8ae985f6747b5f3bbbcf4997357daf299ef0f46cc23534d55b6de91367910eb5`.

No human replay, hidden-state feature, training, outcome-conditioned routing,
within-decision parallel approximation, GPU work, or corpus generation is
permitted.

## Matched namespaces and deterministic smoke

Use engine seed `870000` for a deterministic integrity smoke. Run one paired
seed and an independent from-zero repeat. In each pair the candidate plays once
from each seat against the champion. Excluding wall-clock timing, the two
projected pairs must be byte-identical. All four games must terminate cleanly;
candidate and champion branch accounting must be exact; all continuations must
resolve; hard caps and rollout fallbacks must be zero; and loop adjudications
must be at most 0.1%. Both resolvers must complete at least one search. Smoke
strength is neither summarized nor gated.

Freeze belief root-index start `870000000` and terminal continuation seed start
`8700000000000000000`. For both candidate and champion:

```text
belief root index = 870000000
  + (engine_seed-870000)*1000 + actor_seat*400 + decision_index

continuation seed = 8700000000000000000
  + (engine_seed-870000)*1000000000
  + actor_seat*400000000 + decision_index*1000000
  + determinization*10000 + continuation
```

Thus common champion-prefix leaves use matched randomness across the candidate
and comparator at the same paired seat/decision coordinate. Namespaces are
fresh and disjoint from all prior evidence. PyTorch uses one thread in each
outer worker. Resolver continuation work is serial within a game; outer
parallelism is ten paired-seed workers.

The strength block is forbidden unless the complete smoke passes.

## Fixed fresh paired strength block

Run exactly 200 paired engine seeds `870100--870299`, producing 400 games. For
each seed the expanded resolver plays once from seat 0 and once from seat 1
against the current champion. The paired engine seed is the independent unit.
Do not inspect interim outcomes, stop early, replace seeds, add games, or pool
with `050`, `051`, `084`, `085`, or `086`. An identity-matched progress file is
permitted solely for crash recovery.

After all 200 pairs finish, compute paired score share and final-score margin
with the established 20,000-sample paired bootstrap using seed `870100`.
Seat-specific score shares are mandatory safeguards.

The causal intervention dose is candidate completed searches where its
expanded resolution chooses a different canonical action than the champion
resolution of the same top-four prefix and same leaf table. Report candidate
searches, roots considered, optional roots retained/excluded, expansion
switches, raw-parent switches, and all telemetry by early (`0--19`), mid
(`20--49`), and late (`50+`) phase.

## Frozen gates

All gates are conjunctive:

1. Protocol, implementation configuration, checkpoint, `084`, and `086` hashes
   match, and the exact candidate and comparator contracts reproduce.
2. Exactly 400 strength games terminate through ordinary `completed` or the
   corrected two-sole-`PASS` `forced_pass_deadlock`, with zero timeout, stall,
   invalid cutoff, illegal action, or unexpected resolver failure.
3. Candidate continuations equal the sum over its completed searches of
   `retained_roots*3*4*4`; champion continuations equal completed searches
   times 192. Every continuation resolves, hard caps and rollout fallback
   triggers are zero, and loop adjudications are at most 0.1% jointly and for
   each resolver.
4. The candidate retains at least one optional expanded root and changes the
   champion-prefix action at least once in every phase. Its total expansion
   switch rate among completed candidate searches is at least 15%.
5. Candidate paired score share is at least 55%, and its paired-bootstrap CI95
   lower bound is strictly above 50%.
6. Mean paired candidate final-score margin is strictly positive.
7. Candidate score share from each seat is at least 50%.

## Effect size, power, and compute

As required by `086`, declare 56.5% candidate score share as the expected
alternative and retain the established 55% promotion threshold. With 400
games, an intentionally conservative independent-game normal approximation
gives a standard error near 2.48 percentage points and about 73% probability
of clearing the 55% point gate at a true 56.5% share; the lower-CI-above-50%
gate is of similar order. The paired bootstrap is the actual analysis. This is
moderate rather than high power, but the authorizing protocol fixes 200 pairs;
there is no optional extension after seeing results.

Expected work is approximately 40--60 wall hours on ten CPU workers because
both players resolve and the candidate considers a variable expanded root
set. Freeze a 72-active-wall-hour operational ceiling for the strength block.
There is no retry budget except resuming the same identity-matched progress
artifact after interruption.

## Fixed decisions and falsification

- Smoke or operational-integrity failure:
  `expanded_root_gameplay_087_invalid_no_strength_decision`.
- Valid strength block failing any effect, dose, phase, margin, or seat gate:
  `retain_050_051_research_champion_close_exact_expansion`.
- Every frozen gate passes:
  `promote_expanded_root_public_belief_resolver_as_research_champion`.

On promotion, raw legal `035` remains the fast champion. Failure leaves the
exact `050`/`051` resolver as research champion. No outcome authorizes training,
corpus generation, threshold relaxation, extra games, or a post-hoc pass.

## Evidence

Write smoke, identity-matched strength progress, strength result, final result,
and the pre-evidence manifest under
`outputs/expanded_root_resolver_champion_challenge_087`. Record protocol,
implementation, tests, wrapper, checkpoint, `084`, `086`, smoke, progress,
strength, and final-result SHA-256 hashes as they become available.
