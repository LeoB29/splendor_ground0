# Information-Set Search Mechanism Review After Experiment 133

Date recorded: 2026-08-30

## Decision in one sentence

Do not continue ordinary public-belief PUCT and do not make a neural leaf value
the next dependency. First settle whether the already validated exact
expanded-root `086` resolver is stronger in full games than the `084` research
champion; if it is, preserve its grouped information-set solve and reduce its
cost with structured batch racing.

## What experiment 133 established

The result is valid. Search completed twice on all 36 balanced states, all
110,592 continuations resolved, privacy and semantic-action invariants held,
and the exact repeat matched. The search was not locked to `035`: it moved an
average TV distance of `0.30367` from the prior and changed the prior's top
action on `80.56%` of states.

The decisions were nevertheless poor relative to the frozen `086` expanded-
root oracle. Exact agreement was `41.67%`, mean oracle regret was `0.09244`,
and mean gain over the four-root champion was `-0.02087`, with negative early
and middle means. This reproduces the central failure mode of experiment `060`
at a larger budget and with the root-coverage bottleneck removed: ordinary
visit search can look distributionally stable without producing trustworthy
root values.

## What failed mechanically

The crucial distinction is not simply terminal rollout versus learned leaf.
Both `086` and `133` used the same `035`-based stochastic continuation policy
and score-margin utility.

`086` evaluated a complete Cartesian block for each root:

`root x belief particle x legal opponent response x continuation replicate`

It then chose one shared minimizing opponent action for each identical
opponent information set and averaged that resolved value across belief
particles. Its deployable discovery decision used four continuations for every
leaf. The other twelve continuations per leaf in the audit were independent
evaluation evidence, not part of the deployed decision rule.

`133` instead backed up one terminal sample at a time through ordinary PUCT.
Opponent PUCT changed which responses were sampled, and root Q became the mean
of that evolving sample mixture. This does not preserve the exact grouped
minimization used by the successful resolver. Forced coverage consumed an
average `64.32%` of the budget, after which only the remainder was adaptive.

A post-hoc descriptive check rules out the final visit choice as the sole
problem. Across the 72 search arms, visit argmax equalled Q argmax only
`44.44%` of the time, but Q argmax itself had mean oracle regret `0.09354` and
mean champion-relative gain `-0.02197`. Those are essentially unchanged from
the actual selected action (`0.09244`, `-0.02087`). The unsafe value aggregation
is upstream of the final selector.

Experiments `058`--`060` also warn against a local PUCT repair. Early values
are dominated by belief-batch variation; `12x1` belief breadth helped but did
not make hard choices stable; and the first public-belief visit search already
failed its Q/equivalence gates. More simulations or another PUCT constant
would therefore repeat a rejected family, not test a new scientific idea.

## Why the exact expanded resolver comes next

The project already has a stronger candidate whose scientific status is
unfinished:

- `084` proved that the exact four-root resolver is a real playing-strength
  mechanism: 303-96-1 against raw `035` over 400 games.
- `086` prospectively replicated expanded-root offline headroom on 300 fresh
  states: mean gain `+0.051199`, CI95 `[+0.042185,+0.060546]`, positive in every
  phase, mean regret `0.01673`, and all reliability gates passed.
- `087` did not produce a valid strength answer. Its formal comparison stopped
  on an over-strict smoke condition, and the later partial owner-waived panel
  remains outcome-sealed. It must not be inspected, completed, or pooled.

Until `086` receives a fresh prospective full-game comparison against `084`,
we do not know which planner should define the strength frontier or supply
future learning targets. That question has higher information value than
training a leaf model against an unsettled planner.

## Recommended sequence

### Next experiment: fresh `086` versus `084` gameplay

Freeze a completely new seed namespace and run exact expanded-root `086`
against exact research-champion `084`, balanced across seats. Reuse neither
formal nor owner-waived `087` game outcomes. Require normal completion,
bounded loop handling, material intervention dose, positive score margin,
both-seat safety, a practically meaningful score-share gain, and a paired
confidence interval excluding equality. This is an expensive oracle-calibration
experiment, not a deployment claim and not training.

### If the expanded resolver wins

Treat it as the new search frontier. Build a grouped progressive resolver, not
ordinary MCTS:

1. allocate complete information-set blocks, never isolated rollouts;
2. recompute the exact shared opponent minimum after each block;
3. retain near-equivalent roots under confidence bounds;
4. eliminate roots only when they are safely dominated;
5. spend extra belief blocks early and fewer late;
6. compare strength as a function of wall time against the unchanged exact
   expanded resolver.

This is a best-arm-identification or racing problem over resolved root values.
It preserves the mechanism that has evidence while attacking its cost. Once
that planner is stable and economical, learned high-recall action proposals are
the first neural addition; calibrated leaf value and uncertainty follow.

### If the expanded resolver does not win

Conclude that its local state-panel gain does not compound through play. Do not
distill it. The next genuinely different planner should add horizon while
preserving information-set grouping: depth-limited continual resolving with a
conservative leaf interval, initially using real continuation samples to
validate the leaf contract. A learned leaf model would enter only after that
target is shown to predict the planner's decision-relevant ordering.

## Explicitly rejected immediate moves

- no larger or retuned version of `133`/`060` PUCT;
- no Q-argmax-only patch;
- no new compact AlphaZero value-head or target experiment;
- no training from `133` visits or Q values;
- no inspection or reuse of partial `087` outcomes;
- no claim that information-set search failed as a whole.

## Owner decision required

The next recommended evidence is a fresh, long full-game comparison. The prior
`087` smoke recorded 248,160 candidate continuations versus 24,192 champion
continuations in four games, roughly a 10.3x continuation ratio. Its frozen
engineering estimate was 40--60 wall hours on ten CPU workers, with a 72-hour
active-compute ceiling. That remains the best estimate for an exact rerun.

The proposed fresh experiment uses 200 wholly new paired seeds (400 games),
the exact unchanged `086` candidate and `084` comparator, ten workers, and an
identity-matched crash-resume file. The strength gates should remain the old
prospective standards: at least 55% candidate score share, paired-bootstrap
CI95 lower bound above 50%, positive mean margin, at least 50% from each seat,
at least 15% expansion switches with every phase covered, and complete branch
accounting. The smoke must use the established bounded loop-adjudication rule
rather than `087`'s erroneous zero-fallback-trigger rule. No old `087` outcome
is read or pooled.

Freezing and launching this 40--60-hour comparison requires explicit owner
approval. No launch is automatic.
