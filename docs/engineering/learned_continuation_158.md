# Experiment158: learned continuations in155

2026-09-13. The owner approved implementation and launch of the single experiment
recommended in `docs/strategy/next_after_157.md`. Preserve all intentionally dirty
work and frozen ancestors. This document, the seven-file implementation inventory,
the configuration, checkpoint identities and schedules are frozen before any
off-panel preflight outcome. No training occurs.

## Question and intervention

Does the useful fast policy learned in157 make155 stronger when it plays the
simulated future moves? Candidate and reference both use the unchanged155 real
root operator: maximum four proposals (exact035 best, second distinct canonical
group, two randomly proposed remaining groups; smaller supports retained),32
shared observer-legal worlds, terminal win/draw/loss utility and first-proposal
ties. Reference roots call the frozen155 implementation directly. The candidate
changes only the action selector after the forced root move in each future.

In every candidate continuation, both simulated actors use the exact157
two-network controller. Frozen035 proposes the original candidates. The selected
learned scorer sees **all concrete legal actions** and scores each offered
canonical group by its maximum concrete logit. The first maximizing original
proposal is played, retaining its original concrete representative. No
unrestricted learned argmax, candidate-only network input, recursive search,
new value head, model mixture during a game, or fitted weights.

Use all three sealed157 treatment selections from `selected_lock.json`:

| Controller | Selected update | Checkpoint SHA256 |
| --- | --- | --- |
| treatment_157001 |1220|e2508f56d22e9b9ae2f87004dd9962718e5d3b4f6bd0824088d51f1d27f82b21|
| treatment_157002 |1220|adebd57d57ca53406f5aab01f69a2544885b657b225e88b8c6068e0f6d8ac0f4|
| treatment_157003 |1144|960a258c75d5854da64d5bca98bbfceb8c333bbd95ec845f9ae65206539f6ecb|

The assigned controller stays fixed in both seat games of its initial-seed pair,
and throughout every simulated future. No selection using157 arena ranking.

## Panel and random streams

Main:201 fresh initial-seed pairs, both candidate seats,402 games. Probe:6
off-panel pairs, both seats,12 games. Within each stage sort pair IDs by
`digest(["158-model-assignment-v1", stage, pair])`; rank modulo three selects
the model in the table order. Thus main has67 pairs/134 games per model and
probe has2 pairs/4 games per model. Individual fits use disjoint starting panels.
Pair order and assignment are independent of outcomes. Index is `2*pair+seat`.

Environment seeds: main1580000000+pair; probe1589000000+pair. The actual-root
stream is `digest(["158-root-stream-v1", stage, pair])`, shared between planners
and both paired games. The unchanged155 root scheduler combines this stream
with the acting observer's canonical lawful belief key. Its candidate namespace
remains155-candidate-v1 /155-live-v1; world namespace remains155-world-v1. Actual
hidden deck order and opposing blind identities do not enter those streams.

Each candidate future has an independently reproducible continuation stream:
`digest(["158-continuation-v1", root_stream, world_index, step, actor])`.
World index is0..31; step is the zero-based executed simulated
action index, including the forced root action at0. The first learned decision
is step1. The exact157 controller then incorporates its current acting player's
lawful belief hash through its existing155 scheduler. The forced root
alternative index is absent, coupling corresponding future slots across root
alternatives without candidate-order RNG consumption. The original root belief
is also absent: information lawful to the real root actor may be private from a
simulated opposing actor. Only the exogenous pair stream, slot/step/actor and the
current actor's lawful inner belief determine a continuation proposal. Reuse at
matching slots across different actual roots creates deliberate common random
numbers, without dependence on an ancestral private observation.

Both simulated actors use their own lawful observations. Own blind reserves
are known; opposing blind identities and hidden deck order are unknown. The
world law remains uniform catalog minus legally known cards. No history-based
posterior or omniscient hash is used to decide a proposal.

Unit fixtures use1589800000 and above with a separate158-test-root-stream-v1
namespace. A guarded `test` stage permits tiny constructed states under a
temporary output directory for actual process-spawn replay tests. It is excluded
from the frozen schedule and production lifecycle, which accept only probe/main.

## Endpoint and fixed decision

Candidate score is1 for a win,.5 for a draw and0 for a loss against reference155.
Average the two seats within a pair and67 complete pairs within each model,
then equally average the three models. The estimand is the uniform mixture of
these three fixed planners on new starting positions, conditional on these
checkpoints. It does not treat201 pairs as603 independent pairs or three fitted
models as a population sample of training algorithms.

Use50,000 stratified complete-pair bootstrap replicates with NumPy default_rng
seed1583000000. In chunks of1000, iterate the three models in table order and
draw a `(count,67)` index matrix per model, uniformly with replacement. Both
seat outcomes stay together. Pair units are `2*score_seat0+2*score_seat1`, an
integer0..4. Sum all resampled units across the three strata and divide once
by804 (=4*201), avoiding rounding artifacts at50%. Use linear empirical
quantiles at.025 and.975 for the95% interval. The point estimate uses exact
half-point sums over402 games, identical to the balanced stratum mean.

All of these gates must pass:

- Aggregate score at least55%, with integer half-point comparison.
- Paired95% interval lower bound strictly greater than50%.
- Positive aggregate mean final score margin.
- Aggregate score from each candidate seat at least50%.
- Complete integrity verification.

Per-model results are descriptive. No per-fit gate, selection or veto. All402
valid games and complete replay run on either strength outcome. An inconclusive
interval does not justify extending the panel. Probe outcomes do not filter
strength or change the configuration. There is no automatic promotion,
additional data generation, fitting, dose change or successor.

## Termination and durable evidence

Normal/equal-turn completion and two consecutive turns whose **sole legal
action is PASS** are valid. Forced-pass endings use score, then fewest purchased
cards, then draw. Fourfold repetition,60 no-progress turns, global turn400,
missing/illegal action, numeric failure or exception are failures. Preserve the
failed root and actual game; do not fabricate a score, exclude, replace or
automatically retry a failed game. Learned failed branches retain their action
trace and final snapshot, plus exception diagnostics when applicable. Reference
branches retain the unchanged155 evidence and can be reconstructed from the
saved root/world/action specification.

Every actual move saves its root artifact hash, action and full-state hash.
Each root saves its initial snapshot, lawful RNG schedule, raw035 scores,
proposals, all32 sampled-world identities and every forced-action outcome.
Normal candidate branches save the existing action trace hash, final-state hash,
ending, winner and turn count, plus learned decision count and a streamed
SHA256 of each exact157 evidence dictionary (full legal logits, proposals,
group scores, choice and privacy counts), bound with step, actor and stream.
Hash each event's canonical digest in order, omitting the forced root move from
the learned-decision stream. This permits full reconstruction without storing
hundreds of millions of per-move JSON objects.

The independent auditor rebuilds every actual game, root proposal and world,
then every continuation of both planners. Its action and termination loops are
separate from production. Candidate decisions reuse the frozen157 independent
decision implementation; reference decisions are independently reconstructed.
Every continuation decision checks observer-input invariance under hidden-deck
reordering and replacement of opposing blind reserves by lawful unknown cards.
Actual roots receive the same invariance checks. Worlds are independently
constructed using the frozen154 verifier. Both implementations necessarily
share the engine, canonical keys, serialization, lawful encoders, Torch model
and frozen scorer weights. The auditor independently reconstructs the complete
pair schedule, integer bootstrap, gates and timing/count summaries, comparing
observations/results. Immutable per-game replay certificates bind the complete
root trees. Full reconstruction remains mandatory even if the strength gates fail.

Full repository tests must pass with before/after identity unchanged. Explicit
seven-file158 inventory binds the helper, lifecycle, entry point, independent
verifier, launcher, protocol and tests. The parent check verifies frozen157 and
its selected checkpoint/test/result/audit identities, recursively using the
explicit156->155->154 chain; it does not invoke149's dynamic file inventory.
Mutable agent notes and unrelated dirty files are excluded. Immutable manifest,
schedule, freeze, launch, game, replay, result and final records are never
overwritten. Kernel locks prevent duplicate launch/collection/audit execution.

Mutable progress and per-game telemetry use only a narrowly allowlisted retry
writer: Windows sharing/access errors5/32 get up to50 retries at100ms; other
errors and immutable paths fail immediately. This never reruns a game or model
decision. Status readers must share read/write/delete access briefly or read
append-only logs/immutable artifacts. Spawned verifier tests must submit the
actual game-audit callable and fully replay tiny fixture games, detecting the
module-registration failure missed in157. No automatic recovery command exists.

## Execution and interpretation

Use8 game workers, one Torch thread per worker, deterministic algorithms,
serial continuations, no nested worker pools. The hidden durable launcher
verifies full-test evidence, freezes, runs12 preflight games and full replay,
writes a timing estimate, then runs402 main games, complete replay and report.
Stop after the final report. Record a failure immediately if collection/audit
fails, while allowing already-running workers to preserve their evidence.

The previous estimate was45-60 hours, provisionally two to three days. Every
inner privacy check adds audit work; the complete12-game preflight refines this
estimate. Report measured candidate/reference wall and CPU decision costs and
collection/replay durations. Keep32 worlds regardless of timing. The candidate
can consume more neural compute; this is not an equal-time comparison.

A pass establishes one productive learned-policy-to-stronger-search step for
the complete controller, including its stochastic proposal support. It does
not isolate stronger direct play, stochasticity or calibration as the unique
cause, or establish repeated self-improvement or superhuman performance. A
failure leaves157's verified fast-play gain intact and rejects only this fixed
continuation substitution at the prespecified screen. Further research needs a
new decision; none is launched automatically.
