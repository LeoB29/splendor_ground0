# Useful parallel work during Experiment 162

2026-09-17. Design review only; no new experiment is authorized or launched by
this document. Experiment 162 and its frozen files remain unchanged. No live
162 teacher labels or gameplay outcomes were used for this review.

## Recommendation

Prepare a broader strength benchmark while 162 runs. The immediate question is
whether learning improves play against different opponents, beyond the closely
related players used in recent comparisons. This remains useful whether the
fresh-feedback experiment succeeds or fails. It is preparation for deciding the
next experiment, not a commitment to run this benchmark next.

A second useful deliverable is a design for batching teacher inference. It could
reduce the cost of a teacher already shown to help, without requiring a new
scientific hypothesis about how to choose moves. Implementation and profiling
should wait until they will not compete with the running experiment.

## Why this gap matters

Experiment 106 is a concrete warning: internal AlphaZero progress did not
establish improvement against the separate 035 benchmark; searched gen11 scored
26.5% and its raw policy 12.75%. Here “separate” means outside that AlphaZero
training family, not an independently authored external bot. See
`../engineering/alphazero_gen11_external_benchmark_106_result.md`.

Recent results establish useful components: 160 improved the teacher by spending
more computation, and 161 transferred some of that benefit into fast play.
Neither establishes broad strength, improvement at equal search time, or a
completed learning-to-stronger-search cycle. In particular, 158 did not establish
that replacing simulation continuations with the stronger fast student improves
planning. Win rates from different matchups cannot be multiplied or treated as
a reliable ranking of every player.

These distinctions follow the existing programme goals in
`superhuman_information_set_search_roadmap_after_132.md` and the evaluation
discussion in `whole_programme_direction_review_during_134.md`. The old search
architecture prescriptions must be read with the later stochastic-refill
correction; this note does not reinstate them.

## Draft economical benchmark

This is a proposed design, not a frozen protocol. Final identities, lawful agent
adapters, fresh seed namespaces, termination handling, costs and analysis must be
verified before any launch. Use two-player base Splendor, a 15-point target, and
the repository's validated rules and legal observations. Record exact hardware,
thread counts and per-move time distributions.

### Available opponents and their limits

| Opponent | Purpose | Evidence boundary |
|---|---|---|
| Unchanged raw 035 | Common programme reference | Related to the current students and their teacher; not an independent style |
| Information-legal shallow search from 051 | Handwritten search/evaluation contrast | Existing local opponent, not a verified strong external bot |
| Loop-safe 023 entity scorer from 051 | Different historical representation and policy | Another local learned opponent; strategic independence is limited |
| 016 and gen11 AlphaZero | Reserve diagnostic opponents | Available historical lineages; do not expand the panel merely to accumulate easy wins |
| Exact 084, 155 and 160 planners | Later research-strength comparisons | Expensive; need a separately budgeted study and direct comparisons |

For the initial three-opponent panel, preserve the 051 shallow configuration:
depth 1, maximum branching 6, buy/reserve/take caps 4/2/2, seed 23, one
determinization with seed 29. The exact 023 checkpoint and loop-safe contract are
recorded in `../experiments/broad_public_belief_resolver_confirmation_051.md`.
Reverify present implementations and checkpoint hashes rather than assuming the
old experiment's adapter remains interchangeable.

Use all three fixed 161 students, all three completed 162 refresh students, and
all three matched 162 replay controls. Retain every fit; do not choose the best
one from 162's arena. Including the replay controls distinguishes broader gains
from fresh feedback from gains obtainable through more old-data practice.

A concrete starting proposal is 100 fresh initial-seed clusters, both seats,
shared across all nine players and three opponents: 5,400 games. This is a
calibration of transfer across a small local panel, not a powered claim of tiny
gains or human superiority. Obtain an outcome-blind timing estimate before
committing; shallow search and full replay may materially affect runtime. Do not
infer an ETA from student decision latency alone.

Primary endpoint: change in equally weighted opponent-panel score from matched
161 to 162 refresh students, averaged over the three fixed fits. Score a win as
1, draw as 0.5 and loss as 0. Report the refresh-minus-replay comparison separately
to assess attribution. Bootstrap whole initial-seed clusters jointly, preserving
all models, opponents and both seats within each cluster. These intervals
condition on the chosen fits and opponents, not a population of all possible
training runs or adversaries.

Publish each opponent's result, each fit and each seat, as well as the aggregate,
latency and all integrity failures. A panel average can conceal a bad matchup;
it cannot establish improvement against every opponent. Freeze sample size and
analysis before outcomes. Do not add games to cross a threshold, silently
discard invalid endings or infer promotion from an exploratory panel score.

Keep subsequent confirmation seeds protected. Once these panel results guide
development, the panel is development evidence; new seeds alone do not erase
adaptation to the same opponents. A later confirmation needs protected opponent
diversity as well. This stage would not supersede or reinterpret 162's gates.

### The external-evidence gap

Local diversity is useful, but the eventual superhuman claim needs direct
evidence against independently developed strong bots and multiple competent
human players, at declared time and hardware budgets. The historical ten-game
owner calibrations are descriptive and do not establish that population claim.
Human replay material remains held out; no manual annotation task is proposed.

Two primary-source leads were checked without installing or executing code:

- [RinascimentoFramework](https://github.com/ivanbravi/RinascimentoFramework)
  is an independently developed Splendor research framework with an MIT license.
  It is a candidate for rules comparison and opponent compatibility work, not a
  verified strong opponent in our rules and compute conditions.
- [seal256/splendor](https://github.com/seal256/splendor) describes an MCTS/policy
  system, but its README explicitly reports experiments on a limited variant.
  Its existence does not establish a ready equivalent benchmark.

Before selecting any external opponent, establish exact card/rule sets, two-player
token counts, hidden reserves and deck information, refills, token returns,
equal-turn ending and tiebreaks, legal-action conversion, available weights and
permitted use, and reproducible time controls. A compatible implementation still
needs measured strength; a repository or paper title is not that evidence.

Population evaluation is motivated by the possibility of opponent-specific
strength and strategic cycles; it does not prove cycles in our current agents
or measure their exploitability. See the primary research paper
[Open-ended learning in symmetric zero-sum games](https://proceedings.mlr.press/v97/balduzzi19a.html).

## Interpret 162 before selecting the next expensive study

Use the completed, audited estimates and intervals, not interim outcomes or only
the classification label. “Transfer” means refresh beats the equal-dose replay
control under the frozen gate. “Progress” means its frozen comparison against
unchanged 161 passes. Their definitions remain in
`../engineering/student_refresh_162.md`.

| Valid final result | Interpretation and next decision |
|---|---|
| Both gates pass | One bounded refresh adds useful value and advances 161. Broader transfer becomes a strong candidate for the next check; scaling or replacing search continuations is not automatically justified. |
| Transfer passes, progress misses | Fresh examples help relative to the control, but required progress is unestablished. Check whether replay deteriorated and consider the effect sizes; do not extend the arena to secure a pass. |
| Progress passes, transfer misses | The update improves the anchor, but fresh-data superiority is unestablished. More old-data practice remains a competing explanation and may be cheaper. |
| Neither passes | Distinguish uncertain or small effects from evidence of harm. The tested mixture, sample size and fitting dose failed to establish the required benefit; iterative learning in general is not ruled out. |
| Integrity failure | Resolve validity while preserving the frozen evidence before making any strength claim. |

Even a complete success cannot distinguish student-state relevance from the
benefit of extra independent examples. Nor does it establish learned proposal
generation: the student still chooses within 035's top-two-plus-two-random
offered set. Adding its chosen move back into that same set expands nothing.
Learning to propose useful moves outside the offered set remains a major open
hypothesis, but current evidence does not identify it as the dominant bottleneck.

## Secondary engineering design: cheaper unchanged teacher

Static source review found a concrete opportunity. In
`src/splendor_ai/alphazero/world_budget_160.py`, `choose_root` loops over worlds
and offered moves and calls `trajectory` serially. The continuation in
`src/splendor_ai/alphazero/run_afterstate_advantage.py` calls greedy 035 each turn;
`src/splendor_ai/bots/model_bot.py` builds a single-state tensor batch for that
decision. Independent trajectories could instead advance together and share
batched neural inference.

No profile establishes the fraction of time spent in the network rather than
legal moves, features, simulation, hashing or serialization. Therefore no speedup
or GPU advantage is claimed. First profile a fixed isolated workload later;
prototype CPU batching before choosing a device based on measured throughput.

Any future scheduler must preserve world seeds and draws, lawful observations,
legal-action order, first-maximum ties, each trajectory's repetition/pass/progress
counters, terminal accounting and deterministic record ordering. Test mixed
trajectory lengths, short legal-action lists and hidden information. Padded
batches or device arithmetic can change near-tied moves and whole continuations.
Compare every action trace and final-state digest with the original, not just
the root's winning move. If behavior differs, treat it as a changed teacher that
requires separate validation, not a transparent performance optimization.

No batching implementation, inference, profiling, training, games, monitoring or
changes to 162 were performed for this review. The only new artifact is this
strategy note.
