# Superhuman Information-Set Search Roadmap After Experiment 132

Date recorded: 2026-08-30

## Objective

Build a Splendor agent that can establish superhuman playing strength. Beating
raw legal `035` is an intermediate benchmark, not the definition of success.
The final claim requires decisive performance against every retained AI
champion, robustness across seats and opponent styles, and prospective games
against multiple strong humans under a declared per-move compute budget.

## Strategic decision

Close the present compact AlphaZero **recipe** as the default incremental
research path. Do not claim that AlphaZero is theoretically unsuitable for
Splendor. Experiments 102 and 103 proved two consecutive blocks of searched
self-improvement, and experiment 113 proved substantial deeper-search
headroom. The closed claim is narrower: experiments 104--132 repeatedly failed
to convert additional data, deeper teachers, capacity, replay diversity,
representation changes, alternative value targets, or repeated-return means
into a reliable gameplay successor.

The new default programme is an **information-set search-and-learning loop**:

`planner -> self-play -> learned guidance -> stronger or cheaper planner -> repeat`

Search remains part of the deployed agent. A learned component is not required
to replace search or to win raw-network games. It is accepted only when it
causally improves the complete planner at fixed compute, or preserves planner
strength at materially lower compute.

This is compatible with an AlphaZero-like architecture. A policy/value network
may later supply priors and leaf estimates, while the information-set planner
replaces ordinary perfect-information MCTS as the policy-improvement operator.

## Retained evidence and baselines

- Raw legal `035` remains the fast champion.
- The exact `050`/`051` resolver confirmed by `084` remains the research
  champion. It scored 303-96-1 over 400 games against raw `035`, 75.875% score
  share, with both seat safeguards passing.
- The exact expanded-root `086` candidate remains an unpromoted, expensive
  oracle candidate. On 300 fresh states it produced replicated positive
  untouched payoff gain `+0.051199`, positive effects in every phase, and
  passed the prospective regret/reliability gates. Its valid full gameplay
  strength versus the research champion is unresolved because `087` stopped
  after an invalid smoke and the later incomplete owner-waived panel was
  outcome-sealed.
- Exact AlphaZero generation 11 is retained as the strongest mature checkpoint
  for search and representation research, not as a champion. It remains useful
  evidence that online search can add strength even when compression fails.
- Exact generation 2 remains installed under the existing installation
  contract. This roadmap authorizes no installation or promotion.

## Scientific constraints learned from the resolver programme

The new planner must not simply reproduce the old resolver unchanged.

1. A permanent `035` top-four root set is a strength bottleneck. Expanded-root
   evidence found useful lower-ranked actions, particularly token-taking
   actions.
2. One 192-continuation shallow resolver call is not a stable supervised label.
   Independent calls agreed on the exact best action only 45.83% in experiment
   055. Brute-force averaging converged but remained weakest early and was too
   expensive for a direct teacher.
3. Early instability is mainly belief-sampling instability, not ordinary
   continuation noise. Experiment 058 attributed 61.741% of the balanced
   variation to the sampled public belief, 9.225% to continuation conditions,
   and 29.034% to their interaction.
4. The first public-belief visit search in experiment 060 looked stable mainly
   because its sharp `035` prior dominated visits; its independent Q estimates
   remained unsafe. A new visit distribution must pass an explicit prior-control
   check before it can become a learning target.
5. Direct exact-action distillation, frozen-parent residual correction, public
   history, and the tested learned option selectors did not reliably capture
   resolver decisions. Learning should first target easier, operational jobs:
   high-recall action proposal, leaf evaluation, uncertainty, and search-budget
   allocation.

## Programme stages

### Stage 1: establish the search oracle frontier

Keep the research champion immutable. Resolve the exact `086` expanded-root
candidate's gameplay status only through a separately frozen fresh comparison;
never inspect, pool, or complete the outcome-sealed partial `087` panel as if it
were prospective evidence. This comparison is a bounded calibration of the
available expensive oracle, not the new programme's final architecture.

### Stage 2: build a next-generation public-belief planner

The first new planner must train nothing. It should:

- represent multiple observer-legal possible worlds explicitly;
- share opponent decisions only across identical opponent information sets;
- treat market replenishment as chance without exposing true deck order;
- make every legal public root reachable through forced coverage and
  progressive widening rather than a static top-four prefix;
- use a prior for allocation without allowing the prior to determine the
  result before action values become informative;
- reuse paired randomness across comparable branches where legal;
- retain visit counts, Q estimates, uncertainty, and near-equivalent action
  sets rather than serializing only one noisy winner;
- allocate greater belief breadth early, where replicated evidence found the
  instability;
- report strength and regret as functions of simulations and wall time.

Its first gate is a sealed state-panel feasibility test with independent search
replicates. Require information legality, exact accounting, finite values,
stable semantic action sets, material search-to-prior movement, independent
search agreement that is better than shared-prior agreement, bounded
cross-regret, and positive value relative to the research-champion decision.
Only a complete pass opens fresh full-game comparison against the research
champion.

### Stage 3: learn high-recall action proposals

Do not ask the first learner to reproduce the oracle's exact argmax. Train it
to retain the repeated-oracle near-optimal set while reducing how many roots
receive expensive evaluation. Always union its proposals with forced
exploration. Accept it only if it preserves nearly all full-oracle strength and
materially improves the planner's fixed-time strength frontier on fresh games.

### Stage 4: learn leaf value and uncertainty

Use aggregated planner supervision across belief particles and continuation
conditions, not one terminal outcome or one shallow resolver call. Initially
use the learned value only on ordinary leaves and retain real continuations for
uncertain or close decisions. Offline loss is diagnostic. Acceptance requires
stronger fixed-compute planner gameplay or equal strength at lower compute.

### Stage 5: remove permanent `035` dependence

Begin with the union of `035`, learned proposals, and forced exploration.
Gradually introduce accepted learned and historical policies as priors,
opponents, and continuation policies. Remove an operative `035` dependency
only after a prospective ablation shows no loss. Eventually `035` becomes an
external benchmark rather than the strategic centre.

### Stage 6: iterative information-set AlphaZero

Once the planner is stronger than the research champion and economical enough
for self-play, generate planner self-play, train proposal/policy/value models,
insert them back into the planner, and compare complete searched generations at
equal compute. Require at least two consecutive powered multi-generation gains
before treating the loop as compounding. Raw-network strength remains useful
but is not a gate.

### Stage 7: scale and establish superhuman strength

Only after compounding works should the project batch neural leaf inference,
reuse trees, optimize simulation/state copying, use GPU inference, or
distribute self-play. Evaluate a population of current and historical agents
to expose non-transitivity. A superhuman claim requires a separately designed
human study with multiple strong players, balanced seats, declared compute,
and prospective analysis.

## Stopping and branch rules

- Do not reopen compact AlphaZero by another small target, head, adapter, or
  representation experiment.
- A materially larger fully trainable stochastic AlphaZero run remains a valid
  later falsification/scaling experiment, preferably after shared batching
  infrastructure makes it cheaper. It is not the immediate default.
- Do not train from a new planner until the planner itself passes independent
  stability, prior-control, value, and gameplay gates.
- Do not promote a learned component for offline fit alone.
- If two substantially different, information-legal planner designs fail to
  improve the research champion at a practical compute frontier, perform a
  programme-level review before more local search tuning.
- If learning never improves or accelerates a confirmed planner, retain search
  as the full-strength agent and pursue planner/engine efficiency directly;
  failure to compress is not failure of the playing agent.

## Immediate authorized work

1. Finalize experiment 132 and record the compact-recipe closure.
2. Freeze exact champion/oracle identities and the new programme boundaries.
3. Specify and implement one no-training public-belief planner feasibility
   experiment with a launch guard.
4. Run tests and preflight. The evidence state panel may run after preflight;
   it may not train, install, promote, inspect sealed `087` outcomes, or launch
   gameplay automatically.
5. Return for review if the state-panel evidence fails, or before launching the
   separately frozen expensive full-game comparison that a pass may authorize.

