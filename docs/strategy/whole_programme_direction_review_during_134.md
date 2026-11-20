# Whole-Programme Direction Review During Experiment 134

Date: 2026-09-04  
Status: outcome-blind strategic review; experiment 134 remains running and unchanged

> **Architecture correction recorded later on 2026-09-04:** this review
> over-weighted blind deck-reserves when selecting the primary architecture.
> The project's evidence supports treating public stochastic refills as the
> dominant uncertainty and blind reserves as a bounded secondary case. The
> recommendation to make public-belief GT-CFR experiment 135 is superseded by
> `docs/strategy/stochastic_refill_vs_blind_reserve_architecture_correction.md`.
> The audit of prior work below remains useful, but do not execute its proposed
> experiment 135 without first resolving the chance-aware AlphaZero crux.

## Executive conclusion

The programme is moving in the right general direction, but the destination
should now be stated more precisely.

The strongest long-term route is not ordinary AlphaZero MCTS, and it is not an
indefinite sequence of small improvements to the current exact resolver. It is
a compact, Splendor-specific version of the public-belief, game-theoretic
search-and-learning architecture used by systems such as ReBeL and Player of
Games:

1. represent the public history and each player's possible private states;
2. search a public tree with regret minimization rather than ordinary PUCT;
3. return policies and counterfactual values for all relevant private states,
   rather than one action and one scalar value;
4. re-solve consistently as new public actions and private observations occur;
5. train networks to make that solver faster and deeper, while retaining search
   as the final decision mechanism.

This is still recognizably an AlphaZero-like programme at the highest level:
self-play produces targets, a neural network generalizes them, and search uses
the network to improve decisions. The part that must change is the search
mathematics required by hidden information.

Experiment 134 remains useful whichever way it ends. A win would establish the
expanded exact resolver as a stronger bridge, teacher, and benchmark. A loss or
tie would say that its offline local gains did not convert into game strength.
Neither outcome changes the need for a more principled public-belief solver.

## What the project has established

### The engineering foundation is strong

The repository now has a reliable two-player Splendor engine, legal
player-relative observations, explicit action encoding, equal-turn endings,
the score/card-count tiebreak, reproducible paired matches, frozen experiment
identities, and broad automated tests. That work is not glamorous, but it is
the foundation on which credible research depends.

The programme has also become unusually disciplined about separating training,
selection, and protected evaluation. Failed experiments have generally been
kept as evidence rather than silently tuned away.

### Search adds real value

Several experiments showed that more search can improve the learned agent.
Within the AlphaZero branch, larger searches beat smaller ones. More
importantly, the exact grouped resolver built in experiments 050/051/084 beat
the fast 035 policy clearly, including 75.875% score share in experiment 084.
This is the strongest evidence in the project: planning matters in Splendor.

### The expanded root set is a plausible improvement, not yet a champion

Experiment 086 showed positive offline value from considering more root
actions. It changed the selected action frequently and improved the resolver's
own measured score-margin value across game phases. Experiment 134 is the
proper fresh full-game test of whether that local advantage converts into
strength against 084. Until it finishes, 086 is an oracle candidate, not the
installed research champion.

### The compact AlphaZero loop learned, but did not reach the required level

The AlphaZero line was not a total failure. Early generations improved, and
deeper search helped. But the compact CPU recipe plateaued far below 035 and
later value-target interventions did not repair it. The experiment was also
small by landmark AlphaZero standards: a small network, hundreds rather than
millions of self-play games per generation, and a modest simulation budget.

The correct conclusion is therefore narrow:

- the tested compact recipe is not a promising default path;
- ordinary perfect-information PUCT is not theoretically sound for the actual
  hidden-information game;
- these results do not prove that large-scale self-play and learned search are
  irrelevant to Splendor.

### Start role is not a central obstacle

Experiments 117--126 showed that who started the game carries little useful
extra signal once the current observation is known, except for a small late-game
effect. This was worth checking, but it does not explain the programme's main
performance gap and should not remain a priority.

### Scalar value prediction is not the immediate bottleneck

The recent value studies found substantial target noise and very small or
negative gains from alternative scalar targets. That result is consistent with
the deeper diagnosis: in an imperfect-information game, a single value attached
to a public observation is often not the object the search actually needs. The
planner needs values indexed by private possibilities and reach probabilities.
Refining one scalar target cannot supply that missing structure.

## What has not been established

The project has not yet shown any of the following:

- superhuman play against strong human competition;
- robustness against a diverse population of independently developed agents;
- low exploitability or approximate equilibrium play in full Splendor;
- correct Bayesian beliefs about hidden cards after observing a sequence of
  opponent actions;
- a sound continual re-solving method for the production game;
- a neural evaluator that improves the complete planner at fixed wall-clock
  cost;
- independent agreement between this rules engine and a second implementation.

Beating 035 is a useful internal milestone, but 035 is not an external measure
of superhuman strength. Nearly all current opponents and evaluators also descend
from the same policy family. That creates a risk of measuring progress inside a
self-referential ecosystem.

## Diagnosis of the current resolver

The 084/086 resolver does several important things correctly.

- Its root action is shared across all sampled worlds.
- At the opponent reply, one action is chosen for all worlds that belong to the
  same opponent information set.
- Rollouts use legal player-relative observations.

Those properties prevent the most obvious form of hidden-state cheating. A
future player using their own newly observed private card is also legitimate;
that alone is not strategy fusion.

The remaining limitations are more fundamental.

### Beliefs are based on consistency, not the observed strategy

The current determinizer samples hidden cards uniformly from worlds consistent
with the visible position. It does not condition those probabilities on the
public action history. For example, an opponent's decision to reserve, buy, or
ignore a card can change how likely different hidden holdings are. The present
state key cannot express that inference.

Uniform consistency is a sensible starting approximation, but it is not a full
belief state.

### The game is solved for only a shallow prefix

The resolver performs a robust grouped comparison at the root and opponent
reply, then lets 035 play the continuation. It does not make all later decisions
part of one mutually consistent strategy. As a result, it is a strong local
look-ahead method, not an equilibrium solver.

### Its objective is score margin rather than the exact match result

The rollout utility is normalized final score margin. The actual game objective
is win/draw/loss after the card-count tiebreak. Score margin can be a useful
secondary signal, but a final sound planner should optimize exact match utility.

### It returns a single best move

Hidden-information games can require mixed strategies. A deterministic action
may be perfectly adequate in many Splendor states, but a solver intended to be
robust or approximately unexploitable must be capable of returning and executing
a distribution over actions.

### Its cost grows through complete rollouts

The current expensive resolver improves decisions by running many full games
under 035. That makes experiment 134 slow and creates a poor scaling law. The
long-term purpose of learning should be to replace most of those continuations
with range-conditioned leaf values, permitting a deeper or broader search at
the same decision time.

## Why ordinary AlphaZero search is not the final solver

AlphaZero's tree nodes have a well-defined state value because chess, shogi, and
Go are perfect-information games. In Splendor, the same visible position can
correspond to different private cards and different beliefs about them. The
correct value of an action can also depend on how often that action is chosen.

Ordinary PUCT treats each action as if it has one stable scalar value and backs
that value up through the tree. Determinizing the hidden cards at the root does
not repair this: the search must also track what each player could know, how
likely each private state is, and how strategies behave across information sets.
Experiment 133's instability and negative oracle gain are empirical warning
signs of this mismatch, not merely evidence that its exploration constant was
poorly tuned.

This does not mean the entire AlphaZero idea is discarded. Policy priors,
learned leaf values, self-play targets, replay, and iterative improvement all
remain useful. PUCT can even help decide which branch to expand. It should not
be the rule that claims to solve an imperfect-information public tree.

## The revised target architecture

The proposed destination is a small public-belief continual resolver adapted to
Splendor.

At each decision point it should hold:

- the complete public action and chance history;
- a probability range over each player's possible private reserved cards;
- the public deck uncertainty, represented compactly or with particles;
- a policy for every active information state in the local search tree;
- counterfactual values for those information states at the search boundary.

It should then:

1. grow a shallow public tree;
2. update policies by counterfactual regret minimization;
3. evaluate leaves with exact rollouts at first and a learned
   counterfactual-value-and-policy network later;
4. choose from the solver's mixed root policy;
5. update ranges after the observed action and re-solve at the next public
   state.

Splendor is friendlier to this approach than a generic hidden-information game.
Most of the board is public, private information is sparse, and the important
unknown identities arise mainly from blind reserves. We need not enumerate
every possible ordering of every deck. Hidden reserve identities can be tracked
explicitly while deck refills and longer chance sequences can be sampled.

## Recommended experimental ladder

Each step should be able to fail cheaply and teach us something specific.

### Stage 0: mechanics before training

1. Specify the production game formally as a two-player zero-sum
   extensive-form or factored-observation game. Define public state, private
   state, public history, chance events, information sets, reach probabilities,
   and exact win/draw/loss utility.
2. Reuse the reduced Splendor work from experiments 044--046 as a reference
   game. Confirm that the corrected CFR implementation reduces regret and
   exploitability there.
3. Add an explicit range or particle object. On small enumerated positions,
   test that chance updates and policy-conditioned Bayesian action updates
   exactly match brute-force enumeration.
4. Add mixed-policy execution and deterministic seeded sampling so that paired
   evaluations stay reproducible.

### Stage 1: a minimal production-state solver

5. Implement a depth-two or depth-three public-tree CFR solver on a small frozen
   set of real game states. Initially use exact terminal results or the existing
   full-rollout machinery at the boundary.
6. Return a policy and counterfactual-value vector, not one selected action and
   one averaged scalar.
7. Compare independent runs for regret, policy stability, action-value ranking,
   and runtime. Use exhaustive or very-high-budget evaluation on a small subset
   as the reference.
8. Open a small paired gameplay test only if these mechanism checks pass.

### Stage 2: make the solver affordable with learning

9. Save solver queries consisting of public state/history, ranges, solved
   policies, and counterfactual values.
10. Train a compact counterfactual-value-and-policy network on these vector
    targets.
11. Judge it inside the complete search: either stronger play at equal time or
    equal play at lower time. Standalone mean-squared error is diagnostic, not
    the success criterion.
12. Generate new data through planner self-play and query solving. Keep exact
    search examples in the mixture to prevent drift.

### Stage 3: strength, robustness, and scale

13. Maintain a league of genuinely different opponents and old checkpoints.
    Measure the full cross-play matrix and use a mixture or Nash-style summary
    when non-transitivity appears.
14. Validate the engine against an independent implementation or a carefully
    audited rulebook oracle.
15. Establish an external strength ladder: fast internal champion, search
    champion, diverse external bots, and a powered human or expert evaluation.
16. Move neural inference and training to a GPU when they become the bottleneck.
    More CPUs will continue to help rollouts, but the architecture should aim to
    eliminate most full-game rollouts rather than scale them indefinitely.

## Proposed experiment 135

The best immediate successor is an outcome-blind mechanics bridge, not another
training run and not another long gameplay match.

Working title:

**Experiment 135: reduced-to-production public-belief GT-CFR mechanics bridge**

Its scope should be:

1. write and freeze the public-history, private-range, reach-probability, and
   counterfactual-value contracts;
2. implement exact Bayesian range updates on the reduced game and verify them
   against enumeration;
3. implement a shallow public-tree CFR solver that preserves information sets
   and returns a mixed root policy;
4. run it twice independently on roughly 12--24 frozen, phase-balanced states;
5. require decreasing regret or exploitability on the reduced reference game,
   exact belief-update agreement, no privacy violations, stable production-state
   policies, and a bounded runtime;
6. do no training and open no gameplay until those gates pass.

The purpose of 135 is not to beat 084 immediately. It is to establish that the
new search object is mathematically and mechanically sound before expensive
self-play is built around it.

## How experiment 134 should affect the plan

### If 086 clearly wins

Promote it as the research champion and retain it as a teacher and oracle. One
bounded efficiency experiment may test structured racing or adaptive allocation
of its rollouts. Do not spend a long series tuning rollout caps. Begin 135
regardless.

### If 086 ties or loses

Keep 084 as the research champion. Treat 086's local offline gain as evidence
that the score-margin oracle did not predict full-game strength reliably. Use
that lesson when designing 135's exact WDL targets. Begin 135 regardless.

### If 134 has an operational failure

Separate evidence validity from engineering inconvenience. Repair and rerun only
if the failure compromises identities, game completeness, information legality,
or outcome accounting. Do not repeat a multi-day experiment merely because its
wall-clock estimate or old runtime ceiling was wrong.

## Changes to research practice

### Use two experimental modes

Exploration should use small state panels, modest game counts, effect sizes,
uncertainty, and explicit learning goals. Confirmation should use frozen hashes,
strict integrity gates, protected sets, and sufficiently powered gameplay.

Applying championship-level conjunctions to every exploratory idea wastes
compute and creates many false negatives. Relaxing integrity would be wrong;
relaxing premature strength claims is not.

### Make one endpoint primary

For strength experiments, choose one primary paired gameplay statistic and use
seat balance, phase effects, loops, and expansion dose as guardrails or
diagnostics unless the scientific question makes them primary. A conjunction of
many individually noisy pass/fail thresholds can reject genuinely useful agents.

### Measure cost as part of strength

Record decision latency, CPU-hours, memory, and games or states processed. A
planner that is marginally stronger at fifty times the cost may still be useful
as a teacher, but it should not be confused with the deployment champion.

### Preserve diversity

Do not allow every candidate, continuation policy, and evaluator to descend from
035 indefinitely. Retain old AlphaZero agents, heuristic policies, resolver
variants, deliberately exploitative agents, and future independently trained
policies as a testing population.

## Definition of success

The programme should use several levels of claim.

1. **Internal improvement:** beats the current champion under a frozen paired
   protocol.
2. **Robust internal strength:** performs well against a diverse league and has
   no major cross-play weakness.
3. **Game-theoretic evidence:** low exploitability on exact reduced games and
   improving approximate bounds or best responses on larger abstractions.
4. **External strength:** beats independent bots and performs strongly in a
   powered evaluation against recognized strong human players.
5. **Practical superhuman system:** satisfies the external strength claim at a
   declared decision-time and hardware budget.

The final meaning of “superhuman” should be fixed before making that claim. A
single player's record is valuable engineering evidence, but the scientific
claim should refer to a recognized strong competitive population.

## Paths not recommended now

- Do not resume incremental tuning of the compact ordinary-PUCT AlphaZero line.
- Do not increase experiment 133's simulation budget or tune its exploration
  constant.
- Do not make scalar value-target research the main programme.
- Do not optimize expanded-root rollout engineering through many successive
  experiments before testing the new solver mechanics.
- Do not attempt a full-scale reproduction of ReBeL or Player of Games at once.
- Do not claim superhuman strength from wins against 035 or the current resolver
  family alone.

## Primary scientific references

- Silver et al., *A General Reinforcement Learning Algorithm that Masters
  Chess, Shogi, and Go through Self-Play* (AlphaZero), Science 2018:
  https://discovery.ucl.ac.uk/id/eprint/10069050/
- Brown et al., *Combining Deep Reinforcement Learning and Search for
  Imperfect-Information Games* (ReBeL), 2020:
  https://arxiv.org/abs/2007.13544
- Schmid et al., *Player of Games*, 2021:
  https://arxiv.org/abs/2112.03178
- Moravcik et al., *DeepStack: Expert-Level Artificial Intelligence in
  Heads-Up No-Limit Poker*, 2017:
  https://arxiv.org/abs/1701.01724
- Perolat et al., *Mastering the Game of Stratego with Model-Free Multiagent
  Reinforcement Learning* (DeepNash), 2022:
  https://arxiv.org/abs/2206.15378

## Final recommendation

Let 134 finish without changing it or reading its partial outcomes. Then apply
its result to champion selection only. The next research experiment should be
135 as specified above: a small, no-training public-belief GT-CFR mechanics
bridge. This is the shortest credible step from the current strong local
resolver to a search-and-learning system with a plausible route to superhuman
Splendor.
