# AlphaZero Pivot Engineering Programme `090`

Frozen before programme `090` implementation or sandbox output inspection on
2026-08-09. This is an engineering contract, not a preregistered strength or
promotion experiment. Sandbox results may guide debugging and configuration but
may not be presented as confirmation evidence.

## Objective

Build and exercise the first complete iterative search-learning loop in the
repository:

`network N -> shared-tree search -> self-play visits/outcomes -> network N+1 -> search with N+1`.

The initial deliverable is mechanism validation, not a new champion. Existing
champions and the independently running owner-waived `087` panel are unchanged.

## Architectural decisions

1. Use a randomly initialized `CanonicalEntityActionScorerModel` as a joint
   policy/value network. It emits one logit per structured legal action and a
   scalar value in `[-1,1]`.
2. Neither the new package nor its losses may import a `035` checkpoint,
   `035` scorer, resolver target, human example, rollout continuation policy,
   or parent-KL anchor.
3. Search uses a shared tree, full canonical legal-successor coverage, PUCT
   selection, joint-network leaf evaluation, alternating-player value backup,
   and root visit targets. It performs no terminal policy rollouts and has no
   top-four candidate restriction.
4. Debug mode searches the full simulator state and is mechanics-only. Real
   mode samples a legal observer-relative determinization per simulation and
   keys nodes by information-legal observation. A fixed hidden deck order or an
   opponent's blind-reserved identity may never enter a real-mode key or tensor.
5. Unknown market refills are chance uncertainty. `RESERVE_DECK` remains legal
   and unpenalized; the owner sees the identity and the opponent does not.
6. Initial training loss is soft root-visit cross-entropy plus terminal WDL
   mean-squared value loss and ordinary weight decay. Final margin is recorded
   but is not an initial optimization target.
7. Early work is a reproducible engineering sandbox. Formal promotion testing
   begins only after the improvement cycle works on held-out development seeds.

## New package and interfaces

Create `src/splendor_ai/alphazero/` with narrowly scoped modules:

- `network.py`: checkpoint creation/loading and batched joint inference;
- `search.py`: clean AlphaZero search and a bot wrapper;
- `replay.py`: information-legal trajectory schema and JSONL persistence;
- `selfplay.py`: seeded game generation and terminal target assignment;
- `training.py`: visit/value dataset, collation, loss, fitting, checkpoints;
- `generation.py`: vertical-slice and multi-generation orchestration.

Create one root CLI, `run_alphazero_sandbox_090.py`, with explicit `smoke` and
`pilot` modes. All outputs go under `outputs/alphazero_pivot_090/`.

## Search contract

- All distinct canonical legal successors receive an edge at expansion.
- Priors are the masked softmax of joint-network logits.
- Default PUCT: `c_puct=1.5`, first-play value `0`, maximum depth `96`.
- Self-play root exploration mixes 25% Dirichlet noise. Per-edge alpha is
  `dirichlet_total_concentration / legal_edge_count`, with default total
  concentration `10`.
- Self-play samples the visit distribution through turn 19 and uses visit
  argmax afterward. Evaluation is always visit argmax and has no root noise.
- Root visit probabilities are finite, nonnegative, and sum to one. Search must
  expose priors, visits, Q values, raw root value, selected action, expansions,
  depths, inference calls, and elapsed time.
- Terminal values are WDL from the leaf player-to-move perspective and are
  converted correctly to the root perspective through backup.
- Progressive widening and tree reuse are deferred until profiling shows a
  need. Merely allocating legal edges is not equivalent to rolling them out.

## Replay and loss contract

Each row stores generation, game/turn seed identity, acting player, the
observer-legal entity tensors, the complete structured action tensors, semantic
action payloads, root visits/probabilities, chosen action row, raw network value,
search root value, final winner/scores, WDL target, configuration identity, and
search telemetry. Stored model inputs must be sufficient for training without
reconstructing or reading privileged simulator state.

For a batch with valid legal-action mask `M`, target visit distribution `pi`,
policy logits `p`, scalar value `v`, and WDL outcome `z`:

- `policy_loss = -mean(sum(pi * log_softmax(mask(p))))`;
- `value_loss = mean((v-z)^2)`;
- `total_loss = policy_loss + value_loss` plus optimizer weight decay.

There is no one-hot chosen-action loss. Train/validation separation is by whole
game, never by position.

## Correctness gates before any pilot

Focused tests must establish:

1. a node contains every distinct canonical legal successor exactly once;
2. root visits have exact simulation accounting and normalize to one;
3. terminal and alternating-player backup signs are correct;
4. identical seeds/checkpoints reproduce action and telemetry projections;
5. real mode is invariant to permutations of hidden deck order;
6. an opponent's blind-reserved identity is absent while the owner's identity
   remains available;
7. `RESERVE_DECK` remains in the legal action distribution when legal;
8. replay tensors and probabilities round-trip exactly;
9. the loss is finite and improves on a repeated toy batch;
10. a saved successor reloads byte-equivalent predictions;
11. the end-to-end smoke creates generation zero, self-play rows, generation
    one, and then actually searches with generation one;
12. no programme-090 module imports or loads `035`.

## Vertical slice

The mandatory smoke is deliberately small: create generation zero, run four
debug games with 8--16 simulations per move, fit at least one epoch, save and
reload generation one, then run two games with generation one. This validates
plumbing only.

## Three-generation sandbox pilot

Subject to the correctness gates and smoke:

- generation zero: 32 self-play games, 32 simulations/move;
- generations one and two: target 64 games each, 64 simulations/move;
- retain at most the latest three generations in training replay;
- use root noise and temperature only in self-play;
- run small paired development arenas after each successor where practical:
  search(N) vs raw(N), raw(N+1) vs raw(N), search(N+1) vs search(N), and raw
  N+1 vs `035` as an external diagnostic only.

The runner may reduce pilot games or simulations only for an explicitly
labelled resource-limited smoke/pilot artifact; it must report the realized
dose and may not call the reduced run the target pilot.

## Mechanism report

For every generation report:

- games, positions, termination reasons, turns, and action-family mix;
- `RESERVE_DECK` availability and selection;
- policy/value losses and whole-game validation losses;
- visit/prior divergence, search/raw action switch rate, visit entropy;
- raw and searched network identities used to generate every corpus;
- development arena shares with paired seeds and both seats;
- runtime and search/inference dose.

The first meaningful success requires the qualitative cycle, not a single
threshold: search improves its raw network, the child learns material search
movement and improves on its raw parent, and search can still improve the
child. If none appears after two or three generations, stop scaling and audit
the loop rather than silently adding data or compute.

## Scope exclusions

Programme `090` does not promote a bot, reinterpret prior PUCT failures, inspect
interim `087` outcomes, alter `087`, tune against human games, or authorize a
large formal panel. It builds the missing system and produces explicitly
exploratory engineering evidence.
