# Reduced Splendor CFR Correctness and Viability Gate (`044`)

Frozen before implementing any CFR solver or generating any `044` result. This
is an architectural correctness gate, not a full-game strength experiment and
not permission to reinterpret the closed PUCT, rollout, AWR, preference,
option, value, or rolling-horizon branches.

## Why CFR is the next architectural branch

Two-player base Splendor is a sequential zero-sum win/loss game with chance
and private information. The strongest project diagnostics found stable,
frequently resolved action-conditional returns, while trajectory-level credit,
learned value ordering, determinization search, and compression of local
preferences all failed to improve closed-loop play. Counterfactual regret
minimization addresses actions at legal information sets and produces an
iterative average self-play strategy without requiring a scalar leaf value.

The BGA/expert-replay idea remains a possible later warm start or external
validation source. No BGA data is used or collected in `044`.

## Information-state history contract

Add a versioned acting-player information-state key containing:

- the existing canonical public-belief key, including acting-player private
  reserve identities and masked opponent blind reserves;
- chronological public action events;
- public card identities when a visible card is bought/reserved or a private
  reserve is later purchased/revealed;
- public token movements, tier-only blind-reserve events, noble choice, turn
  order, and start role through the underlying state/key.

The key must never contain hidden deck order or an unrevealed opponent-private
reserve identity. Histories that differ only in those hidden facts must be
identical for an uninformed observer and may differ for the informed owner.

## Frozen reduced Splendor game

Use the production `SplendorEnv` legal-action and transition implementation in
a finite four-action duel:

- fixed start player 0 and no nobles;
- bank: two red, two blue, one gold, and zero other colored tokens;
- visible tier-1 cards `V_RED` and `V_BLUE`, each worth one point and costing
  one matching token;
- hidden tier-1 cards `H_MIX` (two points, cost one red plus one blue) and
  `H_FREE` (one point, zero cost);
- tiers 2 and 3 are empty;
- root chance chooses the hidden order `(H_MIX,H_FREE)` or
  `(H_FREE,H_MIX)` with probability one half;
- ordinary take, visible/deck reserve, visible/reserved buy, replacement, gold,
  and privacy rules come from the production engine;
- after exactly four player actions, adjudicate by score and then the standard
  fewest-purchased-cards tie-break; an unresolved tie is a draw;
- utility to player 0 is `+1/-1/0` for win/loss/draw and player 1 receives its
  negation.

Every root-to-terminal path must finish within four player actions. Semantic
canonical legal-action descriptors, not procedural list indices, identify
information-set actions.

## Reference exact CFR

Run full-tree alternating CFR for exactly 10,000 iterations from zero regrets.
Record cumulative regrets, average strategies, root strategy, expected
self-play utility, terminal/path counts, and normalized positive-regret mass
for each player.

Correctness gates:

- deterministic reruns are byte-equivalent after excluding elapsed time;
- all visited information sets expose one stable semantic action set;
- no privacy-contract violation or illegal action occurs;
- normalized positive-regret mass for each player is below `0.02` at iteration
  10,000;
- each player's normalized positive-regret mass falls by at least 75% from the
  fixed iteration-100 snapshot to iteration 10,000.

## External-sampling MCCFR replication

Run alternating-traverser external-sampling MCCFR for exactly 50,000 iterations
with seed `440044`. Enumerate the traverser's actions; sample chance and the
opponent from their current regret-matched strategy. Store average strategy
only from legally reached information sets.

Replication gates:

- deterministic reruns with seed `440044` are byte-equivalent after excluding
  elapsed time;
- zero illegal actions, privacy failures, or unstable information-set action
  sets;
- both players' final normalized positive-regret mass is below `0.05`;
- root-strategy total variation from exact CFR is at most `0.10`;
- across information sets shared by both solvers with at least 100 external
  average-strategy observations, mean total variation is at most `0.15` and
  the 90th percentile is at most `0.30`;
- at least 75% of exact-CFR information sets are covered by the external
  solver.

## Decision

A complete pass authorizes a separately preregistered full-game neural Deep CFR
pilot with parallel CPU traversals and GPU regret-network training. It does not
authorize checkpoint promotion. Any failure stops before GPU rental and
requires diagnosis of the CFR/game interface rather than threshold tuning or
full-game scaling.

Implementation and result artifacts must live under dedicated `044` names.
The existing champion remains raw legal `035`.
