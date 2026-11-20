# External strength calibration preparation

18 October 2026, preparation revision 2. The next investment is a fair external
benchmark of the complete player. First qualify our unchanged searched players
and an enforced clock on saved positions. Then qualify an external opponent and
screen its competitiveness before allocating a strength arena. No screened
external opponent is yet qualified for a strength result.

The completed main, alternative-ranker and saved-world studies retain their
original conclusions. This preparation changes no frozen player or experiment.
It contains a proposed benchmark and implementation specification, not a new
execution freeze or a completed strength test.

## Player to calibrate

Use the **eight-candidate, 128-world breadth planner** as the research candidate.
It is the strongest verified successor in the local comparison chain: 56.2189%
score against learned128, paired 95% interval [51.6169%, 60.6965%], in 402 games.
It used additional compute. This establishes that matchup, not dominance over
every retained player or independent opposition.

Preserve all three original 161 fits, balanced across seed pairs; do not choose
the best fit. Exact models are recorded in
[`models_lock.json`](../../outputs/proposal_breadth_arena_128/models_lock.json):

- `world128_161001/checkpoint_0563.pt`
- `world128_161002/checkpoint_0488.pt`
- `world128_161003/checkpoint_0563.pt`

These live under `outputs/teacher_budget_161/fits/`. Keep frozen 035 greedy
continuations, original first three proposals, learned fourth, two further
ranked and two deterministic-uniform additions, 128 shared worlds and ordered
ties. The authoritative implementation is
[`arena.py`](../../tools/proposal_breadth_arena_128/arena.py), with its
[frozen protocol](../engineering/proposal_breadth_arena_128.md).

Learned128's 65.796% result was against the original128 planner with a random
fourth suggestion. It is not a direct victory over 084 or an original-ranked
fourth at 128 worlds. The direct 65.125% victory over 084 belongs to the earlier
32-world selector in [experiment 156](../engineering/rollout_selector_champion_156_result.md).

## Timing and hardware

Keep the README's **one-second deployment target** visible. A separate proposed
**120-second research allowance** can assess the fixed-work breadth planner if
it qualifies operationally. Neither track currently has a verified searched
entrant under an enforced deadline. Do not silently replace the research player
with raw 035, reduce its worlds, or return a partial-search fallback.

Saved instrumented timings are substantial: learned128 averaged 75.37 seconds
(p95 144.88); breadth averaged 127.87 seconds against learned128's 66.05 in their
match. Those runs used eight concurrent one-thread workers. They do not predict
isolated-worker latency or establish feasibility at either proposed allowance.
See the [learned128 result](../../outputs/learned_proposal_128/result.json) and
[breadth result](../../outputs/proposal_breadth_arena_128/result.json).

Start qualification on this i5-13600KF CPU with one match active, one verified
physical performance core available to the acting player, one numerical thread,
no GPU, and the same core and priority for both players. Verify topology before
setting affinity. Both players receive the same wall-time allowance; report
delivered CPU time, wall time and work separately. This is equal time on declared
hardware, not equal simulation counts between different algorithms.

Load models before the game under a separate common 60-second initialization
limit. Start the move clock before delivering the legal observation; include
serialization, all within-turn sub-actions, inference and search. End it when a
complete action reaches the referee. A separate supervisor enforces the clock,
including a hung child. No extra clock for token-return or noble-choice steps,
no pondering during the other player's turn, no cache sharing between paired
games. Record initialization costs and any retained within-game search memory.
In a frozen arena, timeout, crash or illegal action is a loss and remains in the
denominator. A referee or audit defect blocks scientific acceptance.

The current general runner cannot supply this contract:
[`match.py`](../../src/splendor_ai/eval/match.py) declares `seconds_per_move` but
does not use it in `play_match`/`play_game`. Its consecutive swapped games also
use different initial seeds. It must not be used unchanged as a timed paired
benchmark. Add a separate harness outside frozen inventories.

## External opponent screening

Repositories were inspected at the following commits on 18 October. Public
weights and source availability do not establish strong play or rules fidelity.

| Candidate | Availability and role | Qualification issue |
| --- | --- | --- |
| [roeey777 Splendor-AI](https://github.com/roeey777/Splendor-AI/tree/95f84d2e6e839c0ef09ca97bdc3b3048a792fb0b) | Explicit MIT license and public PPO weights; first adapter candidate, as an independent policy control | Its source engine omits blind reserve, limits purchased cards per colour and prunes some token returns. Use our referee and qualify a legal subset of actions; do not import that engine as the rules authority. |
| [AhinLendor](https://github.com/inhabae/AhinLendor/tree/96e6f2daff83147495826c4a2073dc3c9c856cc9) | Author reports strong online/human play; promising later strength opponent | No checkpoint or separate LICENSE file was found in the inspected tree, although its README says MIT. Sequential return actions need one shared move clock; current source restricts returning gold. Exact playable weights and rules compatibility remain unresolved. |
| [cestpasphoto Alpha Zero General](https://github.com/cestpasphoto/alpha-zero-general/tree/be3814e077fffcaec978c44e7cd088fd408c9eac) | Public two-player checkpoint | Current game contribution license is PolyForm Noncommercial with upstream MIT portions. Engine/state inspection found nonstandard take/return turns, multiple-noble resolution and private-state exposure risks. Not eligible unchanged. |

The last project's often-quoted 90%/95% performance claims concern Santorini,
not Splendor. AhinLendor's leaderboard/exhibition statements are author reports,
not a reproduced evaluation here. No inference about our player's strength
follows from this screening. An external codebase with modified observation or
action handling must be identified as an adapted version, not the unchanged
published player.

Source details: the proposed PPO's [interface](https://github.com/roeey777/Splendor-AI/blob/95f84d2e6e839c0ef09ca97bdc3b3048a792fb0b/src/splendor/agents/our_agents/ppo/ppo_agent.py),
[features](https://github.com/roeey777/Splendor-AI/blob/95f84d2e6e839c0ef09ca97bdc3b3048a792fb0b/src/splendor/splendor/features.py),
[rules](https://github.com/roeey777/Splendor-AI/blob/95f84d2e6e839c0ef09ca97bdc3b3048a792fb0b/src/splendor/splendor/splendor_model.py)
and [checkpoint](https://github.com/roeey777/Splendor-AI/blob/95f84d2e6e839c0ef09ca97bdc3b3048a792fb0b/src/splendor/agents/our_agents/ppo/ppo_model.pth);
the alternative Alpha Zero General [rules](https://github.com/cestpasphoto/alpha-zero-general/blob/be3814e077fffcaec978c44e7cd088fd408c9eac/splendor/SplendorLogicNumba.py)
and [license](https://github.com/cestpasphoto/alpha-zero-general/blob/be3814e077fffcaec978c44e7cd088fd408c9eac/LICENSE);
AhinLendor's [rules](https://github.com/inhabae/AhinLendor/blob/96e6f2daff83147495826c4a2073dc3c9c856cc9/game_logic.cpp)
and [observer encoder](https://github.com/inhabae/AhinLendor/blob/96e6f2daff83147495826c4a2073dc3c9c856cc9/state_encoder.cpp).

## Bounded qualification before games

Implement the observer/action adapter and clock as a new component. **Run the
local player and clock qualification first**, comparing the exact four-candidate
learned128 and eight-candidate breadth configurations across all three fixed fits.
Verify completed decisions against saved evidence and measure end-to-end timing
on the predetermined positions. This establishes correctness and sampled
latency, not new strength or guaranteed completion on every future position.

Breadth remains the research candidate. If only the four-candidate player fits
the proposed allowance, report that distinction; do not silently substitute it
or claim equal strength. Freeze any subsequent timing-based entrant selection
before external matchup outcomes are collected.

Next qualify roeey777 PPO because its weights and interface are inspectable; it
is an independent-policy control, not an established strong opponent. Pin source,
weights, runtime, feature order and action mapping before running it. Verify
checkpoint structure before loading; do not execute unknown serialized objects.

The exact PPO source, license and checkpoint have now been saved with provenance
in [`upstream_receipt.json`](../../outputs/external_strength_preparation_20261018_01/upstream_receipt.json).
All seven files total 4,375,916 bytes. The 4,311,856-byte checkpoint matches its
pinned Git blob `4c5dad4d7133e5e4d5a12ad5d54dbf4c8643ccb7`. Static archive/pickle
instruction inspection found a tensor parameter dictionary; it did not
deserialize or execute the model. Loadability, runtime, feature compatibility
and dynamic privacy tests remain part of qualification.

The referee supplies only public history, own private reserves, visible opponent
reserves, hidden-reserve counts and deck counts. It must never supply the true
remaining deck or opponent's blind-reserved identities. External search must
sample lawful possibilities itself. The PPO source's apparently public/own-private
feature reads require a dynamic invariance test; source inspection alone is not
that test.

Qualification must cover:

1. For both local configurations, compare a separate adapter against frozen
   scientific records on 18 metadata-selected saved roots, one per fit, seat
   and phase, plus at most six privacy/short-support/tie fixtures. Choose by
   identity hash before inspecting latency or outcomes. Reproduce all offered
   trajectories on these bounded fixtures when claiming exact equivalence.
2. Exercise timely, late, malformed, illegal and hung synthetic players. Measure
   isolated end-to-end decisions, including repeated runs with the same inputs.
   Fix the timeout decision using the trusted supervisor's receive timestamp,
   never a child-supplied timestamp or when the referee later inspects its queue.
3. All 90 card identities, 10 nobles, colour ordering, resource counts, scores,
   seats, purchased bonuses and reserve provenance. Match the two-player base
   rules, end-of-round finish and fewer-purchased-cards tiebreak.
4. Legal buy/payment choices, reserve with and without gold, combined take/return,
   limited bank supplies, noble choice, terminal boundaries and forced pass.
   An opponent may voluntarily use a strict legal subset. An empty translated
   mask must fail qualification rather than silently choose a new policy.
5. Equal adapter inputs and choices under identical player RNG state when only
   inaccessible deck order or opposing blind-reserve identity changes. No
   true-state pointers or real game seed may
   enter the external player. Public actions taken by either player must remain
   representable even when absent from the external policy's selectable subset.
Apply the same observation-privacy requirements to the local adapters. Saved breadth roots
contain the original four-candidate prefix, allowing both fixed selectors to be
checked on common reached positions without claiming a fresh performance result.

Proposed operational envelope for that future qualification: four hours total,
one worker, at most 256 MiB new artifacts, at least 5 GiB disk free, at most 2 GiB
worker memory, and measured memory headroom before launch. Save partial/failure
evidence and stop at the envelope. This preparation does not execute it. Recheck
available space and memory before starting; the much larger data study needs
separate storage provision.
No new simulation-speed campaign or global backend change is needed.

If the exact breadth player exceeds the 120-second qualification allowance,
record that result. A higher-clock comparison or an anytime player is a new
explicit design; do not adjust the clock after seeing matchup scores. The
one-second track remains a separate deployment question until an entrant is
qualified there.

## Prospective strength comparison after qualification

After operational qualification, first screen the adapted PPO against raw 035
on 30 separate development seed pairs at the one-second allowance. This cheap
60-game comparison assesses basic competitiveness and supplies a cost estimate;
it does not establish elite strength. Keep those games out of later calibration.
Choose the strength opponent and final allocation using that development
evidence and measured cost. A weak control does not justify an expensive breadth
arena automatically. An inconclusive screen is not an invented failure gate;
report its estimate and uncertainty before deciding whether more calibration
would be informative.

A concrete research-scale design to price is **201 distinct initial seeds, each
played in both seats**, with 67 pairs per fixed local fit: 402 games total. This
is a conditional allocation, not a committed sample size for the unqualified PPO.
Derive and collision-check new source and player-RNG namespaces only when the
entrant/harness identities are ready to freeze. Keep the same initial deck and
nobles within each pair, but reset both players and use separate declared search
randomness. Balance fit and seat in execution order. Never publish true initial
seeds to the players before play.

First perform 12 disjoint operational pairs, four per fit, excluded from strength
analysis. Admission uses legality, timing, privacy, replay and cost only; keep
their matchup totals closed until the main schedule is locked. Qualification
or calibration games are never training data for the evaluated entrant.

Primary outcome is mean game score, win 1/draw 0.5/loss 0, averaged within seed
pair and equally across fits. Freeze a 50,000-replicate paired bootstrap within
fit, RNG seed and percentile rule before collecting outcomes. Report a 95%
interval, per-seat scores, timeout/illegal-action rates and time distributions.
With 201 bounded pair scores, the approximate worst-case normal 95% half-width
is 6.9 percentage points; this is calibration, not a high-powered test of a
small improvement. Do not add games after inspecting a near miss. Report other
opponents separately; predeclare adjustment if making joint superiority claims.

Check every actual game transition and result independently. Apply the
[verification policy](../engineering/verification_policy.md) to local search
records: all decision checks, eight of 128 worlds at every root, plus deterministic
full stress roots. Freeze sampling identities/rules and fault response. Report
external inference trace checks separately; they do not constitute replay of
unrecorded external search. Resource-stopped games remain unresolved; do not
score them as draws or silently substitute replacements.

Price the actual qualified configuration before freezing this schedule. Even
402 games averaging 60 turns with both players using the full 120 seconds would
require 804 serial collection hours before audits. That is a scenario, not a
runtime prediction or a bound on longer games. Existing 8-worker arena runtimes
are not evidence for a new isolated clocked harness.

## Decision and relation to the learning programme

The immediate engineering step is the bounded local-player and clock
qualification, followed by external-adapter qualification and a small competence
screen. A strength arena is premature while the entrant, clock and opponent
remain unqualified. No benchmark game,
external code execution, installation, human invitation or online play occurred
during this preparation.

Prepare the [larger-data comparison](larger_independent_data_cost_2026-10-18.md)
alongside this work. The previous 16-fold scaling and near-perfect TRAIN fit did
not establish dependable fresh improvement. External calibration can expose
practical weaknesses, while a substantially larger study remains a costed
hypothesis test. Neither branch warrants automatically promoting a ranker,
restarting compact AlphaZero, or claiming superhuman performance.
