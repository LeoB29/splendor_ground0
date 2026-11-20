# Corrected-rules external opponent qualification and 60-game screen

This is the implementation protocol for the user's authorization on 18 October
2026: resolve six card-cost disagreements from authoritative sources, qualify
the external opponent, and run the previously proposed cheap screen if
compatible. Freeze this protocol, code, models, schedule and qualification
receipts before any screen outcome. No training, larger-data collection,
402-game arena or change to historical frozen files is authorized here.

## Component and rule adjudication

The public [licensed Board Game Arena tutorial](https://en.boardgamearena.com/tutorial?game=splendor)
contains all 90 card faces and ten nobles. Its literal component database agrees
with the pinned external catalogue on all 90/10 semantic identities. The old
local catalogue agrees on 84/90 cards and 10/10 nobles. These six white-bonus
card costs are corrected only in the new referee:

| Historical ID | Correct cost |
|---|---|
| T1_05 | blue 2, black 2 |
| T1_07 | white 3, blue 1, black 1 |
| T2_10 | white 2, blue 3, red 3 |
| T2_12 | red 5 |
| T2_13 | red 5, black 3 |
| T2_14 | white 6 |

[Asmodee's partnership announcement](https://www.asmodee.co.uk/blogs/news/splendor-released-on-board-game-arena)
identifies the adaptation as the joint BGA/Space Cowboys version. The
[publisher's classic rulebook](https://cdn.svc.asmodee.net/production-unboxnowcom/uploads/2022/02/Splendor-EN.pdf)
visually corroborates T2_12 in its page-two setup illustration. The complete
primary evidence and independent literal-only checker receipt are saved under
`outputs/component_adjudication_20261018_01/`.

The tutorial embeds release 210916-2237. Its exact-release scripts returned
HTTP 403. Decoding it with the explicitly saved current release 260627-0953
JavaScript/CSS color mapping is a documented inference, corroborated by the
publisher illustration and full 90-face correspondence. Do not represent all
six faces as independently photographed from a publisher component sheet.

The [current publisher rulebook](https://cdn.svc.asmodee.net/production-spacecowboys/uploads/2025/10/SCSPL01EN_SPLENDOR_RULES_LIGHT.pdf)
also makes noble visits compulsory after any turn, with at most one per turn.
The historical engine awarded nobles only after a purchase. The new wrapper
adds eligible noble choices to non-purchase actions and awards them, including
correct full-round termination when those points reach fifteen. Costs, noble
timing and a new catalogue/rules identity are explicit. Historical IDs and
ordering remain stable; historical sources, checkpoints and records are intact.

This screen is a transfer test of frozen raw 035 under corrected rules. The
earlier four/eight-candidate correctness, timing and strength results concern
their historical rules and do not establish corrected-rules playing strength.

## Exact players and external adaptation

Local player: raw frozen 035, checkpoint SHA256
`4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`,
CPU inference without retraining. It receives a reconstructed state containing
only the acting player's lawful observation, with canonical dummy unknowns.

External player: `roeey777/Splendor-AI` commit
`95f84d2e6e839c0ef09ca97bdc3b3048a792fb0b`, checkpoint SHA256
`7fdb58b46d00305af42fad818b05549692e27ff1243cb63a76c6a858ba348df4`.
The separately named selected profile is
`official_rules_buy_and_token_alias_v1`. It preserves weights, normalization,
265 input features, 3,510 output indices, float32 feature rounding followed by
float64 CPU network arithmetic, and first-argmax choice. Safe weights-only
loading is required. The upstream entry point's two-versus-three-output unpack
bug is repaired explicitly. No external engine serves as the referee.

Two additional action adaptations are explicit:

1. Remove the upstream nonstandard seven-purchased-cards-per-color restriction
   only for already represented, officially legal purchase indices. Preserve
   canonical minimum-gold payment handling.
2. For pinned-source-legal token-collection indices lacking a literal official
   action, allow a deterministic official take/return representative only when
   its complete six-color net token change and noble choice are identical.
   These source indices can encode taking one/two colors with a nearly full
   hand; their representative takes the required three and returns tokens to
   reach exactly the same state. Existing literal mappings are preserved.
   No pass, reservation or purchase alias is introduced by this rule.

This is an adapted opponent: affected-state masks, normalized probabilities and
choices can differ from the published agent. The original seven-cap empty-mask
failure and the separate token-return literal-mapping failure remain saved.
No fallback policy, new neural output or arbitrary legal-index activation is
permitted. Blind reserves and some payment variants remain outside its chosen
legal subset. It must nevertheless correctly observe all legal opposing actions.

## Qualification before games

Require a successful selected-profile receipt, bound to its checkpoint,
catalogue and rules identity. Independently execute the pinned feature,
action-vocabulary, mask and neural definitions on fixtures. Verify exact
features, unmasked logits/value, adapted mask/probabilities and first argmax;
verify unchanged original decisions where the original mask is unaffected.
Constructed eighth-buy and token-return witnesses must have legal support.
Check token-alias equality against independent full-state transitions, including
nobles, terminal timing and unchanged card/deck ledgers. This is fixture
qualification, not a proof of support on every reachable state. An empty mask
at any later stage remains a hard prerequisite failure.

Run focused rules, action, neural, independent-ledger and harness tests before
freezing. Convert the already predetermined 18 core positions and five edge
positions into explicitly labelled counterfactual corrected-rules fixtures.
Their old choices are not asserted to be correct under changed costs.
On all 23 fixtures, run both players on the original and a hidden-state variant:
92 timed requests, 46 privacy pairs. Every response must be legal, timely and
identical within its privacy pair. No true seed, hidden deck order or opposing
blind-reserve face enters either player. No screen game starts if a gate fails.

## Frozen development screen and clock

Exactly 30 initial-world pairs, 60 games, with swapped player seats. Pair seeds
are full SHA256 integers from
`splendor_external_corrected_rules_20261018_development_v1:pair:<index>`.
Check collisions against discovered historical manifests, schedules and seed
registries of at most 8 MiB; retain the checked-file inventory and disclose any
oversize omissions. This scan does not claim to inspect every historical game
log. Pair order alternates which PPO seat runs first. Save true seeds only in
trusted referee evidence; never send them to a player.

Reset both policies for each game in one new supervised process, one numerical
thread on a verified performance core, normal priority, with no concurrent
player work or pondering. Both models are stateless with frozen inference
normalization. Charge initialization separately, capped at 60 seconds. Each
move has a real one-second deadline from before request serialization through
receipt of the complete response. Accept only timely legal responses. Preserve
the trusted process-clock receipt and bind it to the per-move evidence.

Player timeout, crash or illegal response loses that game and stays counted.
Referee defects, empty policy support, unexpected protocol/resource statuses,
initialization failure or resource exhaustion stop acceptance of the screen.
Do not replace games. Stop a game still nonterminal after 2,000 plies as
unresolved; there is no invented repetition, no-progress or forced-pass draw.

The new qualification-plus-screen execution has a four-hour wall ceiling, one
active worker under the shared execution lease, 256 MiB evidence ceiling with
8 MiB reserved for final receipts, 2 GiB worker RSS limit and 5 GiB free-disk
reserve. Before each worker require 1 GiB available physical memory and 3 GiB
commit headroom. Poll resource limits during execution. Preserve partial data
and all failures. These bounds do not authorize the deferred larger-data study.

## Results and independent acceptance

Score PPO wins 1, draws 0.5 and losses 0. Average within each seed pair and then
equally across 30 pairs. Freeze 50,000 paired bootstrap resamples, standard
library Random seed 2026101803510, and linearly interpolated 2.5/97.5 percentiles.
Report W/D/L, paired interval, both seats, forfeits and timely latency summaries.
Do not invent a post-outcome pass threshold or extend the sample on a near miss.
An inconclusive estimate is reported as such.

Every accepted move is checked by a separate dictionary ledger implementing
payments, collection/returns, reservation, refill, nobles and terminal scoring.
The final saved replay binds schedule/game identity, initial seed/hash, action
and observation hashes, actor progression, original clock receipts, deadline
flags, final state, forfeit attribution and independently derived scores.
All frozen inputs must pass final hash/size verification. A failed final check
marks the run failed and produces a nonzero exit even if 60 games were played.

The result can assess basic competitiveness and cost of this adapted opponent.
It cannot establish elite strength, promote a learner, or automatically justify
an expensive planner benchmark. Keep these development seeds out of any later
confirmation study and retain all historical qualification failures.
