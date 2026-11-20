# Recommendation after157: test whether learning strengthens the planner

2026-09-13. Recommendation only. The owner asked for the next step after the
verified157 pass. Read AGENT_CURRENT.md first, checked the closed artifacts and
training summaries, reviewed155/156 and the earlier conditional-search failures,
and obtained three independent read-only reviews. No158 implementation,
inference, fitting, simulation, gameplay or launch is authorized by this note.
Preserve all intentionally dirty and frozen work.

## Decision

Recommend one bounded158 experiment:

**Does155 become stronger when its simulated future moves are played by the
verified157 learned controller instead of greedy035?**

Keep155's real-root proposals,32 sampled worlds, terminal-WDL objective and tie
rules fixed. Replace only the continuation player used for both simulated sides.
Play the resulting planner directly against unchanged155 in402 fresh games,
with all three fixed157 treatment checkpoints represented prospectively.

157 established search-to-useful-fast-play transfer.158 would test the next
missing link: useful-fast-play-to-stronger-search. A pass would establish one
productive feedback step, not repeated improvement, full teacher compression,
equal-time superiority or superhuman performance. No new training is needed.

## What157 establishes and what remains open

All six fits were independently reproduced, including bitwise-identical saved
checkpoint tensors, and all4,800 fresh games received full replay. The final
result, audit and frozen/test/parent identities were rechecked for this review.

| Measure | Verified result |
| --- | --- |
| Treatment score against035 |57.5833%, paired95% CI54.7708%-60.3958%|
| Matched035-label control score |49.8542%|
| Treatment-minus-control performance against035 |+7.7292pp, paired95% CI+3.9167 to+11.5208pp|
| Treatment fits157001/157002/157003 |56.8125% /58.5625% /57.375%|
| Mean treatment decision wall time |10.939ms,2.456x matched raw035|
| Treatment selected updates |1220 /1220 /1144, maximum1220|

Every treatment fit scored above50% from both seats. These are three fixed
batch-order seeds, not a characterization of arbitrary training runs. The
controller remains two networks:035 proposes moves; the trained copy ranks the
offered groups. Its unrestricted learned argmax over all legal moves has not
been tested. Its value head has not learned a new value function.

157 is an improvement over035, not evidence that the fast controller matches
155's95.5% against035 or155's65.125% against084. The strong planner remains the
proper next opponent. Playing157 directly against155 would mostly measure the
remaining compression gap rather than whether its knowledge can help planning.

Longer fitting is the strongest cheaper alternative. One independent review
preferred a fixed4-versus12-pass test with fresh extended157-versus157 games.
That would be legitimate, but the evidence for it as the immediate bottleneck
is limited: mean treatment validation CE at passes1/2/3/4 was approximately
1.235156/1.225930/1.219736/1.216565. The last pass improved CE only0.26%, and
about94.4% of total CE improvement occurred by update64. Two endpoint selections
therefore do not establish serious undertraining. There are no saved training
loss curves to establish a train/validation gap. Exact future dose continuation
would require replaying the first four passes to recover AdamW moments, because
the saved checkpoints contain weights but no optimizer state.

Teacher-choice agreement stayed46.25%-46.54%, versus initial035's46.95%, while
fresh gameplay improved. Do not reintroduce an agreement gate or infer failure
from that noisy-label statistic. Roughly18.4%-18.7% of development choices
changed; most switches selected the strongest distinct035 alternative. Those
descriptive patterns neither locate all remaining strategic errors nor justify
a phase filter, removal of random proposals or selection of a favorite seed.

## Proposed158 design

### One behavioral intervention

The reference player is exact155 with fresh experiment RNG streams, keeping its
four-proposal maximum (smaller supports retained),32 shared observer-legal
worlds, greedy035 continuations for both actors, terminal WDL and baseline-first
ties. It is the same planner validated in155/156, not084 or a newly tuned proxy.

The candidate keeps that same real-root operator. In each sampled future, both
simulated actors use the complete157 treatment controller associated with the
game: frozen035 makes the original proposals, the frozen learned scorer sees
all concrete legal actions and maximizes scores within each offered canonical
group, and the selected original representative is played. Keep the original
proposal-order tie behavior. No unrestricted scorer policy and no recursive
calls to155 inside its continuations.

This changes the continuation behavior, including the consequences of157's
random proposal support. It does not separately identify strategic strength,
stochasticity or probability calibration as the causal ingredient. A clone-
continuation arm would further isolate teacher-label training in this role;
it is not part of this first practical comparison of complete planners.

### All three fixed fits, one bounded match

Use the already sealed treatment157001/157002/157003 selected checkpoints.
Before any preflight outcome, hash-assign201 fresh initial-seed pairs into three
equal groups of67. Each pair uses one fixed checkpoint for both seat games and
all of that candidate's simulated futures. This gives402 games total and134
games per checkpoint. Do not select157002 because it scored highest in157, mix
models between simulated turns, form an ensemble or choose a model by results.

The estimand is a uniform mixture of these three fixed planners across new
starting positions. It is not each model on the same starting-position panel,
not three independent population draws of a learning algorithm, and not603
independent seed pairs. Individual-fit results are descriptive; no best/worst
fit selection, additional veto or outcome-triggered panel extension.

Use12 off-panel preflight games: two initial-seed pairs per checkpoint, both
seats. Check actual worker dispatch, exact old155 equivalence,157 reload and
action equivalence, private-information invariance, legal/natural termination
and measured timing. Preflight is an integrity/timing check, not a strength
filter. Freeze all checkpoints, schedules, RNG rules, statistics and gates
before its outcomes; no scientific retuning in response to it.

### Randomness and lawful information

Use fresh random-stream namespaces independent of actual environment seeds.
Couple root proposals and belief worlds wherever the two planners reach the
same legal information, without allowing identity of the actual hidden deck to
influence proposal randomness. Give157's simulated proposals separate fixed
streams, keyed by simulation slot/actor and observer-legal information; couple
corresponding streams across forced root alternatives. Do not consume a shared
mutable RNG in candidate-dependent order or derive seeds from omniscient state
hashes. Exact namespaces and assignment hashes must be frozen at implementation.

Both simulated actors receive their own lawful observations: own blind-reserve
identities are known, opposing blind identities and hidden deck order are not.
Test/replay this invariance inside continuations as well as at actual roots.
Use the existing uniform legal-unknown-card world law; no new history posterior.

### Endpoint, accounting and stop

Primary endpoint: candidate score against155, draw=half. Average the two games
within each initial seed, then equally average the three checkpoint strata.
Bootstrap67 complete seed pairs within each stratum and average the three
stratum means,50,000 resamples using a new frozen bootstrap seed. Keep both
seats together and report the interval conditional on these checkpoints.

Require score>=55%, paired95% CI lower strictly>50%, positive mean final-score
margin, each aggregate seat>=50%, and all integrity checks. Individual-fit
estimates do not add separate gates. An interval too wide to establish the
criteria leaves the benefit unestablished; do not add games after inspecting it.

Retain normal/equal-turn endings and two consecutive sole-legal-PASS endings
with score/fewest-purchased-card adjudication. Fourfold repetition,60no-progress
turns, global turn400, illegal/missing action or exceptions are preserved
failures, never substituted outcomes or replacement games. Independently replay
every actual game, proposal, world and continuation for both players, on either
strength outcome. Shared engine/scorer primitives must be disclosed.

Require full repository tests, explicit frozen file inventory/parent chain,
exclusive execution locks and immutable records. Carry forward the tested
Windows progress-I/O handling and actual spawned-verifier tests before freeze.
Keep status reads compatible with atomic replacement. Both outcomes stop after
the full report; no automatic promotion, new labels, extra learning or successor.

## Cost and why a direct match

157's11ms controller is cheap for direct games but is invoked millions of times
when placed inside a planner. Its measured decision overhead is about2.46x035
in its own arena. It is not an exact simulation-latency multiplier because
state distributions and instrumentation differ.

A rough extrapolation from155's13.5-hour versus-fast035 collection/full-replay
pipeline, replacing one fast opponent with another planner and making one
planner's continuations about2.5x dearer, suggests roughly45-60hours for this
402-game comparison. Budget provisionally two to three days, with the estimate
refined by the complete off-panel timing probe. Do not promise a fixed duration.
There is no training or expensive new teacher-data collection beyond the match.

Both players retain32 worlds. A pass therefore establishes a strength benefit
at fixed rollout count, not equal wall time or equal neural compute. Any later
efficiency claim requires a separately designed matched-time/count frontier.
Do not silently cut the candidate's worlds or add reference worlds here.

A smaller independent-world root diagnostic is cheaper, but its evaluator is
policy-dependent:157 continuations evaluate the new objective,035 continuations
the old one. Fresh worlds remove one source of selection optimism without
settling repeated replanning.134 rejected a locally promising expanded player;
140/141 did not establish the proposed continuation-mismatch explanation. The
direct155/156 and157 gameplay successes now justify testing the complete
learning-to-planning link rather than using another conditional gate as a
substitute for actual playing strength.

This distinction also has primary research precedent: simulation policies can
be trained for useful search estimates, and their direct strength need not be
the appropriate objective. [Silver and Tesauro, ICML2009](https://research.ibm.com/publications/monte-carlo-simulation-balancing--1).
Classification-based policy-iteration analysis motivates the broader loop but
has assumptions and regret-weighted objectives not established by157 or this
proposal; it supplies no guarantee for our unweighted classifier and restricted
stochastic support. [Lazaric, Ghavamzadeh and Munos, JMLR2016](https://www.jmlr.org/beta/papers/v17/10-364.html).

If158 passes, the next decision would concern retaining the stronger operator's
choices in another learned generation and testing repeatability. If it fails,
the exact continuation substitution has failed this screen;157's fast-play
improvement remains valid. Longer fitting, better proposal coverage or different
search estimators remain distinct hypotheses. None is automatically launched.

## Closed157 provenance

- Result: `outputs/fast_selector_transfer_157/result.json`, SHA
  `aac9f292a92b31ae904b6fe7244f1a8ff57b05fd406d9d035d11bc94da18707a`.
- Main audit: `a5057a0aaf0e77c93f340b3046490ad2777b366e9aa8cf9720ad47348a90496c`.
- Final status: `3da68af5ee0a3752b78129d03848d85ba80758497758de9a267f4cd63b0dbae4`.
- Fits audit: `77fc545ba9d460731ec881ecb14baf8f11daf5c7158778728fc7eb6da5df31a8`.
- Selected lock: `4942eebb38de81b40d38315ec4abaf6b306f9340f4fd7051113250d4831044d8`.
- Experiment binding: `3db2246c23aa6e9692a604dd95aa2dc40934e6c024b7eb4aa7fd60474d1c9122`.

157 finished15:02:58 Paris,13September2026. Its monitor is paused; no active
experiment or158 launch is implied by this recommendation.
