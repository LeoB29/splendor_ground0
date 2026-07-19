# Agent Handoff

## 2026-07-19: Outcome sampling is correct but terminal weights are unusable

Experiment `048` tested the non-enumerating alternative required after `047`.
Protocol commit `d127a4c` froze the standard single-trajectory counterfactual
weight, 0.1 full-support behavior mixture, reduced exact checks, an explicit
finite learning-game contract, 256 fresh full-game trajectories, and strict
variance/ESS gates. Implementation commit `d34ef24` preceded evidence, and all
434 tests passed. The first evidence invocation was killed by the shell's
four-minute process limit before producing a result; the identical job restarted
from zero under a longer allowance and completed normally.

Estimator correctness passed decisively. Fixed-uniform estimates from 200,000
samples per traverser covered every eligible exact information set. Mean
absolute regret-component errors were 0.001851 and 0.000709, maximum errors
0.012526 and 0.004920, and earliest-information-set maximum errors 0.003568 and
0.002707. All were well inside the preregistered 0.010/0.050/0.010 limits.

Adaptive reduced-game MCCFR also passed. Three 100,000-iteration seeds reached
98.54%, 99.51%, and 98.54% exact-information-set coverage, with zero illegal or
unstable action sets and normalized positive regrets at most 0.00828. Every
current regret-matched profile had exact exploitability zero; seed 480046
repeated byte-identically. This establishes that the outcome-sampling math is
correct on a tractable game, while retaining the protocol's warning that the
current policy is not a deployment-average strategy.

The finite full-game contract was not the blocker. All 256 fresh trajectories
at seeds 480000--480255 ended through ordinary engine completion; there were no
forced deadlocks, repetition/no-progress/turn adjudications, illegal actions,
or nonfinite targets. They produced 7,915 rows (2,560 early, 3,840 mid, 1,515
late) with legal-action counts 1--470. Privacy, checkpoint identity, semantic
alignment, the repeated first 16 trajectories, and the repeated capacity-512
reservoir all passed. The reservoir serialized to 9,697,929 bytes.

The formal experiment fails only the full-game variance gate, but by enormous
margins. Overall ESS ratio was 0.00016263; phase ratios ranged from 0.000850 to
0.001575, all far below 0.01. P99 absolute importance weight was 2.92e22 versus
the 1e6 cap. Maximum log weight was 72.124, corresponding to roughly 2.1e31,
versus the `log(1e12)` limit 27.631. Split-half median and p95 log-weight deltas
were 1.017 and 2.598, also above their 1.0/2.0 bounds. The unbiased estimator is
thus dominated by extremely rare trajectories; a larger corpus or GPU does not
solve the exponential-horizon variance, and clipping would silently bias CFR.

Do not train on the stored schema/advantage reservoir, tune exploration, add
samples, or rent a GPU. Together, `047` and `048` close both terminal full-game
extremes: enumerating traverser actions explodes computationally, while sampling
one terminal path explodes statistically. The next justified CFR experiment is
a bounded public-belief subgame/continual-resolving target diagnostic. It must
freeze a short resolving horizon, opponent range/belief semantics, and an
information-legal leaf payoff before implementation; validate the leaf contract
independently and compare local regret targets against exact reduced-game and
matched continuation references. Do not reuse the failed learned value as an
unverified leaf oracle. Only stable local targets could authorize neural fitting.
Raw legal `035` remains champion, and BGA remains a separate future option.

Evidence is under `outputs/outcome_sampling_cfr_feasibility_048`. Result SHA-256
is `69c4a976fa121178269a31145c3aa645a7696177155bbab57ec02e35ddf478ff`;
advantage-reservoir SHA-256 is
`52893897cdcdde4f80ddd82201e1ff7fed4a28ade81f164e7225664f865c3277`.

## 2026-07-19: Full-game terminal external Deep CFR is computationally mismatched

Experiment `047` tested the CPU prerequisite authorized by corrected CFR
reference `046`. Protocol commit `c944b4f` froze an information-legal
perfect-recall history representation, variable semantic action rows, bounded
reservoir memories, exact terminal rules, and four production-engine external
traversals. It performed no training, gameplay, or promotion. The implementation
adds `src/splendor_ai/cfr/full_game.py`, a reproducible runner, and six focused
integrity tests. The entire suite passes at 429 tests.

Every representation-side gate passed. Eight fresh loop-safe width-64 `023`
mirror games at seeds 470000--470007 all ended through ordinary completion and
produced 498 schema-only rows (160 early, 240 mid, 98 late), with legal-action
counts from 2 to 186. The exact checkpoint hash/config loaded, and 63 batches of
eight passed finite CPU forward and target-alignment checks. Hidden-deck and
blind-reserve privacy, public reveal timing, state/action permutation invariance,
and privileged-entity rejection all passed. Two capacity-128 Algorithm-R
reservoirs repeated byte-identically, round-tripped losslessly, and serialized
to 4,653,242 bytes, well under 32 MiB.

The formal result nevertheless fails because none of four standard terminal
external traversals completed. Three exhausted exactly 25,000 nodes, at maximum
depths 130, 170, and 262. The fourth encountered the separately invalid
400-turn safeguard after 9,929 nodes. Some traverser nodes exposed up to 550
semantic legal actions. There were zero illegal actions, partial attempts
correctly exposed zero trainable rows, and the first attempt repeated exactly
after excluding elapsed time. Thus the result is not caused by the historical
forced-deadlock mistake: only completed or two-sole-PASS deadlocks are terminal
utilities, while node/turn cutoffs remain invalid.

Do not rent a GPU, train neural Deep CFR on partial trees, increase the node cap
blindly, or revive a learned value bootstrap. The blocker is structural:
external sampling still enumerates every traverser action across a long game,
and uniform policies can enter very long nonterminal behavior. The next
authorized direction should be separately preregistered outcome-sampling CFR
with (1) an explicit finite-game loop/turn utility, (2) full-support behavior
sampling and unbiased importance ratios, (3) an exact reduced-game comparison
to corrected `046`, and (4) full-game variance, effective-sample-size, legality,
and reproducibility gates. Only a pass may authorize neural fitting and GPU
rental. Raw legal `035` remains champion; BGA remains unused.

Evidence is under `outputs/full_game_deep_cfr_feasibility_047`. Result SHA-256
is `7e59f4b8d91c2b2c3be2c593820706cc4167987aa482469b93570abd9e8dfd44`;
the serialized reservoirs SHA-256 is
`9a31a1cd27717c418b5f5be0570bc48382e1f5a539b28f7517786f7555f2d7cd`.

## 2026-07-19: Corrected MCCFR average policy passes every adversarial gate

Experiment `046` repaired the exact defect localized by `045`. The legacy
external solver accumulated average strategy at the traverser's enumerated
nodes. The standard two-player simple rule instead accumulates the sampled
opponent's strategy during the other player's regret pass, where visit
frequency supplies own reach weighting. The new solver is separately named so
`044/045` remain reproducible. Matched-seed tests prove the repair leaves every
regret and current regret-matched strategy unchanged and changes only averaging.

Protocol commit `e20f249` and implementation commit `368df81` preceded
evidence. All 423 tests passed. The run reconstructed the exact `044` hash,
solved corrected external MCCFR for 50,000 iterations at seeds 440046--440050,
and repeated 440046 byte-identically. Every run had zero illegal/unstable
actions, finite metrics, low regret, and 95.63--99.03% exact-information-set
coverage.

All ten gates passed with large margins. Corrected average exploitability was
0.000374336, 0.000383431, 0.000470736, 0.000336231, and 0.000401162; median was
0.000383431, versus frozen limits of 0.020 per seed and 0.010 median. Exact CFR
was 0.000177429. The worst one-player deviation incentive was 0.000618184,
cross-play differed from exact self-play by at most 0.000283, and missing-set
exact substitution changed exploitability by zero.

Rare-state robustness also passed decisively. Replacing every corrected
strategy with fewer than 100 observations by its exact counterpart changed
exploitability by at most 0.00006302, far below 0.005. This contrasts with the
legacy average's 0.293314 exploitability and establishes that the old failure
was an averaging implementation defect, not a failure of CFR or insufficient
sampling iterations.

`046` authorizes a separately preregistered full-game neural Deep CFR
feasibility pilot, not gameplay or promotion. The next step is to freeze the
full-game information-state/action encoding, external traversal and reservoir
memory semantics, network targets, CPU collection smoke, GPU-fit budget, and
offline rare-state/held-out gates. Preserve opponent-node strategy-memory
sampling exactly. Raw legal `035` remains champion. Evidence SHA-256 is
`bc025be45cbb8afd1c907f4e74b587a3647f2414ac11e5d5aba55b2223622449`.
BGA remains a separate future authorized-data option; none was used.

## 2026-07-18: Exact best response exposes rare-state MCCFR averaging defect

Experiment `045` tested whether the two failed raw-policy-TV gates in `044`
were strategically meaningful. Protocol commit `d038a47` froze exact policy
hashes, information-set-consistent best-response semantics, exploitability,
reach/value-weighted discrepancy loss, and cross-play gates. Implementation
commit `6ac8b4e` added the analyzer and tests. All 419 tests passed before the
70.2-second evidence run, and both reconstructed `044` policy hashes matched.

The exact CFR reference passed easily at exploitability `0.000177429`. Every
best response committed to one legal semantic action across all hidden
histories in an information set, the complete analysis repeated identically,
and finiteness/nonnegative-deviation/zero-sum checks passed.

The external sampled average policy failed badly at exploitability `0.293314`
versus its `0.020` limit. Its self-play value was nearly zero, but player 0's
best response achieved `0.558514` and player 1's achieved `0.0281131`. Both the
absolute exploitability and external-minus-exact gap gates failed. This is the
important lesson: self-play or play against an equilibrium-like policy can hide
a large vulnerability because neither opponent deliberately enters the weak
branch.

The seven high-TV states that motivated `045` were not the problem. All passed
the frozen reach-or-value-loss rule; total conservative weighted loss was only
`0.0000171397` for player 0 and zero for player 1. Both cross-play values were
within `0.000523` of exact self-play. Raw policy non-uniqueness was harmless,
but a separate defect existed outside that observed-at-least-100 set.

A bounded post-result localization reconstructed the policies without changing
any gate. Filling all eight missing external information sets from exact CFR
left exploitability unchanged at `0.293314`. Replacing only stored averages
with fewer than 100 observations by exact strategies reduced it to `0.000401`;
thresholds of 1,000/10,000/25,000 yielded approximately
`0.000249/0.000257/0.000267`. The final regret-matched external policy itself
measured exactly zero exploitability on this reduced game. Thus regret learning
converged, but the visit-accumulated average policy was badly estimated in rare,
adversarially reachable information sets.

Do not abandon CFR, loosen `045`, add iterations blindly, deploy the current
regret policy, or rent a GPU. The next experiment should preregister a correct
tabular average-policy estimator following MCCFR reach/sampling weighting or
Deep CFR's explicit strategy-memory approach, then replicate over multiple
fixed seeds with exact exploitability and rare-band gates. Only that pass can
authorize a separately frozen full-game neural Deep CFR feasibility pilot.
Raw legal `035` remains champion. Evidence SHA-256 is
`528e0b0582d2c39c752e88d0cfb5c82188fba510fa0caba8dd915b281a691701`.
BGA remains a separate future authorized-data option; none was used.

## 2026-07-18: Reduced Splendor CFR gate narrowly fails policy-shape replication

Experiment `044` implemented the first direct regret-based self-play machinery:
a versioned history-aware information-state key, a four-action imperfect-
information Splendor duel driven by the production engine, exact alternating
CFR, and external-sampling MCCFR. Protocol commit was `c506cba`; implementation
was checkpointed at `6406c10`. The initial evidence invocation exceeded the
shell limit before writing a result because immutable information keys were
recomputed millions of times. Commit `e072cd9` cached those graph lookups
without changing the game, solvers, iteration counts, seed, metrics, or gates;
the evidence run then restarted from zero. All 415 tests passed first.

The reduced game exhaustively contains 985 nodes, 750 terminal paths, and 206
information sets. Every path ends after exactly four player actions. All
privacy checks passed, including owner-only knowledge of blind-reserved cards,
and neither solver encountered an illegal action or unstable information-set
action set. Both deterministic reruns were byte-identical.

Exact CFR passed every gate after 10,000 iterations. Normalized positive regret
fell from 0.00292944/0.000373672 at iteration 100 to
0.0000292944/0.00000373672. External MCCFR also had very low regret after
50,000 seeded iterations (0.000137200/0.00000775862), covered 96.1165% of exact
information sets, and matched the exact root policy at TV 0.000211297.

The formal result is still a fail because deeper policy shapes did not
replicate closely enough. Across 30 shared information sets with at least 100
external observations, mean TV was 0.1503728 against a maximum of 0.15 and p90
TV was 0.6666667 against 0.30. A post-result descriptive inspection found that
several largest differences were uniform exact mixtures versus concentrated
sampled choices. Together with tiny regrets and root agreement, this may mean
the actions are strategically equivalent and the equilibrium is non-unique;
it may instead expose a sampled-solver issue. `044` cannot decide that because
it did not preregister exploitability or value-equivalence metrics.

Do not retroactively pass `044`, start neural full-game CFR, or rent a GPU. The
next experiment should preregister exact best-response exploitability and
reach/value-weighted action-equivalence diagnostics on this same reduced game.
Only a clean strategic-equivalence result should authorize the full-game neural
pilot. Evidence is under `outputs/deep_cfr_viability_044`; result SHA-256 is
`6b7edee724b98cc149d8bbc1826a75fa7592e246a7d85ea384e703734fbaf632`.
Raw legal `035` remains champion.

BGA/expert replays remain a possible later warm start or independent validation
source if an authorized and reproducible acquisition path is found. No BGA
data was collected or used in `044`.

Simple explanation for the project status: we have not entered full-game
self-play because self-play only improves a bot when its learning rule can
reliably tell which new choices are better. Previous value, search, and
trajectory teachers failed fresh gameplay gates, so using them would risk
amplifying the bot's own mistakes. CFR now works on the small test game, but its
sampled form still differs from the exact reference in some later positions;
we must prove those differences harmless or fix them before scaling to GPU
training and full Splendor self-play.

## 2026-07-18: Large rolling-horizon robustness gate resolves near parity

Because identical 100-pair `041` and `042` blocks estimated score-delta BMRH at
58.5% and 48.0%, respectively, `043` froze a final large estimate rather than
accepting either small block as definitive. It excluded both historical blocks
from promotion evidence and preregistered five non-contiguous fresh 100-pair
bands, with all games required regardless of interim results. Protocol commit
`feb7ae1` and implementation commit `2212606` preceded evidence; all 409 tests
passed.

The complete 500-pair / 1,000-game direct experiment ran cleanly. All games
ended through ordinary `completed`; there were zero deadlocks, invalid
terminations, timeouts, or stalls. Budget accounting was exact: 31,954 planning
decisions, 1,022,528 sequence evaluations, and 3,025,656 simulated actions.

Candidate finished 505-494-1 against raw legal `035`: 50.55% paired share,
CI95 [47.75%, 53.40%], and paired margin +0.211, CI95 [-0.141, +0.561]. Block
shares were 47.00%, 52.50%, 49.25%, 50.00%, and 54.00%; only two of five were
strictly above parity. Candidate share split 53.40% from seat 0 versus 47.70%
from seat 1, which the paired design balanced. Thus aggregate share, confidence,
and four-of-five consistency gates failed; only the positive point-margin gate
passed. Broad games remained locked.

This is the final resolution of `041`/`042`: the operator is near parity and
the 58.5% block was not a stable effect. Across all three experiments the
descriptive, non-promotional total is 718-681-1 / 51.32%, consistent with the
large fresh estimate. Do not collect more BMRH strength games, tune this
formulation, or use it as a teacher. Raw legal `035` remains champion. Evidence
is under `outputs/rolling_horizon_robustness_043`; exact hashes are in the
protocol and `AGENT_CURRENT.md`.

## 2026-07-18: Deadlock-valid rolling-horizon replication rejected

The audit after `041` established that seed `403078` was a genuine locked
Splendor position: each player had exactly one legal action, `PASS`, on
consecutive turns. The repository had already tested and adopted two forced
passes as a clean `forced_pass_deadlock`, adjudicated by score and the standard
fewest-purchased-cards tie-break. Thus `041` cannot be retroactively passed
under its literal frozen rule, but its one non-`completed` match was not a
planner failure.

Experiment `042` corrected only that terminal taxonomy and ran an entirely
fresh replication. Protocol commit `c8ce1bb` and implementation commit
`e6edd6e` preceded evidence; all 405 tests passed. Its 100 paired seeds / 200
games all ended through ordinary `completed`, with zero deadlocks, invalid
terminations, timeouts, or stalls. Candidate accounting was exact at 6,422
planning decisions and 205,504 sequence evaluations.

The strength signal did not replicate. Score-delta BMRH went 96-104-0 versus
raw legal `035`: paired share 48.0%, CI95 [41.0%, 55.0%], and paired score
margin -0.43, CI95 [-1.26, +0.39]. It failed all three strength gates. Candidate
went 46-54 from seat 0 and 50-50 from seat 1; paired seeds contained 23
candidate sweeps, 50 splits, and 27 baseline sweeps. Broad gates remained
locked as required.

This resolves the ambiguity correctly: `041` is positive evidence on one seed
block, but it is not stable enough to promote after a fresh direct replication
lost. Do not pool `041` and `042` for promotion, tune this BMRH formulation, or
use it as a self-play/search teacher. Close rolling horizon and retain raw legal
`035` as champion. Evidence is under
`outputs/deadlock_validated_rolling_horizon_042`; exact hashes are in
`AGENT_CURRENT.md` and the protocol.

## 2026-07-18: Score-delta rolling horizon passed strength, failed integrity

`score_delta_rolling_horizon_041` froze the exact score-delta BMRH control from
`040` and tested it on 100 fresh paired seeds / 200 games against raw legal
`035`. Protocol commit was `25350bd`, implementation commit was `7e9765a`, and
the pre-evidence full suite passed 401 tests.

The planner was statistically stronger: 117-83-0, paired score share 58.5%
with CI95 [52.0%, 65.0%], and mean paired final-score margin +0.91 with CI95
[+0.16, +1.64]. It completed 6,364 planning decisions and exactly 203,648
sequence evaluations. All strength, count, and exact-budget gates passed.

The formal gate nevertheless failed because one game did not have literal
`completed` termination. Seed `403078`, candidate seat 0, reached the match
runner's `forced_pass_deadlock` adjudication at turn 30 and score 2-0; the
candidate was adjudicated winner. It was not a timeout or stall, but the frozen
integrity rule allowed zero non-completed games. Consequently the runner did
not generate any shallow, `016`, or `023` broad block.

Honor the preregistration: keep `035` champion, do not repair or tune `041`
after this result, and close rolling horizon as the active branch. Preserve the
nuance for future strategic review: unlike most rejected directions, this
operator passed every statistical strength gate and was rejected only by one
forced-pass terminal-contract event. Exact hashes are in `AGENT_CURRENT.md` and
the `041` protocol; evidence is under
`outputs/score_delta_rolling_horizon_041`.

## 2026-07-17: Event-value rolling-horizon diagnostic rejected

After `039` closed explicit hierarchical options, the project tested a genuinely
different Splendor-specific planner under the frozen
`event_value_rolling_horizon_040` contract. The implementation evolves two
root-player action-rank decisions with `(1+1)` branching mutation, uses frozen
legal `035` for intervening opponent actions, evaluates 32 sequences over two
observer-relative determinizations, and never chooses cooperative opponent
moves. A matched control changes only the evaluator from published five-event
weights to root-relative score delta.

The implementation was checkpointed before evidence at `6856117`; the complete
suite passed 397 tests. The four-game operational smoke passed every gate in
43.5 seconds with exact budgets and no failed games. The frozen 30-pair/60-game
control also completed without a stall, timeout, or other non-completion, but
event-value lost 15-45-0 to score-delta. Its paired score share was 25.0%, CI95
[15.0%, 35.0%], and mean paired final-score margin was -3.9833, CI95
[-5.3000, -2.6667]. The raw-`035` gate remained locked as required.

Reject the fixed published event weighting. Do not tune its weights, horizon,
or budget after seeing this block. This result does not reject the implemented
rolling-horizon mechanism: the score-delta control dominated event-value, but
its strength against raw `035` is still unknown and no further gameplay is
authorized by `040`. `035` remains champion. Evidence is under
`outputs/event_value_rolling_horizon_040`; hashes are recorded in the protocol
and `AGENT_CURRENT.md`.

## 2026-07-17: Outcome-blind plan persistence preregistration

The suggested high-budget search-oracle diagnostic was discovered to duplicate
the completed PUCT512 action-quality ablation. That prior run already used 320
disjoint states and all-legal matched-continuation returns; PUCT512 slightly
worsened regret and oracle agreement. It must not be repeated as another budget
sweep.

The genuinely new experiment is `outcome_blind_plan_persistence_038`, specified
in `docs/experiments/outcome_blind_plan_persistence_038.md` and implemented by
`src/splendor_ai/training/run_plan_persistence_diagnostic.py`. It uses two fresh,
game-disjoint balanced halves and compares every chosen token take with all
alternative legal takes from the exact same public state, using a later
already-visible/self-reserved card commitment as an outcome-free target. It
stores no outcomes and strips hidden information. Training and promotion are
forbidden. Only a replicated full pass permits a separately preregistered
hierarchical option design; failure closes the proposed hierarchy.

`038` completed and passed every gate. The 256 games were all complete, exactly
balanced in both halves, and yielded 8,059 actor turns. Discovery contained 630
anchors with game-cluster chosen-minus-alternative progress 0.7031, CI95
[0.6638, 0.7433], and rank percentile 0.7895, CI95 [0.7772, 0.8019]. The untouched
confirmation half contained 648 anchors with progress 0.7002, CI95
[0.6596, 0.7402], and rank percentile 0.7879, CI95 [0.7751, 0.8004]. Multi-take
episodes replicated strongly: 58 at 0.6568 [0.5770, 0.7371] and 65 at 0.5863
[0.4947, 0.6709]. Every opponent/seat cell and every phase was directionally
positive.

The stored corpus was regenerated after a privacy audit removed even the
constant `winner: null` key. Resume validation now rejects outcome fields,
nonempty hidden decks, and opponent-private reserves; the final raw corpus
contains none. Interpret the pass narrowly: 1,136/1,278 anchors point to the
next own-turn purchase, 126 span two turns, and 16 span three. This proves a
replicated card-acquisition option signal, not general long-horizon planning.
The only authorization is to preregister a state-only versus option-conditioned
predictive probe. Do not train a gameplay policy or run strength games yet.

That probe is now preregistered as `option_predictive_probe_039`; see
`docs/experiments/option_predictive_probe_039.md` and
`src/splendor_ai/training/run_option_predictive_probe.py`. It uses a new 512-game
outcome-blind corpus at seeds 391000--391511, with exact 192/64/256
train/validation/confirmation splits. The frozen `023` transformer supplies
contextual state, candidate-card, and take embeddings. Two identically
initialized and parameterized heads differ only because the baseline receives
a zero option vector and the conditioned probe receives the selected-take
embedding. Only heads train, and confirmation is not loaded until both epochs
are frozen. The gate requires strong overall NLL/top1/MRR improvement,
multi-turn improvement, and no major-stratum collapse. A pass authorizes only a
separate supervised high-level/low-level hierarchy reconstruction experiment.

`039` completed and failed decisively. All 512 privacy-clean games completed;
the data yielded 934/352/1,216 train/validation/confirmation examples. Both
heads selected epoch 22. The option head was worse on validation NLL (2.0841
versus 2.0635) and on untouched confirmation: NLL 1.9952 versus 1.9784, top1
31.58% versus 32.48%, and MRR 0.5140 versus 0.5184. The game-cluster paired NLL
delta was +0.01790, CI95 [+0.00132, +0.03451], so conditioning significantly
regressed calibration. Top1 and MRR deltas were negative and unresolved.

The 113 multi-turn examples did not rescue the hypothesis: option NLL worsened
1.26%; top1 gained 2.65 points, but its cluster CI touched zero; MRR gained only
1.28 points with a CI spanning zero, below the frozen +2-point gate. Nine gates
failed, including major-stratum noncollapse. The correct synthesis is that
`038` found a real cost-directed action motif, while `039` showed it is not an
incremental latent variable beyond the public state/card representation. Close
hierarchical options; do not tune this head, train a gameplay hierarchy, or use
more compute on this formulation. `035` remains champion.

## 2026-07-17: Start-role value v2 rejected; value/PUCT branch closed

Experiment `037` completed under the preregistered contract. Epoch 1 was the
best of six, but its validation selection score improved only 0.207% versus the
zero-update v1 initialization (required 0.5%). Its overall MAE was marginally
worse than v1 (0.283790 versus 0.283440). Starting-player MAE worsened 1.12%
(0.276430 versus 0.273370), while second-player MAE improved only 0.81%
(0.291216 versus 0.293601); the frozen gate required at least 1% improvement in
both seats. Both seat-correlation gates and every original offline gate passed,
including ordering, calibration, no-collapse, and successor consistency.

The runner correctly stopped before fresh confirmation and gameplay. No v2
deployment checkpoint was written, no PUCT block ran, and `035` remains the
overall champion. Evidence is in
`outputs/start_role_public_belief_value_v2_037/validation_selection.json` and
`gate_final.json`. This is the planned terminal result for the current
public-belief value/PUCT bootstrap. Do not rescue it with threshold changes,
extra epochs, learning-rate tuning, or more compute. Move next to a separately
preregistered hierarchical plan/option diagnostic, beginning with whether
information-legal multi-turn intent is persistent and predictive enough to be
a useful decision variable.

## 2026-07-17: AlphaGo reconstruction lessons and scaled value preregistration

The Dwarkesh Patel / Eric Jang transcript (`https://www.dwarkesh.com/p/eric-jang`)
was reviewed against this project's evidence. The durable lessons are:

- AlphaGo's core loop is search-backed supervised policy iteration, not generic
  trajectory policy gradient. First prove that search improves the current
  network; then search every state, save the full visit distribution from both
  wins and losses, train, and recollect synchronously.
- Value accuracy is a precondition. Limited-budget MCTS with an inaccurate value
  function can degrade a strong policy, which matches this project's weak PUCT
  and failed one-shot-search gameplay results.
- Start from the strongest compatible representation and stay near successful
  play. Do not confuse random initialization with scientific cleanliness.
- Splendor requires a public-belief analogue, because hidden deck/refill state
  makes naive Go-style perfect-information search invalid.
- Lateral hypothesis changes are warranted when repeated local tuning fails.
  Here the new evidence is not a learning-rate tweak: `032`+`033` offer 4,096
  balanced fresh games versus only 240 games in legal value experiment `029`.

Accordingly, experiment `scaled_public_belief_value_036` was preregistered before
training. Its exact splits, distributional margin/WDL target, phase balancing,
fresh confirmation, offline gates, and conditional PUCT sequence are frozen in
`docs/experiments/scaled_public_belief_value_036.md`. The implementation is
`src/splendor_ai/training/run_scaled_public_belief_value.py`. A direct PUCT win
only authorizes the later synchronous AlphaGo loop; it does not automatically
replace champion `035`.

Experiment `036` completed and was rejected offline without gameplay. It selected
epoch 3 after 113,248 train / 16,234 validation positions and generated a fresh
512-game confirmation set (16,164 actor positions, exact eight-cell group
balance, zero failures). It substantially beat the heuristic on aggregate error,
ordering, correlation, calibration, and midgame ordering. The two failures were
the preregistered early-phase no-collapse gate (6.7% MAE regression vs a 5%
ceiling) and successor policy/value top-three consistency (48.05% vs a 50%
floor). All other gates passed. No PUCT block ran. Preserve this as evidence that
the scaled belief-value direction has real signal but is not yet a fully safe
search leaf; do not retrospectively waive the two narrow failures.

The next frozen experiment is `036b`, a no-training reliability blend. Its 12
fixed candidates combine four phase profiles with three WDL-confidence floors,
are selected only on the old validation partition, and must pass the complete
unchanged `036` gate set before a new confirmation corpus is even generated.
Fresh confirmation uses seeds 381000-381127. See
`docs/experiments/reliability_blend_value_036b.md` and
`src/splendor_ai/training/run_reliability_blend_value.py`.

`036b` completed. The selected `(0, 0.5, 0.75, 1)` phase ramp passed all
validation and fresh-confirmation offline gates, repairing both `036` failures.
The resulting PUCT64 was directionally better than heuristic-leaf PUCT64 but not
resolved: 108-92, 54.0% share, CI95 [48%, 60%], +0.595 score margin, zero
failures. It failed the preregistered 55% and CI gates, so raw-035 evaluation was
correctly skipped and the AlphaGo loop was not authorized. Do not pool or extend
this block post hoc merely to cross 55%; `035` remains champion.

The `036b` seat audit found 61% candidate wins from seat 0 but 47% from seat 1.
This led to an objective contract finding: start-player role is present in the
canonical public-belief key but absent from v1 entity tensors. Experiment `037`
is the final value bet and is preregistered in
`docs/experiments/start_role_public_belief_value_v2_037.md`. It keeps v1 frozen,
adds a zero-initialized explicit start-role embedding that initially reproduces
`036` exactly, trains with differential learning rates, and requires both fresh
per-seat offline improvement and at least 50% gameplay share from each seat. If
it fails, close this value/PUCT bootstrap rather than tune further.

Short current-state handoff: read `AGENT_CURRENT.md` first.

This file is now the long historical record of the project. Keep detailed experiment history here, but put the active goals/takeaways/next steps in `AGENT_CURRENT.md`.

This file is the shortest path for a new agent to get productive in this repo.

## Objective

Build a Python AI for the `Splendor` base game that can eventually reach superhuman play through exact simulation, warm-start training, and later self-play.

Current project target:

- `2-player only`
- `legal observation only`
- full legal action space
- final benchmark intent: eventually beat very strong bots and human play fairly
- main training machine: `Windows + NVIDIA RTX 5070 + CUDA`

## Locked Decisions

- Rules authority: the user supplied the base-game rulebook PDF and asked that implementation not deviate from it.
- Rule clarifications explicitly confirmed by the user:
  - distinct token take can be `3`, or `2/1` if fewer colors are available
  - if no take/reserve/buy is legal, the player must `PASS`
- Warm start from heuristics/search is allowed.
- Inference target is around `1 second per move`.
- AMD/Windows/DirectML is only a compatibility path, not the main training route.

## What Exists

Core engine:

- exact base-game data
- deterministic seeded setup
- legal action generation
- transition logic
- noble resolution and end-of-round handling
- replay-oriented diagnostics

Model/training stack:

- fixed flat action codec and legal-action masks
- 295-float public observation tensor
- replay export to JSONL
- supervised policy/value MLP
- supervised training CLI with train/validation metrics, progress logs, final-vs-best-validation checkpoint benchmarking, and saved metrics JSON
- multi-corpus replay loading and repeated `--replay-path` support
- checkpoint-driven replay corpus generation with checkpoint/fallback provenance metadata
- supervised replay filtering by per-game checkpoint fallback trigger count
- optional pre-action replay state snapshots for reconstructing exact training positions
- shallow-search replay action improvement tooling for policy-target relabeling

Bots/eval:

- random legal bot
- greedy heuristic bot
- shallow search bot with anti-token-churn loop penalties
- match runner
- checkpoint benchmark CLI with saved per-game JSON output
- explicit benchmark loop diagnostics: `repetition_cutoff`, `no_progress_cutoff`, per-game termination metadata
- loop-aware checkpoint inference fallback with per-game trigger counts and CLI controls
- replay corpus CLI supports `checkpoint` bot pairings for champion-vs-baseline/champion-vs-champion probes
- Tkinter GUI for human-vs-bot inspection

Infra:

- Windows/CUDA bootstrap script
- environment verify script
- GitHub-friendly `.gitignore`

## Current Reality

Latest continuation:

- Added value-risk audit tooling:
  - `run_value_risk_audit.py`
  - `src/splendor_ai/training/run_value_risk_audit.py`
  - `tests/test_value_risk_audit.py`
- Purpose: compare a neural scorer's top action against plain PGS, while using `value_model_006` only as offline outcome metadata and as a separate "would outcome-PGS change this?" diagnostic. This avoids treating the value head as a live teacher after the failed seat1 confirmation.
- Focused verification passed in the restored Anaconda environment with `PYTHONPATH=src`:
  - `python -m pytest -q tests\test_value_risk_audit.py tests\test_search_value_diff.py tests\test_policy_search_disagreement.py --basetemp .codex_pytest_tmp_value_risk`
  - result: 6 passed
  - `python -m py_compile run_value_risk_audit.py src\splendor_ai\training\run_value_risk_audit.py`
- Ran bounded depth-2 late-seat1 audits over the current-generation value corpus (`max_rows=1200`, analyzed 234 filtered rows):
  - `outputs/analysis/value_risk_audit_022_late40_seat1_sample1200.json`
  - `outputs/analysis/value_risk_audit_022_late40_seat1_sample1200_rows.jsonl`
  - `outputs/analysis/value_risk_audit_016_late40_seat1_sample1200.json`
  - `outputs/analysis/value_risk_audit_016_late40_seat1_sample1200_rows.jsonl`
- Headline result:
  - `022` vs plain PGS: 53/234 disagreements (`22.65%`), mean plain-PGS regret of policy action `372.5`, max `2984.0`.
  - `016` vs plain PGS: 56/234 disagreements (`23.93%`), mean plain-PGS regret of policy action `359.3`, max `3090.0`.
  - Outcome-PGS changed the plain-PGS action on the same 41/234 rows (`17.52%`) for both audits, matching the earlier search-value diff.
  - Value/outcome metadata rarely confirmed the neural disagreement: policy matched outcome-PGS on only 7/53 `022` disagreements and 8/56 `016` disagreements.
  - `016` and `022` chose the same top action on 215/234 rows. They shared 44 plain-PGS disagreement rows, so the issue is mostly inherited/shared neural-vs-PGS behavior, not a new `022`-specific failure.
- Interpretation:
  - Do not train a targeted correction model from this audit. The value-confirmed correction set is only 7-8 rows, far below the "few hundred clean rows" threshold and not reliable enough for a supervised/ranking run.
  - The biggest high-regret neural-vs-PGS disagreements are usually not supported by outcome-PGS; in many cases outcome-PGS agrees with plain PGS against the neural policy.
  - The next useful step should be either a broader, non-value teacher improvement loop or a more direct PGS-vs-policy disagreement diagnosis, not a value-risk targeted training run.
- Follow-up non-value PGS disagreement diagnosis:
  - Extended `run_policy_search_disagreement.py` / `src/splendor_ai/training/run_policy_search_disagreement.py` with:
    - optional `--search-policy-prior-checkpoint`, `--search-policy-prior-device`, and `--search-policy-prior-weight` so it can compare against PGS rather than only unguided shallow search;
    - `--player-id-filter`, `--phase-filter`, and `--min-turn-index`;
    - score-margin buckets in the summary.
  - Focused tests passed for the extended analyzer.
  - Ran PGS016 teacher diagnostics with depth 2, weight 3000, no value head:
    - `outputs/analysis/policy_pgs016_disagreement_022_late_bothseats_sample1800.json`
    - `outputs/analysis/policy_pgs016_disagreement_016_late_bothseats_sample1800.json`
    - `outputs/analysis/policy_pgs016_disagreement_022_allphase_sample1200.json`
    - `outputs/analysis/policy_pgs016_disagreement_016_allphase_sample1200.json`
  - Late/both-seat sample, `min_turn_index=40`, 692 analyzed:
    - `022`: agreement `0.7486`, 174 disagreements, 7 missing from searched candidates.
    - `016`: agreement `0.7182`, 195 disagreements, 1 missing from searched candidates.
    - Seat split for `022`: seat0 agreement `0.7312`, seat1 `0.7659`.
    - Seat split for `016`: seat0 agreement `0.7197`, seat1 `0.7168`.
    - Interpretation: late seat1 is not where `022` looks worse against PGS016; `022` is actually more PGS-aligned there than `016`.
  - All-phase sample, 1,200 analyzed:
    - `022`: agreement `0.7067`, 352 disagreements.
    - `016`: agreement `0.7150`, 342 disagreements.
    - `022` phase agreement: early `0.6048`, mid `0.7370`, late `0.7812`.
    - `016` phase agreement: early `0.6371`, mid `0.7407`, late `0.7674`.
    - Main `022` weakness versus PGS016 is early action selection, especially token-taking and reserve-choice buckets, not late seat1 race decisions.
    - Action-type buckets with weakest `022` all-phase agreement:
      - `TAKE_TOKENS`: n=220, agreement `0.5364`.
      - `RESERVE_DECK`: n=85, agreement `0.5647`.
      - `RESERVE_VISIBLE`: n=172, agreement `0.5872`.
    - Common off-diagonal mismatches for `022`: `RESERVE_VISIBLE -> RESERVE_DECK`, `BUY_RESERVED -> BUY_VISIBLE`, `RESERVE_DECK -> RESERVE_VISIBLE`.
  - Current conclusion:
    - Do not train another late seat1/value-targeted correction.
    - If doing a policy improvement run next, make it a broad non-value PGS/ranking distillation experiment with emphasis on early token/reserve decisions, not a seat1 late-race patch.
- Early token/reserve PGS distillation experiment completed as `028`:
  - Added relabel filters to `run_improve_replay_actions.py` / `src/splendor_ai/training/improve_replay_actions.py`:
    - `--min-turn-index`
    - `--max-turn-index`
    - `--action-type-filter`
    - `--action-type-filter-scope original|selected|either`
  - Focused relabel tests passed.
  - Generated focused non-value PGS016 corpus:
    - `data/corpus_pgs016_early_token_reserve_028/replays.jsonl`
    - `data/corpus_pgs016_early_token_reserve_028/summary.json`
    - Inputs: current-generation value league replay rows, bounded to first 6,000 rows read.
    - Filters: `turn_index <= 39`, action type filter `TAKE_TOKENS RESERVE_VISIBLE RESERVE_DECK`, scope `either`.
    - Teacher: PGS016, depth 2, prior weight 3000, no value head.
    - Rows written: 2,053.
    - Changed by PGS: 789; unchanged anchors: 1,264.
    - Composition: mostly turns 0-19; selected actions were `TAKE_TOKENS` 995, `RESERVE_VISIBLE` 569, `RESERVE_DECK` 482, `BUY_VISIBLE` 7.
    - Main changed transitions: `RESERVE_VISIBLE -> RESERVE_DECK` 165 and `RESERVE_DECK -> RESERVE_VISIBLE` 138.
  - Trained:
    - `outputs/legal_action_scorer_028_early_pgs_distill/legal_action_attention_scorer_best.pt`
    - Parent: `outputs/legal_action_scorer_022_attention_distill/legal_action_attention_scorer_best.pt`
    - Architecture: attention.
    - Device: CPU because CUDA was not visible in this restored shell.
    - Data: six broad current-generation replay shards plus the focused corpus repeated twice.
    - 3 epochs, lr `1e-4`, value loss `0.0`, search distill loss `0.05`, distill temperature `1000`.
    - Dataset: 23,390 samples; best validation checkpoint epoch 2; validation acc `0.7139`.
  - PGS016 agreement diagnostics:
    - All-phase 1,200-row sample:
      - `016`: agreement `0.7150`.
      - `022`: agreement `0.7067`.
      - `028`: agreement `0.7200`.
      - `028` fixed the early alignment regression: early agreement `0.6371` versus `022` early `0.6048`.
      - `028` improved `RESERVE_VISIBLE` agreement from `0.5872` to `0.6471`; `RESERVE_DECK` from `0.5647` to `0.5761`; `TAKE_TOKENS` stayed `0.5318`.
    - Late/both-seat 692-row sample:
      - `016`: agreement `0.7182`.
      - `022`: agreement `0.7486`.
      - `028`: agreement `0.7529`.
  - Benchmarks:
    - `028` vs `022`, 50 games, seed 28000: `24-25-1`; no stalls/timeouts/fallbacks. Candidate by seat: seat0 `14/25`, seat1 `10/25`.
    - `028` vs `016`, 50 games, seed 28100: `25-25-0`; no stalls/timeouts/fallbacks. Candidate by seat: seat0 `13/25`, seat1 `12/25`.
    - Search opponent, same 30-game block seed 28200:
      - `028`: `12-18-0`, no stalls/timeouts/fallbacks; model seat0 `5/15`, seat1 `7/15`.
      - `022`: `18-11-1`, no stalls/timeouts/fallbacks; model seat0 `10/15`, seat1 `8/15`.
      - `016`: `12-17-1`, no stalls/timeouts/fallbacks; model seat0 `7/15`, seat1 `5/15`.
  - Conclusion:
    - `028` is not promotable. It achieved the intended PGS-alignment diagnostic improvement, but it regressed sharply versus search compared with `022`.
    - The run suggests PGS-alignment alone is an insufficient objective: it can move supervised diagnostics while weakening game strength.
    - Do not scale this exact early token/reserve distillation recipe. If continuing from here, either analyze why `022` beats search despite lower all-phase PGS agreement, or use actual game-strength gates as the optimization target rather than PGS agreement.
- `022` vs `028` regression audit on the search benchmark block:
  - Added a root wrapper `run_policy_diff_against_teacher.py`.
  - Extended `src/splendor_ai/training/run_policy_diff_against_teacher.py` with:
    - `teacher_source="search"` for live unguided shallow-search teacher comparisons;
    - `--game-seed-filter`;
    - `--model-turns-only`.
  - Added `tests/test_policy_diff_against_teacher.py`; focused test passed.
  - Differential seeds from the shared 30-game search block seed 28200 where `022` beat search and `028` lost search:
    - `28200, 28204, 28206, 28207, 28208, 28212, 28215, 28218, 28220, 28227`.
  - Regenerated faithful snapshot traces for the same search block:
    - `data/corpus_regression_022_vs_search_seed28200_snapshots/`
      - 30 games, 1,968 rows, 0 stalls/timeouts/fallbacks.
      - Reproduced benchmark result: model `18-11-1` vs search.
    - `data/corpus_regression_028_vs_search_seed28200_snapshots/`
      - 30 games, 1,936 rows, 0 stalls/timeouts/fallbacks.
      - Reproduced benchmark result: model `12-18-0` vs search.
  - Ran policy diffs on model turns only, before=`022`, after=`028`, teacher=`search`:
    - `outputs/analysis/policy_diff_022_to_028_on_022_search_wins_diffseeds.json`
    - `outputs/analysis/policy_diff_022_to_028_on_028_search_losses_diffseeds.json`
    - Derived summary:
      - `outputs/analysis/policy_diff_022_028_search_regression_derived_summary.json`
  - On the `022` winning trajectories:
    - 340 model turns.
    - `022` search-teacher match rate `0.5882`.
    - `028` search-teacher match rate `0.5794`.
    - Top-action agreement `0.8941` (about 36 changed model turns).
    - Early phase was where most changes occurred: early top-action agreement `0.75`; mid `0.9533`; late `0.9556`.
    - Notable losses were early `RESERVE_VISIBLE` positions: search-teacher `RESERVE_VISIBLE:early` match dropped from `0.3824` to `0.2941`, with 4 after-teacher losses.
  - On the `028` losing trajectories:
    - 320 model turns.
    - `022` search-teacher match rate `0.4844`.
    - `028` search-teacher match rate `0.4969`.
    - Top-action agreement `0.8594` (about 45 changed model turns).
    - This means the actual losing trajectories are not explained by `028` simply disagreeing more with immediate shallow search; if anything it matches the one-ply/depth-2 teacher slightly more often on those states.
  - Tempo/conversion extraction on the 10 differential seeds:
    - `022` average model score `16.3`; search score `11.6`.
    - `028` average model score `9.6`; search score `15.6`.
    - First buy turn was the same on average: `7.5`.
    - First scoring buy was slightly earlier for `028` (`32.1`) than `022` (`34.5`), so the issue is not simply delayed first scoring.
    - `022`: average buys `21.1`, reserves `7.4`, scoring buys `7.4`, bought points `16.3`.
    - `028`: average buys `18.8`, reserves `8.1`, scoring buys `5.2`, bought points `9.6`.
  - Current interpretation:
    - `028` learned the targeted early token/reserve alignment but seems to hurt the conversion pipeline after the opening: it reserves a little more, buys fewer total cards, and especially buys far fewer point-scoring cards.
    - The regression is not primarily late seat1, not loop/fallback related, and not just "worse immediate search agreement."
    - Do not train more early reserve/token PGS imitation. The next useful diagnostic should focus on conversion efficiency: how early reserves/takes affect later scoring-buy count and bought points, and why `022` preserves that better than `028`.

The warm-start pipeline is in a stronger place than the prior handoff. Continued sharded `search:greedy` corpus scaling plus multi-corpus supervised training produced a new local champion checkpoint, and a narrow inference-only loop fallback now mitigates the rare checkpoint repetition failures seen in greedy benchmarks.

Strategic assessment:

- The current path is the right bootstrap path: exact simulator first, legal observation/action masking, baselines, benchmark tooling, heuristic/search warm start, and gated policy improvement.
- This path should not be mistaken for the final superhuman training recipe. More shallow-search relabeling against `greedy` can improve the current champion, but it risks producing a strong anti-greedy specialist rather than a broadly superhuman Splendor player.
- Splendor is not Go: it has hidden deck/refill uncertainty, structured economic actions, tactical tempo, and a much more irregular action space. AlphaGo is a useful reference for the overall loop of policy/value learning plus search, but the architecture should be adapted rather than copied.
- The likely superhuman path is: keep the exact simulator and benchmark discipline, harvest one or two more small gated warm-start improvements if they keep working, then pivot toward a structured legal-action scorer and self-play/league training against historical checkpoints and stronger search baselines.
- Greedy is no longer a sufficient long-term benchmark. Keep it as a sanity check, but champion promotion should increasingly depend on stronger tuned search opponents, previous neural champions, fallback-trigger counts, and eventually human/expert-style evaluation.

Current-generation value experiment status:

- Added reproducible value-blended search plumbing to the main CLIs:
  - `run_benchmark.py` now accepts `--search-value-checkpoint`, `--search-value-device`, `--search-value-weight`, and `--search-value-scale` for `--model-kind policy_guided_search`.
  - `run_replay_corpus.py` accepts the same `--search-value-*` flags for `search` and `policy_guided_search` replay actors and records them in metadata.
  - `run_improve_replay_actions.py` accepts `--value-checkpoint`, `--value-device`, `--value-weight`, and `--value-scale` so ranking/distillation rows can use the same value-blended teacher.
- Generated clean current-generation value corpus `data/corpus_value_currentgen_004/`:
  - 300 games, 19,284 replay rows, all with state snapshots.
  - Shards: `016_vs_022`, `016_vs_search`, `022_vs_search`, `pgs016_vs_search`, `016_vs_pgs016`, `022_vs_pgs016`.
  - Seeds 90000-90299, `--source-type currentgen_value_league`, `--swap-seats`.
  - Aggregate hygiene: 0 stalled games, 0 timed-out games, 0 loop-fallback triggers.
- Trained `outputs/value_model_004_currentgen/value_model_best.pt`:
  - 19,284 samples, 17,328 train / 1,956 game-heldout validation, 8 CPU epochs.
  - Best validation checkpoint was epoch 1.
  - Final train metrics overfit heavily (`train_sign_acc=0.976`, `train_pearson=0.984`), but validation was weak (`mse=0.235`, `mae=0.393`, `sign_acc=0.556`, `pearson=0.176`).
- Swept PGS016 + `value_model_004_currentgen` vs plain shallow search on seed block 90300, 20 games each:
  - weight 0.00: 11-9-0, avg margin +1.70; seat0 6-4 margin +24, seat1 5-5 margin +10.
  - weight 0.05: 8-12-0, avg margin -2.50; seat0 3-7 margin -44, seat1 5-5 margin -6.
  - weight 0.10: 4-16-0, avg margin -5.15; seat0 1-9 margin -69, seat1 3-7 margin -34.
  - weight 0.25: 11-9-0, avg margin -0.05; seat0 5-5 margin -8, seat1 6-4 margin +7.
  - All sweep runs had 0 stalls and 0 timeouts.
- Conclusion:
  - Do not generate the 500-1,000 broad value-blended PGS games from this value model.
  - The explicit scaling gate was not met: value-blended PGS did not beat plain PGS; the only matching win-rate point (`0.25`) lost the margin advantage.
  - Best next technical step is value-target/validation work before another value-search generation loop: inspect phase-bucket value error, late-state-only value training, or outcome targets that reduce early-game noise. Broad PGS generation should wait for a value blend that improves plain PGS on at least a 50-100 game confirmation block.
- Follow-up value diagnostics / late-only probe:
  - Added `run_value_diagnostics.py` and module `src/splendor_ai/training/run_value_diagnostics.py`.
    - Reports overall and bucketed value metrics by phase, turn band, seat, source, actor, current score, opponent score, race state, current score margin, and target margin.
    - Can run against `all`, `train`, or reconstructed held-out `validation` split.
  - Added value turn filters:
    - `ValueReplayDataset(min_turn_index=..., max_turn_index=...)`
    - `run_value_supervised.py --min-turn-index/--max-turn-index`
    - `run_value_evaluate.py --min-turn-index/--max-turn-index`
  - Added value-blend turn gates to search:
    - `ShallowSearchBot(value_min_turn_index=..., value_max_turn_index=...)`
    - benchmark/replay/relabel CLIs expose corresponding `--search-value-min-turn-index` / `--search-value-max-turn-index` or `--value-min-turn-index` / `--value-max-turn-index`.
  - Diagnostic report for `value_model_004_currentgen`:
    - `outputs/analysis/value_model_004_currentgen_validation_diagnostics.json`
    - Held-out validation split, 1,956 samples: mse `0.2145`, mae `0.3838`, sign acc `0.5573`, Pearson `0.1319`.
    - Phase buckets showed late positions are more learnable:
      - early: n=600, mae `0.404`, sign `0.525`, Pearson `0.035`
      - mid: n=900, mae `0.393`, sign `0.556`, Pearson `0.121`
      - late: n=456, mae `0.338`, sign `0.603`, Pearson `0.282`
    - Major bias: positions where the actor is already far ahead were badly underpredicted (`current_score_margin=ahead_5plus`: sign `0.273`, bias `-0.445`).
  - Trained late-only value model:
    - `outputs/value_model_005_late40_currentgen/value_model_best.pt`
    - Same current-generation corpus, `--min-turn-index 40`, 7,284 samples, 6,528 train / 756 game-heldout validation, 8 epochs.
    - Best epoch 8; validation mse `0.1826`, mae `0.3375`, sign acc `0.6468`, Pearson `0.3348`.
    - Diagnostic report: `outputs/analysis/value_model_005_late40_validation_diagnostics.json`.
  - Swept PGS016 + gated late-only value vs plain search on seed block 90300, 20 games each, `--search-value-min-turn-index 40`:
    - Plain PGS baseline: 11-9-0, avg margin `+1.70`; seat0 6-4 margin +24, seat1 5-5 margin +10.
    - `value_model_005`, weight 0.05: 11-9-0, avg margin `+1.25`; seat0 6-4 margin +24, seat1 5-5 margin +1.
    - `value_model_005`, weight 0.10: 8-12-0, avg margin `-1.55`; seat0 4-6 margin -22, seat1 4-6 margin -9.
    - `value_model_005`, weight 0.25: 11-8-1, avg margin `+1.05`; seat0 7-3 margin +29, seat1 4-5-1 margin -8.
  - Conclusion:
    - Late-only value training improved held-out value metrics, but still did not produce a better PGS teacher.
    - Do not scale broad PGS generation from `value_model_005`.
    - The next value-specific idea would need to address the ahead-position sign/bias problem directly, e.g. score-margin calibration or a win/loss/outcome head, before another expensive search sweep.
- Outcome-head / value-bias experiment:
  - Added a win/draw/loss outcome head to `ValueMLP`.
    - `ValueMLP.forward(...)` remains backward-compatible and returns the scalar value head.
    - `ValueMLP.outcome_logits(...)` returns 3 logits: loss/draw/win.
    - Old value checkpoints remain loadable via `strict=False` in value eval/diagnostics/search loading.
  - Added optional value multi-task training:
    - `compute_value_losses(...)`
    - `run_value_supervised.py --outcome-loss-weight`
    - metrics now include `total_loss`, `outcome_loss`, and `outcome_class_accuracy`.
  - Extended diagnostics:
    - `run_value_diagnostics.py` now reports outcome-head accuracy and mean loss/draw/win probabilities per bucket.
  - Added search evaluator modes:
    - `CheckpointValueEvaluator(prediction_mode="value"|"outcome"|"blend")`
    - `prediction_mode="outcome"` uses `P(win) - P(loss)` as the normalized value before heuristic scaling.
    - Benchmark/replay/relabel CLIs expose `--search-value-prediction-mode`, `--search-value-outcome-blend-weight`, and corresponding relabel flags.
  - Trained `outputs/value_model_006_late40_outcome_currentgen/value_model_best.pt`:
    - Same current-generation value corpus as 005, `--min-turn-index 40`, `--outcome-loss-weight 0.5`.
    - 7,284 samples, 6,528 train / 756 game-heldout validation, 8 epochs.
    - Best validation checkpoint was epoch 3 by total loss.
    - Validation diagnostics: mse `0.1762`, mae `0.3460`, sign acc `0.6349`, outcome acc `0.6190`, Pearson `0.3404`.
    - Bias buckets improved materially versus 005:
      - `ahead_5plus`: mse `0.2340 -> 0.1546`, Pearson `0.026 -> 0.435`, outcome acc `0.692 -> 0.750`; scalar bias still negative (`-0.266 -> -0.251`).
      - `current_score=10-14`: mse `0.1447 -> 0.1225`, sign `0.609 -> 0.677`, outcome acc `0.489 -> 0.647`.
      - `large_win`: mse `0.6634 -> 0.5828`, sign `0.710 -> 0.806`, outcome acc `0.484 -> 0.871`.
  - Search probe using outcome-mode leaves, gated at `turn_index >= 40`, seed block 90300, 20 games each:
    - plain PGS baseline: 11-9-0, avg margin `+1.70`.
    - `006` outcome mode, weight 0.05: 13-7-0, avg margin `+1.80`; seat0 6-4 margin +19, seat1 7-3 margin +17.
    - `006` outcome mode, weight 0.10: 10-10-0, avg margin `-0.50`.
    - `006` outcome mode, weight 0.25: 7-13-0, avg margin `-2.25`.
  - 50-game confirmation on fresh seed block 90400 rejected the apparent 20-game gain:
    - plain PGS: 29-21-0, avg margin `+1.20`; seat0 15-10 margin +30, seat1 14-11 margin +30.
    - `006` outcome mode, weight 0.05: 24-26-0, avg margin `-0.68`; seat0 15-10 margin +35, seat1 9-16 margin -69.
  - Conclusion:
    - The outcome head fixed several diagnostic bias buckets, but using it in search is still not a stronger teacher.
    - Do not scale broad PGS generation from `value_model_006`.
    - The 20-game apparent gain was noise; the 50-game confirmation exposed a major seat1 regression.
    - The next useful step is not another value-blend sweep. Either diagnose why outcome-guided leaves hurt seat1 late searches, or move the value head out of root gameplay and use it only as an offline diagnostic / data-selection signal.
  - Follow-up search-value decision diagnostic:
    - Added `run_search_value_diff.py` and module `src/splendor_ai/training/run_search_value_diff.py`.
    - It compares plain PGS and value-guided PGS on the same replay snapshots and reports:
      - changed top-action rate;
      - baseline regret of the value-guided action under plain PGS values;
      - value regret of the plain-PGS action under value-guided values;
      - buckets by seat, phase, score-margin bucket, baseline/value action type, and action-type matrix;
      - high-regret changed examples.
    - Full late-seat1 depth-2 run over all rows was too expensive and timed out before writing; stale Python process was stopped.
    - Bounded depth-2 samples over the current-generation corpus, same PGS016 prior and `value_model_006` outcome mode:
      - Seat1 late sample:
        - command output: `outputs/analysis/pgs016_vs_value006_outcome_late40_seat1_diff_sample1200.json`
        - rows analyzed 234, changed 41 (`17.5%`).
        - plain-PGS baseline regret of changed value actions: mean `27.1`, max `107.0`.
        - value-guided regret of plain-PGS actions: mean `77.2`, max `483.7`.
        - highest change rates by score-margin bucket:
          - tied: 14/44 changed (`31.8%`)
          - ahead 1-4: 15/55 changed (`27.3%`)
          - ahead 5+: 3/13 changed (`23.1%`)
          - behind 1-4: 8/95 changed (`8.4%`)
          - behind 5+: 1/27 changed (`3.7%`)
        - Most changes were within buy choices, especially visible-buy to reserved-buy or a different visible buy. Examples include late/tied states where outcome mode preferred a reserved buy over the plain-PGS visible buy despite baseline regret of `65-100`.
      - Seat0 matching sample:
        - command output: `outputs/analysis/pgs016_vs_value006_outcome_late40_seat0_diff_sample1200.json`
        - rows analyzed 234, changed 41 (`17.5%`).
        - plain-PGS baseline regret of changed value actions: mean `45.5`, max `142.5`.
        - change distribution was less concentrated in tied/ahead positions than seat1:
          - tied: 6/40 changed (`15.0%`)
          - ahead 1-4: 16/78 changed (`20.5%`)
          - ahead 5+: 0/16 changed (`0.0%`)
          - behind 1-4: 18/83 changed (`21.7%`)
          - behind 5+: 1/17 changed (`5.9%`)
    - Current interpretation:
      - Outcome-guided leaves are not globally "seat1-only bad"; they perturb a similar fraction of late choices for both seats.
      - The seat1 harm is plausibly from where the perturbations occur: tied/ahead late positions, where seat1 is often responding in the final race and small tempo/card-choice changes can flip the result.
      - Do not keep sweeping live value blend weights. The outcome head is useful as an offline diagnostic/calibration signal, but not as a promoted PGS leaf evaluator yet.
      - If revisiting value search, add a much stricter gate such as applying outcome leaves only in clearly behind late positions, or only when the baseline PGS action and value-guided action agree on action type/card family. This should be treated as a new diagnostic experiment, not a generation loop.

Legal-action scorer pivot status:

- Added `src/splendor_ai/encoding/action_features.py`.
  - Encodes each candidate legal action into a 79-float structured feature vector.
  - Includes action type, board/reserved location, target card, chosen noble, token flow, and derived economics.
  - Latest version also includes opponent-relative context features for score margin, threat state, reserve denial, token overflow, affordability shift, and opponent proximity to the target card.
- Public observation tensors now include opponent reserved-card identities.
  - This is legal public information in Splendor and matters for reserve denial / opponent-threat modeling.
  - Old checkpoints remain loadable because inference and dataset loading truncate/pad encoded vectors to the checkpoint's saved config size.
- Added `LegalActionReplayDataset`.
  - Requires snapshot-enabled replay rows.
  - Reconstructs the pre-action state, regenerates legal actions, encodes legal-action features, and maps the replay `action_index` to `target_action_row`.
- Added variable-length scorer batching through `collate_legal_action_replay_samples`.
- Added `LegalActionScorerModel`.
  - Encodes state observation and each legal action separately.
  - Produces one score per legal action plus a value estimate.
- Added legal-action scorer supervised training.
  - CLI entry point: `run_legal_action_supervised.py`.
  - Saves `legal_action_scorer.pt`, `legal_action_scorer_best.pt`, and `legal_action_scorer_metrics.json`.
  - Supports optional search-ranking distillation:
    - `run_improve_replay_actions.py` now stores `search_action_scores` for ranked searched candidate actions.
    - `LegalActionReplayDataset` loads those stored search values into padded per-legal-action targets.
    - `run_legal_action_supervised.py` accepts `--search-distill-loss-weight` and `--search-distill-temperature`.
    - This lets the scorer learn from the search ranking over legal actions, not only the single top-1 relabeled action.
- Added `LegalActionScorerBot` for checkpoint-backed inference.
- Added benchmark support for scorer checkpoints.
  - Use `run_benchmark.py --model-kind legal_action_scorer`.
  - The default remains `--model-kind flat_policy` for old `CheckpointPolicyBot` checkpoints.
  - Smoke benchmark passed for `outputs/legal_action_scorer_001/legal_action_scorer_best.pt` vs random, 1-0, 0 timeouts.
- Added direct checkpoint-vs-checkpoint benchmark support.
  - Use `--baseline-checkpoint` plus `--baseline-model-kind`.
  - Candidate win/loss accounting is seat-based, so it works even if both sides are model-backed.
- Added `search` as a benchmark opponent.
  - Use `run_benchmark.py --opponents search`.
  - Search settings are recorded in benchmark JSON: `--search-depth`, `--search-max-branching`, `--search-buy-branching`, `--search-reserve-branching`, and `--search-take-branching`.
- Added legal-action scorer checkpoint support to replay corpus generation.
  - Use `run_replay_corpus.py --seat0-bot checkpoint --seat1-bot checkpoint --checkpoint-model-kind legal_action_scorer`.
  - Metadata records checkpoint path, model kind, device, and loop-fallback settings.
  - Added per-seat checkpoint overrides for league corpora:
    - `--seat0-checkpoint-path`
    - `--seat0-checkpoint-model-kind`
    - `--seat1-checkpoint-path`
    - `--seat1-checkpoint-model-kind`
  - Shared `--checkpoint-path` / `--checkpoint-model-kind` remain backward-compatible defaults.
  - Smoke corpus `data/corpus_league_smoke_001` verified `legal_action_scorer_003` vs `legal_action_scorer_002` with 2 completed games, 130 steps, 0 stalls/timeouts/fallback triggers.
- Added structured replay corpus manifests for league/evaluation hygiene.
  - `run_replay_corpus.py` now writes `manifest.json` next to `replays.jsonl` and `summary.json`.
  - CLI flag: `--source-type`, default `generated_replay`; use labels like `league`, `self_play`, or `search_vs_greedy`.
  - Manifest records source type, replay/summary filenames, seed block, aggregate results, matchup-level results, bot classes, checkpoint paths, checkpoint model kinds, and CLI provenance metadata.
  - This is intended to make future league training filters explicit instead of relying on filenames and handoff notes.
- Added manifest reporting utility:
  - `run_manifest_report.py <corpus_dir_or_manifest> [...]`
  - Prints source type, seed block, games, steps, seat W-L-D, stalls/timeouts, fallback triggers, and per-matchup rows.
  - Use the per-matchup rows for actor-vs-actor interpretation when `--swap-seats` is enabled; the top-level W-L-D is seat-based.
- Generated first structured league probe with manifests:
  - All shards used `--source-type league`, `--include-state-snapshots`, `--swap-seats`, 8 games each, 40 games total.
  - `data/corpus_league_manifest_001_001_vs_003`: 8 games, 518 steps, 0 stalls/timeouts/fallback triggers.
    - `001` as seat0 vs `003`: 4-0-0.
    - `003` as seat0 vs `001`: 2-2-0, so actor-level `001` result is 6-2 over this tiny block.
  - `data/corpus_league_manifest_001_001_vs_flat`: 8 games, 484 steps, 0 stalls/timeouts/fallback triggers.
    - Actor-level `001` result vs old flat champion: 6-2.
  - `data/corpus_league_manifest_001_003_vs_flat`: 8 games, 476 steps, 0 stalls/timeouts/fallback triggers.
    - Actor-level `003` result vs old flat champion: 5-3.
  - `data/corpus_league_manifest_001_001_vs_search`: 8 games, 508 steps, 0 stalls/timeouts/fallback triggers.
    - Actor-level `001` result vs shallow search: 3-5.
  - `data/corpus_league_manifest_001_003_vs_search`: 8 games, 508 steps, 0 stalls/timeouts/fallback triggers.
    - Actor-level `003` result vs shallow search: 1-7.
  - These are data-quality probe shards, not promotion benchmarks. They are small but clean and suitable as structured league input for the next training-loop experiment.
- Generated scaled structured league corpus block `corpus_league_manifest_002_*`:
  - 5 shards x 50 games = 250 games total.
  - All shards used `--source-type league`, `--include-state-snapshots`, `--swap-seats`, fresh seed blocks 42000-42449.
  - All shards were clean: 0 stalls, 0 timeouts, 0 fallback triggers.
  - `data/corpus_league_manifest_002_001_vs_003`: 50 games, 3,064 steps.
    - Actor-level `001` vs `003`: 18-32-0. This block favors `003`.
  - `data/corpus_league_manifest_002_001_vs_flat`: 50 games, 3,082 steps.
    - Actor-level `001` vs old flat champion: 27-23-0.
  - `data/corpus_league_manifest_002_003_vs_flat`: 50 games, 3,178 steps.
    - Actor-level `003` vs old flat champion: 29-21-0.
  - `data/corpus_league_manifest_002_001_vs_search`: 50 games, 3,142 steps.
    - Actor-level `001` vs shallow search: 9-39-2.
  - `data/corpus_league_manifest_002_003_vs_search`: 50 games, 3,128 steps.
    - Actor-level `003` vs shallow search: 12-38-0.
  - This is now a usable manifest-audited league data block for the next conservative training experiment. It still should not be treated as a promotion benchmark because each matchup has only 50 games.
- First conservative manifest-league training candidate, `legal_action_scorer_011`:
  - Added legal-action scorer training support for:
    - `--policy-target-filter winner`, which trains only on actions by the eventual game winner.
    - outcome sample weights: `--winner-action-weight`, `--loser-action-weight`, `--draw-action-weight`.
    - `--validation-split-unit row|game|source`; use `game` by default for league training.
  - Trained from `outputs/legal_action_scorer_003/legal_action_scorer_best.pt`.
  - Training data: all five `corpus_league_manifest_002_*` replay files, winner-action rows only.
  - Training setup: 7,735 total rows, 6,987 train / 748 game-heldout validation, 6 epochs, CPU, `--value-loss-weight 0.0`, learning rate `0.0001`.
  - Output: `outputs/legal_action_scorer_011/`; best validation checkpoint was epoch 3.
  - Final train: loss 0.6773, accuracy 0.7736. Final validation: loss 0.7938, accuracy 0.7714.
  - Benchmarks:
    - vs `legal_action_scorer_003`, 100 games seed block 43000: 52-48-0, average margin -0.13, 0 fallback triggers.
    - vs `legal_action_scorer_001`, 100 games seed block 43100: 46-54-0, average margin -0.25, 0 fallback triggers.
    - vs old flat champion `policy_improve_margin500_002`, 100 games seed block 43200: 69-31-0, average margin +2.75, candidate fallback triggers 0, baseline fallback triggers 1.
    - standardized shallow-search block seed 20000: 24-76-0, average margin -4.46, 0 fallback triggers.
  - Conclusion: reject `011`. Winner-only league imitation from this small league block did not improve the stable neural/search gates. It still beats the old flat model, but it is worse than `001`/`003` versus search and not clearly stronger head-to-head.
- Added policy-vs-search disagreement analysis:
  - CLI: `run_policy_search_disagreement.py`
  - Supports `--model-kind legal_action_scorer` and `--model-kind flat_policy`.
  - Reconstructs snapshot replay states, ranks the checkpoint's legal actions, ranks shallow-search actions, and writes:
    - overall agreement rate
    - agreement by model action type
    - agreement by search action type
    - agreement by early/mid/late phase
    - model-action-type vs search-action-type matrix
    - top high-search-margin disagreement examples
  - Analysis outputs:
    - `outputs/analysis/003_search_disagreement.json`
    - `outputs/analysis/001_search_disagreement.json`
  - Both used 1,000 rows from the corresponding `corpus_league_manifest_002_*_vs_search/replays.jsonl` shard and the standard depth-2 search settings.
  - `legal_action_scorer_003` vs search:
    - agreement rate: 0.506
    - disagreements: 494 / 1,000
    - model action missing from searched candidate set: 256 disagreements
    - phase agreement: early 0.325, mid 0.588, late 0.600
    - by model action type:
      - `BUY_VISIBLE`: 520 rows, agreement 0.690
      - `BUY_RESERVED`: 130 rows, agreement 0.785
      - `TAKE_TOKENS`: 180 rows, agreement 0.078
      - `RESERVE_DECK`: 102 rows, agreement 0.157
      - `RESERVE_VISIBLE`: 68 rows, agreement 0.221
  - `legal_action_scorer_001` vs search:
    - agreement rate: 0.526
    - disagreements: 474 / 1,000
    - model action missing from searched candidate set: 252 disagreements
    - phase agreement: early 0.330, mid 0.629, late 0.605
    - by model action type:
      - `BUY_VISIBLE`: 495 rows, agreement 0.711
      - `BUY_RESERVED`: 145 rows, agreement 0.869
      - `TAKE_TOKENS`: 167 rows, agreement 0.084
      - `RESERVE_DECK`: 128 rows, agreement 0.164
      - `RESERVE_VISIBLE`: 65 rows, agreement 0.200
  - Main diagnosis:
    - The neural scorer is reasonably aligned with search when it chooses buy actions.
    - The large mismatch is early-game token-taking and reserve behavior. Agreement is only about one-third in early positions.
    - Many token/reserve disagreements are missing from the pruned search candidate set, which means the model often likes actions that search's heuristic ordering does not even keep.
    - The biggest high-margin examples are late-game tactical misses: buying the wrong visible card or missing a decisive reserve/buy.
  - Implication: the next work should be targeted, not another broad imitation pass. Focus on action feature/label quality for token-taking, reserve decisions, and late-game tactical buy selection.
- Added targeted search-correction corpus generation:
  - CLI: `run_targeted_search_corpus.py`
  - Compares a model's top legal action with shallow search on snapshot replay states.
  - Writes only targeted disagreement rows, with the replay target replaced by the search action.
  - Stores original action, model top action, search margin, full searched action scores, source replay path, and `target_source="targeted_search_disagreement"`.
  - Intended targets are early token/reserve disagreements and late tactical buy/reserve misses, not broad imitation.
- First targeted-corpus smoke results:
  - Completed smoke command on 100 rows from `data/corpus_league_manifest_002_001_vs_search/replays.jsonl`.
  - Output: `data/corpus_targeted_search_smoke_001/`.
  - Summary: 100 rows analyzed, 54 model/search disagreements, 29 rows written.
  - Written target types: `RESERVE_VISIBLE` 13, `TAKE_TOKENS` 11, `RESERVE_DECK` 4, `BUY_RESERVED` 1.
  - Phase split: early 20, mid 8, late 1.
  - A larger CPU attempt capped at 1,000 rows was interrupted after the command timeout; it left a usable partial file with 153 targeted rows at `data/corpus_targeted_search_001/targeted_search_disagreements.jsonl`, but no summary because the command did not finish.
- Smoke fine-tune using the partial targeted file:
  - Output: `outputs/legal_action_scorer_012_targeted_smoke/`.
  - Parent: `outputs/legal_action_scorer_001/legal_action_scorer_best.pt`.
  - Training data: the three original improved checkpoint/greedy corpora plus the 153-row partial targeted file.
  - Total rows: 2,752; game-heldout validation; 2 epochs; CPU.
  - Best checkpoint: epoch 2, validation accuracy 0.7387, validation loss 0.9203.
  - Quick 20-game head-to-head vs `legal_action_scorer_001`, seed 52000: 9-11, 0 stalls/timeouts/fallback triggers.
  - Quick 20-game benchmark vs shallow search, seed 52100: 4-16, 0 stalls/timeouts/fallback triggers.
  - Conclusion: workflow is functional, but this is not a promotion candidate. The targeted file is too small and old-checkpoint fine-tuning cannot use the new 295/79 feature widths.
- Full targeted-corpus fine-tune from `legal_action_scorer_001`, `legal_action_scorer_013_targeted_full_from001`:
  - Full targeted corpus: `data/corpus_targeted_search_full_001/targeted_search_disagreements.jsonl`.
  - Corpus summary: 6,270 rows analyzed, 3,058 disagreements, 1,842 targeted rows written, 0 missing snapshots.
  - Written target types: `TAKE_TOKENS` 803, `RESERVE_VISIBLE` 720, `RESERVE_DECK` 184, `BUY_VISIBLE` 118, `BUY_RESERVED` 17.
  - Phase split: early 939, mid 628, late 275.
  - Training output: `outputs/legal_action_scorer_013_targeted_full_from001/`.
  - Parent: `outputs/legal_action_scorer_001/legal_action_scorer_best.pt`; because of this, the model still uses the old 256/64 feature widths.
  - Training data: three original improved checkpoint/greedy corpora plus the full targeted corpus.
  - Total rows: 4,425; 4 epochs; game-heldout validation; best epoch 4; training time about 184 seconds on CPU.
  - Validation accuracy is low, around 0.366 at best epoch, but this is expected because targeted rows intentionally contradict the old policy distribution.
  - Benchmarks:
    - vs `legal_action_scorer_001`, 100 games seed 53000: 50-50-0, 1 timed-out game.
    - vs shallow search, 20 games seed 53100: 8-12-0, 0 stalls/timeouts/fallback triggers.
    - `legal_action_scorer_001` on the same search seed block 53100: 3-17-0.
    - vs `legal_action_scorer_003`, 100 games seed 53200: 55-45-0, 0 stalls/timeouts/fallback triggers.
  - Current read: promising but not yet promoted. It improves the same-seed 20-game search probe versus `001`, ties `001` directly, and beats `003` directly, but needs a 100-game standardized search benchmark before being treated as a new champion.
- Wide-feature from-scratch targeted model, `legal_action_scorer_014_wide_targeted_fromscratch`:
  - Trained without `--init-checkpoint`, so this is the first candidate that actually uses the new 295-float observation and 79-float legal-action feature vectors.
  - Training data: three original improved checkpoint/greedy corpora plus `data/corpus_targeted_search_full_001/targeted_search_disagreements.jsonl`.
  - Output: `outputs/legal_action_scorer_014_wide_targeted_fromscratch/`.
  - Total rows: 4,425; game-heldout validation; 8 epochs; CPU; best validation epoch 7.
  - Best validation accuracy: 0.5208; final train accuracy: 0.5982; training time about 320 seconds.
  - Quick neural benchmarks:
    - vs `legal_action_scorer_001`, 100 games seed 54000: 57-43-0.
    - vs `legal_action_scorer_003`, 100 games seed 54100: 60-40-0.
  - Quick search probe seed 54200:
    - `014` vs search, 20 games: 11-9-0.
    - `001` same seed block: 7-13-0.
    - `003` same seed block: 6-14-0.
  - Standardized search gate seed 20000:
    - `014`: 28-71-1, avg margin -2.79, avg model score 11.37, 0 timeouts.
    - `013`: 29-70-1, avg margin -3.06.
    - `001`: 30-70-0, avg margin -3.92.
    - `003`: 29-71-0, avg margin -3.86.
  - Current read: `014` is not a clean promotion because it did not improve standardized search win rate, but it is the best model so far by search score margin and clearly beats `001`/`003` head-to-head. It is a useful parent for the next branch, not a final champion.
- Policy-guided search branch:
  - Added `PolicyGuidedSearchBot`.
  - It reuses the existing `ShallowSearchBot` minimax/leaf evaluation but adds a legal-action scorer prior to action ordering/pruning.
  - The policy prior is rank-normalized per state, then blended into the heuristic ordering score with `--policy-prior-weight` / `policy_prior_weight`.
  - Added `CheckpointLegalActionPriorScorer` for loading legal-action scorer checkpoints as search priors.
  - Benchmark support:
    - `run_benchmark.py --model-kind policy_guided_search`
    - Uses the existing `--search-depth`, `--search-max-branching`, `--search-buy-branching`, `--search-reserve-branching`, `--search-take-branching` settings for the guided search.
    - New flag: `--policy-prior-weight`, default `2000`.
  - Replay-corpus support:
    - `run_replay_corpus.py --checkpoint-model-kind policy_guided_search`
    - Metadata records `model_kind="policy_guided_search"` and `policy_prior_weight`.
  - Verification:
    - Full suite passed after implementation: `python -m pytest -q --basetemp .codex_pytest_tmp_all` -> 143 passed.
  - Initial smoke benchmarks with `014` as the prior:
    - `outputs/benchmarks/policy_guided_search_014_vs_search_20g_seed55000.json`
      - Policy-guided search `014` vs plain shallow search: 15-5-0, avg margin +3.25, 0 timeouts.
    - `outputs/benchmarks/policy_guided_search_014_vs_greedy014_20g_seed55100.json`
      - Policy-guided search `014` vs greedy `014`: 14-6-0, avg margin +1.40, 0 timeouts.
  - 100-game confirmation:
    - `outputs/benchmarks/policy_guided_search_014_vs_search_100g_seed56000.json`
      - Policy-guided search `014` vs plain shallow search: 44-55-1, avg margin -0.58, avg model score 12.58 vs 13.16, 0 timeouts.
      - Seat split was asymmetric: model as seat 0 went 26-24-0 with avg margin +0.58; model as seat 1 went 18-31-1 with avg margin -1.74.
  - Policy-prior weight sweep with `014` as the prior, all vs plain shallow search on seed block 57000, 20 games each:
    - weight `0`: 8-12-0, avg margin -0.85, 0 timeouts/fallback triggers.
    - weight `500`: 9-10-1, avg margin +0.25, 0 timeouts/fallback triggers.
    - weight `1000`: 10-10-0, avg margin +0.55, 0 timeouts/fallback triggers.
    - weight `1500`: 13-7-0, avg margin +2.60, 0 timeouts/fallback triggers.
    - weight `3000`: 13-7-0, avg margin +3.35, 0 timeouts/fallback triggers.
  - 100-game confirmation for the best sweep setting:
    - `outputs/benchmarks/policy_guided_search_014_w3000_vs_search_100g_seed56000.json`
      - Policy-guided search `014`, `policy_prior_weight=3000`, vs plain shallow search: 59-40-1, avg margin +1.19, avg model score 13.24 vs 12.05, 0 stalls/timeouts/fallback triggers.
      - Seat split: model as seat 0 went 33-16-1 with avg margin +2.40; model as seat 1 went 26-24-0 with avg margin -0.02.
      - This directly improves over the same seed block at weight `2000` (44-55-1, avg margin -0.58).
    - `outputs/benchmarks/policy_guided_search_014_w3000_vs_search_100g_seed58000.json`
      - Independent confirmation: 56-44-0, avg margin +0.65, avg model score 12.73 vs 12.08, 0 stalls/timeouts/fallback triggers.
      - Seat split: model as seat 0 went 31-19-0 with avg margin +1.76; model as seat 1 went 25-25-0 with avg margin -0.46.
    - Combined weight-3000 confirmation across seed blocks 56000 and 58000: 115-84-1, avg margin +0.92, avg model score 12.98 vs 12.06, 0 stalls/timeouts/fallback triggers.
  - Current read: `policy_prior_weight=3000` is now a confirmed improvement over plain shallow search on these two 100-game blocks and is the best available policy-guided search setting. Use it as the next teacher for a conservative corpus/relabel experiment, while keeping plain shallow search and the existing neural pool as promotion gates. Watch the seat-1 margin: it is near break-even despite the positive aggregate result.
- First policy-guided teacher corpus/probe:
  - Generated `data/corpus_policy_guided_search_014_w3000_vs_search_001`.
    - `source_type=policy_guided_search_vs_search`, 20 games, seed block 59000-59019, `--swap-seats`, `--include-state-snapshots`.
    - Seat0/seat1 top-level result: 12-8-0, 1,262 steps, 0 stalls/timeouts/fallback triggers.
    - Manifest actor rows:
      - `PolicyGuidedSearchBot:policy_guided_search:legal_action_scorer_014_wide_targeted_fromscratch` as seat0 vs `ShallowSearchBot`: 7-3-0.
      - `ShallowSearchBot` as seat0 vs policy-guided search: 5-5-0.
  - Trained small probe `outputs/legal_action_scorer_015_policy_guided_probe/`.
    - Parent: `outputs/legal_action_scorer_014_wide_targeted_fromscratch/legal_action_scorer_best.pt`.
    - Training data: only the 20-game policy-guided-search-vs-search corpus.
    - 1,262 rows, game-heldout validation, 4 CPU epochs, `--value-loss-weight 0.0`, learning rate `0.00005`.
    - Best validation checkpoint: epoch 2, validation loss 1.0555, validation accuracy 0.6270.
  - Quick search check on seed block 60000:
    - `015`: 9-11-0, avg margin -0.90, avg model score 12.75 vs 13.65, 0 stalls/timeouts/fallback triggers.
    - Parent `014` on the same block: 9-11-0, avg margin +0.75, avg model score 14.10 vs 13.35, 0 stalls/timeouts/fallback triggers.
  - Current read: reject `015` as a promotion candidate. The probe proves the corpus/training path works, but training on only this small direct teacher corpus did not improve over `014`. The next useful attempt should either scale the policy-guided teacher corpus substantially or mix it conservatively with the stronger targeted/search-correction corpora rather than fine-tuning on this 20-game corpus alone.
- Scaled policy-guided teacher mix candidate, `legal_action_scorer_016_policy_guided_mix`:
  - Generated `data/corpus_policy_guided_search_014_w3000_vs_search_002`.
    - `source_type=policy_guided_search_vs_search`, 100 games, seed block 61000-61099, `--swap-seats`, `--include-state-snapshots`.
    - 6,436 steps, 0 stalls/timeouts/fallback triggers.
    - Top-level seat W-L-D: 54-44-2.
    - Manifest actor rows:
      - Policy-guided search `014`, `policy_prior_weight=3000`, as seat0 vs plain shallow search: 27-22-1.
      - Plain shallow search as seat0 vs policy-guided search `014`: 27-22-1.
  - Trained `outputs/legal_action_scorer_016_policy_guided_mix/`.
    - Parent: `outputs/legal_action_scorer_014_wide_targeted_fromscratch/legal_action_scorer_best.pt`.
    - Training data:
      - `data/corpus_checkpoint_greedy_snapshots_001/improved_search_policy_margin500_no_take.jsonl`
      - `data/corpus_checkpoint_greedy_snapshots_002/improved_search_policy_margin500_no_take.jsonl`
      - `data/corpus_checkpoint_greedy_snapshots_003/improved_search_policy_margin500_no_take.jsonl`
      - `data/corpus_targeted_search_full_001/targeted_search_disagreements.jsonl`
      - `data/corpus_policy_guided_search_014_w3000_vs_search_002/replays.jsonl`
    - Setup: 10,861 rows, 9,835 train / 1,026 game-heldout validation, 6 CPU epochs, batch size 256, learning rate `0.00005`, `--value-loss-weight 0.0`.
    - Best validation checkpoint: epoch 6, validation loss 1.0585, validation accuracy 0.6384.
  - Benchmarks for `legal_action_scorer_016_policy_guided_mix/legal_action_scorer_best.pt`:
    - vs parent `014`, 100 games seed 62000: 63-37-0, avg margin +1.52, 0 stalls/timeouts/fallback triggers.
    - vs `001`, 100 games seed 62100: 67-33-0, avg margin +3.03, 0 stalls/timeouts/fallback triggers.
    - vs `003`, 100 games seed 62200: 65-35-0, avg margin +2.73, 0 stalls/timeouts/fallback triggers.
    - vs old flat champion `policy_improve_margin500_002`, 100 games seed 62400: 69-31-0, avg margin +3.81, 0 stalls/timeouts/fallback triggers.
    - Standardized shallow-search block seed 20000: 44-55-1, avg margin -1.01, avg model score 12.18 vs 13.19, 0 stalls/timeouts/fallback triggers.
      - Seat split: as seat0 28-21-1, avg margin +0.44; as seat1 16-34-0, avg margin -2.46.
      - This is a large improvement over prior scorer search gates on the same seed block: `014` was 28-71-1, `013` 29-70-1, `001` 30-70-0, `003` 29-71-0.
    - vs policy-guided-search teacher `014` at `policy_prior_weight=3000`, 100 games seed 62300: 42-58-0, avg margin -1.10, 0 stalls/timeouts/fallback triggers.
      - Seat split: as seat0 26-24-0, avg margin +0.84; as seat1 16-34-0, avg margin -3.04.
  - Current read: promote `016` as the current best fast neural scorer / parent checkpoint. It clearly beats `014`, `001`, `003`, and the old flat champion, and it materially improves the standardized shallow-search gate. Do not treat it as stronger than the policy-guided-search teacher; the teacher remains stronger and exposes a persistent seat-1 weakness. The next useful branch is to generate more policy-guided teacher data with stronger seat balance or targeted seat-1 correction, then train a conservative `017` from `016` while keeping seed-20000 shallow search and teacher head-to-head as gates.
- Seat-1-targeted policy-guided correction candidate, `legal_action_scorer_017_seat1_policy_guided_mix`:
  - Generated `data/corpus_policy_guided_search_016_w3000_seat1_vs_search_001`.
    - `source_type=policy_guided_search_seat1_vs_search`, 100 games, seed block 63000-63099.
    - Pairing: plain shallow search as seat0 vs policy-guided search using `016` as the prior at `policy_prior_weight=3000` as seat1.
    - 6,456 steps, 0 stalls/timeouts/fallback triggers.
    - Seat0/seat1 result: shallow search beat the seat1 policy-guided teacher 55-44-1, avg final score 12.70 vs 12.42. This confirms the seat-1 weakness remains even when using `016` as the policy prior.
  - Built filtered weighted seat-1 teacher rows:
    - Output: `data/corpus_policy_guided_search_016_w3000_seat1_vs_search_001/seat1_teacher_weighted.jsonl`.
    - 3,228 rows, all `player_id == 1`.
    - 1,420 rows from teacher wins weighted 1.5; 1,772 rows from teacher losses weighted 1.25; 36 draw rows weighted 1.25.
    - Rows tagged with `target_source="seat1_policy_guided_teacher"`.
  - Trained `outputs/legal_action_scorer_017_seat1_policy_guided_mix/`.
    - Parent: `outputs/legal_action_scorer_016_policy_guided_mix/legal_action_scorer_best.pt`.
    - Training data: the same anchor/targeted/policy-guided mix used for `016`, plus the weighted seat-1 teacher rows.
    - Setup: 14,089 rows, 12,582 train / 1,507 game-heldout validation, 5 CPU epochs, batch size 256, learning rate `0.00003`, `--value-loss-weight 0.0`.
    - Best validation checkpoint: epoch 5, validation loss 1.0762, validation accuracy 0.6324.
  - Benchmarks for `legal_action_scorer_017_seat1_policy_guided_mix/legal_action_scorer_best.pt`:
    - vs `016`, 100 games seed 64000: 44-56-0, avg margin -0.25, 0 stalls/timeouts/fallback triggers.
      - Seat split: as seat0 24-26-0, avg margin +0.78; as seat1 20-30-0, avg margin -1.28.
    - vs `014`, 100 games seed 64100: 55-45-0, avg margin +0.23, 0 stalls/timeouts/fallback triggers.
      - This still beats `014`, but much less convincingly than `016` did in its own checked block.
    - Standardized shallow-search block seed 20000: 43-57-0, avg margin -0.89, 0 stalls/timeouts/fallback triggers.
      - Seat split: as seat0 26-24-0, avg margin +0.06; as seat1 17-33-0, avg margin -1.84.
      - Compared with `016` on the same seed block (44-55-1, avg margin -1.01; seat0 28-21-1, seat1 16-34-0), `017` slightly improved score margin/seat1 win count but lost win rate and regressed seat0.
    - vs policy-guided-search teacher using `016` prior at `policy_prior_weight=3000`, 100 games seed 64200: 41-58-1, avg margin -1.05, 0 stalls/timeouts/fallback triggers.
      - Seat split: as seat0 19-31-0, avg margin -1.82; as seat1 22-28-0, avg margin -0.28.
      - The seat1-focused training improved the candidate's seat1 margin against this teacher but damaged seat0 enough that the total result remained worse than the teacher.
  - Current read: reject `017`; keep `016` as the current best fast neural parent. The weighted seat-1 imitation pass did not clear the promotion bar and appears to trade away seat0 strength for only modest seat1 gains. The next useful step should be diagnostic rather than another blind weighted imitation run: split policy/search disagreements and action-type frequencies by seat, especially early token/reserve choices and late race decisions, then generate targeted correction rows only for high-margin seat1 mistakes instead of all seat1 teacher actions.
- Seat-aware disagreement diagnostics after rejecting `017`:
  - Extended `run_policy_search_disagreement.py` to report:
    - `by_seat`
    - `by_seat_phase`
    - `by_seat_model_action_type`
    - `by_seat_search_action_type`
    - `seat` on stored high-margin examples
  - Focused tests passed:
    - `python -m pytest tests/test_policy_search_disagreement.py -q --basetemp .codex_pytest_tmp_policy_disagreement` -> 2 passed.
  - Ran `016` disagreement analysis on two 2,000-row slices:
    - `outputs/analysis/016_search_disagreement_policy_guided_corpus_seat_split.json`
      - Source: `data/corpus_policy_guided_search_014_w3000_vs_search_002/replays.jsonl`
      - Overall agreement: 0.6055.
      - Seat0 agreement: 0.6240; seat1 agreement: 0.5870.
      - Early phase is weak for both seats, and weaker for seat1:
        - seat0 early: 0.4677
        - seat1 early: 0.4355
      - Seat1 model-action agreement:
        - `RESERVE_DECK`: 0.1915, many model actions missing from search candidates.
        - `TAKE_TOKENS`: 0.3371.
        - `RESERVE_VISIBLE`: 0.4331.
        - `BUY_VISIBLE`: 0.7258.
        - `BUY_RESERVED`: 0.8323.
    - `outputs/analysis/016_search_disagreement_seat1_teacher_corpus_seat_split.json`
      - Source: `data/corpus_policy_guided_search_016_w3000_seat1_vs_search_001/replays.jsonl`
      - Overall agreement: 0.5780.
      - Seat0 agreement: 0.5700; seat1 agreement: 0.5860.
      - Seat1 model-action agreement:
        - `RESERVE_DECK`: 0.2045.
        - `TAKE_TOKENS`: 0.2697.
        - `RESERVE_VISIBLE`: 0.4385.
        - `BUY_VISIBLE`: 0.7659.
        - `BUY_RESERVED`: 0.7683.
  - Diagnosis:
    - Early token-taking and reserve decisions remain the most frequent alignment problem, but many are low-margin or pruned/missing from search candidates.
    - The largest search margins are mostly late tactical buy/reserve decisions, especially choosing the wrong buy target, buying when search wants a reserve, reserving when search wants a buy, or choosing the wrong reserved/visible buy.
    - This explains why broad seat1 imitation (`017`) was not enough: it added many ordinary seat1 rows, while the high-value mistakes are sparse and late.
  - Generated a high-margin seat1 correction corpus:
    - Input: `data/corpus_policy_guided_search_016_w3000_seat1_vs_search_001/seat1_teacher_weighted.jsonl` (seat1-only rows).
    - Output: `data/corpus_targeted_seat1_high_margin_001/targeted_search_disagreements.jsonl`.
    - Summary: `data/corpus_targeted_seat1_high_margin_001/summary.json`.
    - Filter: `min_search_margin=1000`, search target action types `RESERVE_VISIBLE RESERVE_DECK BUY_VISIBLE BUY_RESERVED`, `late_buy_turn_min=50`, no missing-model-action inclusions.
    - Result: 3,228 rows analyzed, 1,396 disagreements, 55 written correction rows.
    - Written rows: 55 all seat1; 53 late, 2 mid.
    - Search target types: `BUY_VISIBLE` 26, `RESERVE_DECK` 12, `RESERVE_VISIBLE` 9, `BUY_RESERVED` 8.
    - Model top types on those rows: `BUY_VISIBLE` 32, `RESERVE_VISIBLE` 13, `BUY_RESERVED` 7, `RESERVE_DECK` 3.
    - Margin mean is high because one race/tactical miss is near terminal scale; minimum is 1000 by filter.
  - Current read: do not train another candidate on just these 55 rows. They are useful as diagnostics and as a seed for a larger correction set. Next training attempt should first scale seat1 data collection, then mine high-margin late buy/reserve corrections across a larger seat1 corpus. A reasonable next corpus target is 300-500 seat1 policy-guided-search games, then run the same high-margin filter and only train if the correction file reaches a few hundred rows without excessive duplication by seed/position.
- Scaled high-margin seat1 correction candidate, `legal_action_scorer_018_seat1_high_margin`:
  - Continued the seat1 policy-guided data collection after a machine crash. Completed five 100-game shards total:
    - Existing `data/corpus_policy_guided_search_016_w3000_seat1_vs_search_001`, seeds 63000-63099: search seat0 vs policy-guided `016` seat1, search won 55-44-1.
    - `data/corpus_policy_guided_search_016_w3000_seat1_vs_search_002`, seeds 65000-65099: search won 55-44-1.
    - `data/corpus_policy_guided_search_016_w3000_seat1_vs_search_003`, seeds 65100-65199: policy-guided `016` seat1 won 55-45-0.
    - `data/corpus_policy_guided_search_016_w3000_seat1_vs_search_004`, seeds 65200-65299: search won 62-38-0.
    - `data/corpus_policy_guided_search_016_w3000_seat1_vs_search_005`, seeds 65300-65399: search won 53-45-2.
    - All shards were clean: 500 games, 32,186 steps total, 0 stalls/timeouts/fallback triggers.
    - Combined actor result: policy-guided `016` as seat1 went 227-270-3 vs plain shallow search. This confirms the seat1 weakness is stable across larger data.
  - Built combined seat1-only weighted file:
    - Output: `data/corpus_policy_guided_search_016_w3000_seat1_vs_search_500/seat1_teacher_weighted.jsonl`.
    - 16,093 seat1 rows from 500 games.
    - 7,298 rows from seat1 teacher wins weighted 1.5; 8,659 from losses weighted 1.25; 136 from draws weighted 1.25.
  - Mined high-margin seat1 corrections from the combined file:
    - Output: `data/corpus_targeted_seat1_high_margin_500/targeted_search_disagreements.jsonl`.
    - Summary: `data/corpus_targeted_seat1_high_margin_500/summary.json`.
    - Filter: `min_search_margin=1000`, target action types `RESERVE_VISIBLE RESERVE_DECK BUY_VISIBLE BUY_RESERVED`, `late_buy_turn_min=50`, no missing-model-action inclusions.
    - 16,093 rows analyzed, 6,991 disagreements, 285 correction rows written.
    - Written rows: all seat1; 257 late, 28 mid; 204 unique game seeds; max 5 rows from one seed.
    - Search target types: `BUY_VISIBLE` 171, `RESERVE_VISIBLE` 45, `RESERVE_DECK` 44, `BUY_RESERVED` 25.
    - Model top types on those rows: `BUY_VISIBLE` 191, `BUY_RESERVED` 53, `RESERVE_VISIBLE` 35, `RESERVE_DECK` 6.
  - Trained `outputs/legal_action_scorer_018_seat1_high_margin/`.
    - Parent: `outputs/legal_action_scorer_016_policy_guided_mix/legal_action_scorer_best.pt`.
    - Training data: same anchor/targeted/policy-guided mix used for `016`, plus only the 285 high-margin seat1 correction rows. The broad seat1 imitation rows from `017` were not included.
    - Setup: 11,146 rows, 9,859 train / 1,287 game-heldout validation, 4 CPU epochs, batch size 256, learning rate `0.00003`, `--value-loss-weight 0.0`.
    - Best validation checkpoint: epoch 3, validation loss 1.2947, validation accuracy 0.6200.
  - Benchmarks for `legal_action_scorer_018_seat1_high_margin/legal_action_scorer_best.pt`:
    - vs `016`, 100 games seed 66000: 53-47-0, avg margin +0.19, 0 stalls/timeouts/fallback triggers.
      - Seat split: as seat0 28-22-0, avg margin +0.58; as seat1 25-25-0, avg margin -0.20.
    - vs `014`, 100 games seed 66100: 54-46-0, avg margin +0.72, 0 stalls/timeouts/fallback triggers.
      - Seat split: as seat0 27-23-0, avg margin +1.22; as seat1 27-23-0, avg margin +0.22.
    - Standardized shallow-search block seed 20000: 43-56-1, avg margin -1.06, 0 stalls/timeouts/fallback triggers.
      - Seat split: as seat0 28-21-1, avg margin +0.78; as seat1 15-35-0, avg margin -2.90.
      - This is a slight regression from `016` on the same search gate: `016` was 44-55-1, avg margin -1.01, seat1 16-34-0, seat1 avg margin -2.46.
    - vs policy-guided-search teacher using `016` prior at `policy_prior_weight=3000`, 100 games seed 66200: 45-54-1, avg margin -0.57, 0 stalls/timeouts/fallback triggers.
      - Seat split: as seat0 23-27-0, avg margin -0.10; as seat1 22-28-0, avg margin -1.04.
  - Current read: reject `018`; keep `016` as the current best fast neural parent. The high-margin seat1 correction set produced a narrow direct head-to-head edge over `016`, but it did not preserve the standardized search gate and worsened seat1 search performance. The likely issue is still not enough reliable supervised signal for seat1 race/tactical decisions, or the correction rows are too sparse/off-policy to train by simple imitation without damaging general search behavior. Next step should not be another small supervised fine-tune on this same signal. Prefer one of:
    - generate a much larger policy-guided/search league with both seats and train with game/result-aware weighting, not seat1-only rows;
    - improve the search teacher/feature representation for final-round timing and race threats, then regenerate corrections;
    - use the policy-guided search bot itself as a stronger inference-time player while separately working on neural distillation.
- Mixed league/result-weighted candidate, `legal_action_scorer_019_league_result_weighted`:
  - Generated a 300-game manifest-audited league block around `016`, all with `--source-type league`, `--include-state-snapshots`, `--swap-seats`, 0 stalls/timeouts/fallback triggers:
    - `data/corpus_league_019_016_vs_014`: 50 games, 3,182 steps, top-level seat W-L-D 28-22-0.
      - Actor-level `016` vs `014`: 27-23-0.
    - `data/corpus_league_019_016_vs_001`: 50 games, 3,180 steps, top-level 28-22-0.
      - Actor-level `016` vs `001`: 33-17-0.
    - `data/corpus_league_019_016_vs_003`: 50 games, 3,130 steps, top-level 29-20-1.
      - Actor-level `016` vs `003`: 34-15-1.
    - `data/corpus_league_019_016_vs_flat`: 50 games, 3,144 steps, top-level 30-20-0.
      - Actor-level `016` vs old flat champion: 39-11-0.
    - `data/corpus_league_019_pgs016_vs_search`: 50 games, 3,282 steps, top-level 32-18-0.
      - Actor-level policy-guided search using `016` at `policy_prior_weight=3000` vs plain shallow search: 33-17-0.
    - `data/corpus_league_019_016_vs_pgs016`: 50 games, 3,158 steps, top-level 29-21-0.
      - Actor-level fast `016` vs policy-guided search using `016` at `policy_prior_weight=3000`: 16-34-0.
  - Trained `outputs/legal_action_scorer_019_league_result_weighted/`.
    - Parent: `outputs/legal_action_scorer_016_policy_guided_mix/legal_action_scorer_best.pt`.
    - Training data: same anchor/targeted/policy-guided mix used for `016`, plus all six `corpus_league_019_*` replay files.
    - Setup: 29,937 rows, 26,994 train / 2,943 game-heldout validation, 5 CPU epochs, batch size 256, learning rate `0.00003`, `--value-loss-weight 0.0`.
    - Result-aware weights: `--winner-action-weight 1.5`, `--loser-action-weight 0.5`, `--draw-action-weight 1.0`.
    - Best validation checkpoint: epoch 5, validation loss 0.8728, validation accuracy 0.7227.
  - Benchmarks for `legal_action_scorer_019_league_result_weighted/legal_action_scorer_best.pt`:
    - vs `016`, 100 games seed 68000: 47-53-0, avg margin +0.01, 0 stalls/timeouts/fallback triggers.
      - Seat split: as seat0 30-20-0, avg margin +1.38; as seat1 17-33-0, avg margin -1.36.
    - vs `014`, 100 games seed 68100: 50-50-0, avg margin -0.24, 0 stalls/timeouts/fallback triggers.
      - Seat split: as seat0 26-24-0, avg margin +0.22; as seat1 24-26-0, avg margin -0.70.
    - Standardized shallow-search block seed 20000: 40-60-0, avg margin -1.29, 0 stalls/timeouts/fallback triggers.
      - Seat split: as seat0 23-27-0, avg margin -0.68; as seat1 17-33-0, avg margin -1.90.
      - This is a regression from `016` on the same gate: `016` was 44-55-1, avg margin -1.01.
    - vs policy-guided-search teacher using `016` prior at `policy_prior_weight=3000`, 100 games seed 68200: 30-70-0, avg margin -2.07, 0 stalls/timeouts/fallback triggers.
      - Seat split: as seat0 16-34-0, avg margin -1.94; as seat1 14-36-0, avg margin -2.20.
  - Current read: reject `019`; keep `016` as the current best fast neural parent. The mixed league/result-weighted recipe improved validation metrics but failed the real gates. It appears to overfit or imitate weaker/ambiguous league actions, diluting the search-strength signal from `016`. Do not keep scaling this exact result-weighted supervised recipe. Next main path should be either:
    - use policy-guided search directly for stronger play while designing a better distillation objective;
    - add richer action/state features for final-round timing, race threats, and denial value before more supervised distillation;
    - or build a true self-play/league RL-style update where game outcomes drive policy improvement, rather than row-level imitation of mixed actors.
- Action-set-aware scorer candidate, `legal_action_scorer_020_set_scorer`:
  - Added `LegalActionSetScorerModel` / `LegalActionSetScorerConfig`.
    - Encodes state and legal actions like the existing scorer.
    - Pools masked mean/max/count context across the whole legal action set.
    - Scores each action using state encoding + action encoding + set-context encoding.
  - Added checkpoint kind `legal_action_set_scorer` across:
    - training CLI: `run_legal_action_supervised.py --architecture set`
    - inference bot: `LegalActionSetScorerBot`
    - benchmark CLI
    - replay corpus generation
    - policy/search disagreement and targeted-corpus tooling
    - policy-guided search prior loading
  - Trained `outputs/legal_action_scorer_020_set_scorer/`.
    - Init checkpoint: `outputs/legal_action_scorer_016_policy_guided_mix/legal_action_scorer_best.pt`.
    - Warm-start loaded 24 compatible tensors and skipped 4 old score-head tensors.
    - Training data: same five-corpus mix used for `016`.
    - Setup: 10,861 rows, 9,835 train / 1,026 game-heldout validation, 6 CPU epochs, batch size 256, learning rate `0.00005`, `--value-loss-weight 0.0`.
    - Best validation checkpoint: epoch 6, validation loss 1.2333, validation accuracy 0.6170.
      - This is worse than `016` validation accuracy 0.6384, so validation did not indicate a win.
  - Benchmarks for `legal_action_scorer_020_set_scorer/legal_action_set_scorer_best.pt`:
    - vs `016`, 100 games seed 70000: 41-57-2, avg margin -1.21, 0 timeouts.
      - Seat split: as seat0 19-30-1, avg margin -1.16; as seat1 22-27-1, avg margin -1.26.
    - Standardized shallow-search block seed 20000: 40-60-0, avg margin -1.46, 0 timeouts.
      - Seat split: as seat0 24-26-0, avg margin -0.44; as seat1 16-34-0, avg margin -2.48.
      - This is worse than `016` on the same gate: `016` was 44-55-1, avg margin -1.01.
    - vs policy-guided-search teacher using `016` prior at `policy_prior_weight=3000`, 100 games seed 70100: 28-72-0, avg margin -2.65, 0 timeouts.
      - Seat split: as seat0 14-36-0, avg margin -2.64; as seat1 14-36-0, avg margin -2.66.
  - Current read: reject `020`; keep `016` as the best fast neural scorer. The architecture wiring is useful and should stay, but the naive pooled-set scorer did not improve strength. The likely issue is that mean/max set context is too weak or too abruptly introduced by partial warm-start. If revisiting this branch, try a real pairwise/attention action-set ranker or distill from policy-guided search rankings directly into the set-aware model, instead of treating pooled context as a small drop-in replacement.
- Attention action-set scorer candidate, `legal_action_scorer_021_attention_scorer`:
  - Added `LegalActionAttentionScorerModel` / `LegalActionAttentionScorerConfig`.
    - Encodes state and each legal action like the previous scorer family.
    - Adds a state-conditioned masked self-attention encoder over the legal action rows.
    - Scores each action after it has directly interacted with the other legal alternatives.
  - Added checkpoint kind `legal_action_attention_scorer` across:
    - training CLI: `run_legal_action_supervised.py --architecture attention`
    - inference bot: `LegalActionAttentionScorerBot`
    - benchmark CLI
    - replay corpus generation
    - policy/search disagreement and targeted-corpus tooling
    - policy-guided search prior loading
  - Trained `outputs/legal_action_scorer_021_attention_scorer/`.
    - Init checkpoint: `outputs/legal_action_scorer_016_policy_guided_mix/legal_action_scorer_best.pt`.
    - Warm-start loaded 24 compatible tensors and skipped 4 old score-head tensors.
    - Training data: same five-corpus mix used for `016` and `020`.
    - Setup: 10,861 rows, 9,835 train / 1,026 game-heldout validation, 6 CPU epochs, batch size 256, learning rate `0.00005`, `--value-loss-weight 0.0`, attention layers 2, heads 4, feedforward 256.
    - Best validation checkpoint: epoch 6, validation loss 1.1085, validation accuracy 0.6287.
      - Better than `020` validation accuracy 0.6170, still below `016` validation accuracy 0.6384.
  - Benchmarks for `legal_action_scorer_021_attention_scorer/legal_action_attention_scorer_best.pt`:
    - vs `016`, 100 games seed 71000: 54-46-0, avg margin +0.35, 0 timeouts.
      - Seat split: as seat0 29-21-0, avg margin +0.94; as seat1 25-25-0, avg margin -0.24.
    - Standardized shallow-search block seed 20000: 42-58-0, avg margin -0.72, 0 timeouts.
      - Seat split: as seat0 24-26-0, avg margin +0.12; as seat1 18-32-0, avg margin -1.56.
      - This improves over `020` (40-60-0, avg margin -1.46) but remains below `016` win rate on the same gate (`016` was 44-55-1, avg margin -1.01).
    - vs policy-guided-search teacher using `016` prior at `policy_prior_weight=3000`, 100 games seed 71100: 38-62-0, avg margin -2.00, 0 timeouts.
      - Seat split: as seat0 23-27-0, avg margin -0.16; as seat1 15-35-0, avg margin -3.84.
  - Current read: do not promote `021`; keep `016` as the best fast neural scorer. `021` is architecturally more promising than `020` and beats `016` directly on one 100-game block, but it still fails the standardized search gate and has a severe seat1 failure against policy-guided search. The next useful experiment should not be another plain imitation run. If continuing this branch, use the attention scorer with an explicit search-ranking objective, ideally from policy-guided-search ranked action scores, to preserve the direct head-to-head gain while repairing search/teacher alignment.
- Attention + search-ranking distillation candidate, `legal_action_scorer_022_attention_distill`:
  - Trained `outputs/legal_action_scorer_022_attention_distill/`.
    - Init checkpoint: `outputs/legal_action_scorer_021_attention_scorer/legal_action_attention_scorer_best.pt`.
    - Warm-start loaded 56 compatible tensors, skipped 0.
    - Training data: same five-corpus mix used for `016`/`020`/`021`, plus `data/corpus_search_distill_full_001/improved_search_policy_margin500_no_take.jsonl`.
    - Distillation signal:
      - `data/corpus_targeted_search_full_001/targeted_search_disagreements.jsonl`: 1,842 rows with `search_action_scores`, all with at least two scored actions, avg 4.80 scores/row.
      - `data/corpus_search_distill_full_001/improved_search_policy_margin500_no_take.jsonl`: 4,454 rows with scores, 4,440 rows with at least two scored actions, avg 7.04 scores/row.
    - Setup: 15,315 rows, 13,888 train / 1,427 game-heldout validation, 4 CPU epochs, batch size 256, learning rate `0.00003`, `--value-loss-weight 0.0`, `--search-distill-loss-weight 0.05`, `--search-distill-temperature 1000`.
    - Best validation checkpoint: epoch 4, validation loss 0.9847, validation accuracy 0.7022.
  - Benchmarks for `legal_action_scorer_022_attention_distill/legal_action_attention_scorer_best.pt`:
    - vs `016`, 100 games seed 72000: 48-52-0, avg margin -0.64, 0 timeouts.
      - Seat split: as seat0 29-21-0, avg margin +1.02; as seat1 19-31-0, avg margin -2.30.
      - This is worse directly than `021` vs `016` (54-46-0), so distillation traded away some direct neural-pool strength.
    - vs `021`, 100 games seed 72200: 52-48-0, avg margin +0.21, 0 timeouts.
      - Seat split: as seat0 32-18-0, avg margin +2.18; as seat1 20-30-0, avg margin -1.76.
      - This suggests `022` is not purely a search-specialist regression, but the edge is small and seat-sensitive.
    - Standardized shallow-search block seed 20000: 55-45-0, avg margin +1.01, 0 timeouts.
      - Seat split: as seat0 29-21-0, avg margin +1.72; as seat1 26-24-0, avg margin +0.30.
      - This is the first clear win over the standardized search gate. It substantially improves over `016` (44-55-1), `020` (40-60-0), and `021` (42-58-0).
    - vs policy-guided-search teacher using `016` prior at `policy_prior_weight=3000`, 100 games seed 72100: 41-59-0, avg margin -0.94, 0 timeouts.
      - Seat split: as seat0 24-26-0, avg margin +0.06; as seat1 17-33-0, avg margin -1.94.
      - Better than `021` on this teacher gate (38-62-0, avg margin -2.00), but still not beating the teacher and still seat1-limited.
  - Confirmation after the initial 022 result:
    - vs `016`, fresh 200 games seed 73000: 109-91-0, avg margin +0.34, 0 timeouts.
      - Seat split: as seat0 55-45-0, avg margin +0.90; as seat1 54-46-0, avg margin -0.22.
      - Combined direct vs `016` across seed 72000 + 73000 blocks: 157-143-0 over 300 games, avg margin about +0.01.
      - Conclusion: the initial 48-52 direct result was likely noise; direct strength against `016` is roughly even to slightly positive.
    - vs shallow search, fresh 100 games seed 73200: 39-61-0, avg margin -1.18, 0 timeouts.
      - Seat split: as seat0 24-26-0, avg margin +0.16; as seat1 15-35-0, avg margin -2.52.
      - Combined search vs seed 20000 + 73200 blocks: 94-106-0 over 200 games, avg margin about -0.09.
      - Conclusion: the seed-20000 55-45 search win did not fully generalize. 022 is much more search-competitive than prior models, but not a confirmed search-beater.
  - Current read: `022` is a real improvement on search alignment and is not clearly weaker than `016`, but it is still not a clean champion. Keep `016` as the conservative best fast neural scorer; keep `022` as the best search-aligned neural candidate and a promising parent for the next distillation branch. The next useful experiment should target the remaining seat1 gap without destroying the search gain:
    - run a small distillation-weight sweep from `021` or `022` (`0.02`, `0.05`, `0.1`) using the same data and gates; or
    - generate policy-guided-search ranked action scores from the stronger PGS016 teacher, especially for seat1 positions, then train attention scorer on those rankings rather than plain shallow-search rankings.
- Policy-guided-search ranked distillation candidate, `legal_action_scorer_023_attention_pgs_distill`:
  - Extended `run_improve_replay_actions.py` / `improve_replay_actions.py` to support policy-guided search ranking:
    - `--policy-prior-checkpoint`
    - `--policy-prior-weight`
    - `--policy-prior-device`
    - optional `--player-id-filter`
    - metadata source is `policy_guided_search` when a prior is used.
    - The prior checkpoint is loaded once and reused across ranked rows.
  - Generated `data/corpus_pgs016_ranked_023/seat1_pgs_ranked.jsonl`.
    - Input: `data/corpus_policy_guided_search_016_w3000_seat1_vs_search_500/seat1_teacher_weighted.jsonl`.
    - Ranked first 3,000 seat1 positions with PGS016:
      - prior checkpoint `outputs/legal_action_scorer_016_policy_guided_mix/legal_action_scorer_best.pt`
      - `policy_prior_weight=3000`
      - depth 2 / max branching 10 / buy 6 / reserve 3 / take 3.
    - Output summary:
      - 3,000 rows written, all seat1, all source `policy_guided_search`.
      - 300 top-action changes, 2,700 unchanged.
      - 2,998 rows have at least two ranked actions.
      - Average ranked actions per row: 5.84.
  - Trained `outputs/legal_action_scorer_023_attention_pgs_distill/`.
    - Parent: `outputs/legal_action_scorer_022_attention_distill/legal_action_attention_scorer_best.pt`.
    - Training data: same 022 mix plus the new PGS016-ranked seat1 corpus.
    - Setup: 18,315 rows, 16,683 train / 1,632 game-heldout validation, 3 CPU epochs, batch size 256, learning rate `0.00002`, `--value-loss-weight 0.0`, `--search-distill-loss-weight 0.03`, `--search-distill-temperature 1000`.
    - Best validation checkpoint: epoch 3, validation loss 1.1822, validation accuracy 0.6330.
  - Benchmarks for `legal_action_scorer_023_attention_pgs_distill/legal_action_attention_scorer_best.pt`:
    - vs `016`, 200 games seed 81000: 99-100-1, avg margin +0.31, 0 timeouts.
      - Seat split: as seat0 52-48-0, avg margin +0.59; as seat1 47-52-1, avg margin +0.03.
      - Directly, 023 is essentially tied with 016.
    - vs `022`, 100 games seed 81200: 50-50-0, avg margin +0.05, 0 timeouts.
      - Seat split: as seat0 26-24-0, avg margin +0.16; as seat1 24-26-0, avg margin -0.06.
      - 023 does not clearly improve over 022.
    - Standardized shallow-search block seed 20000: 45-55-0, avg margin -0.50, 0 timeouts.
      - Seat split: as seat0 25-25-0, avg margin +0.06; as seat1 20-30-0, avg margin -1.06.
      - This regresses from 022's 55-45-0 on the same seed block.
    - Fresh shallow-search block seed 73200: 38-62-0, avg margin -1.32, 0 timeouts.
      - Seat split: as seat0 23-27-0, avg margin -0.26; as seat1 15-35-0, avg margin -2.38.
      - This does not repair 022's fresh-block search instability.
    - vs policy-guided-search teacher using `016` prior at `policy_prior_weight=3000`, 100 games seed 81300: 40-59-1, avg margin -0.78, 0 timeouts.
      - Seat split: as seat0 25-25-0, avg margin +0.52; as seat1 15-34-1, avg margin -2.08.
      - This is not better than 022's PGS teacher gate (41-59-0).
  - Current read: reject `023`; keep `016` as conservative best fast scorer and `022` as the best search-aligned neural candidate. The first PGS-ranked seat1 distillation attempt did not fix seat1 and damaged the strongest seed-20000 search result. Likely causes:
    - only 3,000 PGS-ranked seat1 rows, with just 300 top-action changes, may be too weak/sparse;
    - training only on top-action relabels plus soft scores may still overfit seat1 distribution without improving general search alignment;
    - PGS-ranked labels from positions where the teacher is seat1 may not be enough; need mixed-seat PGS rankings and/or higher-quality high-margin PGS disagreement rows.
  - Next useful direction should be diagnostic before more training:
    - compare 022 vs 023 policy/search disagreement by seat/action type on the same rows;
    - inspect where 023 changed decisions versus 022, especially seat1 token/reserve and late buy actions;
    - if continuing PGS-ranked data, mine high-margin PGS-vs-022 disagreements instead of ranking broad teacher trajectory rows.
- 022 vs 023 policy-diff diagnostics:
  - Wrote:
    - `outputs/analysis/022_vs_023_policy_diff_pgs016_ranked_seat1.json`
    - `outputs/analysis/022_vs_023_policy_diff_search_distill_full.json`
    - `outputs/analysis/022_vs_023_policy_diff_targeted_search_full.json`
  - On the new PGS016-ranked seat1 corpus:
    - 3,000 rows, all seat1.
    - 022/023 top-action agreement: 95.63%.
    - PGS teacher top-match rate:
      - 022: 70.27%.
      - 023: 70.77%.
    - 023 teacher gains/losses: 57 gains vs 42 losses.
    - Average teacher-value gap improved only slightly: 421.6 -> 419.2.
    - Largest shifts:
      - `RESERVE_VISIBLE -> RESERVE_DECK`: 49.
      - `RESERVE_VISIBLE -> RESERVE_VISIBLE`: 25.
      - `TAKE_TOKENS -> TAKE_TOKENS`: 23.
    - By phase:
      - early teacher match improved 60.99% -> 62.27%.
      - mid improved 77.33% -> 77.78%.
      - late slightly regressed 71.08% -> 70.87%.
  - On `corpus_search_distill_full_001`:
    - 4,454 rows.
    - 022/023 top-action agreement: 97.53%.
    - Shallow-search teacher top-match rate:
      - 022: 79.91%.
      - 023: 80.60%.
    - 023 teacher gains/losses: 52 gains vs 21 losses.
    - Average teacher-value gap improved only slightly: 1960.5 -> 1954.4.
    - Seat1 match improved 79.39% -> 80.18%, but seat1 value gap slightly worsened 64.5 -> 66.1.
  - On `corpus_targeted_search_full_001`:
    - 1,842 rows.
    - 022/023 top-action agreement: 92.78%.
    - Search teacher top-match rate:
      - 022: 34.91%.
      - 023: 35.56%.
    - 023 teacher gains/losses: 36 gains vs 24 losses.
    - Early rows improved 27.37% -> 29.29%, but late rows regressed 45.00% -> 43.82%.
  - Diagnostic conclusion:
    - 023 did learn a small amount from the PGS-ranked data, but the actual policy changed very little.
    - The PGS-ranked broad seat1 corpus mainly nudged reserve choice variants (`RESERVE_VISIBLE -> RESERVE_DECK`) and same-type token/reserve choices.
    - It did not create enough high-value tactical changes to improve benchmarks, and it slightly damaged the strongest 022 search result.
    - Do not continue by adding more broad teacher-trajectory rows in the same format. If continuing PGS distillation, mine high-margin PGS-vs-022 disagreements and overweight only those corrections, especially seat1 late/race decisions.
- High-margin PGS-vs-022 targeted correction candidate, `legal_action_scorer_024_attention_pgs_targeted`:
  - Extended `run_targeted_search_corpus.py` so targeted disagreement mining can use policy-guided search directly:
    - `--policy-prior-checkpoint`
    - `--policy-prior-weight`
    - `--policy-prior-device`
    - `--player-id-filter`
    - `--phase-filter`
    - `--training-weight`
    - Summary now includes rows skipped by player/phase and written rows by seat.
  - Verification after the tooling change:
    - `python -m py_compile src\splendor_ai\training\run_targeted_search_corpus.py` passed.
    - `python -m pytest -q tests\test_targeted_search_corpus.py tests\test_improve_replay_actions.py --basetemp .codex_pytest_tmp_targeted_final` passed: 8 passed.
    - Full suite passed: `python -m pytest -q --basetemp .codex_pytest_tmp_all` -> 152 passed.
  - Generated `data/corpus_pgs022_targeted_024/targeted_pgs_disagreements.jsonl`.
    - Input: `data/corpus_policy_guided_search_016_w3000_seat1_vs_search_500/seat1_teacher_weighted.jsonl`.
    - Compared 022 top action against PGS016 depth-2 search with:
      - prior checkpoint `outputs/legal_action_scorer_016_policy_guided_mix/legal_action_scorer_best.pt`
      - `policy_prior_weight=3000`
      - search branching depth 2 / max 10 / buy 6 / reserve 3 / take 3.
    - Filters: `--player-id-filter 1`, `--phase-filter mid late`, `--min-search-margin 500`, `--include-missing-model-action`, `--late-buy-turn-min 45`, `--training-weight 2.0`, first 8,000 source rows.
    - Summary:
      - 8,000 rows seen, 5,520 analyzed, 2,480 skipped by phase.
      - 1,448 PGS-vs-022 disagreements.
      - 346 high-margin correction rows written.
      - All written rows are seat1.
      - Phase split: late 226, mid 120.
      - Target action types: `BUY_VISIBLE` 150, `RESERVE_VISIBLE` 82, `RESERVE_DECK` 58, `TAKE_TOKENS` 37, `BUY_RESERVED` 19.
  - Trained `outputs/legal_action_scorer_024_attention_pgs_targeted/`.
    - Parent: `outputs/legal_action_scorer_022_attention_distill/legal_action_attention_scorer_best.pt`.
    - Training data: exact 022 mix plus only the 346-row targeted PGS-vs-022 correction corpus. The broad 023 PGS-ranked corpus was intentionally excluded.
    - Setup: 15,661 rows, 13,943 train / 1,718 game-heldout validation, 3 CPU epochs, batch size 256, learning rate `0.00002`, `--value-loss-weight 0.0`, `--search-distill-loss-weight 0.03`, `--search-distill-temperature 1000`.
    - Best validation checkpoint: epoch 3, validation loss 1.0866, validation accuracy 0.6420.
  - Benchmarks for `legal_action_scorer_024_attention_pgs_targeted/legal_action_attention_scorer_best.pt`:
    - vs `016`, 200 games seed 83000: 95-103-2, avg margin +0.10, 0 timeouts.
      - Seat split: as seat0 63-37-0, avg margin +2.24; as seat1 32-66-2, avg margin -2.04.
      - This fails to improve the conservative fast neural champion and still has a severe seat1 weakness.
    - vs `022`, 100 games seed 83200: 47-53-0, avg margin -0.37, 0 timeouts.
      - Seat split: as seat0 27-23-0, avg margin +1.26; as seat1 20-30-0, avg margin -2.00.
      - The targeted PGS correction did not improve over its parent.
    - Standardized shallow-search block seed 20000: 47-51-2, avg margin +0.22, 0 timeouts.
      - Seat split: model as seat0 26-23-1, avg margin +1.06; as seat1 21-28-1, avg margin -0.62.
      - This is worse than 022 on the same seed block (55-45-0).
    - Fresh shallow-search block seed 73200: 37-63-0, avg margin -1.47, 0 timeouts.
      - Seat split: model as seat0 20-30-0, avg margin -0.90; as seat1 17-33-0, avg margin -2.04.
      - This is also worse than 022's fresh search block (39-61-0, avg margin -1.18).
    - vs policy-guided-search teacher using `016` prior at `policy_prior_weight=3000`, 100 games seed 83300: 34-66-0, avg margin -2.00, 0 timeouts.
      - Seat split: as seat0 21-29-0, avg margin -1.06; as seat1 13-37-0, avg margin -2.94.
      - This is clearly worse than 022 and 023 on the PGS teacher gate.
  - Current read: reject `024`. The experiment was useful because it tested the narrow high-margin correction hypothesis cleanly, but simple supervised fine-tuning on 346 targeted PGS-vs-022 rows did not fix seat1 and damaged the search/PGS gates. Keep `016` as conservative best fast scorer and `022` as the best search-aligned neural parent. Do not immediately scale more of the same targeted PGS rows; the failure mode points to optimization/representation or teacher-target mismatch, not just too few rows.
  - Best next direction:
    - Diagnose why the 346 correction rows move behavior in the wrong direction before training again.
    - Compare 022 vs 024 top-action changes on the targeted corpus and on search benchmarks, by phase/action/seat.
    - Measure whether 024 actually increased teacher top-match on the 346 targeted rows; if it did but benchmarks regressed, the rows are overfitting/off-policy.
    - Then either use a much lower correction weight / replay ratio, or move to online evaluation-style updates instead of more supervised correction rows.
- 022 vs 024 correction-learning diagnostics:
  - Added `run_policy_diff_against_teacher.py`.
    - Compares two checkpoints against either stored `search_action_scores` or a live policy-guided-search teacher.
    - Reports teacher top-match rates, before/after top-action agreement, gains/losses, teacher-value gaps, phase/action/seat buckets, and action-type shift matrices.
  - Verification after adding the diagnostic CLI:
    - `python -m py_compile src\splendor_ai\training\run_policy_diff_against_teacher.py` passed.
    - Full suite passed: `python -m pytest -q --basetemp .codex_pytest_tmp_all` -> 152 passed.
  - Stored-teacher diagnostic on the 346 rows actually used to train `024`:
    - Output: `outputs/analysis/022_vs_024_policy_diff_pgs022_targeted_346.json`.
    - Dataset: `data/corpus_pgs022_targeted_024/targeted_pgs_disagreements.jsonl`.
    - 346 rows analyzed, all rows are by construction PGS-vs-022 disagreements.
    - 022 teacher top-match: 0 / 346 = 0.00%.
    - 024 teacher top-match: 43 / 346 = 12.43%.
    - 022/024 top-action agreement: 291 / 346 = 84.10%.
    - 024 changed top action on only 55 rows; 43 of those changes hit the teacher top action and 12 changed to another non-teacher action.
    - Teacher-gap mean improved from 8,851.7 to 8,121.9, but the max-gap terminal-scale misses remained.
    - By teacher target type:
      - `RESERVE_VISIBLE`: 26 / 82 learned = 31.7%.
      - `RESERVE_DECK`: 6 / 58 learned = 10.3%.
      - `BUY_RESERVED`: 2 / 19 learned = 10.5%.
      - `BUY_VISIBLE`: 9 / 150 learned = 6.0%.
      - `TAKE_TOKENS`: 0 / 37 learned = 0.0%.
    - Interpretation: `024` did not strongly memorize the targeted correction rows. It mostly learned visible-reserve corrections and barely learned the larger late visible-buy or token-take corrections.
  - Live-PGS diagnostic on the broader first-8,000-row mined slice:
    - Output: `outputs/analysis/022_vs_024_policy_diff_pgs016_live_first8000_midlate.json`.
    - Input: `data/corpus_policy_guided_search_016_w3000_seat1_vs_search_500/seat1_teacher_weighted.jsonl`.
    - Same slice/filter as the 024 mining pass: first 8,000 rows, seat1 only, mid/late only, PGS016 prior weight 3000, depth 2 / max 10 / buy 6 / reserve 3 / take 3, seed 82000.
    - 8,000 rows seen, 2,480 skipped by phase, 5,520 analyzed.
    - 022 teacher top-match: 4,072 / 5,520 = 73.77%.
    - 024 teacher top-match: 4,089 / 5,520 = 74.08%.
    - Net improvement: +17 teacher matches, only +0.31 percentage points.
    - 022/024 top-action agreement: 5,321 / 5,520 = 96.39%, so the trained policy barely moved on the broader distribution.
    - Gains/losses: 90 teacher gains vs 73 teacher losses.
    - Phase split:
      - Mid: 74.97% -> 75.13%, 54 gains / 48 losses.
      - Late: 71.28% -> 71.89%, 36 gains / 25 losses.
    - By teacher target type:
      - `RESERVE_VISIBLE`: 73.24% -> 75.71%, 28 gains / 15 losses.
      - `RESERVE_DECK`: 30.73% -> 37.61%, 16 gains / 1 loss.
      - `BUY_VISIBLE`: 75.70% -> 75.79%, 30 gains / 27 losses.
      - `BUY_RESERVED`: 90.15% -> 89.29%, 4 gains / 12 losses.
      - `TAKE_TOKENS`: 56.91% -> 56.04%, 12 gains / 18 losses.
    - Interpretation: the broad effect is not overfitting in the strong sense; `024` did not over-learn the correction file. It made a very small teacher-alignment improvement, concentrated in reserve targets, while slightly regressing buy-reserved and token-take teacher alignment. That tiny net alignment gain was not enough to improve play and coincided with worse search/PGS benchmarks.
  - Current read after diagnostics:
    - The failure is not simply "the model memorized 346 off-policy rows and overfit." It barely learned those rows.
    - The supervised update mostly nudged reserve choices, did not solve late buy/race misses, and caused small offsetting losses elsewhere.
    - Do not fix this by just adding more of the same rows at the same recipe. The next training attempt should either:
      - lower the correction weight/replay ratio and train as a gentler regularizer, then gate immediately; or
      - change the target/optimization style, e.g. pairwise/ranking loss focused on teacher-vs-model alternatives, or online league/search evaluation, rather than top-1 imitation of sparse corrections.
- Stronger correction-pressure ablation, `legal_action_scorer_025_attention_pgs_targeted_pressure`:
  - Purpose: test whether the current CE/distillation objective can actually move onto the 346 high-margin PGS-vs-022 correction rows when given much stronger pressure.
  - Trained `outputs/legal_action_scorer_025_attention_pgs_targeted_pressure/`.
    - Parent: `outputs/legal_action_scorer_022_attention_distill/legal_action_attention_scorer_best.pt`.
    - Training data: exact 022 anchor/search-distill mix, plus `data/corpus_pgs022_targeted_024/targeted_pgs_disagreements.jsonl` repeated 8 times.
    - The targeted rows already have `training_weight=2.0`, so effective correction pressure was much higher than `024`.
    - Setup: 18,083 rows, 16,211 train / 1,872 game-heldout validation, 2 CPU epochs, batch size 256, learning rate `0.00005`, `--value-loss-weight 0.0`, `--search-distill-loss-weight 0.03`, `--search-distill-temperature 1000`.
    - Best validation checkpoint: epoch 2, validation loss 1.2215, validation accuracy 0.5999.
    - Validation accuracy dropped materially, as expected from heavy correction pressure.
  - Diagnostic on the 346 targeted rows:
    - Output: `outputs/analysis/022_vs_025_policy_diff_pgs022_targeted_346.json`.
    - 022 teacher top-match: 0 / 346 = 0.00%.
    - 025 teacher top-match: 123 / 346 = 35.55%.
    - 022/025 top-action agreement: 180 / 346 = 52.02%.
    - Teacher-gap mean improved from 8,851.7 to 4,171.1.
    - By teacher target type:
      - `RESERVE_VISIBLE`: 58 / 82 learned = 70.7%.
      - `RESERVE_DECK`: 26 / 58 learned = 44.8%.
      - `TAKE_TOKENS`: 11 / 37 learned = 29.7%.
      - `BUY_VISIBLE`: 25 / 150 learned = 16.7%.
      - `BUY_RESERVED`: 3 / 19 learned = 15.8%.
    - Interpretation: stronger pressure can move the policy, but still learns reserve corrections much more readily than late/race buy corrections.
  - Live-PGS diagnostic on the broader first-8,000-row mined slice:
    - Output: `outputs/analysis/022_vs_025_policy_diff_pgs016_live_first8000_midlate.json`.
    - Same slice/filter as `024`: first 8,000 source rows, seat1 only, mid/late only, PGS016 prior weight 3000, depth 2 / max 10 / buy 6 / reserve 3 / take 3, seed 82000.
    - 5,520 rows analyzed.
    - 022 teacher top-match: 4,072 / 5,520 = 73.77%.
    - 025 teacher top-match: 3,918 / 5,520 = 70.98%.
    - Net broad alignment change: -154 teacher matches, -2.79 percentage points.
    - 022/025 top-action agreement: 85.25%.
    - Gains/losses: 248 teacher gains vs 402 teacher losses.
    - Phase split:
      - Mid: 74.97% -> 72.42%, 138 gains / 233 losses.
      - Late: 71.28% -> 68.00%, 110 gains / 169 losses.
    - By teacher target type:
      - `RESERVE_VISIBLE`: 73.24% -> 77.04%, 63 gains / 43 losses.
      - `RESERVE_DECK`: 30.73% -> 49.54%, 44 gains / 3 losses.
      - `BUY_VISIBLE`: 75.70% -> 70.99%, 98 gains / 247 losses.
      - `BUY_RESERVED`: 90.15% -> 83.77%, 9 gains / 68 losses.
      - `TAKE_TOKENS`: 56.91% -> 55.90%, 34 gains / 41 losses.
    - Interpretation: `025` passes the "can the loss move the policy?" check but fails the "does movement preserve broad teacher alignment?" check. It over-rotates into reserve/deck-reserve behavior and damages buy alignment, especially `BUY_VISIBLE` and `BUY_RESERVED`.
  - Current read: reject `025` without full benchmarks. The pre-benchmark diagnostic failed: broad PGS alignment regressed substantially, so spending time on 100-game gates is not justified unless specifically requested.
  - Next useful technical step:
    - Stop increasing top-1 correction replay pressure.
    - Implement a pairwise/ranking preference objective for correction rows: teacher action should score above the model's previous/top competing action by a margin, without forcing the entire distribution to imitate the sparse correction row.
    - Use the 346 correction rows as a small controlled test for the new loss:
      - require meaningful targeted-row improvement;
      - require no broad 5,520-row PGS alignment regression;
      - only then benchmark.
- Pairwise preference-loss implementation and first candidate, `legal_action_scorer_026_attention_pgs_pairwise_pref`:
  - Implemented first-class pairwise/ranking support in legal-action scorer training:
    - `LegalActionReplayDataset` now detects rows with a valid `model_top_action_index`.
    - For those rows, the replay `action_index` is the teacher/preferred action and `model_top_action_index` is the competitor.
    - Batches now include:
      - `preference_competitor_row`
      - `preference_mask`
      - `policy_sample_weight`
      - `distill_sample_weight`
      - `preference_sample_weight`
    - Added `compute_pairwise_preference_loss`, using a softplus margin loss: competitor score should be below teacher score by the configured margin.
    - Added CLI flags to `run_legal_action_supervised.py`:
      - `--preference-loss-weight`
      - `--preference-margin`
      - `--preference-rows-policy-weight`
      - `--preference-rows-distill-weight`
      - `--preference-rows-preference-weight`
    - This allows correction rows to contribute pairwise preference loss while their CE/distillation weights are set to zero, avoiding sparse top-1 imitation on those rows.
  - Verification after implementation:
    - Focused tests passed: `python -m pytest -q tests\test_legal_action_supervised.py tests\test_legal_action_scorer.py --basetemp .codex_pytest_tmp_pref_focused` -> 21 passed.
    - Full suite passed: `python -m pytest -q --basetemp .codex_pytest_tmp_all` -> 154 passed.
  - Trained `outputs/legal_action_scorer_026_attention_pgs_pairwise_pref/`.
    - Parent: `outputs/legal_action_scorer_022_attention_distill/legal_action_attention_scorer_best.pt`.
    - Training data: exact 022 anchor/search-distill mix plus `data/corpus_pgs022_targeted_024/targeted_pgs_disagreements.jsonl` repeated 4 times.
    - Preference row settings:
      - `--preference-loss-weight 0.5`
      - `--preference-margin 1.0`
      - `--preference-rows-policy-weight 0.0`
      - `--preference-rows-distill-weight 0.0`
      - `--preference-rows-preference-weight 1.0`
    - Other setup: 16,699 rows, 14,915 train / 1,784 game-heldout validation, 3 CPU epochs, batch size 256, learning rate `0.00003`, `--value-loss-weight 0.0`, anchor `--search-distill-loss-weight 0.05`.
    - Best validation checkpoint: epoch 3. Validation loss 1.2506 and accuracy 0.5975; the total loss scale includes the preference term, so compare diagnostics rather than validation loss.
  - Diagnostic on the 346 targeted rows:
    - Output: `outputs/analysis/022_vs_026_policy_diff_pgs022_targeted_346.json`.
    - 022 teacher top-match: 0 / 346 = 0.00%.
    - 026 teacher top-match: 115 / 346 = 33.24%.
    - 022/026 top-action agreement: 124 / 346 = 35.84%.
    - Teacher-gap mean improved from 8,851.7 to 4,012.1.
    - By teacher target type:
      - `RESERVE_DECK`: 41 / 58 learned = 70.7%.
      - `RESERVE_VISIBLE`: 39 / 82 learned = 47.6%.
      - `TAKE_TOKENS`: 9 / 37 learned = 24.3%.
      - `BUY_VISIBLE`: 23 / 150 learned = 15.3%.
      - `BUY_RESERVED`: 3 / 19 learned = 15.8%.
    - Interpretation: pairwise loss can move the model, but this setting still mostly learns reserve/deck-reserve corrections and barely learns late/race buy corrections.
  - Live-PGS diagnostic on the broader first-8,000-row mined slice:
    - Output: `outputs/analysis/022_vs_026_policy_diff_pgs016_live_first8000_midlate.json`.
    - Same slice/filter as `024`/`025`: first 8,000 source rows, seat1 only, mid/late only, PGS016 prior weight 3000, depth 2 / max 10 / buy 6 / reserve 3 / take 3, seed 82000.
    - 5,520 rows analyzed.
    - 022 teacher top-match: 4,072 / 5,520 = 73.77%.
    - 026 teacher top-match: 3,557 / 5,520 = 64.44%.
    - Net broad alignment change: -515 teacher matches, -9.33 percentage points.
    - 022/026 top-action agreement: 74.11%.
    - Gains/losses: 280 teacher gains vs 795 teacher losses.
    - Phase split:
      - Mid: 74.97% -> 67.85%, 181 gains / 446 losses.
      - Late: 71.28% -> 57.39%, 99 gains / 349 losses.
    - By teacher target type:
      - `RESERVE_DECK`: 30.73% -> 55.50%, 71 gains / 17 losses.
      - `BUY_VISIBLE`: 75.70% -> 62.10%, 106 gains / 536 losses.
      - `BUY_RESERVED`: 90.15% -> 83.33%, 10 gains / 73 losses.
      - `RESERVE_VISIBLE`: 73.24% -> 64.52%, 43 gains / 89 losses.
      - `TAKE_TOKENS`: 56.91% -> 52.55%, 50 gains / 80 losses.
    - Action shifts show the failure mode clearly:
      - From 022 `BUY_VISIBLE`, 026 changed 498 rows to `RESERVE_DECK`.
      - From 022 `RESERVE_VISIBLE`, 026 changed 138 rows to `RESERVE_DECK`.
      - From 022 `BUY_RESERVED`, 026 changed 51 rows to `RESERVE_DECK`.
    - Interpretation: this first pairwise setting over-rotates hard toward `RESERVE_DECK` and damages broad PGS alignment. Reject `026` without full benchmarks.
  - Current read:
    - Pairwise preference loss is implemented and functional.
    - The first setting is too strong/broad and should not be benchmarked.
    - If continuing this branch, use a much gentler ablation before changing the objective again:
      - no repeated correction rows or at most 1-2 repeats;
      - `--preference-loss-weight` around `0.02` to `0.1`;
      - consider filtering preference rows to late `BUY_VISIBLE` / `BUY_RESERVED` corrections only, because reserve corrections are easy to learn but harmful when over-amplified.
    - Gate any such run with the same two diagnostics before benchmarks.
- Gentle buy-only pairwise preference candidate, `legal_action_scorer_027_attention_buy_pairwise_pref`:
  - Purpose: test the safer version of the pairwise idea after `026` over-rotated into reserves.
  - Created filtered buy/race preference corpus:
    - Output: `data/corpus_pgs022_buy_pref_027/buy_preference_rows.jsonl`.
    - Source: `data/corpus_pgs022_targeted_024/targeted_pgs_disagreements.jsonl`.
    - Filters: teacher action type `BUY_VISIBLE` or `BUY_RESERVED`, `turn_index >= 45`, valid `model_top_action_index`.
    - 346 rows seen, 169 rows written.
    - Target types: `BUY_VISIBLE` 150, `BUY_RESERVED` 19.
    - Phase split: late 136, mid 33.
  - Created sanitized anchor copy:
    - Output: `data/corpus_pgs022_buy_pref_027/targeted_search_full_no_pref_anchor.jsonl`.
    - Source: `data/corpus_targeted_search_full_001/targeted_search_disagreements.jsonl`.
    - Removed `model_top_action_index`, `model_top_action_payload`, `model_top_score`, and `model_top_search_value` from 1,842 rows.
    - Reason: the global preference-row flags would otherwise turn this older targeted-search anchor into preference-only training rows. The sanitized copy keeps it as a normal CE/distillation anchor.
  - Trained `outputs/legal_action_scorer_027_attention_buy_pairwise_pref/`.
    - Parent: `outputs/legal_action_scorer_022_attention_distill/legal_action_attention_scorer_best.pt`.
    - Training data: exact 022 mix, except using the sanitized targeted-search anchor, plus the 169-row buy-only preference corpus once.
    - Preference settings:
      - `--preference-loss-weight 0.03`
      - `--preference-margin 1.0`
      - `--preference-rows-policy-weight 0.0`
      - `--preference-rows-distill-weight 0.0`
      - `--preference-rows-preference-weight 1.0`
    - Other setup: 15,484 rows, 13,731 train / 1,753 game-heldout validation, 3 CPU epochs, batch size 256, learning rate `0.00003`, `--value-loss-weight 0.0`, anchor `--search-distill-loss-weight 0.05`.
    - Best validation checkpoint: epoch 3, validation loss 1.0509, validation accuracy 0.6828.
  - Tests for the pairwise implementation were already completed before this run:
    - Focused tests: 21 passed.
    - Full suite: 154 passed.
  - Diagnostics:
    - Buy-only targeted rows:
      - Output: `outputs/analysis/022_vs_027_policy_diff_buy_pref_169.json`.
      - 022 teacher top-match: 0 / 169 = 0.00%.
      - 027 teacher top-match: 26 / 169 = 15.38%.
      - 022/027 top-action agreement: 129 / 169 = 76.33%.
      - Gap mean improved from 15,921.9 to 1,407.7.
      - By target type: `BUY_VISIBLE` 24 / 150 = 16.0%; `BUY_RESERVED` 2 / 19 = 10.5%.
    - Full 346 targeted rows:
      - Output: `outputs/analysis/022_vs_027_policy_diff_pgs022_targeted_346.json`.
      - 027 teacher top-match: 32 / 346 = 9.25%.
      - 022/027 top-action agreement: 84.10%.
    - Broader first-8,000-row live-PGS slice:
      - Output: `outputs/analysis/022_vs_027_policy_diff_pgs016_live_first8000_midlate.json`.
      - 5,520 rows analyzed.
      - 022 teacher top-match: 4,072 / 5,520 = 73.77%.
      - 027 teacher top-match: 4,073 / 5,520 = 73.79%.
      - Net broad alignment change: +1 teacher match; 115 gains vs 114 losses.
      - 022/027 top-action agreement: 94.75%.
      - Useful targeted shift:
        - `BUY_VISIBLE` teacher alignment improved 75.70% -> 77.21%, 76 gains / 28 losses.
      - Harmful offsetting shifts:
        - `RESERVE_VISIBLE` teacher alignment regressed 73.24% -> 65.65%, 3 gains / 43 losses.
        - `TAKE_TOKENS` regressed 56.91% -> 55.90%.
        - `BUY_RESERVED` regressed 90.15% -> 89.29%.
      - Interpretation: this gentle buy-only setting is diagnostically much safer than `026` and does improve broad `BUY_VISIBLE` alignment, but the effect is small and offset by reserve/token regressions.
  - Benchmarks:
    - vs `016`, 200 games seed 87000: 95-105-0, avg margin 0.00, 0 timeouts.
      - Seat split: as seat0 50-50-0, avg margin +0.48; as seat1 45-55-0, avg margin -0.48.
    - vs `022`, 100 games seed 87200: 46-54-0, avg margin -0.02, 0 timeouts.
      - Seat split: as seat0 20-30-0, avg margin +0.10; as seat1 26-24-0, avg margin -0.14.
    - Standardized shallow-search block seed 20000: 44-55-1, avg margin +0.37, 0 timeouts.
      - Seat split: model as seat0 22-28-0, avg margin +0.72; as seat1 22-27-1, avg margin +0.02.
      - This is worse in win rate than `022` on the same seed block (55-45-0) and only matches the older `016` win/loss shape despite positive score margin.
  - Current read: reject `027` as a promotion candidate.
    - It is the first correction run that preserved broad PGS alignment and improved broad `BUY_VISIBLE` alignment, so the buy-only/gentle direction is technically sound.
    - It did not produce playing-strength gains. Direct gates vs `016`/`022` and standardized search failed.
    - Do not run more broad benchmarks for this checkpoint unless needed for a specific comparison.
  - Next useful direction:
    - The supervised correction path is producing useful local alignment but not stronger play.
    - If continuing pairwise, tune only one small change at a time:
      - slightly raise buy-only preference pressure (`0.05` or `0.08`) while keeping no repeats; or
      - restrict even further to late `BUY_VISIBLE` rows where the model top was a reserve, since that is the clearest harmful race miss.
    - A stronger alternative is to stop supervised patching and move to online league/search evaluation updates, because local teacher-match improvements have repeatedly failed to translate into benchmark gains.
- Diagnostic-gated candidate harness:
  - Added `run_candidate_harness.py`.
    - Config-driven harness for training variants, running diagnostics, applying JSON gates, and only then running benchmarks.
    - Supports `module` stages via `python -m ...` or raw `command` stages.
    - Supports `skip_if_outputs_exist` so expensive completed stages can be reused.
    - Supports gates over JSON paths such as `overall.after_teacher_match_rate`, `by_teacher_action_type.BUY_VISIBLE.after_teacher_match_rate`, or `matchup.candidate_wins`.
    - Variant status becomes `diagnostic_failed`, `benchmark_failed`, `passed`, etc., based on gates.
  - Tests:
    - Added `tests/test_candidate_harness.py`.
    - Focused harness tests passed: 4 passed.
    - Full suite passed after adding the harness: `python -m pytest -q --basetemp .codex_pytest_tmp_all` -> 158 passed.
  - Added reusable config:
    - `configs/candidate_harness_027_existing.json`.
    - This validates the harness on existing `027` outputs without retraining.
  - Harness validation run:
    - Command:
      - `$env:PYTHONPATH='src'; python -m splendor_ai.training.run_candidate_harness --config configs\candidate_harness_027_existing.json --output-path outputs\harness\candidate_harness_027_existing_summary.json`
    - Summary:
      - Training skipped because the `027` checkpoint already exists.
      - Diagnostics skipped because JSON outputs already exist, then gates were evaluated from disk.
      - Diagnostic gates passed:
        - buy-target teacher match: actual 0.1538 >= 0.10.
        - broad teacher alignment no-drop: actual 0.73786 >= 0.7376.
        - broad `BUY_VISIBLE` alignment improved: actual 0.77212 >= 0.7569.
      - Benchmark gate failed:
        - `matchup.candidate_wins` vs `022`: actual 46 < required 50.
      - Final variant status: `benchmark_failed`, matching the manual `027` verdict.
  - Current read:
    - Future supervised candidates should go through this harness rather than manual train/diagnose/benchmark loops.
    - The next useful use of the harness is a small sweep over buy-only pairwise settings, e.g. `preference_loss_weight` 0.05 and 0.08, or a stricter late-buy-visible-where-model-reserved corpus. But given repeated benchmark failures, consider moving to league/search evaluation updates before spending many more CPU hours on supervised patch variants.
- Smoke-trained the first real scorer on `data/corpus_checkpoint_greedy_snapshots_003/improved_search_policy_margin500_no_take.jsonl`.
  - Output: `outputs/legal_action_scorer_smoke_001/`
  - 800 samples, 720 train / 80 validation, 1 epoch, CPU.
  - Small model config: state hidden 64, action hidden 32, scorer hidden 64.
  - Result: train loss 4.3837, train accuracy 0.2542, validation loss 4.0197, validation accuracy 0.2125.
  - This is a functionality smoke test only, not a strength candidate.
- Verification after scorer scaffolding:
  - `python -m pytest -q --basetemp .codex_pytest_tmp_all`
  - 111 tests passed.
- Verification after scorer benchmark support:
  - `python -m pytest tests/test_run_benchmark.py tests/test_legal_action_supervised.py tests/test_model_bot.py -q --basetemp .codex_pytest_tmp_benchmark`
  - 13 tests passed.
- First full legal-action scorer candidate:
  - Training output: `outputs/legal_action_scorer_001/`
  - Trained on snapshot/improved corpora `001`, `002`, and `003`, 2,583 total rows, 8 epochs, CPU.
  - Best validation checkpoint: `outputs/legal_action_scorer_001/legal_action_scorer_best.pt`.
  - Validation accuracy reached 0.6705 by epoch 7; best validation loss was epoch 4.
  - 100-game sanity benchmark, seed block 8000: 100-0 vs random, 96-4 vs greedy, 0 timeouts/stalls/fallback triggers.
  - Direct 100-game head-to-head vs old flat champion `outputs/policy_improve_margin500_002/supervised_policy_value.pt`, seed block 9000: 64-32-4 for the legal-action scorer, 0 timeouts/stalls/fallback triggers.
  - Direct-match average score margin was +2.25; candidate score average 13.91 vs baseline 11.66.
  - Seat split: candidate as seat 0 went 36-11-3, candidate as seat 1 went 28-21-1. This shows a first-player effect but still a positive result for the scorer from both seats.
  - Confirmation direct 100-game head-to-head on seed block 10000: 63-37-0 for the legal-action scorer, 0 timeouts/stalls/fallback triggers.
  - Confirmation average score margin was +3.14; candidate score average 13.93 vs baseline 10.79.
  - Confirmation seat split: candidate as seat 0 went 35-15-0, candidate as seat 1 went 28-22-0.
  - Shallow-search benchmark, depth 2 / max branching 10 / buy 6 / reserve 3 / take 3, seed block 11000, 20 games: scorer 4, search 16, draws 0, 0 timeouts/stalls/fallback triggers.
  - Search benchmark average score margin was -4.25 for the scorer; scorer average score 10.00 vs search 14.25.
  - Search benchmark seat split: scorer as seat 0 went 4-6, scorer as seat 1 went 0-10. This is the clearest current weakness and confirms search remains a stronger benchmark than old neural/greedy opponents.
  - Old flat champion `policy_improve_margin500_002` on the same search benchmark/seed block went 2-18, draws 0, 0 timeouts/stalls/fallback triggers.
  - Old flat champion average search margin was -5.85; model average score 9.20 vs search 15.05. The scorer therefore improved against search, but search remains clearly stronger than both neural checkpoints.
- First scorer self-play/search-relabel iteration, `legal_action_scorer_002`:
  - Generated `data/corpus_legal_scorer_selfplay_snapshots_001` from `legal_action_scorer_001` self-play.
  - 10 games, 626 steps, 0 stalled, 0 timed out, 0 fallback triggers.
  - Relabeled with shallow search using depth 2 / max branching 10 / buy 6 / reserve 3 / take 3, `--min-search-margin 500`, `--exclude-changed-action-types TAKE_TOKENS`, and `--write-unchanged`.
  - Relabel output: `data/corpus_legal_scorer_selfplay_snapshots_001/improved_search_policy_margin500_no_take.jsonl`.
  - 340 rows written: 25 high-confidence changed labels and 315 unchanged anchors.
  - Filtered 112 low-margin changes, 79 changed rows where the original action was pruned/missing from search candidates, and 95 selected-token-take changes.
  - Fine-tuned from `outputs/legal_action_scorer_001/legal_action_scorer_best.pt` on the three prior improved corpora plus the new scorer self-play relabel corpus.
  - Output: `outputs/legal_action_scorer_002/`; best validation checkpoint was epoch 4.
  - Training dataset: 2,923 total rows, 2,631 train / 292 validation, 4 epochs, CPU, `--value-loss-weight 0.0`.
  - Final train accuracy 0.7355; validation accuracy 0.6952; validation loss 0.9917.
  - Direct 100-game head-to-head vs `legal_action_scorer_001`, seed block 13000: `legal_action_scorer_002` went 54-45-1, 0 timeouts/stalls/fallback triggers.
  - Confirmation direct 100-game head-to-head vs `legal_action_scorer_001`, seed block 14000: `legal_action_scorer_002` went 49-51-0, 0 timeouts/stalls/fallback triggers.
  - Direct 100-game head-to-head vs old flat champion `policy_improve_margin500_002`, seed block 13000: `legal_action_scorer_002` went 62-38-0, 0 timeouts/stalls/fallback triggers.
  - Shallow-search benchmark on the same seed block/settings as `legal_action_scorer_001`, 20 games seed 11000: `legal_action_scorer_002` went 6-14, improving over `legal_action_scorer_001` at 4-16 and old flat champion at 2-18.
  - Search benchmark average score margin improved to -2.90 from `legal_action_scorer_001` at -4.25, but search remains clearly stronger.
  - Conclusion: `legal_action_scorer_002` is directionally useful because it improved the search benchmark, but it is not clearly stronger than `legal_action_scorer_001` in direct neural head-to-head yet.
- Second scorer self-play/search-relabel iteration, `legal_action_scorer_003`:
  - Generated `data/corpus_legal_scorer_selfplay_snapshots_002` from `legal_action_scorer_002` self-play.
  - 25 games, 1,528 steps, 0 stalled, 0 timed out, 0 fallback triggers.
  - Relabeled with shallow search using the same settings and gate as `002`: depth 2 / max branching 10 / buy 6 / reserve 3 / take 3, `--min-search-margin 500`, `--exclude-changed-action-types TAKE_TOKENS`, and `--write-unchanged`.
  - Relabel output: `data/corpus_legal_scorer_selfplay_snapshots_002/improved_search_policy_margin500_no_take.jsonl`.
  - 834 rows written: 68 high-confidence changed labels and 766 unchanged anchors.
  - Filtered 279 low-margin changes, 219 changed rows where the original action was pruned/missing from search candidates, and 196 selected-token-take changes.
  - Fine-tuned from `outputs/legal_action_scorer_002/legal_action_scorer_best.pt` on the three prior improved corpora plus both scorer self-play relabel corpora.
  - Output: `outputs/legal_action_scorer_003/`; best validation checkpoint was epoch 1.
  - Training dataset: 3,757 total rows, 3,382 train / 375 validation, 4 epochs, CPU, `--value-loss-weight 0.0`.
  - Epoch 1 best validation: train accuracy 0.7732, validation accuracy 0.7840, validation loss 0.9121. Final epoch validation accuracy stayed 0.7840 but validation loss rose to 0.9191.
  - Direct 100-game head-to-head vs `legal_action_scorer_002`, seed block 16000: `legal_action_scorer_003_best` went 46-54-0, 0 timeouts/stalls/fallback triggers.
  - Confirmation direct 100-game head-to-head vs `legal_action_scorer_002`, seed block 17000: `legal_action_scorer_003_best` went 60-40-0, 0 timeouts/stalls/fallback triggers.
  - Final epoch checkpoint also went 60-40-0 vs `legal_action_scorer_002` on seed block 17000.
  - Direct 100-game head-to-head vs `legal_action_scorer_001`, seed block 16000: `legal_action_scorer_003_best` went 47-53-0, 0 timeouts/stalls/fallback triggers.
  - Direct 100-game head-to-head vs old flat champion `policy_improve_margin500_002`, seed block 16000: `legal_action_scorer_003_best` went 64-35-1, 0 timeouts/stalls/fallback triggers.
  - Shallow-search benchmark on the same seed block/settings as `001` and `002`, 20 games seed 11000: `legal_action_scorer_003_best` went 10-10, improving over `002` at 6-14, `001` at 4-16, and old flat champion at 2-18.
  - Search benchmark average score margin improved to -0.35 from `002` at -2.90 and `001` at -4.25.
  - Confirmation shallow-search benchmark, 20 games seed 12000: `legal_action_scorer_003_best` went 3-17, average score margin -5.25, 0 timeouts/stalls/fallback triggers.
  - Combined shallow-search result across seed blocks 11000 and 12000: `legal_action_scorer_003_best` went 13-27, average score margin -2.80.
  - Standardized 100-game shallow-search benchmark, seed block 20000:
    - `legal_action_scorer_001`: 30-70-0, average margin -3.92.
    - `legal_action_scorer_002`: 25-75-0, average margin -3.99.
    - `legal_action_scorer_003`: 29-71-0, average margin -3.86.
  - Conclusion: the larger search benchmark says `001` and `003` are effectively tied against search, while `002` is slightly worse. The earlier `003` 10-10 result was noise. `003` is not a clean promotion because it is volatile against `002`, lost to `001` on the checked direct block, and does not clearly beat `001` on the standardized 100-game search block.
- First mixed league/search-relabel iteration, `legal_action_scorer_004`:
  - Generated four 5-game league corpora with per-seat checkpoint pairings:
    - `data/corpus_league_004_003_vs_001`
    - `data/corpus_league_004_003_vs_002`
    - `data/corpus_league_004_003_vs_flat`
    - `data/corpus_league_004_001_vs_002`
  - Combined corpus size: 20 games, 1,258 steps, 0 stalled, 0 timed out, 0 fallback triggers.
  - Relabeled all four corpora together with shallow search using depth 2 / max branching 10 / buy 6 / reserve 3 / take 3, `--min-search-margin 500`, `--exclude-changed-action-types TAKE_TOKENS`, and `--write-unchanged`.
  - Relabel output: `data/corpus_league_004_improved/improved_search_policy_margin500_no_take.jsonl`.
  - 699 rows written: 68 high-confidence changed labels and 631 unchanged anchors.
  - Filtered 211 low-margin changes, 180 changed rows where the original action was pruned/missing from search candidates, and 168 selected-token-take changes.
  - Fine-tuned from stable parent `outputs/legal_action_scorer_001/legal_action_scorer_best.pt` on the three prior improved corpora, both scorer self-play relabel corpora, and the new league relabel corpus.
  - Output: `outputs/legal_action_scorer_004/`; best validation checkpoint was epoch 3.
  - Training dataset: 4,456 total rows, 4,011 train / 445 validation, 4 epochs, CPU, `--value-loss-weight 0.0`.
  - Best validation epoch 3: train accuracy 0.7886, validation accuracy 0.7753, validation loss 0.9639.
  - Direct 100-game head-to-head vs `legal_action_scorer_001`, seed block 25000: `legal_action_scorer_004_best` went 48-52-0, average margin -0.12, 0 timeouts/stalls/fallback triggers.
  - Direct 100-game head-to-head vs `legal_action_scorer_003`, seed block 25000: `legal_action_scorer_004_best` went 51-49-0, average margin +0.07, 0 timeouts/stalls/fallback triggers.
  - Direct 100-game head-to-head vs old flat champion `policy_improve_margin500_002`, seed block 25000: `legal_action_scorer_004_best` went 63-37-0, average margin +1.69, 0 timeouts/stalls/fallback triggers.
  - Standardized 100-game shallow-search benchmark, seed block 20000: `legal_action_scorer_004_best` went 21-79-0, average margin -4.54, 0 timeouts/stalls/fallback triggers.
  - Conclusion: do not promote `legal_action_scorer_004`. The mixed league corpus did not improve strength; `004` is tied-ish with `001`/`003`, still beats the old flat champion, but regressed versus search compared with `001` at 30-70 and `003` at 29-71 on the same 100-game search block.
- Search-ranking distillation support, ready for the next candidate:
  - Implemented after rejecting `004`.
  - Future relabeled corpora include `search_action_scores` for the searched candidate list.
  - Training can add a soft cross-entropy term from those search values using `--search-distill-loss-weight`.
  - Tiny smoke run succeeded:
    - relabel: `data/distill_smoke_001/improved_search_scores.jsonl`, 12 rows
    - train: `outputs/legal_action_scorer_distill_smoke_001/`, 1 CPU epoch from `legal_action_scorer_001`
  - This smoke checkpoint is only a functionality check, not a strength candidate.
- First search-ranking distillation candidate, `legal_action_scorer_005`:
  - Attempted to regenerate one full combined depth-2 search-relabeled corpus with `search_action_scores` from the existing snapshot sources.
  - CPU relabeling was too slow for a full pass in-session. The combined relabel job was stopped after 502 valid written rows.
  - Distillation slice:
    - `data/corpus_search_distill_005/improved_search_policy_margin500_no_take.jsonl`
    - 502 rows, all with `search_action_scores`
    - 216 changed labels, 286 unchanged anchors
    - no normal summary JSON, because the long relabel was intentionally stopped before completion
  - Trained from `outputs/legal_action_scorer_001/legal_action_scorer_best.pt`.
  - Training used the prior top-1 improved corpora plus the 502-row scored distillation slice:
    - `corpus_checkpoint_greedy_snapshots_001`
    - `corpus_checkpoint_greedy_snapshots_002`
    - `corpus_checkpoint_greedy_snapshots_003`
    - `corpus_legal_scorer_selfplay_snapshots_001`
    - `corpus_legal_scorer_selfplay_snapshots_002`
    - `corpus_league_004_improved`
    - `corpus_search_distill_005`
  - Training command used `--search-distill-loss-weight 0.1`, `--search-distill-temperature 1000`, `--value-loss-weight 0.0`, 4 epochs, CPU.
  - Output: `outputs/legal_action_scorer_005/`; best validation checkpoint was epoch 4.
  - Dataset: 4,958 rows, 4,463 train / 495 validation.
  - Best validation: loss 1.2618, accuracy 0.7515.
  - Direct 100-game head-to-head vs `legal_action_scorer_001`, seed block 26000: `005` went 48-52-0, average margin -0.57, 0 fallback triggers.
  - Direct 100-game head-to-head vs `legal_action_scorer_003`, seed block 26000: `005` went 47-53-0, average margin -0.72, 0 fallback triggers.
  - Standardized 100-game shallow-search benchmark, seed block 20000: `005` went 25-75-0, average margin -4.38, 0 fallback triggers.
  - Conclusion: do not promote `legal_action_scorer_005`. The small scored distillation slice did not improve over `001`/`003`; it is below `001` at 30-70 and `003` at 29-71 on the same search block.
- Full search-ranking distillation candidate, `legal_action_scorer_006`:
  - Completed the full overnight depth-2 scored relabel pass:
    - `data/corpus_search_distill_full_001/improved_search_policy_margin500_no_take.jsonl`
    - summary: `data/corpus_search_distill_full_001/improved_search_policy_margin500_no_take_summary.json`
    - 8,326 rows read
    - 4,454 rows written
    - 1,223 high-confidence changed labels
    - 3,231 unchanged anchors
    - 1,113 low-margin changes filtered
    - 1,787 changed rows filtered because original action was pruned/missing
    - 972 selected-token-take changes filtered
    - 0 missing snapshots, 0 rows without search candidate
    - all 4,454 written rows have `search_action_scores`
  - Trained from `outputs/legal_action_scorer_001/legal_action_scorer_best.pt`.
  - Training used only the new full scored relabel corpus, with `--search-distill-loss-weight 0.1`, `--search-distill-temperature 1000`, `--value-loss-weight 0.0`, 4 epochs, CPU.
  - Output: `outputs/legal_action_scorer_006/`; best validation checkpoint was epoch 3.
  - Dataset: 4,454 rows, 4,009 train / 445 validation.
  - Best validation: loss 1.1412, accuracy 0.8022.
  - Direct 100-game head-to-head vs `legal_action_scorer_001`, seed block 27000: `006` went 49-50-1, average margin -0.03, 0 fallback triggers.
  - Direct 100-game head-to-head vs `legal_action_scorer_003`, seed block 27000: `006` went 53-47-0, average margin -0.20, 0 fallback triggers.
  - Standardized 100-game shallow-search benchmark, seed block 20000: `006` went 28-72-0, average margin -4.13, 0 fallback triggers.
  - Conclusion: do not promote `legal_action_scorer_006`. Full search-score distillation with this recipe is not clearly better than `001` or `003`; it remains slightly below `001` at 30-70 and `003` at 29-71 on the same standardized search block.
- Distillation ablation candidates, `legal_action_scorer_007` and `legal_action_scorer_008`:
  - Both used the same full scored corpus:
    - `data/corpus_search_distill_full_001/improved_search_policy_margin500_no_take.jsonl`
  - Both trained from `outputs/legal_action_scorer_001/legal_action_scorer_best.pt`, 4 epochs, CPU, `--value-loss-weight 0.0`.
  - `legal_action_scorer_007`:
    - `--search-distill-loss-weight 0.03`, `--search-distill-temperature 1000`
    - output: `outputs/legal_action_scorer_007/`; best validation checkpoint was epoch 4
    - best validation: loss 0.9200, accuracy 0.7865
    - standardized 100-game shallow-search benchmark, seed block 20000: 22-78-0, average margin -4.80, 0 fallback triggers
    - conclusion: reject
  - `legal_action_scorer_008`:
    - top-1 only on the same corpus, `--search-distill-loss-weight 0.0`
    - output: `outputs/legal_action_scorer_008/`; best validation checkpoint was epoch 4
    - best validation: loss 0.8038, accuracy 0.8022
    - standardized 100-game shallow-search benchmark, seed block 20000: 28-72-0, average margin -4.41, 0 fallback triggers
    - conclusion: reject
  - Overall conclusion from `006`/`007`/`008`: this full scored relabel corpus improves supervised validation metrics but does not improve playing strength versus the shallow-search benchmark. Do not keep tuning this exact supervised/distillation recipe as the main path.
- Weighted gated-relabel candidates, `legal_action_scorer_009` and `legal_action_scorer_010`:
  - Added `training_weight` support for legal-action scorer training.
    - `run_improve_replay_actions.py` can write `training_weight`.
    - `LegalActionReplayDataset` loads it as `sample_weight`.
    - legal-action scorer policy/value/distillation losses use weighted means.
    - Old corpora remain backward-compatible with default weight 1.0.
  - Built weighted copy of the full scored corpus:
    - `data/corpus_search_distill_weighted_001/improved_search_policy_margin500_no_take_weighted.jsonl`
    - source: `data/corpus_search_distill_full_001/improved_search_policy_margin500_no_take.jsonl`
    - 4,454 rows total
    - 1,223 changed rows weighted 2.0
    - 3,231 unchanged anchors weighted 0.5
  - `legal_action_scorer_009`:
    - parent: `outputs/legal_action_scorer_001/legal_action_scorer_best.pt`
    - top-1 only, no soft distillation, 4 epochs, CPU
    - output: `outputs/legal_action_scorer_009/`; best validation checkpoint was epoch 3
    - best validation: loss 1.0685, accuracy 0.7955
    - standardized 100-game shallow-search benchmark, seed block 20000: 27-73-0, average margin -4.04, 0 fallback triggers
    - conclusion: reject
  - `legal_action_scorer_010`:
    - parent: `outputs/legal_action_scorer_003/legal_action_scorer_best.pt`
    - same weighted corpus and training setup
    - output: `outputs/legal_action_scorer_010/`; best validation checkpoint was epoch 2
    - best validation: loss 1.0450, accuracy 0.8090
    - standardized 100-game shallow-search benchmark, seed block 20000: 20-80-0, average margin -5.26, 0 fallback triggers
    - conclusion: reject
  - Overall conclusion from `009`/`010`: weighting confident search corrections more heavily did not improve strength. It may overfit to the shallow-search correction style or overweight noisy teacher corrections.
- Value-model branch:
  - Added a dedicated state-value model path.
    - `ValueMLP` / `ValueModelConfig` in `src/splendor_ai/training/model.py`
    - `ValueReplayDataset` derives targets from final score margin from the acting player's perspective.
    - `run_value_supervised.py` trains value checkpoints.
    - `run_value_evaluate.py` evaluates value checkpoints on arbitrary replay corpora.
    - `CheckpointValueEvaluator` can load a value checkpoint and feed a heuristic-scale value into `ShallowSearchBot`.
  - `outputs/value_model_001/`:
    - trained on the stronger snapshot/league set only, 8,326 rows, margin target scaled by 15, 12 epochs.
    - random row-split validation looked strong: MSE 0.0606, MAE 0.1308, sign accuracy 0.9167, Pearson 0.9239.
    - held-out `data/corpus_search_greedy_001/replays.jsonl` exposed poor generalization: MSE 0.2171, MAE 0.3669, sign accuracy 0.6789, Pearson 0.4074.
    - conclusion: not suitable for search integration.
  - `outputs/value_model_002/`:
    - trained on broader mixed data, 66,701 rows, holding out `data/corpus_search_greedy_001/replays.jsonl` for external evaluation.
    - random validation: MSE 0.0387, MAE 0.1132, sign accuracy 0.9551, Pearson 0.9543.
    - external eval on `corpus_search_greedy_001`: MSE 0.1556, MAE 0.3070, sign accuracy 0.7071, Pearson 0.5704.
    - external eval on `corpus_search_greedy_cutoff_001`: MSE 0.0775, MAE 0.1842, sign accuracy 0.8481, Pearson 0.8350.
    - external eval on `corpus_pass_smoke_001`: MSE 0.1244, MAE 0.2673, sign accuracy 0.7472, Pearson 0.8193.
  - Added game/source-level validation splitting for value training:
    - `run_value_supervised.py` now accepts `--validation-split-unit row|game|source`.
    - `game` groups by `(source replay path, game_seed)`, so positions from the same game cannot leak across train and validation.
    - Use `--validation-split-unit game` for normal value experiments; `source` is stricter and useful when testing whole-corpus generalization.
  - `outputs/value_model_003/`:
    - same broad mixed data as `value_model_002`, but trained with `--validation-split-unit game`.
    - dataset split: 66,701 total rows, 60,132 train / 6,569 validation, 12 epochs, CPU.
    - game-heldout validation selected epoch 1 as best; final epoch overfit the training games.
    - final epoch train: MSE 0.0205, MAE 0.0839, sign accuracy 0.9719, Pearson 0.9750.
    - final epoch game-heldout validation: MSE 0.2405, MAE 0.3557, sign accuracy 0.8159, Pearson 0.6994.
    - best-checkpoint external eval on `corpus_search_greedy_001`: MSE 0.1910, MAE 0.3441, sign accuracy 0.6999, Pearson 0.5063.
    - best-checkpoint external eval on `corpus_search_greedy_cutoff_001`: MSE 0.1723, MAE 0.3146, sign accuracy 0.6994, Pearson 0.6573.
    - best-checkpoint external eval on `corpus_pass_smoke_001`: MSE 0.1437, MAE 0.3013, sign accuracy 0.7415, Pearson 0.7858.
    - conclusion: do not promote `value_model_003`; it validates the split fix and shows the previous row-level validation was too optimistic, but `value_model_002` remains the stronger external-eval value checkpoint.
  - Value-assisted search probes using `value_model_002`:
    - value blend: `leaf = 0.9 * heuristic + 0.1 * neural_value * 15000`
    - 100-game direct search-vs-value-search benchmark, seed block 30000: value-search went 48-52-0 vs normal search, average margin -0.72.
    - stronger value blend 0.25, 20-game probe seed block 30100: 10-10-0, average margin +0.35.
    - conclusion: do not promote value-assisted search yet. The learned value is helpful enough to experiment with, but it does not clearly improve depth-2 search.

Important local artifacts:

- `data/corpus_greedy_random_003/replays.jsonl`
  - 2,000 games, 145,570 steps, 0 stalled, 0 timed out
- `data/corpus_search_greedy_002/replays.jsonl`
  - 500 games, 31,954 steps, 0 stalled, 0 timed out
- `data/corpus_search_greedy_003a/replays.jsonl`
  - 125 games, 7,992 steps, 0 stalled, 0 timed out
- `data/corpus_search_greedy_003b/replays.jsonl`
  - 125 games, 7,960 steps, 0 stalled, 0 timed out
- `data/corpus_search_greedy_003c/replays.jsonl`
  - 125 games, 7,942 steps, 0 stalled, 0 timed out
- `data/corpus_search_greedy_003d/replays.jsonl`
  - 125 games, 8,012 steps, 0 stalled, 0 timed out
- `data/corpus_search_greedy_004[1-5]/replays.jsonl`
  - 125 games total, 7,964 steps, 0 stalled, 0 timed out
- `outputs/warmstart_search_003/supervised_policy_value_best.pt`
  - prior champion checkpoint
- `outputs/warmstart_search_004/supervised_policy_value_best.pt`
  - previous champion checkpoint before gated policy improvement
- `outputs/policy_improve_margin500_001/supervised_policy_value.pt`
  - current local champion checkpoint, local/ignored artifact
- `outputs/benchmarks/warmstart_search_003_best_100g_seed5000.json`
  - 100-game benchmark: 98-2 vs random, 83-16-1 vs greedy, 2 repetition-cutoff games vs greedy
- `outputs/benchmarks/warmstart_search_003_best_100g_seed6000.json`
  - 100-game benchmark: 100-0 vs random, 86-14 vs greedy, 1 repetition-cutoff game vs greedy
- `outputs/benchmarks/warmstart_search_003_final_100g_seed5000.json`
  - final checkpoint benchmark: worse than `best_validation`
- `outputs/benchmarks/loop_fallback_enabled_100g_seed5000.json`
  - local/ignored artifact, 100-game benchmark vs greedy with fallback enabled: 85-15-0, 0 timed out, 100 completed, 8 fallback triggers
- `outputs/benchmarks/loop_fallback_enabled_100g_seed6000.json`
  - local/ignored artifact, 100-game benchmark vs greedy with fallback enabled: 85-15-0, 0 timed out, 100 completed, 5 fallback triggers
- `outputs/benchmarks/loop_fallback_disabled_seed5008_probe.json`
  - local/ignored artifact, confirms seed 5008 still hits `repetition_cutoff` when fallback is disabled
- `outputs/warmstart_mix_001/supervised_metrics.json`
  - local/ignored artifact, naive mixed search+greedy_random training; benchmark champion final checkpoint went 17-3 vs random, 5-15 vs greedy on seed block 7000
- `outputs/warmstart_mix_002_search_weighted/supervised_metrics.json`
  - local/ignored artifact, search-weighted mixed training; benchmark champion final checkpoint went 19-1 vs random, 9-11 vs greedy on seed block 7000
- `outputs/benchmarks/warmstart_search_003_best_20g_seed7000.json`
  - local/ignored artifact, then-current champion comparison on same seed block: 20-0 vs random, 16-4 vs greedy
- `outputs/warmstart_search_004/supervised_metrics.json`
  - local/ignored artifact, search-only training on `002 + 003[a-d] + 004[1-5]`; best-validation checkpoint selected as benchmark champion
- `outputs/benchmarks/warmstart_search_004_best_100g_seed5000.json`
  - local/ignored artifact, 100-game benchmark vs greedy: 85-15-0, 0 timed out, 4 fallback triggers
- `outputs/benchmarks/warmstart_search_004_best_100g_seed6000.json`
  - local/ignored artifact, 100-game benchmark vs greedy: 86-14-0, 0 timed out, 1 fallback trigger
- `outputs/benchmarks/warmstart_search_004_best_random_100g_seed5000.json`
  - local/ignored artifact, 100-game sanity benchmark vs random: 100-0-0, 0 timed out, 4 fallback triggers
- `outputs/warmstart_search_005/supervised_metrics.json`
  - local/ignored artifact, search-only training on `002 + 003[a-d] + 004[1-5] + 005[1-5]`; benchmark champion final checkpoint went 20-0 vs random and 15-5 vs greedy on seed block 7000, so it was not promoted
- `data/corpus_checkpoint_greedy_probe_001/`
  - local/ignored artifact, 2-game checkpoint-vs-greedy end-to-end replay probe: 126 steps, 0 stalled, 0 timed out, 0 fallback triggers
- `data/corpus_checkpoint_greedy_00[1-5]/`
  - local/ignored artifacts, 125 games total, 8,156 steps, 0 stalled, 0 timed out, 4 fallback triggers across 4 games
- `outputs/warmstart_champion_mix_001/supervised_metrics.json`
  - local/ignored artifact, search-only champion data plus checkpoint-vs-greedy data with fallback-triggered games excluded; benchmark champion final checkpoint went 20-0 vs random and 15-5 vs greedy on seed block 7000, so it was not promoted

Observed results:

- Scaling clean `greedy:random` helped reach roughly random-level play, but did not reliably beat `greedy`.
- Search bot token-loop behavior was improved with game-history and take/return churn penalties.
- After the search fix, `search:greedy` corpora remained clean through `corpus_search_greedy_002` and sharded `corpus_search_greedy_003[a-d]`.
- Added benchmark-side loop diagnostics so repeated-state issues surface as explicit `termination_reason` fields instead of opaque 400-turn outcomes.
- Added multi-corpus supervised training support; `run_supervised.py` now accepts repeated `--replay-path` arguments and records per-source sample counts in metrics.
- Training on `002 + 003[a-d]` produced `warmstart_search_003`, where the `best_validation` checkpoint is materially stronger than the final checkpoint.
- `warmstart_search_003_best.pt` beat `GreedyHeuristicBot` in two 100-game runs at 83% and 86%, while remaining near-perfect vs random.
- Added a conservative `CheckpointPolicyBot` loop fallback:
  - tracks repeated public state signatures and the bot's own consecutive non-progress actions
  - only intervenes when legal buy actions exist
  - chooses the model's highest-logit legal buy, and only if it is within the configured logit-gap threshold
  - can be disabled/tuned through benchmark CLI flags
  - reports per-game and aggregate fallback trigger counts in benchmark JSON
- Reused a single loaded checkpoint bot across benchmark games in both standalone and post-training benchmark paths, avoiding repeated checkpoint reload overhead while resetting per-game loop counters.
- Validated fallback on the two known greedy seed blocks:
  - seed block 5000: 85-15-0, 0 timeouts, 8 fallback triggers
  - seed block 6000: 85-15-0, 0 timeouts, 5 fallback triggers
  - combined: 170-30-0 across 200 games, 0 timeouts, 13 fallback triggers
- Ran two mixed-data warm-start experiments after accepting the fallback:
  - `warmstart_mix_001`: existing search corpora plus `corpus_greedy_random_003`, 209,430 samples, 8 epochs, final checkpoint selected by benchmark, 5-15 vs greedy on seed block 7000
  - `warmstart_mix_002_search_weighted`: search corpora repeated 3x plus `corpus_greedy_random_003`, 337,150 samples, 8 epochs, final checkpoint selected by benchmark, 9-11 vs greedy on seed block 7000
  - then-current champion `warmstart_search_003/supervised_policy_value_best.pt` went 16-4 vs greedy on the same seed block, so neither mixed checkpoint should be promoted
- Generated clean `search:greedy` shard block `004[1-5]`:
  - 125 games, 7,964 steps, 0 stalled, 0 timed out
- Trained `warmstart_search_004` on search-only data (`002 + 003[a-d] + 004[1-5]`):
  - 71,824 total samples, 8 epochs, best validation epoch 6
  - post-train 20-game seed-7000 benchmark: best-validation checkpoint went 20-0 vs random and 19-1 vs greedy
  - 100-game greedy validation: 85-15 on seed block 5000, 86-14 on seed block 6000
  - 100-game random sanity benchmark: 100-0 on seed block 5000
  - combined vs greedy: 171-29-0 across 200 games, 0 timeouts, 5 fallback triggers
  - compared to fallback-enabled `warmstart_search_003_best`: 170-30-0 across 200 games, 0 timeouts, 13 fallback triggers
  - promote `warmstart_search_004/supervised_policy_value_best.pt` as the new local champion; the win-rate gain is small, but fallback use is materially lower
- Added checkpoint-driven replay generation:
  - `run_replay_corpus.py` now accepts `checkpoint` as a bot name in `--seat0-bot`, `--seat1-bot`, and `--pairing`
  - checkpoint bots use `--checkpoint-path` and `--checkpoint-device`
  - checkpoint loop fallback can be enabled/disabled/tuned for replay generation
  - checkpoint models are cached per bot factory/seat, avoiding repeated checkpoint reloads while keeping separate per-seat state
  - replay rows now include `game_bot_metadata`, `game_loop_fallback_triggers_by_seat`, and `game_model_loop_fallback_triggers`
  - corpus `summary.json` now includes aggregate `loop_fallback_triggers`, `loop_fallback_games`, and CLI provenance metadata
- Added supervised replay filtering for checkpoint fallback games:
  - `SupervisedReplayDataset` accepts `max_model_loop_fallback_triggers`
  - `run_supervised.py` exposes `--max-game-model-loop-fallback-triggers`
  - rows without checkpoint fallback metadata count as 0, preserving compatibility with older search corpora
- Trained `warmstart_search_005` after generating clean `corpus_search_greedy_005[1-5]`, but it regressed in the quick seed-7000 benchmark:
  - benchmark-selected final checkpoint: 20-0 vs random, 15-5 vs greedy
  - `warmstart_search_004_best` on comparable post-train seed-7000 benchmark: 20-0 vs random, 19-1 vs greedy
  - do not promote `warmstart_search_005`
- Validated the new path with `data/corpus_checkpoint_greedy_probe_001`:
  - pairing `checkpoint:greedy`, then-current champion checkpoint, CUDA, swap seats
  - 2 games, 126 steps, 0 stalled, 0 timed out, 0 fallback triggers
- Generated `data/corpus_checkpoint_greedy_00[1-5]`:
  - 125 games, 8,156 steps, 0 stalled, 0 timed out
  - 4 fallback triggers across 4 games
- Trained `warmstart_champion_mix_001` on search data through `004[1-5]` plus checkpoint-vs-greedy data, excluding any game with more than 0 checkpoint fallback triggers:
  - dataset after filtering: 79,708 samples
  - benchmark champion final checkpoint: 20-0 vs random, 15-5 vs greedy on seed block 7000
  - current champion `warmstart_search_004_best` on the comparable seed block was 20-0 vs random, 19-1 vs greedy
  - do not promote `warmstart_champion_mix_001`
- Added the first policy-improvement tooling:
  - `run_replay_corpus.py --include-state-snapshots` writes compact pre-action state snapshots into replay rows without changing the default corpus format
  - `deserialize_state_snapshot` reconstructs exact `SplendorState` objects from those rows
  - `ShallowSearchBot.rank_actions(...)` exposes ranked/scored root actions
  - `run_improve_replay_actions.py` reads snapshot-enabled replay JSONL, reranks each position with shallow search, and writes relabeled policy rows plus a summary JSON
  - improved rows preserve `original_action_index`, `original_action_payload`, and `improvement_metadata`
  - metadata explicitly marks `target_semantics` as `policy_relabel_only_original_final_value_retained`
- Ran the first policy-improvement probe:
  - generated `data/corpus_checkpoint_greedy_snapshots_001`
  - 25 games, 1,650 steps, 0 stalled, 0 timed out, 0 loop fallback triggers
  - relabeled all 1,650 rows with depth-2 shallow search into `improved_search_policy_full.jsonl`
  - search changed 1,123 rows and left 527 rows unchanged
  - added `run_supervised.py --init-checkpoint` and fine-tuned from `outputs/warmstart_search_004/supervised_policy_value_best.pt`
  - trained `outputs/policy_improve_001` for 4 policy-only epochs with `--value-loss-weight 0.0`
  - quick seed-7000 benchmark: 19-1 vs random with 1 `repetition_cutoff`, 18-2 vs greedy with 0 timeouts
  - do not promote `policy_improve_001`; then-current champion remained `warmstart_search_004_best`, whose comparable quick seed-7000 result was 20-0 vs random and 19-1 vs greedy
- Added confidence-gated shallow-search relabeling:
  - `run_improve_replay_actions.py` now supports `--min-search-margin` and `--exclude-changed-action-types`
  - the gate requires the original action to be present in the pruned search candidate set when a margin is requested
  - filtered rows are counted separately for low margin, missing original score, and excluded selected action type
- Ran `policy_improve_margin500_001`, the first successful policy-improvement candidate:
  - relabel settings: `--min-search-margin 500 --exclude-changed-action-types TAKE_TOKENS --write-unchanged`
  - relabel output: `data/corpus_checkpoint_greedy_snapshots_001/improved_search_policy_margin500_no_take.jsonl`
  - 901 rows written: 374 high-confidence changed labels and 527 unchanged anchors
  - filtered 193 low-margin changes, 399 changes where the original action was pruned from search candidates, and 157 selected-token-take changes
  - fine-tuned from `outputs/warmstart_search_004/supervised_policy_value_best.pt` for 4 policy-only epochs with `--value-loss-weight 0.0`
  - quick seed-7000 benchmark: 20-0 vs random, 19-1 vs greedy, 0 timeouts
  - 100-game random sanity seed block 5000: 100-0-0, 0 timeouts, 0 fallback triggers
  - 100-game greedy seed block 5000: 94-6-0, 0 timeouts, 0 fallback triggers
  - 100-game greedy seed block 6000: 95-5-0, 0 timeouts, 1 fallback trigger
  - combined greedy validation: 189-11-0 across 200 games, 0 timeouts, 1 fallback trigger
  - this beats `warmstart_search_004_best` on the same two 100-game greedy blocks: 171-29-0 with 5 fallback triggers
  - promoted `outputs/policy_improve_margin500_001/supervised_policy_value.pt` as the new local champion after checkpointing the code/handoff
- Ran a second gated policy-improvement iteration, `policy_improve_margin500_002`:
  - generated `data/corpus_checkpoint_greedy_snapshots_002` from `policy_improve_margin500_001` vs greedy
  - 25 games, 1,634 steps, 0 stalled, 0 timed out, 0 fallback triggers
  - relabel settings unchanged: `--min-search-margin 500 --exclude-changed-action-types TAKE_TOKENS --write-unchanged`
  - relabel output: `data/corpus_checkpoint_greedy_snapshots_002/improved_search_policy_margin500_no_take.jsonl`
  - 882 rows written: 362 high-confidence changed labels and 520 unchanged anchors
  - filtered 164 low-margin changes, 426 changes where the original action was pruned from search candidates, and 162 selected-token-take changes
  - fine-tuned from `outputs/policy_improve_margin500_001/supervised_policy_value.pt` for 4 policy-only epochs with `--value-loss-weight 0.0`
  - quick seed-7000 benchmark: 20-0 vs random, 19-1 vs greedy, 0 timeouts
  - 100-game random sanity seed block 5000: 100-0-0, 0 timeouts, 0 fallback triggers
  - 100-game greedy seed block 8000: 97-3-0, 0 timeouts, 0 fallback triggers
  - 100-game greedy seed block 5000: 96-4-0, 0 timeouts, 1 fallback trigger
  - this improves over `policy_improve_margin500_001` on the checked blocks: seed 8000 improved from 87-13 to 97-3, seed 5000 improved from 94-6 to 96-4
  - recommend promoting `outputs/policy_improve_margin500_002/supervised_policy_value.pt` as the current local champion

Judgement calls made for checkpoint replay support:

- Used one shared `--checkpoint-path` for all `checkpoint` bots rather than separate per-seat checkpoint flags. This keeps the first champion-assisted workflow simple and unambiguous; per-seat checkpoint paths can be added later when we intentionally compare two different model generations.
- Kept the checkpoint loop fallback enabled by default for replay generation, but recorded fallback counts in every game and summary. This avoids known loop failures while preserving the ability to filter out or separately analyze fallback-influenced games.
- Cached checkpoint bots per factory/seat instead of reloading the model every game. This is much faster and relies on the existing per-game reset logic in `CheckpointPolicyBot`; separate factories still give each seat its own bot instance.
- Added provenance to both replay rows and `summary.json` rather than only printing it in logs. Future training filters can then exclude high-fallback games or select by checkpoint source without rerunning generation.
- Ran only a tiny 2-game checkpoint probe for validation. Checkpoint-generated data can reinforce model habits, so the next useful step is a controlled probe block and analysis, not immediate large-scale self-play.
- Trained one controlled champion-data mix after the 125-game probe, but rejected it on benchmark strength. This suggests naive imitation of the current champion against greedy is not enough; the next champion-assisted step needs a better target, weighting, or policy-improvement signal rather than simply adding model-play rows.

Judgement calls made for replay action improvement:

- Made state snapshots opt-in through `--include-state-snapshots`. This avoids bloating ordinary replay corpora while giving improvement scripts exact hidden decks, visible markets, nobles, and player zones when needed.
- Stored card/noble IDs rather than full card payloads. The base component table is canonical in code, so ID snapshots are smaller and less likely to drift.
- Added a search ranker instead of only calling `choose_action`. This lets relabeled rows record search value, heuristic score, loop penalty, and whether the original action was considered by the search candidate cap.
- Defaulted `run_improve_replay_actions.py` to writing changed rows only. This keeps improvement corpora focused and reduces duplicate imitation rows; `--write-unchanged` is available for diagnostics or full relabeled datasets.
- Kept original `final_value` on relabeled rows and marked that fact in metadata. This is the important caveat: if the action is changed, the old game outcome is no longer a clean value target. Use these rows primarily as policy-improvement targets, or train with reduced/zero value loss for this source until we add rollout-backed value labels.
- Defaulted relabeling to stateless per-row search. Because changing an action makes later original replay states off-policy, carrying a search history across all rows could inject inconsistent loop context. `--stateful-search-history` exists for sequential diagnostics.
- For `policy_improve_001`, fine-tuned from the current champion rather than training from scratch. A tiny relabeled corpus is not enough to relearn the whole policy; the right test was whether it can nudge the champion upward.
- After a power outage interrupted the first relabel attempt, verified the 25-game snapshot corpus was complete and restarted only the relabel step. The half-written relabel output was empty, so there was no useful partial data to salvage.
- Used `--write-unchanged` for the first training probe. Since 68% of actions changed, keeping unchanged rows gave the model both positive corrections and explicit "search agrees" anchors.
- Used `--value-loss-weight 0.0` for the fine-tune because changed actions retain original game outcomes and are not clean value targets.
- For the successful gated run, used `--min-search-margin 500` and excluded selected `TAKE_TOKENS` labels. This preserved the useful tactical corrections, mostly buy/reserve over original token takes, while avoiding the noisy high-volume token-take relabels that likely hurt the ungated run.
- Reused the same gate for `policy_improve_margin500_002` because the relabel distribution stayed similar to the first successful run. The model was slower in greedy benchmarks, so future iterations should track elapsed time / turns as secondary signals, not just win rate.

Current practical recommendation:

- Treat `outputs/legal_action_scorer_001/legal_action_scorer_best.pt` as the current leading candidate.
- Keep `outputs/policy_improve_margin500_002/supervised_policy_value.pt` as the old flat-policy champion baseline.
- Keep `outputs/policy_improve_margin500_001/supervised_policy_value.pt` as the previous flat-policy champion baseline.
- Keep `outputs/warmstart_search_004/supervised_policy_value_best.pt` as the older pre-policy-improvement baseline.
- Use the benchmark CLI for future checkpoint comparisons instead of ad hoc inline Python.
- Use direct head-to-head benchmark mode for promotion decisions against neural checkpoints:
  - `--checkpoint <candidate> --model-kind <candidate_kind> --baseline-checkpoint <baseline> --baseline-model-kind <baseline_kind>`
- Keep the loop fallback enabled for GUI/benchmark play, but continue recording trigger counts so it remains measurable and removable.
- Important caveat: the fallback is a pragmatic guardrail, not a pure policy improvement. It may slightly affect marginal benchmark strength by overriding the model in rare loop-risk states, so compare future champions using both win rate and fallback trigger counts.
- Keep using sharded search corpora plus multi-path training instead of single giant replay jobs.
- Continue excluding stalled/timeout games when training.
- Watch raw checkpoint loop behavior separately from guarded benchmark behavior. The fallback mitigates observed loops, but the long-term fix should come from stronger data/training/self-play targets.

Machine transfer checklist:

- Git contains the code, tests, and handoff, but `data/` and `outputs/` are intentionally ignored. To continue from the same state on another machine, copy at minimum:
  - `outputs/policy_improve_margin500_002/supervised_policy_value.pt`
  - `outputs/policy_improve_margin500_002/supervised_metrics.json`
  - `outputs/policy_improve_margin500_001/supervised_policy_value.pt`
  - `outputs/warmstart_search_004/supervised_policy_value_best.pt`
  - `outputs/benchmarks/policy_improve_margin500_002_100g_seed5000.json`
  - `outputs/benchmarks/policy_improve_margin500_002_100g_seed8000.json`
  - `outputs/benchmarks/policy_improve_margin500_002_random_100g_seed5000.json`
- Optional but useful if continuing the exact data lineage without regenerating:
  - `data/corpus_checkpoint_greedy_snapshots_001/`
  - `data/corpus_checkpoint_greedy_snapshots_002/`
- User said they plan to copy the full suggested set above, including the optional corpus lineage directories. On the next machine, assume these artifacts may be present, but still run the `Test-Path` checks before launching training/eval.
- On the new machine, first verify:

```powershell
Test-Path outputs\policy_improve_margin500_002\supervised_policy_value.pt
.venv\Scripts\python.exe -m pytest -q --basetemp .codex_pytest_tmp_all
```

- NVIDIA CUDA remains the validated fast path. CPU can run benchmarks if needed; training and relabel loops are much slower without CUDA.

## Known Technical Debt

1. The flat action space is very large.
   - It is workable for the first supervised model.
   - It is not the preferred final architecture for serious self-play scale-up.
   - The likely future refactor is a legal-action scorer instead of a giant sparse softmax.

2. Replay validation loss is only loosely correlated with benchmark strength.
   - The training CLI now benchmarks both final and best-validation checkpoints.
   - For `warmstart_search_003`, the `best_validation` checkpoint is the true playing-strength winner.
   - Pick champions by benchmark results, not validation loss alone.

3. Search corpus generation is slow.
   - `search:greedy` is much slower than `greedy:random`.
   - Sharding is the preferred operational pattern now.

4. Search bot quality may still have hidden loop or style issues.
   - The obvious `repetition_cutoff` token-churn pattern was improved.
   - Corpus generation is currently clean, but continue watching new search styles if branching/depth change.

5. Raw checkpoint policy can still produce rare non-progress benchmark games.
   - The loop fallback is an inference guardrail, not a training fix.
   - It is intentionally narrow and has validated well so far, but it can still change move choice at the margin when it fires.
   - Track `model_loop_fallback_triggers`; a future stronger model should need it less often, and ideally not at all.
   - With fallback disabled, seed `5008` still ends at turn 39 with `termination_reason='repetition_cutoff'`, `final_scores=(0, 0)`, and `no_progress_streak=27`.
   - With fallback enabled, seed `5008` completed in 82 turns, model won 15-7, and the fallback fired once.
   - This remains a learned policy behavior / benchmark cutoff issue, distinct from the earlier `ShallowSearchBot` corpus-generation loop fix.

## Recommended Next Step

Near-term:

0. Current immediate recommendation after `024`:
   - Do not promote `outputs/legal_action_scorer_024_attention_pgs_targeted`.
   - Keep `outputs/legal_action_scorer_016_policy_guided_mix/legal_action_scorer_best.pt` as the conservative best fast scorer.
   - Keep `outputs/legal_action_scorer_022_attention_distill/legal_action_attention_scorer_best.pt` as the best search-aligned neural parent.
   - The latest hypothesis tested cleanly and failed: high-margin seat1 PGS-vs-022 corrections, trained as a small overweighted supervised corpus, did not fix seat1 and regressed search/PGS gates.
   - Next do diagnostics before training again:
     - compare 022 vs 024 top-action changes on `data/corpus_pgs022_targeted_024/targeted_pgs_disagreements.jsonl`;
     - measure 022/024 teacher top-match on those 346 rows and on the broader 8,000-row mined slice;
     - inspect whether the bad shift is concentrated in late buys, reserves, or token takes;
     - if 024 matches the correction rows better but benchmarks worse, reduce correction weight/replay ratio or abandon this supervised correction style in favor of online league/search evaluation.

1. Treat the current local neural champion set as unresolved between:

```powershell
outputs\legal_action_scorer_003\legal_action_scorer_best.pt
```

and:

```powershell
outputs\legal_action_scorer_002\legal_action_scorer_best.pt
```

and:

```powershell
outputs\legal_action_scorer_001\legal_action_scorer_best.pt
```

`legal_action_scorer_001` and `legal_action_scorer_003` are the best current neural baselines. `legal_action_scorer_004` should not be promoted because it regressed versus search on the standardized 100-game block. Keep `001`, `002`, and `003` as active baselines rather than discarding them.

The old flat-policy champion baseline remains:

```powershell
outputs\policy_improve_margin500_002\supervised_policy_value.pt
```

2. The legal-action scorer has now confirmed as the preferred neural architecture.
   - First direct 100-game result, seed block 9000: scorer 64, old flat champion 32, draws 4.
   - Second direct 100-game result, seed block 10000: scorer 63, old flat champion 37, draws 0.
   - First scorer self-play/search-relabel iteration produced `legal_action_scorer_002`, which improved vs search but remains roughly tied with `legal_action_scorer_001` directly.
   - Second scorer self-play/search-relabel iteration produced `legal_action_scorer_003`, which improved the first 20-game search benchmark to 10-10 but fell to 3-17 on the confirmation block; on the standardized 100-game search block it was effectively tied with `001`.
   - First mixed league/search-relabel iteration produced `legal_action_scorer_004`, which did not improve over the `001`/`003` baseline set and regressed against search.

3. Do not promote the mixed-data experiments.
   - Adding `greedy:random` data diluted the stronger search policy target and regressed benchmark strength.
   - Even a 3x search-weighted mix only reached 9-11 vs greedy on seed block 7000, while the current champion reached 16-4.
   - The lesson is that broader data needs quality control or weighting by playing strength, not just more rows.

4. Do not spend the next main effort on more flat-policy gated policy-improvement unless the scorer confirmation fails.
   - `warmstart_search_005` was trained after another clean search-only block but regressed in the quick seed-7000 benchmark, so do not promote it without further study.
   - The first controlled checkpoint-vs-greedy mix also regressed in the quick seed-7000 benchmark, so do not promote it.
   - Do not add more naive imitation data. The useful recipe is:
     - generate a snapshot-enabled champion-vs-greedy corpus from `policy_improve_margin500_002`
     - run `run_improve_replay_actions.py` with the margin gate
     - train a small policy-focused candidate from `policy_improve_margin500_002` using `--value-loss-weight 0.0`
     - benchmark against `policy_improve_margin500_002` before promoting
   - First ungated version of this probe was completed as `policy_improve_001` and rejected. The useful lesson is that shallow-search relabeling changed many champion actions, but directly fine-tuning on all labels slightly hurt benchmark strength and reintroduced one random-game repetition cutoff.
   - The gated version `policy_improve_margin500_001` succeeded, and `policy_improve_margin500_002` improved again on the checked blocks.
   - Next improvement attempt can either repeat the same gated recipe for one more 25-game shard, or run a small margin sweep before scaling heavily.
   - If repeating directly, use `data/corpus_checkpoint_greedy_snapshots_003`, seed start `9500`, and output `outputs/policy_improve_margin500_003`.
   - If doing a margin sweep, reuse one snapshot corpus and compare at least `--min-search-margin 250`, `500`, and `1000` before training larger candidates.
   - Stop scaling this recipe aggressively if gains flatten or benchmark strength becomes opponent-specific.

Strategic next phase after the immediate gated experiment:

5. Continue the structured legal-action scorer path.
   - Score the legal actions present in a state rather than learning a huge sparse softmax over every possible encoded action.
   - Splendor actions have natural structure: buy board card, buy reserved card, reserve, take tokens, return tokens, and pass. Action features plus state features should generalize better than a flat index.
   - The first trained scorer already beat the old flat-policy champion head-to-head, so the next work should improve and validate this architecture rather than reverting to flat softmax by default.

6. Use the new search benchmark as one promotion gate, not the only one.
   - `legal_action_scorer_003` is useful as the search-improved candidate, but direct neural strength versus `legal_action_scorer_001`/`002` is mixed.
   - The next scorer candidate should be judged against both `policy_improve_margin500_002` and `--opponents search`.
   - The first scorer self-play/search-relabel recipe worked directionally but was small: `002` went 54-45-1 then 49-51-0 vs `001`, while improving vs search to 6-14 from 4-16.
   - The second scorer self-play/search-relabel recipe improved search on one block but was unstable: `003` reached 10-10 then 3-17 vs search, and on the 100-game search block scored 29-71 vs `001` at 30-70. Direct neural results were 46-54 then 60-40 vs `002`, and 47-53 vs `001`.
   - The first mixed league recipe `004` did not work: it scored 21-79 vs search on the standardized 100-game block.
   - Promotion should require a clearer head-to-head result against the neural pool plus at least matching `001`/`003` on the standardized search block.

7. Search distillation with the current recipe should not be the next main path.
   - `005` showed that a 502-row scored slice is not enough.
   - `006` showed that even the full depth-2 scored corpus, trained with `--search-distill-loss-weight 0.1`, does not clearly beat `001`/`003`.
   - `007` with lower distillation weight `0.03` got worse versus search.
   - `008` top-1-only on the same full corpus matched `006` but still failed the promotion bar.
   - `009`/`010` tested explicit example weighting, with changed rows weighted 2.0 and unchanged anchors 0.5; this also failed.
   - Keep the full scored corpus because it is reusable for ablations.
   - Do not spend the next main effort on more single-corpus supervised tuning from this data.
   - Do not use `004`, `005`, `006`, `007`, `008`, `009`, or `010` as parents unless a later benchmark promotes them. The stable parents remain `001` and maybe `003`.

8. Value learning is implemented but not yet a stronger search evaluator.
   - `value_model_002` is the best current value checkpoint, but it did not improve search in the 100-game probe.
   - Value training now supports game/source-level heldout validation. Prefer `--validation-split-unit game` for value experiments because random row validation leaks positions from the same games.
   - `value_model_003` confirmed the leak concern but did not beat `value_model_002` on external corpora.
   - The next value-model work should focus on target quality and game-outcome/value usage in the league loop, not simply plugging this value into search again.

8. Move the main learning signal from greedy/search imitation to self-play and league evaluation.
   - Train against a pool of historical checkpoints, the current champion, tuned search bots, and baseline bots.
   - Keep greedy/random as regression tests, not as the main strength target.
   - Compare candidates by benchmark win rate, fallback-trigger count, game length/tempo, and performance against previous neural champions.

9. Use search selectively rather than copying AlphaGo-style MCTS wholesale.
   - Shallow/medium search remains useful for relabeling, benchmark opponents, and possibly inference under the 1-second move budget.
   - Be careful with hidden deck/refill information and off-policy value labels when search changes replay actions.
   - The long-term fix for loops should be learned policy quality and stronger self-play targets, not reliance on the inference fallback.

Future directions worth keeping from the external expert feedback:

- Expert iteration may become useful later: periodically review high-margin model/search disagreements and encode the missing Splendor concepts explicitly, especially token-taking plans, denial reserves, and race-to-15 tactical buys.
- Consider policy-guided search ordering once the scorer is stronger. The current shallow search is a strong opponent/teacher, but a scorer-guided move ordering could make deeper or broader search fit the 1-second inference budget.
- Revisit value learning only when target quality improves. Game-level validation is now available; the next useful value work is better outcome/tempo targets from league/self-play, not just plugging the current value model into search.
- Keep league data structured by source, matchup, seed block, and checkpoint identity. This is important for avoiding leakage and for diagnosing whether a candidate is a general improvement or only an anti-teacher specialist.
- Do not over-scale one supervised relabel recipe. If targeted corrections do not improve strength after a properly sized run, move toward league/self-play policy improvement or policy-guided search rather than more variants of the same corpus.

Operational notes:

10. Use the sharded replay pattern if continuing data scaling:

```powershell
.venv\Scripts\python.exe run_replay_corpus.py --output-dir data\corpus_search_greedy_004a --games 125 --seed-start 8000 --seat0-bot search --seat1-bot greedy --swap-seats --max-turns 400 --repetition-limit 4 --no-progress-limit 60 --search-depth 2 --search-max-branching 8 --search-buy-branching 5 --search-reserve-branching 2 --search-take-branching 2 --log-every 10
```

11. Train with repeated replay paths rather than merging corpora by hand.

12. Benchmark the resulting checkpoint with fallback metrics enabled, and choose champions by 100+ game benchmark performance rather than validation loss alone.

Checkpoint replay command that was validated:

```powershell
.venv\Scripts\python.exe run_replay_corpus.py --output-dir data\corpus_checkpoint_greedy_001 --games 25 --seed-start 9100 --seat0-bot checkpoint --seat1-bot greedy --checkpoint-path outputs\warmstart_search_004\supervised_policy_value_best.pt --checkpoint-device cuda --swap-seats --max-turns 400 --repetition-limit 4 --no-progress-limit 60 --log-every 5
```

Train on checkpoint-generated data only after inspecting `loop_fallback_triggers`, `timed_out_games`, and the per-row `game_model_loop_fallback_triggers` fields.
Use `--max-game-model-loop-fallback-triggers 0` to exclude all fallback-triggered games from supervised training.

Snapshot-enabled champion replay command template:

```powershell
.venv\Scripts\python.exe run_replay_corpus.py --output-dir data\corpus_checkpoint_greedy_snapshots_001 --games 25 --seed-start 9300 --seat0-bot checkpoint --seat1-bot greedy --checkpoint-path outputs\warmstart_search_004\supervised_policy_value_best.pt --checkpoint-device cuda --swap-seats --max-turns 400 --repetition-limit 4 --no-progress-limit 60 --include-state-snapshots --log-every 5
```

Action-improvement command template:

```powershell
.venv\Scripts\python.exe run_improve_replay_actions.py --input-path data\corpus_checkpoint_greedy_snapshots_001\replays.jsonl --output-path data\corpus_checkpoint_greedy_snapshots_001\improved_search_policy.jsonl --summary-path data\corpus_checkpoint_greedy_snapshots_001\improved_search_policy_summary.json --search-depth 2 --search-max-branching 10 --search-buy-branching 6 --search-reserve-branching 3 --search-take-branching 3 --log-every 1000
```

## Files Worth Reading First

- `README.md`
- `docs/environment_spec.md`
- `docs/windows_cuda_setup.md`
- `src/splendor_ai/engine/env.py`
- `src/splendor_ai/encoding/action_codec.py`
- `src/splendor_ai/training/run_supervised.py`
- `src/splendor_ai/training/replay.py`
- `src/splendor_ai/training/improve_replay_actions.py`
- `src/splendor_ai/eval/run_benchmark.py`
- `src/splendor_ai/eval/match.py`
- `src/splendor_ai/diagnostics.py`
- `src/splendor_ai/bots/heuristic_bot.py`
- `src/splendor_ai/bots/search_bot.py`

## Test Status

Direct verification completed in this session:

- `python -m pytest -q --basetemp .codex_pytest_tmp_all` passed: 141 passed after the observation/action feature update and targeted-corpus tooling.
- `.venv\Scripts\python.exe -m pytest -q --basetemp .codex_pytest_tmp_all` passed: 85 passed
- `.venv\Scripts\python.exe -m py_compile` over modified source files succeeded
- Focused tests cover normal masked argmax behavior, fallback activation after loop evidence, and fallback logit-gap refusal.
- Benchmark seed 5008 before/after:
  - fallback enabled: completed, model won 15-7, 1 fallback trigger
  - fallback disabled: `repetition_cutoff` at turn 39, greedy adjudicated winner, 0 fallback triggers
- 100-game fallback validation:
  - seed block 5000: 85-15-0 vs greedy, 0 timeouts, 8 fallback triggers
  - seed block 6000: 85-15-0 vs greedy, 0 timeouts, 5 fallback triggers
- Mixed-data experiments:
  - `warmstart_mix_001`: 85 passed before run, trained successfully, benchmark champion final checkpoint 17-3 vs random and 5-15 vs greedy on seed block 7000
  - `warmstart_mix_002_search_weighted`: trained successfully, benchmark champion final checkpoint 19-1 vs random and 9-11 vs greedy on seed block 7000
  - then-current champion comparison on seed block 7000: 20-0 vs random and 16-4 vs greedy
- `warmstart_search_004`:
  - generated `corpus_search_greedy_004[1-5]`: 125 games, 7,964 steps, 0 stalled, 0 timed out
  - trained successfully on 71,824 search-only replay samples
  - post-train benchmark selected `supervised_policy_value_best.pt`
  - seed block 5000 vs greedy: 85-15-0, 0 timeouts, 4 fallback triggers
  - seed block 6000 vs greedy: 86-14-0, 0 timeouts, 1 fallback trigger
  - seed block 5000 vs random: 100-0-0, 0 timeouts, 4 fallback triggers
- Checkpoint replay support:
  - `.venv\Scripts\python.exe -m pytest -q --basetemp .codex_pytest_tmp_all` passed before filtering work: 89 passed
  - focused replay/checkpoint tests passed: 16 passed
  - `.venv\Scripts\python.exe -m py_compile` over modified replay/corpus/checkpoint files succeeded
  - `warmstart_search_005` completed but was rejected: final checkpoint 20-0 vs random and 15-5 vs greedy on seed block 7000
  - `data/corpus_checkpoint_greedy_probe_001`: 2 games, 126 steps, 0 stalled, 0 timed out, 0 fallback triggers
- Champion-data filtering and mix:
  - focused dataset/training/replay tests passed: 28 passed
  - `data/corpus_checkpoint_greedy_00[1-5]`: 125 games, 8,156 steps, 0 stalled, 0 timed out, 4 fallback triggers
  - `warmstart_champion_mix_001`: trained with `--max-game-model-loop-fallback-triggers 0`, final checkpoint 20-0 vs random and 15-5 vs greedy on seed block 7000; rejected
- Replay action improvement tooling:
  - focused tests passed: `.venv\Scripts\python.exe -m pytest -q tests\test_replay_export.py tests\test_improve_replay_actions.py --basetemp .codex_pytest_tmp_improve` -> 6 passed
  - full suite passed: `.venv\Scripts\python.exe -m pytest -q --basetemp .codex_pytest_tmp_all` -> 94 passed
  - CLI smoke probe passed:
    - generated 1 snapshot-enabled random-vs-random probe game
    - relabeled first 5 rows with `run_improve_replay_actions.py`
    - cleaned temporary probe output afterward
- Policy-improvement probe:
  - `data/corpus_checkpoint_greedy_snapshots_001`: 25 games, 1,650 steps, 0 stalled, 0 timed out, 0 fallback triggers
  - `improved_search_policy_full_summary.json`: 1,650 rows written, 1,123 changed, 527 unchanged, 0 missing snapshots
  - `outputs/policy_improve_001`: policy-only fine-tune from current champion, best validation epoch 4
  - benchmark final/best checkpoints both produced 19-1 vs random with 1 repetition cutoff and 18-2 vs greedy with 0 timeouts on seed block 7000
  - `.venv\Scripts\python.exe -m pytest -q --basetemp .codex_pytest_tmp_all` passed after the probe: 94 passed
- Gated policy-improvement probe:
  - focused gated relabel tests passed: `.venv\Scripts\python.exe -m pytest -q tests\test_improve_replay_actions.py --basetemp .codex_pytest_tmp_improve` -> 4 passed
  - `improved_search_policy_margin500_no_take_summary.json`: 901 rows written, 374 changed, 527 unchanged, 193 filtered by margin, 399 filtered by missing original score, 157 filtered by selected action type
  - `outputs/policy_improve_margin500_001`: policy-only fine-tune from `warmstart_search_004_best`, best validation epoch 4
  - quick seed-7000 benchmark: 20-0 vs random, 19-1 vs greedy, 0 timeouts
  - 100-game seed-5000 random sanity: 100-0-0, 0 timeouts, 0 fallback triggers
  - 100-game seed-5000 greedy: 94-6-0, 0 timeouts, 0 fallback triggers
  - 100-game seed-6000 greedy: 95-5-0, 0 timeouts, 1 fallback trigger
  - full suite passed after the gated probe: `.venv\Scripts\python.exe -m pytest -q --basetemp .codex_pytest_tmp_all` -> 96 passed
- Second gated policy-improvement probe:
  - `data/corpus_checkpoint_greedy_snapshots_002`: 25 games, 1,634 steps, 0 stalled, 0 timed out, 0 fallback triggers
  - `improved_search_policy_margin500_no_take_summary.json`: 882 rows written, 362 changed, 520 unchanged, 164 filtered by margin, 426 filtered by missing original score, 162 filtered by selected action type
  - `outputs/policy_improve_margin500_002`: policy-only fine-tune from `policy_improve_margin500_001`, best validation epoch 4
  - quick seed-7000 benchmark: 20-0 vs random, 19-1 vs greedy, 0 timeouts
  - 100-game seed-5000 random sanity: 100-0-0, 0 timeouts, 0 fallback triggers
  - 100-game seed-8000 greedy: 97-3-0, 0 timeouts, 0 fallback triggers
  - 100-game seed-5000 greedy: 96-4-0, 0 timeouts, 1 fallback trigger
  - full suite passed after the second gated probe: `.venv\Scripts\python.exe -m pytest -q --basetemp .codex_pytest_tmp_all` -> 96 passed

Plain pytest without `--basetemp` can still hit Windows temp permission issues around local pytest temp directories. Use a repo-local basetemp:

```powershell
.venv\Scripts\python.exe -m pytest -q --basetemp .codex_pytest_tmp_all
```

## Recent Commit Summary

Changes prepared in this session:

- Added conservative loop-aware inference fallback to `CheckpointPolicyBot`.
- Added fallback trigger counts to `GameResult` and benchmark JSON summaries.
- Added benchmark CLI flags to disable/tune the loop fallback.
- Reused loaded checkpoint bots across benchmark games in standalone and post-training benchmark paths.
- Added focused tests for fallback behavior and updated benchmark summary expectations.
- Validated fallback on greedy seed blocks 5000 and 6000, removing observed repetition cutoffs while preserving roughly the same win rate.
- Ran and rejected `warmstart_mix_001` and `warmstart_mix_002_search_weighted`; then-current champion remained `warmstart_search_003/supervised_policy_value_best.pt`.
- Generated clean `corpus_search_greedy_004[1-5]`, trained `warmstart_search_004`, and promoted `outputs/warmstart_search_004/supervised_policy_value_best.pt` based on 200-game greedy validation.
- Generated clean `corpus_search_greedy_005[1-5]` and trained `warmstart_search_005`, but rejected it because the quick greedy benchmark regressed.
- Added checkpoint bot support to replay corpus generation, including checkpoint provenance and fallback-trigger metadata in replay rows and corpus summaries.
- Added supervised filtering by checkpoint fallback trigger count, generated `corpus_checkpoint_greedy_00[1-5]`, trained `warmstart_champion_mix_001`, and rejected it because the quick greedy benchmark regressed.
- Added optional replay state snapshots, exact snapshot reconstruction, shallow-search action ranking, and `run_improve_replay_actions.py` for policy-improvement relabeling.
- Added `run_supervised.py --init-checkpoint`, ran `policy_improve_001`, and rejected it because the quick benchmark was slightly worse than `warmstart_search_004_best`.
- Added confidence/margin-gated relabeling, trained `policy_improve_margin500_001`, and promoted it because it beat the previous champion by 18 games over the same two 100-game greedy validation blocks while reducing fallback use.
