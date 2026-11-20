# Why stronger direct play did not improve planning: first diagnosis

## Owner accepted the narrowed experiment159

The owner subsequently authorized execution: "ok, let's do this". The fixed
design is in `docs/engineering/planner_preference_159.md`:180 distinct-source
positions balanced across36 cells,32 fresh selection worlds,128 independent
evaluation worlds, simultaneous primary uncertainty and an inconclusive result.
The protocol includes a cross-half interaction diagnostic for effects that
cancel in the average. No2x2 role ablation, new arena or training is authorized
as part of159. Execution status lives at the top of `AGENT_CURRENT.md`.

## Recommendation review after owner challenge

The owner asked whether the proposed2x2 role probe is really the best next step.
Two independent read-only reviews support only moderate confidence in that
proposal. It remains a possible exploratory mechanism test, but is superseded
as the immediate recommendation by a narrower independent-world replication.
No new experiment is authorized or launched by this review.

Frequent argmax changes can be harmless near ties. The16/16 split diagnostic
shows sampling sensitivity, not that it caused158's null arena result. Further,
a stable role-induced ranking shift does not establish a better ranking. The
original conditional rule below, "stable role effect -> test that role", needs
an explicit independent-quality question; neither continuation policy is truth.

For the owner's current explanatory goal, first compare the existing two
planners on fixed positions from distinct source pairs across all game phases.
Retain the original32-world selection budget and freeze the selected moves
before fresh-world evaluation. Evaluate the same fixed action differences under
both existing continuation objectives, with disjoint selection/evaluation worlds.
Measure the size and repeatability of action differences, rather than only move
agreement. Choose the number of independent source pairs from a prespecified
precision target; more streams within a root do not replace more source pairs.
Keep an explicit inconclusive outcome. This is not a new gameplay-quality oracle.

If a material policy-dependent difference is repeatable with sufficient
precision, the role probe has a concrete effect to dissect. If adequate precision
excludes such differences on the tested distribution, role attribution becomes
less valuable and estimation/selection efficiency merits attention. This cannot
by itself establish that more worlds, asymmetric rollouts or any alternative
planner wins more games. Each practical claim still needs a direct match.

For the separate objective of directly improving the strongest player, one
reviewer instead preferred a prospective155-at128-worlds versus155-at32-worlds
arena. That isolates a practical sampling-budget intervention but costs more
and does not explain158. A longer157 training dose is cheaper, but limited
evidence of undertraining and its indirect relation to the present question
make it a different research choice. No option is established as optimal.

The remainder records the original investigation and the now-deferred role-probe
proposal. Its measurements stand; its causal interpretation is narrowed above.

2026-09-14. The owner stopped the remaining158 audit and authorized investigation.
The recovery parent and eight workers were stopped after identity checks; all
402 game records,24,496 roots and19 completed main replay certificates remain.
The full-replay requirement was explicitly waived for this screen. The48.00995%
score (paired95% interval43.28358%-52.73632%) is a **provisional negative result**,
not a fully audited result, a proved regression, or proof of exact equivalence.
No new simulation, inference, training or arena ran in this investigation.

## What the saved evidence reveals

The first diagnosis is that stronger direct play did not visibly improve the
reliability of the rollout comparisons. Both planners already have substantial
sampling instability. We have not identified a unique causal explanation for
the game-level result.

The new diagnostic hashes every game/root against its saved bindings and
recomputes every saved32-world mean and selected action. It examines all24,496
roots. In the paired games,308 roots share exactly the same full state, legal
proposals, RNG schedule and sampled hidden worlds across the two planners.
Exactly201 are initial positions, one per prespecified pair. These201 are the
cleanest matched panel; later shared positions are selected by preceding play.

For each root,20 fixed random partitions divide the32 worlds into disjoint
16-world halves. Each half selects its own move using the original tie rule;
the other half evaluates that selection. Both directions are averaged. This
measures **16-world** sampling stability, not independent32-world repeatability.
Held-out gains are relative to the original035 proposal and concern each
continuation policy's own outcome objective, not actual planner win probability.

| On201 matched initial positions | Learned continuations | Greedy035 continuations |
| --- | ---: | ---: |
| Different move chosen by the two16-world halves |70.35%|68.16%|
| Apparent selected-move gain within its selection half |+11.46pp|+11.46pp|
| Gain when evaluated on the other half |-0.318pp|-0.003pp|
| Mean sampled best-versus-runner-up gap, all32 worlds |6.69pp|7.28pp|

The exploratory paired, model-stratified interval for the disagreement-rate
difference is-3.43 to+7.76pp; the point difference is+2.19pp. Thus this diagnostic
does not establish that the learned controller adds more instability. It also
does not prove the two variances equal. The held-out-gain difference is-0.315pp,
with exploratory interval-2.05 to+1.42pp. These are descriptive follow-up
intervals, not new confirmatory gates and not replacements for the full arena.

Changing continuations changes the selected initial move in127/201 positions
(63.18%). About49.36% of corresponding forced-action/world outcomes differ.
This shows that changing the simulation policy substantially changes the
estimated futures. It does **not** distinguish systematic policy bias from
sampling variation: only one recorded continuation-policy stream is available
per hidden world, and even the within-policy halves disagree frequently.

Across all roots, half-sample disagreement is55.60% versus54.01%. Average
held-out gains remain positive:2.56pp versus2.27pp. These are different visited
state distributions, so their difference is not a causal comparison. Later
positions have larger held-out gains (4.30pp versus3.96pp) than initial roots;
this is compatible with useful tactical search despite unreliable openings.
It does not authorize an opening filter or imply search is useless.

## Strongest remaining explanations

1. **Long-horizon rollout estimates remain a weak way to rank close moves.**
   This has the clearest descriptive support. Many apparent opening advantages
   vanish on the other world half. A modestly better continuation player may
   change which noisy move wins without improving the comparison. Increasing
   the number of worlds could help sampling error, but cannot repair policy
   bias; these data do not establish that more compute will improve gameplay.
2. **Direct strength and usefulness as an evaluator are different objectives.**
   157 learned restricted teacher-choice classification, not value calibration
   or low-variance action differences. It beat035 at57.58%, while155 was much
   stronger against035 in its own earlier panel. Partial compression success
   therefore provides no guarantee of a better rollout evaluator. This general
   distinction is also motivated by simulation-balancing research, which
   optimized the accuracy of simulated outcomes rather than direct policy
   strength. [Silver and Tesauro, ICML2009](https://research.ibm.com/publications/monte-carlo-simulation-balancing--1).
3. **Changing both simulated players may change the wrong future assumptions.**
   In158, both sides inside each candidate rollout became157, while actual play
   continually replans. Better self-continuations and better opponent responses
   can affect alternatives differently; improving both sides need not improve
   their relative evaluation. The records cannot separate these two effects.
   Neither cancellation nor opponent-model mismatch has been established.
4. **The learned policy may transfer poorly to imagined intermediate states.**
   Its training rows are real155/156 decisions; long imagined branches can
   differ. Normal branch records retain decision digests and counts, not all
   intermediate features and choices. Those digests cannot diagnose distribution
   shift without reconstructing targeted trajectories. Undertraining, harmful
   action families or hidden-information subgroups remain untested explanations.

The explanation "random proposals uniquely made the candidate much noisier"
currently has weak support. Reference search shows nearly the same instability.
The explanation "157 learned nothing" conflicts with its fully verified direct
play result. A single bad checkpoint also looks unlikely as a complete account:
all three158 point estimates lie near47.4%-48.9%, although their panels differ.

Earlier140 did not establish a benefit from its stronger continuation policy
(strong-minus-greedy normalized-margin interval-0.075 to+0.0296).141 bounded its
specific response-rule difference below the designated material size. Those
were different operators and conditional diagnostics; they neither solve nor
rule out the specific158 mechanisms. Do not revive them as established causes.

## Highest-value next causal diagnostic

Recommend one small **2x2 continuation-role probe with repeated proposal streams**,
before another full arena or training run. It should ask what changes in the
estimator, rather than claim to settle playing strength by another local proxy.

Use a fixed36-root panel drawn without outcomes, confidence gaps or selected
move filters: two distinct seed-pair sources in each of three fixed models,
three game phases and two acting seats. Freeze hash-based selection and use
disjoint source pairs where possible. Keep each root's original proposals and
assigned learned checkpoint. This panel should include later states, where the
saved evidence shows more useful signal than at the opening.

At each root compare greedy/greedy, learned-self/greedy-opponent,
greedy-self/learned-opponent and learned/learned. Share64 fresh lawful hidden
worlds across all cells. For cells using the learned controller, repeat four
independent proposal streams on each identical world; greedy/greedy is
deterministic and needs only one run. Use actor-lawful information and preserve
the original controller/termination rules. At most119,808 terminal continuations
for36 roots with four proposals: time a small fixed fixture before committing
to a runtime estimate. This proposal has not been launched.

Measure within-world proposal-stream variance separately from between-world
variance, repeatability of move rankings, and each role's effect on **action
differences after removing any common value shift**. Use separate streams to
check whether apparent rank changes persist. A higher average self win estimate
alone does not make a policy more useful for selecting between actions.

If repeatability is the main problem, test a fixed variance-reduction or budget
intervention next. If stable rank changes concentrate in one role, test that
role substitution in a bounded direct match. If both effects are small, revisit
training coverage/objective. None of those follow-ups is authorized by this
diagnostic alone, and none should treat either rollout policy as ground truth.

## Reproducible artifacts

- `scripts/analysis/diagnose_saved_learned_continuation_158.py`: saved-root scan,
  deterministic partitions and matched-state checks. Completed in24seconds.
- `outputs/learned_continuation_158/diagnostics_saved_evidence_20260914/summary.json`:
  descriptive summaries, input/script hashes, units and limitations.
- `roots.json` and `matched_roots.json` in that folder: per-root data.
- `initial_root_intervals.json`: exploratory20,000 paired stratified resamples,
  seed1584000001. Root-split partitions used seed1584000000.
- `recovery_main_replay/owner_stop.json`: authorized stop and protocol deviation.

Scheduled monitoring remains disabled. The incomplete integrity audit remains
explicitly disclosed in all uses of158; all original evidence is preserved.
