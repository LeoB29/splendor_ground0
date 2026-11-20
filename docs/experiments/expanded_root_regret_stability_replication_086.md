# Experiment 086: Expanded-Root Regret-Stability Replication

**Preregistered:** 2026-08-07, before implementing `086`, generating any
`086` source game, or inspecting any post-hoc `085` regret/equivalence metric.

## Why a fresh replication is justified

`085` is permanently invalid under its own frozen conjunction. Expanded
discovery/evaluation exact best-root agreement was 179/300 = 59.667%, one state
below its 60% floor. Do not round, pool, or reinterpret that result as a pass.

The invalidity nevertheless exposed a measurement problem worth testing
prospectively. Exact argmax agreement treats two roots separated by numerical
or practically negligible payoff exactly like a catastrophic mismatch. In
`085`, expanded root-value Pearson was 0.9758, mean expanded-selected versus
champion-selected untouched gain was +0.061875 with CI95
[+0.050431,+0.073736], movement was 74.33%, and every phase was positive.
Those observed aggregates motivate a fresh replication; no `085` row-level
regret is opened to choose `086` thresholds.

`086` asks:

> On wholly fresh states and more precise untouched evaluation, does the exact
> four-discovery-continuation expanded selector choose roots that are
> practically equivalent to the evaluation oracle and still improve on the
> current four-root resolver?

This does not retroactively pass `085`. It is a new validation experiment with
a more decision-relevant, prospectively frozen argmax endpoint.

## Exact candidate and fresh sample

The deployed selector is unchanged from `085`: raw `035`'s first four distinct
canonical roots plus every lower-ranked, individually structurally stable
non-`RESERVE_DECK` public root. Root ordering, structural exclusion reasons,
three observer-relative determinizations, top-four `035` opponent responses,
information-set consistency, stochastic-`035` continuation policy, utility,
and canonical tie rule are identical to `085`.

Generate exactly 600 new raw-`035` self-play games at seeds
`860000--860599`. Freeze the same outcome-blind assignment and selection rule:
`(seed-860000)%6` assigns seat 0/1 and early `0--19`, mid `20--49`, or late
`50+`; take the first 50 structurally champion-eligible games in each cell,
with the within-game median-turn ordering from `085`. Require exactly 300
unique games/states, 50 per cell. All source integrity, hidden-information, and
no-human-input rules from `085` remain mandatory. No `085` source game, state,
branch, continuation, or payoff is reused.

Frozen checkpoint:
`outputs/legal_action_scorer_035_legal_observation/`
`legal_action_attention_scorer_best.pt`, SHA-256
`4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`.

## Discovery and three untouched evaluation halves

Evaluate each retained leaf with exactly 16 terminal continuations. Indices
`0--3` are the only discovery table and therefore define the exact candidate
action that later gameplay would deploy. Indices `4--7`, `8--11`, and `12--15`
are three separately reported untouched evaluation halves. Their pooled 12-
continuation mean is the primary evaluation table. No evaluation continuation
may affect selection.

Freeze epsilon 0.1, temperature 0.5, top-k 3, 400 additional turns, at most 64
loop-fallback actions, and score margin divided by 20. Seeds are shared across
root/response alternatives for matched variance reduction.

- belief root index:
  `860000000 + (engine_seed-860000)*400 + turn`;
- continuation seed:
  `8600000000000000000 + sample_index*1000000000 +`
  `determinization*10000 + continuation`.

Use ten CPU state workers, identity-matched progress, and a from-zero repeat of
the complete first state. The namespaces are disjoint from all prior evidence.

## Primary headroom endpoint and power

For each state, use the four-continuation discovery table to select the champion
root and expanded root. The primary effect is pooled-12 evaluation value of the
expanded-selected root minus pooled-12 evaluation value of the champion-
selected root. Bootstrap 300 state clusters 20,000 times with seed `860086`.

All primary headroom gates are required:

1. mean pooled evaluation gain at least `+0.05`;
2. bootstrap CI95 lower bound strictly above zero;
3. expanded selection changes at least 15% of states and at least one state in
   every phase; and
4. each of the three separate four-continuation evaluation halves has positive
   mean expanded-selected versus champion-selected gain.

`085` observed about +0.062 but is not pooled. Declare +0.060 as the replication
alternative. With 300 fresh clusters and conservative effect SD 0.20, standard
error is 0.01155 and the probability of clearing the binding +0.05 point gate
is about 80.7%; the CI-lower-above-zero gate is weaker. This directly powers
replication at the observed-order effect without treating the boundary itself
as the alternative.

Phase guardrails match `085`: a phase fails only if its fresh pooled-gain CI95
upper bound is at or below -0.05.

## Prospective argmax regret validation

For every state define pooled evaluation regret as:

```text
best pooled expanded-root evaluation value
minus pooled evaluation value of the discovery-selected expanded root
```

This is always nonnegative and measures the consequence of an argmax mismatch.
Thresholds are inherited from the established `059` practical-equivalence
standards, not fitted to unseen `085` regrets. All are required:

1. at least 60% of states have regret at most `0.025`;
2. mean regret is at most `0.025`;
3. 90th-percentile regret is at most `0.075`;
4. pooled discovery/evaluation expanded root-value Pearson is at least 0.60;
   and
5. discovery versus each separate evaluation half has expanded root-value
   Pearson at least 0.60.

Exact discovery/evaluation best-root agreement is reported, including its
Wilson interval, but is explicitly **not a decision gate**. This does not relax
`085`; it prospectively replaces a brittle proxy with bounded evaluation loss.

## Integrity and evaluation-table reliability

All are required before either headroom or regret can be interpreted:

1. exact clean 600-game seed range; exact 300-state six-cell balance; frozen
   protocol/config/checkpoint; stable root/action mappings; no illegal action;
   no human path or hash; and hidden-deck-order invariance;
2. exact variable accounting of `retained_roots*3*4*16` continuations per state;
   all resolve, hard caps and rollout fallback triggers are zero, and loop
   adjudications are at most 0.1%;
3. for discovery versus each evaluation half across leaves: Pearson at least
   0.60, MAE at most 0.10, and sign agreement at least 70% where both absolute
   half-means are at least 0.025;
4. the complete first state repeats byte-identically, excluding timing; and
5. evaluation completes within 12 active wall hours. Expected work is roughly
   1.25 million terminal continuations and 6--8 wall hours on ten CPUs.

An evaluation-table reliability failure makes the instrument invalid. Failure
of a prospective argmax regret gate is instead a valid negative result about
the candidate; it is not labelled operational invalidity.

## Frozen decisions

- Operational, privacy, determinism, mapping, accounting, or evaluation-table
  reliability failure:
  `expanded_root_086_invalid_no_replication_decision`.
- Valid primary headroom pass but any argmax regret failure:
  `expanded_root_086_argmax_unreliable_close_exact_expansion`.
- Valid argmax pass but any primary headroom failure:
  `expanded_root_086_headroom_not_replicated_close_exact_expansion`.
- Valid simultaneous primary-headroom and argmax-regret failure:
  `expanded_root_086_headroom_and_argmax_failed_close_exact_expansion`.
- Valid pass of every primary, argmax, and phase condition:
  `expanded_root_086_replication_pass_authorize_gameplay_preregistration`.

A full pass authorizes only a separately preregistered 200-pair gameplay panel
of this exact four-discovery-continuation expanded resolver against the current
resolver research champion. That panel must use fresh seeds, retain at least
15% action-change dose, declare 56.5% expected share, and require score share at
least 55% with paired-bootstrap CI95 lower bound above 50%. It is expected to
cost 40--60 wall hours on ten CPUs.

No `086` result changes either champion, authorizes training/corpus generation,
or makes `085` valid. Do not add states, pool experiments, tune roots, alter
continuation counts, or substitute exact-agreement thresholds after outcomes.

## Evidence

Write source selection, resumable evaluation, and final result under
`outputs/expanded_root_regret_stability_replication_086`. Record protocol,
implementation, tests, checkpoint, selection, progress, and result hashes.
