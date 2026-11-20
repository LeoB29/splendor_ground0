# AlphaZero Head-Attribution Audit 096

Status: frozen engineering protocol  
Date frozen: 2026-08-12  
Scope: no-new-corpus mechanism audit; no promotion claim

## Purpose

Experiment 094 proved that 256-simulation, `c_puct=2.25` search improves raw
generation 2. Experiment 095 then produced a child with substantially better
held-out replay losses but no independently confirmed searched improvement.
Experiment 096 attributes that failure between policy learning, value learning,
and their interaction before any further self-play is authorized.

No new training or self-play occurs. The only gameplay is a matched hybrid-head
audit using the already frozen parent and rejected child.

## Frozen inputs

- Parent generation 2 checkpoint:
  `outputs/alphazero_pivot_090/pilot/generation_002.pt`, SHA-256
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`.
- Rejected 095 child:
  `outputs/alphazero_confirmed_operator_successor_095/candidate/generation_003.pt`,
  SHA-256
  `36476fa8333a208c6d14559894cc491f9aa8f75a0a3c0c1b2972090d073090a4`.
- Fresh 095 replay:
  `outputs/alphazero_confirmed_operator_successor_095/selfplay/generation_002_high_dose.jsonl`,
  SHA-256
  `c58f9955362d1bb82b47ed34dd4bc0358cdf9a00116fca733b0171cb6d2cae47`.
- Controlling 095 result:
  `outputs/alphazero_confirmed_operator_successor_095/result.json`, SHA-256
  `6d2ec319982ced65855d526e1fe2c692c1f4bda49d8e3f60c473b693b4714655`.

The implementation must verify all hashes and verify 095's decision is
`searched_confirmation_failed_keep_generation_2`.

## Stage A: locked replay attribution

Run both exact checkpoints over every one of the 4,106 complete 095 replay
positions. Compare each network to the stored 256-simulation visit target and
terminal WDL target. Report:

- policy cross-entropy, target-top agreement, target total variation and
  Jensen-Shannon divergence;
- value mean, bias, MAE, MSE, and outcome-sign accuracy;
- child-minus-parent changes overall and by phase (`<20`, `20--49`, `>=50`),
  outcome, visit-selected action family, whether visit argmax changed the raw
  parent action, and quartiles of parent raw-policy confidence;
- on decisive states where visit argmax differs from parent raw top, each
  network's top agreement with the visit argmax and probability assigned to it;
- reproduction error between freshly inferred parent probabilities and stored
  parent root priors.

Strata are diagnostic and cannot select or modify gameplay dose.

## Stage B: matched hybrid-head gameplay

Construct three fixed candidates:

1. `full_child`: child policy and child value;
2. `child_policy_parent_value`;
3. `parent_policy_child_value`.

Each candidate plays the exact full parent (parent policy and parent value) on
the same 24 fresh paired seeds `1670000--1670023`, both candidate seats: 48
games per candidate, 144 games total. Every agent uses matched real PUCT with
256 simulations, `c_puct=2.25`, no root noise, depth 96, and otherwise the
confirmed 094 evaluation contract. Panels are independent comparisons and are
not pooled as strength evidence.

For each panel report W-D-L, share, margin, seat shares, normal terminations,
and a 50,000-resample paired-seed CI with fixed seeds 960096, 960097, and
960098. Also bootstrap same-seed score-share differences between each hybrid
and `full_child`, using seeds 961096 and 961097.

## Frozen attribution rubric

Let `V-swap` be child-policy/parent-value minus full-child paired score share,
and `P-swap` be parent-policy/child-value minus full-child share.

- `child_value_head_primary_blocker` if the V-swap CI95 lower bound is above
  zero and the P-swap lower bound is not.
- `child_policy_head_primary_blocker` if the P-swap CI95 lower bound is above
  zero and the V-swap lower bound is not.
- `both_heads_or_joint_incompatibility` if both lower bounds are above zero.
- `no_clear_gameplay_head_attribution` otherwise.

Offline diagnostics explain the result but do not override this rubric. A
hybrid whose parent-comparison CI lower bound exceeds 50% is labelled
`restores_confirmed_transfer`; this remains an audit label, not installation
authorization.

## Boundaries and decisions

- Exact generation 2 remains installed under every outcome.
- No checkpoint is trained, installed, or promoted.
- No self-play replay or `035` arena is generated.
- No seed replacement, dose extension, configuration tuning, or pooling with
  095 is allowed.
- 087 remains stopped and outcome-sealed.

The result should recommend the smallest targeted 097 correction consistent
with the attribution. If no head is clearly identified, 097 should improve the
training objective/representation using offline tests before another corpus,
not repeat the 095 recipe.

## Verification

Freeze protocol, implementation, and tests before gameplay. Verify input and
output hashes, exact replay membership, hybrid policy/value identities, exact
arena schedules, bootstraps, termination reasons, installed-parent hash,
stderr, elapsed time, and tests.
