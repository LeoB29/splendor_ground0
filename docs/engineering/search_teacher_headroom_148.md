# Experiment 148: public-value search-teacher headroom

Owner authorized `docs/strategy/next_after_147.md` with "ok, please do it" on
2026-09-08. Run exactly the two-stage design below, including the conditional
stage only after the primary stage passes. No training or promotion is included.
Preserve all intentionally dirty work and frozen previous experiments.

## Question and fixed panels

Does extra decision time improve the exact 035/public-value PUCT player, and,
if so, does the resulting player beat raw 035? This tests additional compute,
not equal-time efficiency, value attribution or an isolated depth mechanism.

| Panel | Candidate | Reference | Starting seeds | Games |
|---|---|---|---|---:|
| compute | public PUCT, 1.6 seconds | same public PUCT, .4 seconds | 1480000000--1480000199 | 400 |
| competence, conditional | same public PUCT, 1.6 seconds | exact greedy raw 035 | 1481000000--1481000199 | 400 |

Every starting seed has two games. Game index 2k places the candidate in seat 0,
2k+1 in seat 1. No opening exploration. Search RNG is game_seed*10000 +
acting_seat*1000 + turn_index + 148013, independent of treatment. Full pair is
the resampling unit. The two panels are disjoint; do not pool them or historical
144/145 outcomes. No games are added, excluded, replaced or chosen from outcomes.

## Unchanged player and search

Reuse `PublicValueInference`, exact 035 full-legal-set canonical-max priors and
the selected 143 public 15-feature evaluator, with existing float32 feature
rounding, float64 dot product, zero intercept and clipping to [-1,1]. No refit.

- Checkpoint SHA256:
  `4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`.
- Public evaluator SHA256:
  `a5c87ab3fb0f3f22a0e0e25a5adbfcad1bb3422c7f40cdc060d33a0d3a9b720c`.
- BudgetSearch, c_puct=1.5, max_depth=96, real observer-legal determinization,
  no noise, no temperature, no root cap, fresh tree at each move. Keep exact
  terminal/deadlock recognition and perspective/sign handling.
- Time includes root encoding/expansion. At least one simulation completes;
  no subsequent simulation starts after the deadline. One simulation plus
  assembly can overrun. The config's simulations=64 is a fixed-count testing
  default, not an arena cap. Save completed count and CPU/wall times at every move.
- Eight CPU worker processes, one Torch thread per worker. Both timed arms use
  the same policy/value object and implementation; allowance is the treatment.
  Exact original LegalActionAttentionScorerBot.choose_action supplies raw 035.
  Caller legal enumeration and artifact serialization are outside decision timing.

No 146/147 neural checkpoint enters either panel. Their failed prediction gates
remain closed. Search has no access to hidden deck order or opponent blind-card
identity. The shared rules/search implementation remains the existing audited
implementation; this experiment is not an independent rules validation.

## Endpoints and stop rules

Win=1, draw=.5, loss=0. Calculate candidate mean score share and a percentile
95% CI from 50,000 whole-seed-pair bootstrap resamples. Use NumPy default_rng,
seed 148999 for compute and 148998 for competence. Require in each panel:

1. All trajectories, model/source identities and independent audits valid.
2. Each timed arm's mean decision time <=1.10*allowance and p95 <=1.25*allowance,
   and every search used its full allowance. For long these limits are 1.76 and
   2.00 seconds; for short .44 and .50 seconds. A fourfold measured time ratio
   or simulation ratio is not required; report both actual budgets and cost.
3. Candidate score share >=.55 and CI lower >.50.

The strength gate proves superiority with a material point estimate, not a
five-point advantage with 95% confidence. If upper <.50, label candidate weaker;
if lower >.50 but share <.55, positive below practical threshold; otherwise
improvement not established. Report whether upper <.55 excludes the practical
target. Any integrity/budget failure blocks a positive conclusion.

Natural completion and existing two-sole-PASS forced deadlock are valid; record
deadlocks separately. Use max_turns=400, repetition_limit=4, no_progress_limit=60.
Arbitrary cutoff/stalled games are saved with no valid winner, .5 solely for a
labeled operational summary, and invalidate a positive strength conclusion.
Never substitute their diagnostic score-adjudicated winner. No replacements.

Complete all 400 compute games, reveal its aggregate result, then independently
audit. Only a matching complete successful audit AND the strength/timing gates
permit the competence panel. Audit failure stops the pipeline. Failed or
inconclusive compute strength stops after that stage. Freeze competence before
the first compute game; it gets no candidate changes. Both panel passes are
required for `teacher_headroom_confirmed`. All other outcomes stop without an
automatic budget sweep, further games, training, installation or successor.

Seat shares, score margin, depth, simulations and cost are descriptive. Depth
is each search's maximum reached depth, not average path length. No subgroup
or root statistic can substitute for the primary endpoint. A pass does not
prove distillation, repeated improvement, isolated value utility, or beating 084.

## Preparation, execution and independent audit

Run focused 148 tests and the full repository suite. Preflight four complete
games: compute seeds 1489000000 in both seats, competence seed 1489001000 in both
seats. Those probes never enter protected results; preflight timing is used only
for feasibility. Verify every probe trajectory and prescribed search reproductions.
Probe output reports natural completion and integrity, not strength selection.

Freeze both panel configs, code, protocol, verifier, tests, wrapper, Python/Torch/
NumPy versions and inputs before protected games. Preflight must match that exact
identity. Validate 147's explicit frozen dependencies and inputs; do not recompute
an earlier dynamically enumerated identity after adding new files. Use a process
lock, refusal to overwrite evidence, per-game atomic artifacts and manifest binding.
During collection inspect only progress/integrity, never partial win rates.

The independent verifier reconstructs every trajectory, legal action, state hash,
termination and outcome; every original raw-035 action; and every timed root's
full canonical set, original priors, independently calculated public value, visit
accounting, selected move, bounded values and allowance record. Reproduce first
and last decision of each timed arm in every game using recorded RNG and completed
simulation count. Require exact edges, values, actions, expansions and inference
counts. Middle trees receive structural/prior/value audits rather than complete
regeneration. Recalculate paired statistics and gates independently, verify game
hashes and all frozen dependencies, and bind audit to result SHA256.

The verifier runs as a separate process. It shares the existing engine and search
code, and does not reproduce OS scheduling. The parent verifies audit/result
binding before launching the conditional stage and records both final hashes.
No default restart on failure. Preserve failure traces and inspect causes without
changing protected settings or selecting replacements.

Runner: `run_search_teacher_headroom_148.py preflight`, then `run`. Artifacts:
`outputs/search_teacher_headroom_148/`, with separate compute/competence game,
result and verification files and a top-level final_status.json. Champions
remain 084 research and 035 fast, regardless of this experiment's outcome.
