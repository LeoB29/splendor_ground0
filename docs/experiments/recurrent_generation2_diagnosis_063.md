# Recurrent Generation-2 Failure Diagnosis (`063`)

Frozen after `062` rejected generation 2 and before reconstructing its decision
corpus or computing any new breakdown. This is a CPU-only, no-training,
no-new-gameplay diagnostic. It cannot promote a checkpoint or retroactively
change the `062` decision.

## Question

Why did the same bounded recurrent update recipe pass generation 1 but produce
both a negative held-out surrogate and a 17-23 immediate-parent result in
generation 2?

The audit distinguishes:

1. small-validation-set variance or cell imbalance;
2. train-to-validation overfitting;
3. a policy direction that reverses the useful generation-1 direction;
4. failure to use public history despite the recurrent architecture; and
5. a broad systematic generation-2 regression.

## Frozen prerequisites

- `062` result:
  `outputs/recurrent_league_stage_b_062/result.json`
- Result SHA-256:
  `2c4d245eb683d2008f84a3ac705c1fa684817e0fac3b7b000b07d7112d7eeb5d`
- Epoch zero SHA-256:
  `2d807231666420bb221049aa7e2a4a28358ef62618cf1c652ede67dd7249acef`
- Accepted generation 1 SHA-256:
  `e1805b4ff76cbe43f8ff19f3d38e94ef2ced87713109a451c9c249006945c485`
- Rejected generation 2 SHA-256:
  `b2d4f85b126b469eb70931a92cb9c808c44ceb743eb640803f97495530373d10`

Also require the exact `035`, `016`, clean canonical `023`, and Stage A hashes
already frozen in repaired `062`.

## Exact corpus reconstruction

Reconstruct generation 2 only, with accepted generation 1 acting on the exact
original league:

- `035`, `016`, clean canonical `023`, legal shallow, and generation 1;
- original generation number `2`;
- original seed start `621000`;
- original sampling-seed formula;
- original 240-game balance and split;
- original complete-trajectory WDL/margin returns and per-cell normalization.

Perform no optimizer step. Require exact equality with the stored `062`
generation-2 collection summary and per-game summaries, exact behavior
log-probability reconstruction within `1e-5`, and all 240 valid games. Any
failure yields `generation2_diagnosis_invalid_corpus_not_reproduced`.

## Surrogate localization

For frozen generation 1 as parent and rejected generation 2 as candidate,
compute the game-balanced chosen-action surrogate

`advantage * log(candidate probability / parent probability)`

separately on train and validation games. Report:

- overall train and validation values;
- opponent, actor seat, opponent-seat cell, phase (`0-19`, `20-49`, `50+`),
  and chosen action type;
- per-game contributions and positive-game fraction;
- leave-one-cell-out validation values;
- 20,000-sample game-cluster bootstrap CI95, seed `630063`;
- split-half validation values using alternating game ordinal within every
  opponent-seat cell.

The validation split has only four games per cell, so all uncertainty is
clustered by game, never by decision.

Apply the same generation-2 validation advantages to epoch-zero versus
generation-1 action changes. This asks whether generation 2's sampled outcomes
still support the prior successful direction.

## Direction audit

For all trainable parameters, report generation-1 delta
`g1 - epoch0`, generation-2 delta `g2 - g1`, their norms and cosine. Repeat for:

- public-history embeddings/projection/GRU;
- residual hidden/output layers.

On generation-2 validation states, compute the cosine between centered
epoch0-to-generation1 and generation1-to-generation2 legal-logit deltas,
overall and by phase.

## Public-history ablation

Repeat parent-versus-candidate policy diagnostics on the same validation rows
after replacing every public transcript with an empty transcript while
leaving current public state and legal actions unchanged. Report:

- real-history versus empty-history KL, total variation, argmax movement, and
  surrogate;
- correlation and mean absolute difference between real- and empty-history
  chosen-action log ratios;
- the fraction of candidate movement removed by emptying history.

This is an architecture-use diagnostic only; the empty-history policy is not a
legal gameplay candidate.

## Existing parent-gate uncertainty

Use only the 20 paired seeds already stored in `062`. Report candidate score
share and paired score-margin bootstrap CI95 with 20,000 paired-seed samples,
seed `630064`, plus seat breakdown. Run no additional games.

## Interpretation

Report nonexclusive flags:

- `train_validation_reversal`: train surrogate positive and validation
  surrogate nonpositive;
- `validation_sign_uncertain`: validation game-bootstrap CI includes zero;
- `validation_systematically_negative`: validation CI upper bound below zero;
- `direction_reversal`: parameter-delta or mean logit-delta cosine below zero;
- `history_material`: emptying history removes at least 25% of real-history KL
  or changes chosen log ratios with correlation below `0.80`;
- `history_mostly_unused`: empty-history KL is at least 80% of real-history KL
  and chosen-log-ratio correlation is at least `0.90`;
- `parent_gate_uncertain`: paired score-share CI includes `0.50`;
- `parent_gate_systematically_worse`: paired score-share CI upper bound below
  `0.50`.

Final decision:
`generation2_failure_diagnostic_complete_no_scale_authorized`.

The diagnostic may recommend the next experiment, but it authorizes no
training, rerun, Stage C scale collection, promotion, or GPU rental.
