# Deadlock-Validated Score-Delta Rolling-Horizon Gate (`042`)

Frozen before implementing this gate or generating any fresh `042` game. This
is a corrected-terminal, fresh-seed replication of `041`; it is not a parameter
repair or strength-threshold change.

## Why `041` did not answer the strength question formally

`041` passed every statistical strength gate against raw `035` (117-83,
58.5% share, CI95 [52.0%, 65.0%]) but rejected one
`forced_pass_deadlock` as a non-completed game. That integrity rule contradicted
the repository's already-tested engine contract from DAgger `026`/`027`, under
which two consecutive forced passes are a valid clean terminal while stalls and
true cutoffs remain invalid.

The known seed `403078` was reconstructed only to audit semantics; it is
forbidden from `042` evidence. At turns 28 and 29 respectively, both players'
complete legal action set was exactly `[PASS]`. Every colored bank stack was
empty, the bank held only five gold, each player held ten colored tokens and
three reserved cards, and neither could purchase. The two passes were forced;
this was not a policy loop, timeout, hidden-information issue, or search-budget
failure.

Preserve the established engine outcome: `forced_pass_deadlock` is adjudicated
by current prestige score and the normal fewest-purchased-cards tie-break. Do
not reclassify it as a draw. The only correction in `042` is to count this
existing clean terminal reason as valid in the integrity gate.

## Frozen candidate

The candidate is literally the `040`/`041` score-delta BMRH planner:

- legal `035` prior and simulated opponent;
- actor horizon 2, top four distinct canonical successors;
- 32 sequence evaluations over two observer-relative determinizations;
- unchanged branching mutation, tie-breaking, score-delta evaluator, terminal
  override, checkpoint, and seed derivations;
- CPU, max 400 turns, repetition limit 4, no-progress limit 60.

No planner parameter or statistical gate may change from `041`.

## Corrected terminal integrity

Valid clean terminal reasons are exactly:

- `completed`;
- `forced_pass_deadlock`.

Every `forced_pass_deadlock` must have `timed_out=false` and `stalled=false`, as
guaranteed by the tested match contract. `stalled`, `repetition_cutoff`,
`no_progress_cutoff`, `max_turns`, and every unknown reason are invalid.

## Fresh direct confirmation

Run 100 new paired seeds / 200 games against raw legal `035` at seeds
`404000--404099`. Require:

- exactly 200 games and zero invalid terminations;
- zero timed-out or stalled games;
- exact `32 x planning decisions` sequence accounting;
- score share at least 55%;
- paired-bootstrap CI95 lower bound strictly above 50%;
- positive mean paired final-score margin.

Failure closes rolling horizon. Do not combine `041` and `042`, replace a seed,
or tune after the result.

## Conditional broad confirmation

Only a complete direct pass unlocks fresh 50-pair / 100-game blocks:

- gate-`030` information-legal shallow search, seeds `404200--404249`;
- historical scorer `016`, seeds `404300--404349`;
- loop-safe clean canonical policy `023`, seeds `404400--404449`.

Every block independently requires the same integrity, exact-budget, 55%
share, lower-CI-above-50%, and positive-margin conditions. All three must pass.
A complete pass establishes the planner as a stronger search teacher and only
then authorizes a separately preregistered search-every-move corpus. It does not
promote a neural checkpoint itself. GPU use is not authorized.
