# Option-Conditioned Predictive Probe (`039`)

Frozen before fresh collection, probe training, validation selection, or
confirmation access. This experiment trains diagnostic heads only. It cannot
update a gameplay policy, run strength games, or promote a checkpoint.

## Question

Does the token-take representation contain incremental information about the
later card objective beyond the complete public state and candidate card
representations?

Both probes use the frozen canonical entity/action transformer from `023`.
For every qualifying take, its target is the first card bought or visibly
reserved within four own turns that was already visible or self-reserved at the
take. Candidate card vectors and the pooled state vector come from frozen `023`.

Two identically initialized, identically parameterized MLP heads are trained:

- state-only: its option-vector input is exactly zero;
- option-conditioned: the option vector is frozen `023`'s contextual embedding
  of the selected take.

Only the heads train. This is a decodability test, not a policy update.

## Fresh splits

Generate 512 outcome-blind greedy-035 games at seeds 391000--391511 against the
four frozen league opponents with exact actor-seat balance:

- games 0--191: training, 24 games per opponent/seat cell;
- games 192--255: validation, 8 games per cell;
- games 256--511: untouched confirmation, 32 games per cell.

The raw corpus inherits `038`'s privacy guard: no winner/final fields, true deck
order, or opponent-private reserved identities. Confirmation examples are not
loaded until both heads and their independently selected minimum-validation-NLL
epochs are frozen. Train 30 paired epochs at learning rate `1e-3`, weight decay
`1e-4`, batch size 64, and fixed seed 390717.

## Confirmation gate

Require at least 800 confirmation examples and 80 examples whose target is at
least two own turns away. On all confirmation examples, option conditioning
must achieve:

- at least 5% relative NLL improvement, with game-cluster paired NLL-delta CI95
  upper bound below zero;
- at least 5 percentage points top-1 improvement, with paired CI95 lower bound
  above zero;
- at least 3 points MRR improvement, with paired CI95 lower bound above zero.

On the multi-turn subset, require at least 3% relative NLL improvement, top-1
nonregression, and at least 2 points MRR improvement. Every major
opponent/seat and phase stratum with at least 50 examples must avoid more than
2% NLL regression.

A complete pass authorizes a separately preregistered supervised high-level
card-option model and option-conditioned low-level behavior-reconstruction
probe. Failure closes the hierarchical direction. Thresholds and splits may
not be changed after collection.
