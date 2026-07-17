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

## Result

Rejected on 2026-07-17. The privacy-clean collection contained 512 completed
games, 16,201 actor turns, and zero stalls or timeouts. Feature splits contained
934 training, 352 validation, and 1,216 untouched confirmation examples. Both
heads independently selected epoch 22; validation NLL was `2.063525` for the
state-only head and a worse `2.084111` for the option-conditioned head.

Untouched confirmation:

- state-only: NLL `1.978413`, top1 `32.48%`, MRR `0.518431`;
- option-conditioned: NLL `1.995238`, top1 `31.58%`, MRR `0.514029`;
- paired game-cluster NLL delta: `+0.017896`, CI95
  `[+0.001319, +0.034511]`;
- top1 delta: `-1.20` points, CI95 `[-2.46, +0.07]`;
- MRR delta: `-0.005291`, CI95 `[-0.013354, +0.002690]`.

The multi-turn subset had 113 examples. Option NLL worsened from `2.278359` to
`2.306954` (`-1.26%` relative improvement). Top1 increased from `18.58%` to
`21.24%`, but its clustered CI touched zero. MRR increased only `0.01275`, with
an interval spanning zero and below the frozen `0.02` floor.

Nine gates failed: every overall improvement/significance gate, multi-turn NLL
and MRR, and major-stratum noncollapse. Coverage and multi-turn top1
nonregression passed. Decision: `explicit_option_not_incremental_close_hierarchy`.
No policy was updated and no gameplay ran.

`038` therefore measured a genuine card-cost-directed behavioral motif, but
`039` shows it is not a useful incremental latent variable beyond contextual
public state and card features. Do not tune this probe or proceed to a
hierarchical gameplay model.
