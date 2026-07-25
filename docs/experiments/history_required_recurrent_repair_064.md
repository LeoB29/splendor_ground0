# History-Required Recurrent Succession Repair (`064`)

Frozen after the valid `063` diagnosis and before implementation, collection,
or learning evidence. This is a fresh CPU-only two-generation repair of `062`,
not a rerun. It cannot promote a checkpoint.

## Question

Can recurrent population learning compose across two generations when:

1. policy movement is structurally impossible without public history;
2. weakly attributable early actions receive less whole-game credit; and
3. every opponent-seat validation cell contains eight fresh games rather than
   four?

## Frozen prerequisites

- `063` result:
  `outputs/recurrent_generation2_diagnosis_063/result.json`
- `063` SHA-256:
  `ee6b5275ff012081b7bce97637ccd64fd1bae25ae74e07c434f71915c6882a73`
- Frozen legal `035`:
  `4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`
- Frozen `016`:
  `876ac23a005d367ac48b3619bb7cdcb6e0230a3e9e97943d7eecdc6d07d1b056`
- Frozen clean canonical `023`:
  `29ef7c13c5553ec88c8bcb98f7378c925f0e600e2fcf3d944df5926efed7a070`

Do not reuse `062` games, validation states, or candidates for fitting or
selection.

## History-required residual

Initialize a new recurrent checkpoint from exact frozen `035`, using the same
public event schema and 64-wide GRU as `061`.

For every legal action form a frozen action/state feature vector from:

- frozen `035` state hidden features;
- frozen contextual action features;
- frozen parent action score.

Project that vector to width 128. Separately project the public-history GRU
context to width 128 with no bias. Multiply the two projections elementwise,
apply GELU, and map to one residual scalar with a bias-free output layer.

The output layer starts at exact zero. Because the history projection and
output are bias-free, an empty history produces an exact zero residual for all
future parameter values. Therefore any learned deviation from `035` requires
a nonempty public transcript; the optimizer cannot fall back to the
state-only shortcut diagnosed in `063`.

Require epoch zero to reproduce `035` scores, value, greedy choices, and
checkpoint reload exactly.

## Fresh generations

Use the unchanged static league:

- raw legal `035`;
- `016`;
- clean canonical `023`;
- exact information-legal shallow depth-one search.

Generation 1:

- 320 eligible games from seed band `640000--640999`;
- 40 games per opponent-seat cell;
- first 32 per cell train, last 8 validate.

Generation 2 conditionally adds accepted generation 1:

- 400 eligible games from seed band `641000--641999`;
- 40 games per opponent-seat cell;
- first 32 per cell train, last 8 validate.

Eligibility remains outcome-blind and accepts only `completed` or
`forced_pass_deadlock`. Attempt ceilings are 400 and 480. All other lifecycle,
sampling, legality, WDL/margin return, and per-cell normalization contracts
remain those frozen in `062`.

## Phase-weighted credit

Every game still has total decision weight one, but distribute it by phase:

- turns `0-19`: weight factor `0.25`;
- turns `20-49`: factor `1.0`;
- turns `50+`: factor `1.0`.

Normalize these factors within each game. The game advantage remains constant
within a trajectory. This reduces early-action dominance without selecting or
relabeling actions from `062`.

Use the unchanged four-epoch clipped update, learning rate `3e-4`, batch size
128, entropy `0.002`, gradient cap `1.0`, and validation-KL projection target
`0.008`.

## History-use diagnostics

On every validation decision after each update:

1. Empty every history and require candidate and embedded raw `035` scores and
   values to be exactly equal.
2. Deterministically rotate histories across games within exact
   `(opponent, actor seat, turn)` cells, preserving history length and public
   schema. Compare real-history and rotated-history action distributions.

Require mean real-versus-rotated total variation at least `0.002`. This ensures
the policy uses transcript content rather than merely detecting that a history
exists.

## Generation gates

Require:

- exact valid collection size, opponent-seat balance, and 32/8 split;
- zero illegal actions and no ineligible game in a split;
- exact behavior reconstruction within `1e-5`;
- frozen embedded `035`, trainable history/residual parameters, finite metrics,
  nonzero gradient, and exact checkpoint reload;
- validation raw immediate-parent KL in `[0.0001,0.0100]`;
- validation argmax movement in `[0.005,0.12]`;
- positive overall validation surrogate;
- positive combined mid/late validation surrogate;
- normalized validation entropy at least `0.30`;
- exact empty-history fallback to raw `035`;
- real-versus-rotated-history total variation at least `0.002`.

Then run 40 untouched paired seeds against the immediate parent:

- generation 1: `642100--642139`;
- generation 2: `642200--642239`.

Require all 80 games valid, candidate score share at least `0.45`, and mean
paired score margin at least `-1.5`. Stop if generation 1 fails.

## Conditional final population gate

If both generations pass, compare generation 2 and epoch zero against each
static opponent on the same ten paired seeds:

- `035`: `643000--643009`;
- `016`: `643100--643109`;
- `023`: `643200--643209`;
- shallow: `643300--643309`.

Require all 160 games valid, aggregate generation-2 minus epoch-zero
score-share delta strictly positive, at least two nonnegative opponent deltas,
and no opponent delta below `-0.10`.

## Decisions

Integrity failure:
`history_required_repair_invalid_no_design_decision`.

Any learning or gameplay failure:
`history_required_repair_failed_diagnose_before_scale`.

All gates pass:
`history_required_repair_composes_authorize_stage_c_preregistration`.

A pass authorizes only a separately frozen scale protocol. It does not promote
the candidate or authorize GPU rental.
