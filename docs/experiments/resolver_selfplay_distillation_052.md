# Synchronous Resolver Self-Play Distillation (`052`)

Frozen before instrumenting the proven resolver, generating any `052` game,
training a student, or inspecting any `052` outcome. This is the first complete
teacher-collection and supervised-update iteration authorized by the broad
`051` pass. It is CPU-only and permits exactly one student proposal.

## Question and prerequisite

`049b`, `050`, and `051` established that the exact bounded public-belief
resolver has stable local targets, beats raw legal `035` 138-62, and scores
74.5--79.0% against three broad opponents. `052` asks whether one synchronous
generation of resolver self-play can transfer that strength into the fast
legal-action attention model without privileged observations or outcome-only
credit assignment.

Require `outputs/broad_public_belief_resolver_confirmation_051/result.json` to
have SHA-256
`1a731e5edf3c63cffb2b46d8adaebba9fc50b81e490367bb0c8448ef6a885ed4`,
report `passed=true`, and authorize a separately preregistered resolver teacher
corpus. Require the frozen `035` checkpoint SHA-256
`4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`.

No `052` target, validation metric, confirmation metric, or gameplay outcome
may tune a temperature, blend, learning rate, epoch count, seed, split, or gate.

## Honest target semantics

The public-belief resolver is not MCTS and produces no visit counts. Do not
label invented quantities as visits. For every legal action, store a complete
`resolver_policy_distribution`; separately store the exact four searched
canonical successor values. The policy target is fixed as follows.

1. Compute frozen raw-`035` logits for every concrete legal action and their
   softmax distribution at temperature `1.0`, `p035`.
2. For a completed resolver search, softmax its four adversarially resolved
   normalized score-margin values at temperature `0.10`.
3. If multiple concrete legal actions have the same canonical public
   successor, allocate that successor's resolver mass among them in proportion
   to their conditional `p035` mass. This avoids arbitrary supervision among
   strategically identical actions.
4. Form the complete target over every legal action as
   `0.75 * presolver + 0.25 * p035` and renormalize.
5. On a preregistered structural resolver fallback, store exactly `p035` as an
   anchor target with training weight `0.25`; completed-search rows have weight
   `1.0`.

Every row stores reconstructable state snapshot, acting seat, game seed, turn,
phase, chosen action, raw-`035` action, fallback reason if any, all legal action
indices and probabilities, all four canonical resolved values where available,
final winner and actor-relative result, and exact resolver telemetry. Rows from
both the eventual winner and loser are retained. No outcome filters or
winner-weight changes are allowed.

The snapshot may retain simulator state solely for exact reconstruction. Model
features must be regenerated through the existing acting-player public
observation and legal-action encoders. Hidden-deck order, opponent blind reserve
identity, terminal outcome, future state, and resolver branch truth are never
model inputs.

## Frozen teacher and synchronous collection

Both seats use the exact `050`/`051` `FrozenPublicBeliefResolverBot` backed by
raw `035`: three observer-relative determinizations, top-four canonical root
successors, top-four information-set-consistent opponent responses, four
matched stochastic terminal continuations, and 192 branches per completed
search. Preserve every leaf parameter, structural fallback, loop rule,
canonical tie, belief namespace, and continuation namespace literally.

Use CPU, ten workers, PyTorch one thread per worker, game turn cap 400,
repetition limit 4, and no-progress limit 60. Valid game terminations are
exactly ordinary `completed` and corrected two-sole-`PASS`
`forced_pass_deadlock`.

Run these stages in order:

1. Recording smoke: play seed `610000` twice from zero with resolver on both
   seats. Exclude both games from learning and require byte-identical projected
   games, target rows, and telemetry.
2. Development collection, with no model update during collection:
   - training games: seeds `610100--610179` (80 games);
   - validation games: seeds `610180--610199` (20 games).
3. Train and select exactly one student as specified below.
4. Only after model selection, collect a fresh untouched confirmation corpus at
   seeds `610300--610319` (20 games), using the unchanged teacher.

Collection is resumable by completed game and exposes only completion counts,
errors, and ETA before a stage finishes. Training and validation targets may be
inspected after development collection; confirmation targets and metrics remain
untouched until the best epoch is fixed.

## Collection and privacy gates

Every collection stage must satisfy all applicable conditions:

- exact checkpoint/configuration/result hashes and exact seed membership;
- every game has a valid clean termination, with zero timeout, stall, cutoff,
  illegal action, exception, or unknown terminal;
- exact 192-branch accounting, all continuations resolved, zero hard caps, and
  loop-adjudicated continuations at most 0.1%;
- one row for every teacher decision, with completed-search plus structural-
  fallback rows exactly equal to decisions;
- regenerated legal action indices exactly equal the stored target support,
  with finite non-negative probabilities summing to one within `1e-6`;
- completed-search rows contain exactly four distinct canonical successor
  values and fallback rows contain none;
- target reconstruction is invariant to true hidden-deck permutation and
  opponent blind-reserve identity wherever those facts are unobservable;
- at least 4,000 training rows, 1,000 validation rows, and 1,000 confirmation
  rows, with at least 50 completed-search rows in every seat, phase, and
  winner/loser role in each non-smoke split.

Any smoke or development collection failure stops `052` before training.

## Frozen student and optimization

Initialize the exact legal-action attention architecture and every parameter
from frozen `035`; do not reset a head or add a value model. Train only against
the stored complete policy distributions:

- policy hard-label weight `0.0`;
- value loss weight `0.0`;
- resolver policy-distribution cross-entropy weight `1.0`;
- all search-score, preference, trajectory, and outcome losses `0.0`;
- AdamW, learning rate `1e-5`, weight decay `1e-4`;
- batch size 256, four epochs, deterministic shuffle seed `52`, CPU;
- train only on the 80-game training file;
- select the single epoch with lowest weighted validation target
  cross-entropy on the fixed 20-game validation file, breaking ties toward the
  earlier epoch.

Save the final and selected checkpoints, every epoch metric, optimizer recipe,
source/result hashes, row counts, phase/seat/outcome breakdowns, and parameter
movement. Do not interpolate, ensemble, resume with a changed optimizer, or
train a second proposal.

## Untouched offline gate

For parent and selected student, evaluate the stored full distributions using
acting-player legal inputs only. Primary error is
`KL(target || model_policy)`, not hard-action accuracy. All conditions pass:

1. On completed-search rows, student KL is at most 80% of parent KL on both
   validation and untouched confirmation.
2. On confirmation completed-search rows, student agreement with the target
   top action exceeds parent agreement by at least five percentage points.
3. Student-versus-parent confirmation argmax movement is between 5% and 50%.
4. Student completed-search KL is strictly below parent KL independently in
   early, mid, late, seat 0, seat 1, eventual-winner, and eventual-loser
   strata. Every stratum must contain at least 50 rows.
5. On confirmation structural-fallback rows, `KL(p035 || student)` is at most
   `0.02`; the learned correction may not destroy the parent anchor.
6. Every score is finite, every selected action is legal, and hidden-state
   invariance passes. No phase, seat, or outcome role may be omitted.

Failure rejects the student and stops before gameplay. Do not change the target
temperature/blend or retrain after seeing it.

## Conditional paired promotion gates

Only a complete offline pass unlocks 100 fresh paired seeds / 200 games against
raw `035` at `610500--610599`, with seats reversed per seed. Require exactly
the `050` clean terminal contract plus:

- score share at least 55%;
- paired-bootstrap CI95 lower bound strictly above 50%;
- positive mean paired final-score margin;
- score share from each seat at least 50%.

Only a complete parent pass unlocks three independent fresh 50-pair / 100-game
broad blocks:

- gate-`030` shallow search: `610700--610749`;
- historical `016`: `610800--610849`;
- loop-safe canonical `023`: `610900--610949`.

Every broad block independently requires 100 clean games, zero timeout/stall or
invalid terminal, score share at least 55%, CI95 lower bound above 50%, positive
mean margin, and at least 50% from each seat. All three must pass; aggregate
panel statistics cannot rescue one failed opponent.

## Decision

A complete collection, offline, parent, and broad pass promotes the selected
student as the new fast checkpoint champion and authorizes a separately
preregistered synchronous generation 2 using that promoted student as the
resolver prior. It does not authorize an open-ended autonomous loop.

Any failure retains raw `035`, rejects this single student, and stops the stage
chain at the specified boundary. Do not add games, reuse confirmation for
training, tune the target, change learning rate, or train another epoch after
the result. Corpus artifacts remain valid diagnostic evidence even if the
student fails. GPU rental is not authorized for `052`; corpus generation is
engine/rollout bound and this one small model update is appropriate on CPU.
