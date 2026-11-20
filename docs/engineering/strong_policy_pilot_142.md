# Experiment 142: one learning cycle initialized from strong policy 035

Prospective protocol, 2026-09-08. The owner authorized the proposed experiment.
Preserve the intentionally dirty worktree and all prior experiments. This is an
isolated branch; the original randomly initialized 090 protocol remains intact.
084 remains research champion and 035 fast champion. No installation, promotion,
second generation, additional evaluation, or outcome-driven tuning is included.

## Question

Can one search-and-learning cycle improve a player whose initial move policy is
exactly 035? The primary estimand is successor score share against its own
value-bootstrapped parent, both using the same search and 0.4 seconds per decision
on the same CPU host. This is a local learning-pilot test, not a comparison with
084, evidence of superhuman strength, or an isolated causal test of initialization
versus the old 090 architecture. No 086/084-derived labels enter the new corpus.

035 already has a value head but its training assigned value loss zero weight.
Copy the checkpoint with SHA256
`4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`.
First train only its value head, preserving all policy-affecting tensors exactly.
The parent is this model; the successor is one end-to-end update of that parent.

## Policy bridge and search

The original 295 public-observation and 79 action features are unchanged. The
attention network always sees the complete concrete legal action list, including
when search requests only canonical representatives. Each canonical successor
gets its maximum concrete logit. This preserves greedy successor preferences and
avoids alias-count weighting. The training policy loss uses exactly the same
canonical maximum; no representative-only attention or fabricated alias labels.
Raw 035 logits/actions must match the adapter exactly. Policy tensor hashes must
be identical before and after value bootstrap. Existing information-legal search
samples hidden worlds and uses the original finite-game termination rules.

Search: real information-set mode, PUCT 1.5, max depth 96, no tree reuse between
moves; 128 simulations for learning games. Self-play has root Dirichlet mixing
.25 with total concentration 10, temperature 1 for the first 20 turns, then greedy
visits. Arena has no noise or temperature. Root visit sums must equal completed
simulations. Arena deadlines include feature encoding/root expansion, permit at
least one complete simulation, and start no subsequent simulation after expiry.
One simulation plus result assembly may overshoot. Log measured decision wall
time, CPU time, simulations and depth for every move. Both players run within
each worker, with the same deadline and architecture; eight workers, Torch one
thread each. Timing variability makes arena choices nondeterministic; fixed
self-play and the recorded arena action trajectories remain reproducible.

## Fixed data, learning, and evaluation

1. **256 bootstrap games**, seeds 1,420,000,000 through 1,420,000,255. Both seats
   use 035; during the first 20 turns, with probability .2 sample among its top
   three concrete actions using softmax temperature .5; otherwise greedy. Train
   value from completed-game W/D/L (-1/0/+1 from the acting player's view), keeping
   losses and wins. Zero the last value linear layer before fitting; freeze every
   other parameter. Eight epochs, AdamW LR .001, weight decay .0001, batch 128.
   Frozen state representations may be cached. Parent = fixed final epoch.
   Halt if its held-out value MSE does not beat the initial zero predictor, or if
   any policy-affecting tensor changes. This gate validates a useful bootstrap,
   not a claim that the value function estimates optimal play.
2. **128 searched self-play games**, seeds 1,421,000,000 through 1,421,000,127,
   all from the frozen parent, 128 simulations per decision. Store complete
   root visit distributions and actual final W/D/L. No 084/086 labels, no older
   replay, no extra generation. Require at least 5,000 naturally completed-game
   positions before training. Opening search noise provides exploration.
3. **One end-to-end update**, four epochs, AdamW LR .00001, weight decay .0001,
   batch 64, global gradient norm cap 1. Joint batches group positions by legal
   action count to limit padding; shuffle ties and batch order each epoch. Every
   training position is used once per epoch. Loss = canonical soft-target policy CE
   plus value MSE, equal weights, all parameters trainable, no KL anchor or
   selective label filtering. The fixed final epoch is evaluated, independent
   of offline losses. No best-checkpoint/arena selection. Nonfinite loss or
   gradients abort. Bootstrap split/training seed 142010; joint seed 142011.
   Each corpus has its own 20% held-out split by whole game, never by position.
   Validation is diagnostic and is not recycled into training.
4. **400 primary arena games**, 200 fresh seeds 1,422,000,000 through
   1,422,000,199, each played with successor in both seats. Each side uses the
   same 0.4-second search budget per decision. A win scores one, draw one half.
   Report W/D/L, score share, seat results, time, simulations and depth. The
   95% percentile interval resamples the 200 entire seed pairs 50,000 times,
   NumPy default_rng seed 142999. No interim winners, losses, gains, subgroup
   comparisons or optional stopping are inspected during collection.

All games have absolute turn cap 400, repetition limit four and no-progress
limit 60. Natural outcomes include an actual completed game and a verified two
sole-pass deadlock scored by points then fewest purchased cards. Cutoff games
remain recorded and are never assigned training outcomes. Stop a learning stage
if more than 2% of its games fail to finish naturally; otherwise use all valid
games and report the exclusions. No replacement games. Every arena game is kept;
cutoffs score one half for an operational summary but ANY arena cutoff prevents
a valid superiority claim. No engine/legal-action failures are tolerated.

## Decision and integrity

Before arena outcomes are interpreted, require complete dose, legal replay,
correct outcome/signs/visit accounting, unchanged source/dependency/data/model
hashes, exact policy retention, finite training, and natural arena completion.
Time matching also requires successor/parent mean decision-time ratio in
[.95,1.05], with both means between .4 and .5 seconds. Report tails even if this
gate passes. A failed gate is an invalid pilot, not a strength conclusion.

For a valid arena: CI lower bound > .5 and point estimate >= .55 is material
improvement; CI lower > .5 with smaller point estimate is positive but below the
material point estimate; CI upper < .5 is regression; otherwise improvement is
not established. This sample resolves sizeable gains, not tiny improvements.
Report the interval, not just the label. Success licenses recommending a fresh
replication, not automatic promotion or another generation. A failure bounds
only this architecture, dose, search and optimization recipe, not all strong
initialization, value learning or self-play programmes.

## Execution and review

Freeze all source Python, runner, tests, verifier, protocol, configuration,
software versions and 035 hash in `outputs/strong_policy_pilot_142/manifest.json`
before protected collection. Stage records bind input checkpoint and manifest
hashes; save tensors and full legal-action traces atomically per game. Resume
only the identical schedule/identity; no replacement of completed work. Preserve
all old frozen dependencies. Full-suite tests and outcome-blind off-schedule
timing/structural probes precede collection. Probe seeds occupy the separate
1,429,000,000 range and never enter learning or arena. Initial adapter probes at
turns 0/20/40 took .095-.150 seconds for 32 simulations on an idle single core.

The independent verifier reconstructs every action trajectory, legal move,
state hash, termination, training input feature, outcome label and search visit
target; checks artifact identities/retention; and independently reconstructs
paired statistics. It does not independently reimplement Splendor rules or
repeat every internal search simulation. Wall-time variability is measured,
not silently treated as deterministic search.
