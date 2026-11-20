# Experiment157: fast selector transfer

Owner authorized the recommendation in `docs/strategy/next_after_156.md` with
"ok, do it please". This protocol and its explicit seven-file implementation
inventory are frozen before exports, fitting or experiment games. Preserve all
dirty work and frozen ancestors. No promotion, successor or adaptive rescue.

## Scientific question and immutable design

Can the155 terminal-rollout choices teach a cheap ranking among its own offered
groups? Both controllers retain the unchanged035 proposer (greedy concrete
action, highest-ranked distinct group, two uniformly sampled remaining groups;
supports1-4; original representatives and ordering). One copied035 attention
network learns to rank; another exact035 network proposes. All concrete legal
actions enter attention. Maximum concrete logit per offered group determines
loss and play, with original candidate-order ties, baseline first. No candidate
indicator enters the network. Return the offered representative, never an alias.
At initial weights the complete controller must exactly reproduce greedy035.

Reuse all24,342155-selector roots from main155 (12,185) and main156 (12,157),
all800games/400initial pairs. Exclude preflights and084 roots. Pair-grouped80/20
split separately within each experiment:160train/40validation pairs.156 pair0
is forced training due prior inspection. Sort remaining identifiers by SHA256
of canonical JSON ["157-pair-split-v1",experiment,pair]; first40 are validation.
All turns and both seats stay together. No outcome filtering or balancing.
The full source inventory, split, manifest and arena schedules are sealed in
freeze.json before export. Existing data are reused development evidence.

Export observer-relative035 observations (295) and complete action features
(79) in immutable per-game Torch shards. Game IDs/splits/teacher indices/group
memberships/phase/privacy are metadata only, never tensors passed to the model.
Snapshots contain hidden state but are not inputs. Own blind reserves remain
known; opposing blind identities, actual deck order/refills, search outcomes,
seed/stream IDs and final game outcomes are forbidden inputs. No pre-action
feature is constructed by stepping the actual hidden snapshot. Reconstruct
every source trajectory, feature, candidate and label independently; check
hidden alternatives at every root and exact initial035 equivalence. Bind the
already audited155/156 terminal labels without repeating millions of rollouts.

## Six fixed fits

Treatment: actual155 choice; control: candidate0. Ordinary float32 CE over only
offered maximum-group logits; one-support CE is zero and the row is retained.
Each row contributes equally to minibatch loss. No value loss, blended parent
target, discovery-max regression, return weight, extra corpus or sweep.

Three fixed order seeds157001/157002/157003 for each arm. Every fit starts from
exact035 SHA4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd.
All parameters except value_head.* train; that head's tensors stay unchanged.
Four complete passes, AdamW lr1e-5/weight_decay1e-4/default remaining parameters,
batch64 including final short batch, clip global gradient norm1.0. CPU one Torch
thread per worker, deterministic algorithms; three simultaneous fit workers.
Float32, no mixed precision. Training mode for updates, eval/no_grad validation.

Freeze batch construction: preserve export order (155game order then156game
order, turns ascending); each epoch uses Python Random string
"157-batches-v1|seed|epoch", shuffles indices, stable-sorts by legal-action
count, chunks64, then shuffles batches with the same generator. Right-pad action
features with zeros to the largest count in each batch; masks exclude padding
from attention and group membership. All concrete legal actions remain present.
This controls padding cost and pairs identical batches between arms. Validation
uses the same batching construction with seed157000/epoch-1, then sorts row
records by identifier for reduction. Batch numerical scores can differ slightly
from single-position inference; the actual controller uses no_grad/unpadded035
execution. No dropout is added and no validation-label training occurs.

Checkpoints: initial0; updates1,2,4,8,16,32,64; each quarter of each epoch at
ceil(q*ceil(N/64)/4), deduplicated. Finish all four epochs. Select smallest
validation CE averaged first within initial pair, then equally across80pairs;
exact ties choose earlier update. Every checkpoint and complete validation
prediction record is immutable; report teacher/own-label agreement, movement,
seat/phase/support/private-card summaries for every seed at every checkpoint.
No offline fit/accuracy/movement gate prevents fresh games for a numerically
valid selected checkpoint, including selected0. No best seed or ensemble.

Independent verifier implements its own split, batch/padding, group/loss,
selection, complete training loop and validation reductions. Shared primitives:
engine, canonical keys, serialization, encoders, Torch model and CE/AdamW.
It requires bitwise-identical tensors at every saved checkpoint in every one
of the six complete replayed fits, verifies the unused value head, selection,
paired batch sequence, all predictions and finite values. All selected models
are sealed together after the full fit audit and before any arena outcome.

## Fresh games, inference and timing

Six controllers each play greedy035 on the same400fresh seeds1570000000..399,
both controller seats:800/model,4,800 games total. Off-panel preflight uses
four seeds1579000000..003, both seats for all six models:48 games. No search or
teacher continuations are invoked. The155 candidate operator retains its frozen
internal tags, but its new stream is SHA256 canonical JSON
["157-proposer-stream-v1",stage,pair,seat], shared across arms/order seeds where
the same legal information is encountered, independent of actual hidden seeds.
Index order: pair, treatment/control,157001/2/3,seat0/1. All schedules freeze
before training. Test-only environments use1579800000+ and are excluded.

Eight game workers, one Torch thread, no nested pool/BLAS environment changes.
Per decision save full concrete learned scores, proposals, nonce, group scores,
choice, privacy, state hash, action, phase and wall/CPU timing. Complete traces,
final states and failures are immutable. Time controller from proposal entry
through learned decision/evidence construction, excluding disk writes; raw035
from choose_action through return using already enumerated legals. Controller
timing includes its own proposal/legal/encoding overhead. Report mean/p95/max
wall and CPU by each fit, role, phase and offered support. Latency gate uses
equal average of the three treatment per-decision wall means, divided by equal
average of raw035 wall means in those same treatment games; require<=10x and
strictly<0.1s. Control timings also reported. This is host/instrumentation/game
distribution dependent, not identical-position microbenchmarking. No numerical
or timing tweak follows preflight. Preflight integrity/reload/privacy must pass
before main; latency is a final gate, no model is removed for slow probe timing.
Report measured collection+replay throughput to refine remaining time.

Every actual game/decision gets independent full replay, legal input and hidden
invariance checks. Normal/equal-turn completion and two consecutive sole-legal
PASS turns are valid, latter adjudicated by score then fewest purchased cards.
Fourfold repetition,60no-progress turns,global turn400, illegal/missing action,
exception or invalid numerics fail the experiment. Save failures, no fabricated
winner, exclusion, replacement, retry or automatic resume. Full protected arena
and replay complete regardless of statistical/latency result if numerically and
legally valid. The sole pipeline/verifier kernel locks prevent concurrent runs.

## Statistics and stopping

For each of400initial seeds, average its two seats and three fixed fitted
policies within each arm. Bootstrap50,000 complete starting-seed clusters with
NumPy default_rng1573000000, chunks1000, retaining arms/seats/fit pairing. Report
percentile95% CI for treatment share and treatment-minus-control share. Fixed
training seeds are not extra independent games; the interval conditions on
these three fits. Also report all six per-fit shares, seats and margins.

Practical gate: treatment>=55%, CI lower strictly>50%, mean final score margin>0
and each aggregate seat>=50%. Attribution: treatment's share against035 minus
control's share against035>=5pp and paired difference CI lower>0. Point gates
use exact half-outcome sums divided by2400 to avoid threshold rounding from
intermediate pair averaging. Both, latency and all integrity gates required.
This is performance against a common opponent, not treatment-vs-control play.
No arena extension, optional stopping, training/model/seed changes or successor.

Full repository tests precede freeze and exact test evidence is rebound at
stage boundaries. Parent156->155->154->earlier explicit inventories are checked;
never invoke149's dynamic identity scan. The seven157 files are data/fit helper,
runner, auditor, root entry, tests, this protocol and durable launcher. Each is
hash-bound with config/Python/Torch/NumPy and closed155/156 artifacts. Reports and
progress are outside code identity. If interrupted, preserve evidence; an
identical crash resume would require explicit review/authorization and separate
recovery implementation. Both valid outcomes write report/final status and stop.
