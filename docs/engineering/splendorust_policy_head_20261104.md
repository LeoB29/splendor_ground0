# Policy-head-only fixed-corpus study

4 November 2026. The user authorized the recommended next study and its launch:
“ok let's do the next step you recommend.” No further launch confirmation is
required after qualification. The original model remains the reference; larger
data, external opponents and six-world work remain deferred.

## Scientific question and sole intervention

Test whether learning the 512-simulation teacher's policy can improve play while
preserving the original value function. Warm start from original checkpoint
EF8A4521CD6C03C7075F15EFEE23F4CDE6EC94D1B5398765CF0C24A09AD745CB.
Only policy.weight and policy.bias (20,817 parameters) are trainable. Every shared
trunk and value parameter stays byte-identical to the original. This limits
learning to existing features; a negative result does not rule out joint learning.

Keep the original corrected corpus:256 games,192 training games/10,660 roots,
32 validation games and32 offline games. Reuse the already accepted data at its
original paths and hashes, including the earlier48-game prefix. No source games
are generated and no historical labels are relabelled. All64 held-out games have
already been opened and are descriptive diagnostics, never fresh confirmation.
The previously trained joint-update checkpoint is not an initialization.

Keep two complete epochs, original shuffle seed/order, batch32, AdamW learning
rate1e-5, decay1e-4, betas0.9/0.999, epsilon1e-8, gradient cap1, float32 and CPU0.
Keep the original legal policy mask, game weights and two-head objective. The
value contribution is constant with respect to trainable parameters. One fixed
final checkpoint; no sweep, early stopping or selection on held-out results.

## Qualification before scientific training and play

Every actual fitter/evaluator calls verify_runtime_sources() once at startup.
Retain the corrected catalogue, noble rules and atomic turn contract from the
accepted pilot. Run the actual corrected loader/collation counterexamples,
including all six corrected faces, nobles, final-round termination, gold/returns,
private reserves and rejection of incompatible data.

Before the668 scientific optimizer steps, use a separate disposable original
model and the first32 examples in the fixed epoch0 order, with their official
corrected terminal labels. Perform one engineering optimizer step, check the
exact trainable parameter set, genuine policy change, byte-identical shared/value
tensors and identical value outputs. Discard this model/optimizer and reload the
pristine original before the full training run. The engineering update and its
three forwards are counted separately. It is not a second trained candidate.

Check frozen tensor hashes after every main update, and policy changes. Check
parent/candidate value equality on the fixed parity inputs and all3,557 saved
held-out inputs. Retain the six eager/compiled learned-adapter decisions with
externally observed QPC clocks, and the actual native graph/scientific suite at
every assessment worker startup. Same-input native value probes must match the
accepted original parent. Values on differing search trajectories are not compared.

## Fresh matched strength assessment

If qualification passes, run six fresh operational games, then400 fresh games
in200 swapped-seat pairs, with both candidate and original parent at128 simulations,
three worlds,cpuct0.4,depth64 and the same execution route. Freeze all seeds and
check disjointness against prior registered studies before training. Analysis
bootstrap seed20261104128,50,000 paired resamples. Operational games are not pooled
with main outcomes. Preserve the accepted official/conservative analyses and all
outcome-blind replay requirements (up to96 exact reproductions).

A useful gain requires score at least55%, paired lower95%bound above50%, at least
50%score in each seat and the existing interval safeguards, under both official
and conservative scoring. No automatic model promotion. A valid negative ends
this study; do not extend the sample or launch another candidate automatically.

## Timing, resource policy and ownership

The one-second target and700ms margin are descriptive. Keep9seconds internal and
10seconds complete-response research guards; completed late moves stay in the
sample and classified hard-guard failures count under the existing loss policy.
Never reset or retry an active/failed decision or replace a game. Infrastructure
failures stop and preserve the attempt with all evidence.

Use a new retained owner, fresh monotonic/UTC origins and a72-hour total window,
with60hours for model work. All admission, fitting, playing and final checks share
these clocks. Stage ceilings: fit8h, assessment qualification2h, operational2h,
main assessment40h, reproduction8h, final saved work12h; independent reader6h,
always clipped to the global limits. These are generous ceilings, not estimates.
Available RAM reserves, stability waits and resource forecasts cannot reject or
stop the study. Keep actual5GiB free-disk floor,4GiB fitter RSS,2GiB combined
inference RSS,48GiB new evidence/8GiB final reserve, maximum two resident inference
trees and804 assessment lifecycles. Existing corpus bytes are read-only inputs,
not copied into new evidence. No scheduled monitor is created.

Use only tools/splendorust_policy_head_20261104 and
outputs/splendorust_policy_head_20261104_01, plus the separate retained owner
outputs/splendorust_policy_head_owner_20261104_01. Freeze exact code/tests/protocol,
source receipts, candidate-independent assessment choices and resource policy
before any model work. Require independent implementation reviews. All steps
advance automatically after passing checks. Completion requires independent
acceptance, actual clean owner exits and cleanup, not a completion file alone.
