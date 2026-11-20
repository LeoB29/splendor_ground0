# Experiment 142 preparation

2026-09-08. The owner authorized the recommended strong-policy-initialized,
bounded learning pilot. Implementation is separate from the original 090 model
and does not modify any existing source file, checkpoint, or result. All 268
dependencies frozen by experiment 141 still matched during preparation.

The concrete bridge copies 035's existing legal-action attention model. Its
checkpoint records value loss weight zero, so a copied value head is not assumed
to be useful. Bootstrap only that head from completed games; preserve the entire
policy path bit-for-bit. Search scores canonical successors using maximum logits
over the complete legal-action context, and the student learns the same canonical
objective. End-to-end training then releases every model parameter.

Initial outcome-blind probes used seed 1,429,000,000 and positions at turns
0, 20, and 40. Full concrete score vectors matched the installed 035 scorer
exactly; legal counts were 30, 13, and 15, with logit ranges 5.67, 8.65, and 6.32.
Thirty-two simulations took .150, .097, and .095 seconds respectively on one
idle CPU core. No learning or arena position was inspected. These probes justify
the fixed local workload, not a playing-strength conclusion.

The primary test compares successor search with parent search at the same
0.4-second decision allowance. Exact retention verifies the parent's raw policy
competence; it does not independently establish that adding search to the
bootstrapped value model improves over raw 035. No comparison with champion 084
is included. A positive result would therefore demonstrate one-cycle improvement
within this system, requiring external-strength checks and fresh replication
before any promotion or superhuman claim.

The fixed stages are 256 bootstrap games, 128 self-play games at 128 simulations,
one four-epoch end-to-end update, and 400 seat-paired primary games. The fixed
sample is intended to detect sizeable gains; uncertainty around smaller gains
is an allowed result. The training and evaluation schedules are disjoint. There
is no adaptive extension, checkpoint selection on arena results, or automatic
second generation.

Sixteen focused tests passed before the full-suite run. They check exact 035
retention across 50 successive decisions; preservation of full attention context;
hidden-deck invariance; canonical gradients; policy-preserving value fitting;
timed/fixed search agreement at the same completed simulation count; deadlines;
whole-game data splits; pair-level bootstrap; invalid-result gates; cutoff label
handling; and finite end-to-end policy/value gradients. The independent verifier
will additionally reconstruct every collected game and learning tensor.

The full suite passed 970/970 in 283.76 seconds. An off-schedule full-game probe
then exposed integer-key/string-key inequality after JSON round-trip in the
verifier's final-state comparison. Every individual move/state hash had matched.
Comparison now uses canonical serialized state equality; a seventeenth focused
test checks both a valid round-trip and rejection of a changed final state. All
17 focused tests passed. The initial manifest, probe and verifier are preserved
in `outputs/strong_policy_pilot_142_preflight_attempt1`; no protected collection
or training had occurred. The final source/test/verifier identity was re-frozen.

Final preflight passed with manifest digest
`71b52489ee0dd15d412738819bd67dd81386a20f4ce52d7c6c8107f1ce7fda17`.
128-simulation probes took .560, .353, and .637 seconds; timed .4-second probes
finished in .4044, .4004, and .4025 seconds, completing 88, 107, and 73 simulations.
Two complete off-schedule probe trajectories and all saved feature tensors and
targets were reconstructed. The single frozen pipeline launched around 10:47
Europe/Paris on 2026-09-08, session 83234, eight workers and Torch one thread each.
