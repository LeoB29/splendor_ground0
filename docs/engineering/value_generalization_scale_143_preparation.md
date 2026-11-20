# Experiment 143 preparation

2026-09-08. The owner authorized the proposed controlled value-learning test:
5,120 fresh games, 256 versus 4,096 training games, 512 shared validation games
and 512 untouched test games, with neutral and public score/resource references.
No search, policy update, arena or promotion is included. The 142 failure and
all its artifacts are preserved; all 271 frozen dependencies matched before work.

The new code reuses the existing 035 scorer bridge, replay collector, finite-game
termination rules and atomic/hash utilities. Only a new experiment module,
runner, verifier, tests and documents are added. 035's behavior distribution is
unchanged from 142; only fresh seeds are used. Saved data omit unnecessary full
legal-action tensors, while keeping a complete behavior trace and all inputs
needed to audit value fitting.

The two neural fits have identical initialization, optimizer, maximum update
budget, game-balanced minibatches and validation selection rule. The large
candidate is primary before any outcomes exist. Test games are generated only
after both heads and the simple baseline are selected and sealed. All training
diagnostics are persisted at each validation checkpoint, including a neutral
step-zero selection. Actual selected data exposure is recorded.

Fifteen focused tests passed. During unit tests, NumPy matrix multiplication
triggered the host's duplicate OpenMP runtime error after Torch initialization.
All dense linear algebra now uses Torch, with no unsafe duplicate-runtime
environment override. The full-game test also detected that the shared replay
collector returns a diagnostic endpoint schema, distinct from its complete
pre-move state snapshot. The verifier now compares the correct endpoint schema
and rejects a corrupted endpoint. Both issues were found before freezing or
collecting any protected data.

The tests cover disjoint/nested schedules, game-balanced loss and sampling,
public-feature information boundaries, selection and stopping, paired bootstrap,
invalid-result precedence, ridge fitting, preserved policy tensors, saved fit
diagnostics, sealed-test ordering, checkpoint tampering, complete 035 behavior
reconstruction, cutoff targets and preservation of saved tensors.

The final frozen manifest digest is
`72b0878756dee34e1be59151e76fc157366287b5a2ebc240d255c6a29804ed02`.
Two off-schedule probes at seeds 1,439,000,000 and 1,439,000,001 passed the full
behavior/trajectory/observation/latent/public-feature/label audit, covering 124
positions in .781 seconds. The serial collection-plus-audit projection is .555
hours and corpus projection 912,386,560 bytes; both engineering gates passed.
The full suite is running before protected collection. Preflight SHA256:
`c6153060acad8e51452f5e78d385ca550e7a289c7488feb2cd205c25d72ef25a`.

The full suite passed **986/986 in 258.03 seconds**. The single authorized
pipeline launched at 11:50:03 Europe/Paris, parent PID 36824, execution session
97865, with eight workers and Torch one thread each. No protected outcomes were
inspected before launch. The manifest/protocol/source identity is frozen.
