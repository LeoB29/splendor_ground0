# Network execution attempt: exactness rejection

31 October 2026. The one authorized implementation candidate was built, frozen
and tested. It failed the required exact-output comparison, so the owner stopped
before scientific decision qualification or speed testing of the candidate.
The failed attempt and both earlier construction-test failures are preserved.
The unchanged 128 reference and verified stronger 256 challenger are unchanged.

## What was measured and changed

The preceding accepted diagnostic placed roughly 78% of complete 256 response
time inside compiled network evaluation. This study then measured operations
inside that network on six fixed lawful saved inputs, without changing weights,
precision, search, rules, information, threads or processor placement.

All 11 original graph cases and 67 corrected-rule scientific decisions passed for
the localization worker. All 60 raw forwards (30 profiler ON / 30 OFF) reproduced
complete numerical outputs and float32 bits exactly. On 24 repeated profiled
calls, the feed-forward transformations across the six encoder layers accounted
for a median 50.09% of compiled-forward elapsed time. The final layer's two
feed-forward matrix operations accounted for a median 7.17% (451.15 microseconds).
These direct child scopes exclude nested double counting.

Profiling increased the median measured forward duration by 21.14%. The whole
profiled call, including setup and teardown, took a median 6.18 times the plain call. These are diagnostic
operator measurements, not an unprofiled speed estimate or OS thread-CPU share.

The sole candidate computed the last layer's feed-forward output only for the
summary token, token 0. The original network uses only that token after the final
layer, and no later attention mixes information across tokens. All earlier five
fused layers, full final attention, both full normalization operations, original
weights and final output heads were retained. The factory copied the frozen
execution graph; it did not invoke a model, trace or add warmup calls.

Removing 30 of 31 rows from that final feed-forward calculation offered only
limited headroom: about 5.4% ideal complete-response saving using the median
operator share and the prior 78% network share. This was a cross-diagnostic
opportunity estimate, not measured acceleration. Extra execution overhead could
consume it. A changed matrix shape also risked different floating-point rounding.

## Qualification result

The original 133 model-free protocol tests passed. The candidate added 22 passing
construction tests after two preserved model-free test failures. Independent
code review and a separate exact-supplement review approved the one attempt.

The actual control then passed all 11 graph cases and 67 corrected-rule scientific
decisions. The candidate completed all 11 prescribed graph fixtures, covering
batch sizes 1, 2 and 8, and failed exact float and float32-bit equality in every case.

| Saved output comparison | Unequal values | Maximum absolute difference |
|---|---:|---:|
| Policy outputs (raw scores) | 1,390 / 2,187 | 0.000002384185791015625 |
| Value outputs | 35 / 54 | 0.00000014901161193847656 |
| All outputs | 1,425 / 2,241 | 0.000002384185791015625 |

The candidate's eager reference outputs exactly match unchanged localization,
control, the preceding accepted efficiency study and the original 256 reference.
Independent saved-only recomputation checked all 55 comparison rows across those
five records. The discrepancy is in the candidate execution outputs, not a
changed reference or missing fixture.

No candidate scientific decisions, paired timing measurements or games followed.
The 144 planned unprofiled decisions were correctly not started. No failed request
was retried, no second implementation was selected and no tolerance was relaxed.
The total performed coverage was 33 graph cases, 134 scientific qualification
decisions, 60 localization forwards and zero ordinary timing decisions.

## What this result establishes

The measured large cost lies in the network's matrix transformations. A specific
mathematically motivated reduction was feasible to implement, but did not retain
bit-for-bit output equivalence in this installed runtime. It therefore cannot be
accepted as an unchanged-player acceleration under the frozen requirement.

The tiny discrepancies are consistent with changed floating-point execution;
the saved outputs alone do not identify the exact internal operation causing
them. They do not establish worse play, different final search decisions or a
speedup. Policy raw-score maxima are not winning-probability errors. The study
also does not prove that every possible exact optimization is impossible.

## The next step

Close this bounded engineering branch and park this candidate. Do not start a
second implementation hunt or change the current attempt's exactness rule.
Recommend preparing one fixed 192-simulation versus unchanged 128 strength study,
with fresh paired whole games, the established useful-gain criteria and timing
reported separately. 192 is a predetermined intermediate search budget, not a
quota selected by a timing sweep. Its strength and one-second practicality are
unknown. No new study is launched by this recommendation.

The reason is practical: 256 already has a verified strength gain, but even an
ideal saving near 5% would not generally bring the recent 1.20-second systematic
mean below one second. An intermediate budget can directly test whether part of
the strength gain is available at a more useful cost. Keep the same checkpoint,
three worlds and exploration coefficient 0.4, and accept completed late moves
under the established research guards while reporting their frequency.

## The broader picture

The project seeks strong two-player Splendor from lawful information with a
practical response time near one second. Extra search has delivered verified
progress: 256 scored 61.0% against 128 in 400 games, but its additional work usually
takes more than one second. 128 remains the research reference; 256 remains the
verified stronger research challenger.

The last two diagnostics first located most of that cost in the network, then
tested one concrete way to reduce it without changing the player. This attempt
has ruled out that exact implementation route under its strict requirement.
The useful next question is which search budget gives a worthwhile strength gain
at an acceptable measured cost, rather than extending a timing-admission campaign.
External testing, six-world work, training and larger-data work remain deferred.
Scheduled chat monitors remain cancelled.

## Evidence and closeout

- Protocol: `docs/engineering/splendorust_network_execution_20261031.md`.
- Preserved attempt: `outputs/splendorust_network_execution_20261031_01`.
- Phase 1 freeze: `ce9b995aafe8794f17317fe8cbdb57eef3f1a8fbf01bc504cc57ce24a0125b4f`.
- Candidate supplement: `218fddf2f51c56aa2f888834c9a29b70aadd180c8e2de34fef93edaa7c305cf7`.
- Operator result: `localization_result.json`; passed saved localization:
  `localization_verification.json`; selection calculations:
  `candidate_preparation/decision.json`.
- Candidate values: `workers/candidate/graph_parity.json`.
- Separate saved-only closeout:
  `outputs/splendorust_network_execution_20261031_01_closeout`.

The original controller recorded `incomplete` and actual exit 1 after the exact
parity gate failed. That failure remains unchanged. The retained owner session
also returned actual exit 1, observed directly; this is not inferred from a
completion file. The separate closeout verifies an early candidate rejection,
not a completed or accepted speed experiment. Both independent saved-only reviews
passed. All 2,409 original frozen pins and supplemental inputs are intact. All
18 worker/helper processes have recorded exit 0; all three owning processes have
recorded exit 1 after the numerical gate rejection. All 21 exact process identities
are absent, with no cleanup errors. Six parent transport requests were delivered
successfully; no timeout occurred. The owners stopped about 23 minutes 46 seconds
after launch, well within the original cap. No model work ran during closeout.

Final receipts are `numerical_rejection_review.json`,
`owner_rejection_review.json`, `actual_guardian_exit.json` and
`final_closeout.json` in the separate closeout directory. One model-free audit
argument error is also preserved there; the corrected saved-only check passed.
