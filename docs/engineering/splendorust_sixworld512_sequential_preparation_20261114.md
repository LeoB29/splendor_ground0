# Six versus three worlds at 512 simulations: prospective comparison

14 November 2026. Preparation only: no execution freeze, new games, models,
training, production promotion or scheduled monitor. The user supports a larger
test with planned interim checks and separate teacher/player decisions. The user
explicitly selected a two-percentage-point noninferiority margin for teacher use.
This document proposes the remaining concrete design; numerical boundaries and
the execution implementation must be qualified before launch.

## Question and unchanged comparison

Compare the original EF8A network with six versus three root worlds, both using
512 TOTAL simulations, corrected two-player rules and the same other search
settings. Use the original shared-stream mode. Split-stream experiments remain
diagnostic only. Use fresh independent setups, each played twice with swapped
seats. Draws count half a point; the complete pair is the statistical unit.
No old games or operational qualification games enter this sample.

The previous diagnostic found 23.7% lower visit-distribution variability at512,
but did not establish better play or more accurate training targets. The old
64-simulation six-world comparison was inconclusive. Evidence here concerns the
512-simulation setting and does not promote six worlds at all search budgets.

## Size and interim schedule

Recommend at most **3,600 games / 1,800 complete pairs**, with analyses after
400, 800, 1,200, 1,600, 2,000, 2,400, 2,800, 3,200 and 3,600 games. Only verified
complete pairs enter a decision. No result-dependent sample extension.

Target about80% power for a true score of52.4445% (about+17 logistic Elo). Preserve
the previous confidence standard: one-sided2.5% false-positive allowance for the
lower bound, corresponding to the lower endpoint of a two-sided95% interval.
Use an O'Brien-Fleming-type group-sequential boundary over pair scores, with
confidence limits adjusted for all planned looks. Ordinary repeated95% intervals
are not the decision procedure. Qualify the finite paired-score implementation,
draw handling, null error, power and stopping logic before freezing it.

Model-free Gaussian planning with the November1 pair variance0.1151005 and a
second variance0.125 (independent decisive-game equivalent) gave approximately
82–85% power at this maximum before futility, and82–85% rounded after the proposed
futility checks. These are planning assumptions, not a guaranteed property of
the new matchup. A3,200-game maximum was approximately77–81% across those same
assumptions. The user's2,700-game estimate is reasonable at one-sided5% error;
preserving the prior lower-bound confidence standard requires more games.

## Prospective outcome rules

Let L and U be the qualified sequentially adjusted lower and upper score bounds.

| Result | Required evidence | Consequence after full verification |
| --- | --- | --- |
| Superior | L > 50% | Adopt six worlds for the tested512 player setting and provisionally for the next teacher cycle. No55% observed-score floor. |
| No material regression | L > 48%, superiority not established | Eligible as the provisional teacher for the next bounded learning cycle; retain the three-world playing reference. |
| Materially worse | U < 48% | Reject six worlds for this change. |
| Unresolved | Neither adoption criterion established | Retain three worlds; do not call this equivalence or noninferiority. |

The two-point teacher margin is the user's explicit tolerance: it rules out a
loss exceeding about14 Elo, not every possible loss. Even if true strength is
exactly equal, planning suggests only about64–68% probability of certifying this
margin by the proposed maximum. An unresolved teacher decision is possible.

The teacher permission is provisional for the next bounded cycle, not proof
that every future corpus or learner will improve. Retain the original network
and three-world reference, and assess the trained learner normally. No separate
standalone teacher-validation project is required before that bounded cycle.

These are new prospective scientific rules. Historical55% gates and results
remain unchanged. Report seat-specific scores and costs as diagnostics; do not
silently retain an additional each-seat-score-above50% promotion requirement
that would confound small strength gains with first-seat advantage. Integrity,
corrected-rule, exact reproduction, source and process-cleanup checks remain.

## Early stopping

Stop for superiority when its adjusted boundary is crossed. Beginning at800
games, the proposed futility rule uses a10% conditional-power threshold. Stop
only when the prospect of establishing superiority under the+17 Elo planning
alternative is below10%, AND either teacher noninferiority has already been
established or its prospect under a true50% score is also below10%.

Define the conditional-power calculations, harm boundary and tie ordering in
the implementation before freezing. Qualify the whole rule against paired-score
distributions. Superiority futility alone must not prematurely terminate an
otherwise useful teacher assessment. A futile stop without a passed adoption
criterion is unresolved, never evidence of equality or absence of small gains.
Any early scientific stop is followed by the full required evidence audit.

Planning simulations suggest about2,600–2,660 games on average at+17 Elo and
3,050–3,070 at equality under these assumptions. Do not promise an early stop
when the result is flat: the teacher question still requires information.

## Time and execution preparation

November1 recorded55.3 decisions/game and2.371 seconds per512 decision. At that
speed,3,600 games cost approximately131 serial search-hours. At the recent
diagnostic's roughly2.9 seconds, the estimate is about160 search-hours, plus
qualification, recording, replay and final verification. These are estimates,
not RAM-reserve or forecast-based refusal criteria. Qualified parallel execution
may reduce wall time, but its throughput and unchanged behavior are not assumed.

Six worlds was close in observed cost, not literally free: legacy512 means were
2.855 seconds for three worlds and2.902 for six (+1.65%). Report actual thinking
time, including one-second and700ms exceedances, separately from strength.

Before launch: qualify the live adapter and sequential checker in separate files,
freeze the fresh schedule and all boundaries, set actual resource/time bounds
large enough for the full planned study and verification, and retain durable
dispatch/checkpoint ownership. No active decision-clock reset or silent retry;
preserve every failure. This preparation does not launch the comparison.

## Methodological references

- [rpact authors: group-sequential boundaries and alpha spending](https://www.rpact.org/vignettes/planning/rpact_boundary_examples/)
- [gsDesign2 authors: deriving spending boundaries](https://merck.github.io/gsDesign2/articles/usage-gs-spending-bound.html)

Historical planning evidence: outputs/splendorust_search512_20261101_01/
main_analysis.json and timing_analysis.json. Repeatability evidence: the accepted
amended teacher-variability recovery and its14November closeout. Neither supplies
fresh outcome evidence for the proposed live comparison.
