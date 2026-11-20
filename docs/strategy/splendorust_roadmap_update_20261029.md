# Roadmap: 128 becomes the reference; return to strength improvement

Execution update, 29 October: the user has now authorized implementation,
qualification and execution of the bounded exploration study below. Its full
[protocol](../engineering/splendorust_exploration128_20261029.md) fixes the two
alternatives at0.2 and0.8, a200-game development screen and conditional independent
400-game confirmation. Implementation and model-free checks are in progress;
the experiment has not yet been frozen or launched. The recommendation and
original documentation-only status below record the preceding decision.

29 October 2026. The user adopts the unchanged 128-simulation player as the
new reference and defers work on occasional timing overruns. This supersedes
the prospective reference choice and fresh-versus-retained timing recommendation
in the earlier roadmap and repeat report. Their evidence remains unchanged.

## Current reference and timing decision

The reference is the exact verified three-world 128-simulation player:

- Profile: `outputs/splendorust_search128_strength_20261028_01/profiles/puct128.json`.
- Profile SHA256: `048fea7e9d9377f5b3e477d4bafc96d3e96321661d2146e299a95cdc51b36edb`.
- Player identity: `strength_search_puct128_pool3_20261028_v1`.
- Same inherited network and weights; 128 total simulations, three root worlds,
  depth64, float32, batch1, exploration coefficient0.4, dynamic first-play urgency
  with reduction0.02965, and unchanged auxiliary decisions.

This is a change of research reference, not new weights, a new implementation,
or a claim that128 beats the much higher-budget player or all external players.
Preserve optimized64 as a historical control and fallback; it is no longer the
default comparator for new strength improvements.

The accepted400-game comparison scored56.125% against optimized64, with a paired
95% interval51.5–60.75%, and passed all predefined useful-gain criteria. The latest
120-decision repeat reproduced full outputs exactly and had no response above
0.9seconds. The128 median was0.725seconds and maximum0.816seconds. Neither result
identifies the original timing cause. Host conditions are plausible; they remain
a hypothesis rather than a verified diagnosis.

Park the proposed worker-lifetime diagnostic and further timing optimization.
For future research strength tests, retain completed late moves under the existing
symmetric9-second internal/10-second response guards and report timing separately.
One second remains the practical objective;700ms remains descriptive, never a gate.
Do not change any historical deadline, failure, accepted result or frozen profile.

Evidence: [accepted strength audit](../engineering/splendorust_search128_saved_audit_20261029_results.md)
and [verified timing repeat](../engineering/splendorust_search128_repeat_20261029_results.md).

## Recommended next strength experiment

Test whether a different exploration balance makes better use of the same128
simulations. Keep the network, weights, precision, worlds, seed scheme, depth,
first-play urgency and auxiliary handling fixed. Compare the incumbent
exploration coefficient (`cpuct=0.4`) with only two predetermined alternatives,
provisionally0.2 and0.8.

A lower coefficient gives less weight to the policy-guided exploration bonus
relative to estimated move value; a higher coefficient gives it more weight.
This may change how broadly the search investigates alternatives before choosing
a move. Neither direction is known to be better. The evidence supports a small,
testable calibration question, not a diagnosed defect or a large parameter sweep.
The simulation ceiling is unchanged; actual neural calls and elapsed time can
still vary with search trajectories and must be recorded.

The next design should specify:

1. Separate additive implementation and correctness qualification, reproducing
   the unchanged0.4 reference exactly and proving that alternative settings reach
   actual native search. The inherited bridge hard-codes0.4; editing profile
   metadata alone would not establish a different player.
2. One bounded development screen using paired whole games, with an advance rule
   selecting at most one challenger or stopping without a candidate. Fix samples,
   seeds and total time/memory/storage caps before launch. Saved-position agreement
   and same-network Q estimates cannot substitute for winning games.
3. A separate fresh paired confirmation against unchanged128 if a challenger is
   selected. Recommend400 games/200 swapped-seat pairs, disjoint from development.
   Preserve the established useful-gain criteria: score at least55%, paired95%
   lower bound above50%, score at least50% from each seat, and complete verification.
   Do not pool selection and confirmation games or extend a sample after results.

Corrected-rule startup guards and actual-entrypoint counterexamples remain
mandatory. Preserve failures, lawful-information checks, raw evidence and full
verification. This document recommends a study; it does not fix its complete
execution design, create seeds, implement a challenger or authorize a launch.

## Why this direction now

The32-to64 and64-to128 results show that additional search can improve the inherited
network's decisions. The next question is whether better allocation can improve
strength without another general increase in compute.

A256-simulation study remains a possible later research question, but the earlier
saved panel had a1.423-second median: a systematic extra cost rather than merely
an occasional spike. Six worlds scored50.5% in400 games at64 simulations, with no
demonstrated gain; there is no new evidence justifying that branch as the immediate
priority. A larger training programme would require corrected-data migration,
justified teacher targets and storage. Keep these directions deferred.

The wider objective remains exceptional two-player Splendor using lawful
information and practical response times. Improving against a frozen parent is
local progress. Diverse strong opponents and eventual expert-human comparisons
remain necessary for broader claims; external-candidate testing stays paused.
Later, a bounded corrected-rules learning cycle may improve the network at the
same inference cost, but this promotion does not launch collection or training.

## Execution status

Documentation and reference choice only. No experiment, model, game or training
job was launched. All frozen players and evidence remain untouched.
