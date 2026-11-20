# Practical network follow-up: qualification passed, timing interrupted

31 October 2026. The final-token feed-forward candidate passed the separately
frozen numerical tolerances and corrected-rule qualification. A controller bug
stopped the experiment before its first ordinary timing request. There is no
speed result or full-panel decision-agreement result. The failed attempt remains
incomplete; independent saved-evidence verification is recorded separately.

## What ran and what passed

Both players retained 256 simulations, three worlds, the same weights, float32
precision, exploration coefficient 0.4 and native search. Only the candidate's
last feed-forward calculation was narrowed to summary token 0. This is exactly
the implementation rejected by the earlier bitwise-equality study, with no new
implementation change or retrospective reclassification of that rejection.

The new study fixed its numerical tolerances before execution: policy error at
most `1e-5 + 1e-6 * abs(reference)` and value error at most `1e-6`.

| Check, per player | Unchanged control | Candidate |
|---|---:|---:|
| Original compatibility fixtures | 11/11 exact | 11/11 within tolerance |
| Fresh numerical fixtures | 38/38 exact | 38/38 within tolerance |
| Actual corrected-rule qualification decisions | 67/67 passed | 67/67 passed |
| Ordinary timing requests | 0/72 | 0/72 |

The 38 fresh fixtures comprise 36 distinct saved batch-one inputs and two fixed
batch composites. Their games and tensors were excluded from the earlier
development panel, but they come from the same accepted 400-game source corpus.
They are not a new independent whole-game strength study.

Candidate maximum absolute differences were **3.8146973e-6 for policy scores**
and **1.7881393e-7 for values**, with zero tolerance violations. None of its 49
fixture outputs was fully bit-identical. The control was exact on all 49.
The historical batch-one eager outputs were also reproduced exactly.

An additional saved comparison found matching complete actions and callback
input paths on all 37 comparable actual-handler cases: 18 search256 cases and
19 reference16 cases. These are purposive correctness fixtures, including related
cases, not 37 independent representative positions. Search traces and numerical
outputs were not generally identical. The prescribed fresh 18-position decision
panel and its 144 counterbalanced measurements never began.

## Why the run stopped

The controller's handoff from qualification to evaluation read
`child.initialization['identity']`. The live initialization object contains only
the READY response; its process identity is stored separately in `child.identity`.
The saved initialization file adds an outer identity field, which explains why
the on-disk verifier's corresponding access is correct.

The original unit-test mock supplied the richer saved shape as the live object
and therefore missed this mismatch. The failure was a `KeyError` in the
controller, after both players qualified and before request 4. The original code
had changed one idle phase-deadline attribute before raising, but no request
clock or new decision began. Both workers then closed cleanly.

This was not a numerical rejection, player timeout, RAM refusal or exhausted
study budget. The owning run lasted about 159 seconds. Exactly six successful
startup requests ran, with zero ordinary requests, partial responses or timeouts.
All 15 exact owner/worker/helper identities are absent. All 2,625 frozen inputs
remain unchanged, and the failed output's 155 files (36,721,151 bytes) are preserved.
All three owners exited 1; both six-process worker trees exited 0.

## Prepared technical correction

The separate helper `tools/splendorust_network_practical_handoff_20261031.py`
reads the live identity, binds it to the owned process and saved initialization,
and validates and publishes the handoff receipt before changing the idle deadline.
Existing active-request, failed-request and original-clock guards remain.
The saved verifier is unchanged.

All 36 model-free regression tests passed. They reconstruct both actual live
receipts using the constructor's own saved-field definitions, reproduce the old
exception, pass the corrected receipt through the unchanged saved verifier, and
reject missing, changed or reused identities before deadline mutation. Publication
failure also leaves that deadline unchanged. Independent review approved this
held helper. An audit of all 21 related identity accesses found one faulty live
access. The helper has no launch/retry entry point and is not a complete newly
frozen experiment; no second model run has started.

## What this result establishes

The earlier bitwise failure did not prevent this same candidate from passing
explicit practical numerical bounds on fresh saved inputs. It also preserved
the tested rule behavior and selected actions on the correctness suite. This
supports continuing the practical assessment. It does not establish unchanged
future decisions, equal strength, a speedup or one-second deployment readiness.
The original exactness rejection remains valid under its original criterion.

## The next step

Recommend one separately frozen corrected attempt, integrating the tested handoff
helper into versioned controller/owner/reader entry points. Preserve this attempt
and use the same candidate, tolerances, 18 predetermined roots, order and speed
criteria. Repeat full startup qualification and then the 144 ordinary requests;
label numerical fixtures as repeated assessment, not new independent data.

Retain the three-hour overall/two-hour model allowance, two-worker limit,
resource guards and independent saved verification. Keep the nine-second internal
and ten-second complete-response research guards; one second and 700 ms remain
descriptive. Legal candidate action disagreements remain observations, not reasons
to discard a measurement. No fallback candidate, training or games are included.

The current frozen attempt explicitly permits no automatic retries and has used
its two worker lifecycles. Its clock has not been reset and its resource limits
have not been extended. This report recommends a new attempt; it does not launch
one or promote a player.

## The broader picture

128 simulations remains the reference. The verified 256-simulation challenger
was stronger in the previous 400-game comparison, but generally slower than the
one-second target. This branch asks whether a small network execution change can
reduce that cost while preserving useful decisions. We now have reassuring
numerical and correctness evidence; the missing evidence is measured speed and
agreement on the broader saved panel. External opponents, six worlds, training
and larger-data work remain deferred, and scheduled monitors remain cancelled.

## Evidence

- Original protocol: `docs/engineering/splendorust_network_practical_20261031.md`.
- Frozen attempt: `outputs/splendorust_network_practical_20261031_01`;
  freeze SHA256 `e16b9a5afbf962da976f42d64670a2048e2f5c760865c3a80e2545977e56fbf3`.
- Saved-only qualification and owner reviews:
  `outputs/splendorust_network_practical_20261031_01_closeout`.
- Prepared correction, tests, diagnosis and independent review:
  `outputs/splendorust_network_practical_20261031_01_postfailure/handoff_fix`.
- Actual retained guardian exit: session 48705, exit 1, tool chunk `c59271`.
