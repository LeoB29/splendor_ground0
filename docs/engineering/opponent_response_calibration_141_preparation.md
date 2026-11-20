# Experiment 141 preparation

**Completed:** the authorized run finished all 120 states and 190,076 natural
continuations in 43m24s, ending 2026-09-08 01:54:36 Europe/Paris. Parent 59860,
session 7833 and all ten workers exited successfully. Independent verification
passed. Do not restart. See `opponent_response_calibration_141_result.md` for the
precise bounded primary effect and uncertain actual-policy absolute gain. Earlier
preparation/launch entries below are preserved as history.

2026-09-08. Owner authorized the calibration proposed after 140. The frozen
protocol is `docs/engineering/opponent_response_calibration_141.md`.

Implementation uses a new module and wrapper; no existing source is edited.
At preparation start, all 267 dependency files frozen by 140 were byte-identical.
Existing tracked and untracked work is intentionally preserved.

The source has 138 eligible games after exclusion of the 138--140 panels and
all their timing probes: 23 per phase/seat cell. Identifier-only selection fixes
120 games (20 per cell), plus six separate timing probes. The budget is 16
three-world blocks, four stochastic replicas, with at most 230,400 continuations.
Only weak 035 runs. Sampling, reply-set handling and grouped resolution reuse the
unchanged 138/139 components. A shared tracking adapter checks natural completion,
absolute turn count, repetition and no-progress compatibility. It records scores,
card counts, seeds, action-sequence digests and the policy arm's immediate reply.

The initial 57 focused tests passed; two subsequent terminal/policy-law checks
bring the focused total to 59. The 15 new tests passed independently. These
include comparison to 139 under matched worlds/seeds, information/legal-world
checks, seed separation, unchanged coincident-root observations, group-before-min
arithmetic, corrupted endpoint rejection, terminal roots, and same-seed agreement
with 140's stochastic policy. The full suite passed **954/954 in 249.12 seconds**
before preflight. Logs: `outputs/opponent_response_calibration_141/tests.log`.

The frozen panel contains 78 differing move pairs and 42 coincident pairs;
198 unique roots give a deduplicated maximum of 190,080 continuations before
terminal/reply structure is accounted for. The off-panel probe sample indices
are 173, 37, 85, 194, 153 and 178. Selection did not use these agreement counts.

No protected outcome has been generated or inspected. The outcome-blind
structural/timing preflight is running, followed by authorized fixed-budget launch.
Timing changes do not authorize sample or threshold changes after outcomes.
The independent verifier is written before launch and will reconstruct all 18
reported mean/CI numbers, all per-block and per-state values, classifications and
variance estimates. This is endpoint reconstruction, not full trajectory replay.

Runtime probes and launch identities will be recorded in the experiment output
directory; champions remain 084 and 035. No automatic follow-on experiment.

## Preflight passed and authorized launch

All 1,920 protected block structures passed in a preflight taking 63.62 seconds,
including six off-panel probes. All probe paths completed naturally without
fallbacks, adjudications or hard caps; all six policy-arm action sequences and
endpoints matched 140 from the same seed. The late-seat0 repeated block was exact.
Probe payoffs/winners were discarded; no protected outcomes were inspected.

Expected work is **152,060 sampled-min paths plus 38,016 policy-reply paths**,
total **190,076**. One branch has three instead of four sampled replies, reducing
work by four continuations; it is retained. Conservative serial wall projection
is 22,859.30 seconds (6.35 aggregate hours), below the 24-hour prelaunch ceiling.
Runtime is now telemetry. The fixed 120-state run was launched with ten workers.

- Protocol SHA256: `2832213514fe82b1c30b4a9b814e7c51ec0a25db19ea18d77e3b7576c7874033`.
- Preflight SHA256: `d7ad6c8fe089202ed55c0b7d85da622eee16cee9fdfd03ef87b4e3ae8abda161`.
- Manifest canonical digest: `c227063d255909e9fc6819cbb274362770573064e8315561d7c591e86d353fce`.
- Manifest file SHA256: `c641fb3026b2886602750c277e55faed6cd2ca134a22ea5be3c2007ecce204cf`.

Running execution session: 7833. Continue the existing run; do not start a duplicate.
Inspect only process health, counts and integrity until all states complete.

During execution, before any protected outcome inspection, a reporting-label
overlap was documented in `opponent_response_calibration_141_interpretation_note.md`.
It preserves the frozen classifier and all calculations; the final report must
interpret the actual CI directly if a nonzero difference is also bounded below
the material threshold. No code/protocol hash or execution decision changed.
