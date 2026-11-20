# Saved-decision diagnostic: corrected checker, attempt 02

27 October 2026. The separate v2 checker completed the unchanged bounded
diagnostic. Both checker defects were repaired without changing the players,
weights, native search, precision, positions, seeds or limits. Attempt 01 remains
incomplete and preserved; it has not been retrospectively accepted.

## Execution and verification

The new attempt ran from 11:10:21 to 11:20:52 UTC (12:10:21 to 12:20:52 Paris),
about **10 minutes 31 seconds**, within its unchanged two-hour cap. Qualification,
collection and finalization used about 150.6, 470.5 and 8.9 seconds respectively.
The original one-second frozen-player allowance and separate 30-second research
allowance were enforced. No new game or training was run.

- **152 model-free tests passed**, including regressions using the authentic
  rejected response and its saved evidence. Earlier test-harness failures remain
  in the new preparation evidence.
- **123 qualification decisions passed:** 52 actual-entrypoint counterexamples
  on each route and 19 parity decisions. The corrected catalogue, noble, lawful
  information and negative-request cases ran through the actual worker.
- **156/156 diagnostic requests passed:** 24 exact historical reproductions,
  30 fixed pilot requests, 90 remaining main-panel requests and 12 exact repeats.
- All **24 positions** have all five research conditions: 64-A, 256-A, 1024-A,
  1024-B and 1024-C. All six phase/seat cells retain four positions.
- **44 graph cases**, four closed workers and **181 parent requests** reconcile
  with the planned workload. The 104 entrypoint decisions are carried by two
  suite requests; therefore parent requests and scientific decisions differ.
- No timeout, rejected decision, mismatch, retry or replacement position was
  recorded. All three actual owning process exits were zero; cleanup reported
  no errors. Internal disposition is `completed_verified`.
- Peak sampled combined worker/helper RSS was 326,701,056 bytes (311.6 MiB).
  The final seal binds 1,320 artifacts totaling 152,156,684 bytes (145.1 MiB).
  These are distinct from conservative logical evidence accounting used by the
  resource guard. The pilot passed every time and evidence gate.

The execution freeze binds 1,930 inputs, including the prior attempt's 1,776
frozen inputs and all 145 prior attempt artifacts. Independent postrun review
**accepted the complete attempt**, rechecked the finite inventories, recomputed
saved science/parity/analysis and confirmed that all 27 recorded owner/worker
process identities were absent. The new attempt's `postrun` directory records
that review separately. A request-file glob assumption in the review reader was
corrected and preserved as a reader-only note; it changed no experimental file
and required no new model execution.

## What changed in the checker

**Value semantics.** Upstream `SearchTargets.network_value` is the mutable
post-search root-node mean under dynamic first-play urgency, not the initial
network prediction. V2 keeps that raw field unchanged and labels it correctly.
It derives the initial win credit from the first network callback and checks the
post-search mean against `(initial + N * root_visit_weighted_mean) / (N + 1)`.
The declared arithmetic bound accounts only for float32 reporting and binary64
arithmetic. Exact outputs, decisions and deterministic repetitions remain exact.

**Rejected-response preservation.** The unchanged receiver legitimately keeps
a decoded rejected response with `accepted=false`. V2 preserves that response,
its raw-packet/hash binding and original timing, records the first stop before
secondary checks, and verifies the rejected prefix without accepting it as a
successful decision. The regression deliberately feeds the saved rejected packet
through this failure path and requires an incomplete, non-retried disposition.
Corruption tests still reject altered packets, values and inconsistent receipts.

These are Python checker repairs. The native build was reused; the four research
profile files are byte-identical to attempt 01. Optimized64 and unchanged32 remain
frozen. External-candidate testing is paused and larger-data work is deferred.

## Descriptive decision results

| Comparison | Changed main and complete actions | Positions |
|---|---:|---:|
| 64-A to 256-A | 4 | 24 |
| 256-A to 1024-A | 7 | 24 |
| 64-A to 1024-A | 9 | 24 |

At 1,024 simulations, all three seed families chose the same main action on
**16/24 positions**. At least one differed on **8/24**; B and C differed on
**6/24**. These seeds jointly vary sampled worlds and search randomness, so this
does not isolate either source or show that three sampled worlds are inadequate.

For each of the nine changed 64-A versus 1024-A action pairs, both actions were
visited in both held-out seed families. Their internal Q estimates agreed on the
new alternative in **4 cases**, agreed on the original action in **2**, and had
mixed preferences in **3**. No missing Q was treated as zero. Four different
actions also had unanimous 1,024-seed action agreement; five did not.
Some eligible edges had only one to three visits in a reference search. Visited
support therefore permits a comparison but does not make its estimate precise.

All these assessments use the same inherited network. They describe search
stability and internal preferences, not proven mistakes, factual regret, actual
win probabilities or independent evidence of improved play.

| Research condition | Median complete response | Maximum | Positions |
|---|---:|---:|---:|
| 64-A | 0.389 s | 0.558 s | 24 |
| 256-A | 1.423 s | 1.823 s | 24 |
| 1024-A | 5.404 s | 7.903 s | 24 |
| 1024-B | 5.414 s | 6.721 s | 24 |
| 1024-C | 5.423 s | 7.086 s | 24 |

These are observed research-route timings with its diagnostics. They do not
qualify a new deployment player. The separate 24 frozen-route historical
reproductions had a maximum response of 0.616 seconds. Median callback counts
were 65, 257 and 1,025 at the corresponding budgets; some later positions reused
terminal values and needed far fewer network calls. This describes work done,
not a new attribution of elapsed time among inference, supervision and I/O.

Both 256-A and 1024-A exceeded one second on 20/24 positions. The same four
positions completed below one second at both budgets; all four kept the original
action and agreed across the three deeper seed families. On this panel, useful
stable alternatives were not concentrated in these cheaply extended decisions.
That weakens the immediate case for simply giving extra search to cheap moves.

For transparency, the following is a postrun breakdown of the preselected cells;
it is descriptive development evidence, not a subgroup selection rule.

| Absolute ply / seat | 64-A to 1024-A changes | All three 1024 seeds agree | B/C differ |
|---|---:|---:|---:|
| 0-19 / 0 | 2/4 | 3/4 | 0/4 |
| 0-19 / 1 | 1/4 | 2/4 | 1/4 |
| 20-39 / 0 | 2/4 | 3/4 | 1/4 |
| 20-39 / 1 | 0/4 | 2/4 | 2/4 |
| 40+ / 0 | 2/4 | 3/4 | 1/4 |
| 40+ / 1 | 2/4 | 3/4 | 1/4 |

## What this result establishes

The repaired evaluator now passes actual-model qualification and complete
verification on the frozen panel. Extra search sometimes changes the decision,
but its preferred replacement is not uniformly stable across random samples.
Some stable alternatives are worth investigating, while the mixed cases make
simply treating a deeper search as a teacher unjustified. The measured deeper
conditions also exceed the one-second practical target on typical positions.

The panel is small, historical and deliberately balanced by phase and seat. Its
fractions are not estimates for every future game. This run neither identifies
a causal bottleneck in the network nor establishes a strength gain from a new
player. No player has been promoted.

## The next step

Recommend **one bounded study of sampling stability under a fixed compute
budget**: test whether changing how the same affordable search budget is spread
across sampled worlds/search repetitions can make choices more robust. Its
design should separate world-sampling randomness from search randomness where
the implementation permits, retain corrected rules and lawful information, and
measure the complete response against one second. The present run varied those
sources jointly and cannot choose a new world count or repetition count for us.

This is a follow-up hypothesis, not a new launch or proof that stability implies
strength. Any resulting candidate must retain the frozen optimized64 control and
earn acceptance in a fresh paired whole-game comparison at one second. These
diagnostic positions are development evidence, not that independent confirmation.
Do not launch a broad parameter sweep, increase simulations globally, enlarge the
network, collect a larger training set or resume external screens on this result.

## The broader picture

We moved to the verified pretrained SplendoRust policy/value network because it
provided a much stronger starting player. Optimized64 then improved on the
32-simulation parent, and the separated-supervisor work established practical
reliability. The two screened external policies were too weak to test whether
that gain generalizes, so external adapter work is paused.

This diagnostic directs the next attempt at improving strength per second. It
provides evidence about where search changes its mind and where those changes
remain sensitive to randomness, before committing to expensive training or
another large experiment. The wider goal remains exceptional, independently
demonstrated two-player Splendor strength with lawful information and practical
response times. Neither a local parent win nor this saved-position study alone
establishes that wider claim.

## Evidence

- Source: `tools/splendorust_decision_quality_20261027_v2`.
- Execution: `outputs/splendorust_decision_quality_20261027_02`.
- Freeze SHA256: `2382d92fb7bdfea2835abfc2a3e8dcd0da769f4940921de8032e766bbebab65c`.
- Analysis SHA256: `1722d7f8585475e67594699945f3222cf7d047b24e078dd27a9f71cc50f680e8`.
- Completion SHA256: `e4dec93aa164d8f9fc7b8cc6a47b49963edcd0a9d7dbd05c80a08015f8af9828`.
- Verification SHA256: `c6083e0af71bb4b8b8fd5c564d568ab75c9fb4ce10e10571f744d080c3f2e7c2`.
- Evidence seal SHA256: `90ff708845468422b4e66d15906ee0d95c263cfe0c878549d58900373ba5b695`.
- Actual outer exit SHA256: `9a38c69affafd8f0f95d19b71b4b3e6a161ef73aea742752865187e5dbba4e3c`.
- Independent acceptance: `outputs/splendorust_decision_quality_20261027_02/postrun/independent_completion_review.json`,
  SHA256 `92a84da454acc395e653268548f653b56a5285faa6fa082875120a360d18a7e2`.
- Preserved prior failure: [attempt 01 report](splendorust_decision_quality_20261027_attempt01_results.md).
