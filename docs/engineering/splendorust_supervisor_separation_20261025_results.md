# Supervisor CPU separation: verified stopped qualification

Report prepared 26 October 2026. The bounded run completed on 25 October at
23:57:56 Paris. Supervisor separation is implemented separately, but this
attempt did not qualify it: the original-placement control timed out, activating
the frozen stop rule before unchanged32 timing or fresh operational games.

## What changed

New version: `tools/splendorust_supervisor_separation_20261025`.
Both players and their helpers remain on logical processor 0. Isolated
supervision, including the owning watchdog and reader/writer threads, uses
processors 2–19, excluding both logical processors of the player's physical core.
The original comparison condition permits supervisory processors 0–19.

Weights, float32 precision, numerical thread count, graph64/eager32 backends,
64/32 simulations, three worlds, search, corrected rules, minimal observer,
normal priority and original 1.000s/.900s clocks remain unchanged. Explicit host
topology admission prevents the supervisor's narrower mask from reselecting the
player's core. Original request/receipt/reader/sampler/cleanup methods remain
inherited. Process and thread placement proofs are bound to owned identities,
requests and watchdog acknowledgments. This does not reserve a core against
other applications.

## Frozen plan and outcome

Protocol: `splendorust_supervisor_separation_20261025.md` in this directory.
Evidence: `outputs/splendorust_supervisor_separation_20261025_01`.
Freeze SHA256:
`b6e9073a83d2bf00e4d1ac16d249e9bc75fc0219730f7d4781fb264d80fbf855`.
The freeze binds 4,227 inputs. All 229 model-free tests passed, including real
Windows placement fixtures; independent implementation review passed.

The actual-model correctness suites all passed: 52 decisions for each of the two
players under each placement, 208 total. The planned timing workload was 24 saved
positions × 2 players × 2 placements × 2 passes = 192 requests. The run stopped
after 74 requests, preserving 73 successful decisions and one control timeout.

| Timing condition | Attempted | Completed | Slowest successful arrival |
|---|---:|---:|---:|
| Optimized64, separated supervision | 48 | 48 | 624.031 ms |
| Optimized64, original supervision | 26 | 25 | 596.241 ms |
| Unchanged32, either placement | 0 | 0 | Not reached |

Every completed separated64 decision was below the frozen 700 ms engineering
margin. The original-placement timeout occurred at schedule index 73 (the 74th
request), block 3, second pass, saved position `main_g72_p0`. This is an early
control position, not the previous live failure at `main_g72_p28`.

For the new failing position, the first original pass completed in 491.254 ms;
the two isolated passes completed in 429.870 ms and 452.855 ms. The second original
pass timed out. These are descriptive within-position observations, not proof
of a scheduling cause.

## Failure evidence and interpretation

The trusted receipt records `timeout`, no accepted result, and no accepted
arrival time. The retained raw error packet arrived at 1.0235925s. Its child
sidecar reports the unchanged internal deadline guard, with 937.1159 ms inside
the handler. Parent start to child input read was 51.8167 ms; parsing to handler
start was 1.8746 ms. Sampled request CPU was 421.875 ms, with the usual different
scope and sampling limitations: subtracting it from handler time would not
establish an exact scheduling wait. Partial simulation/callback progress is not
available and must not be invented.

The 1.0429946s receipt elapsed field includes the parent's timeout handling; it
is not a successful response time. There was no valid action. This is a timing
qualification failure, not an invented game loss: no games were played.

The frozen plan required every control request to complete, so this failure
correctly stopped the comparison and prevented all six conditional games.
No retry, replacement or continuation occurred. The incomplete control panel
cannot establish the planned mean/p95 nonregression gates. Nor can isolated64's
48 successes qualify unchanged32 or prove that rare stalls have been eliminated.
The observation is encouraging for isolated64, while full qualification remains
unachieved. Do not classify the control timeout as an isolated-placement failure.

## Verification and resources

Final saved-data verification passed for the entire collected prefix:

- 73 completed scientific decisions, 24 observed state/quota cells, one complete
  four-repetition cell and 37 matching-quota historical comparisons. The planned
  48 complete cells and 92 historical comparisons were not claimed.
- All four actual entrypoint suites; 8 workers including 6 graph workers; exact
  graph/weight proofs; original raw transport, compressed logs and cleanup.
- Per-request placement proofs, complete profile/placement inventories and
  six watchdog/runner placement exchanges.
- 613 sealed artifacts with readback, frozen sources/runtime and prior accepted
  evidence. No failed decision was replayed; no operational reproduction was
  required because no operational game was admitted.

The owning watchdog exited 0 with no cleanup errors. Recorded watchdog 38004 and
runner 39908 were absent after completion. Charged duration 254.094s (4m14s),
inside the one-hour cap; final verification took place within the original
window. Maximum sampled combined worker/helper RSS was 326,475,776 bytes. Evidence
was approximately 40.7 MB before the separate post-run review/report receipts.
Completion status is `timing_failure_verified`: a verified stopped qualification,
not a qualified separated player.

A separate independent saved-data review passed after completion. Its receipt,
`independent_results_review.json` in the evidence directory, verifies all 613
sealed artifacts and the exact directory inventory, rehashes 647 consumed files,
and independently reconciles the timing prefix and reported outcome. Its SHA256
is `ab9a118a9d136a170a0ce33cb0c98090fff7e486aa1c6ee5ef4a28dc23e718ca`.
It discloses two corrected review-reader assumptions about wrapper structure and
canonical versus raw request hashes. These were read-only audit issues, with no
change to experimental evidence, results or clocks. No unresolved evidence
concern remains; the review does not convert the stopped trial into a pass.

## Recommended next step

Prepare a **new**, bounded qualification of separated supervision for both
players. Its primary question should be whether that configuration meets the
existing absolute correctness, one-second deadline and 700 ms engineering margin
on a fixed panel. Keep original-placement failures as explicit comparison
outcomes, rather than allowing a failure of the control configuration to cancel
all remaining tests of the proposed configuration.

This changes the prospective qualification question and must be frozen openly
before another run. Specify how control failures affect relative timing claims;
do not silently drop them, fabricate completion times, or claim that this
attempt's original mean/p95 gates passed. Predeclare any post-failure worker
lifecycle, without retrying or replacing a failed request. Admit six fresh
operational games only after both separated players qualify. No new run is
authorized or launched by this recommendation.

## Broader picture

The project now uses the verified SplendoRust network/search system as a strong
starting point. Optimized64 seeks more search within the same one-second move
budget while preserving the network and decisions; unchanged32 remains the
reference. Recent failures have made runtime reliability the immediate obstacle
to a trustworthy strength comparison. CPU separation tests whether supervisory
work can be moved away from the player's core without changing how it plays.
This result narrows the next qualification task but does not yet establish a
speedup, a causal cure or improved playing strength. The previous failed attempt
and all losses remain intact; larger-data and training work remain deferred.
