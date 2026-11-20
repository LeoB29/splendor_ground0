# One bounded 512-versus-256 strength comparison

1 November 2026. The user authorized “ok lets do the 512 comparison you suggest”.
This authorizes preparation, correctness qualification, six operational games,
and one fixed 400-game paired comparison if qualification passes. It does not
authorize training, another search-budget sweep or reference promotion.

## Research question and fixed players

Does doubling search from 256 to 512 simulations improve the playing strength
of the same pretrained SplendoRust network? The accepted 256-versus-128 study
found a useful head-to-head gain. The recent neural execution optimization
provided a modest speed improvement and passed six operational games; it did
not establish strength equivalence. Both arms here therefore use the original
full network execution, exactly the implementation of verified256.

The control uses the existing, byte-identical 256 profile; the candidate uses
a separately declared 512 profile. Both keep the same checkpoint, float32,
three hidden worlds, cpuct0.4, depth64, dynamic FPU settings, native search
binary, complete-action mapping, observer-derived RNG and one-thread graph
execution. The scientific difference is the simulation count alone. Player
CPU0 and separated supervisory CPUs2–19 remain fixed. No final-token FFN
optimization, new weights, altered precision, extra information or clock-based
adaptive search is introduced.

New source: `tools/splendorust_search512_20261101`. New output:
`outputs/splendorust_search512_20261101_01`. All prior frozen sources, failed
attempts and accepted evidence remain unchanged. Original128 remains the
reference; verified256 remains the stronger research challenger until a
separate promotion decision.

## Qualification before play

Every new model worker calls `verify_runtime_sources()` once at startup.
The actual collector and evaluator use corrected two-player rules,
ruleset `splendor_base_qualified_rules_20261018_v1`, catalogue SHA256
`db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312`, and
turn contract `atomic_prereveal_completed_turn_v1`. Historical default engines
and legacy decoders are not used for fresh play. Previously disclosed search
deadlock conventions and pre-reveal auxiliary choices stay unchanged.

Every worker lifecycle first passes 11 exact original-graph parity fixtures
and the complete 67-decision actual-entrypoint suite, including all six
corrected card faces, nobles after nonpurchase, multiple eligible nobles,
noble-triggered final-round termination, payments, token returns, privacy,
legal encoding and corrected hidden-world reconstruction. This includes 15
direct native/host checks at the worker's actual quota, not merely at16.
The control must reproduce the historical original256 scientific suite after
normalizing only explicitly declared route/schema labels. No numerical
tolerance relaxation is used in this study.

After startup qualification, play six fresh operational games: three full
initial states, each played with seats swapped. Require six natural terminal
games with all checks passing. These games are excluded from the strength
estimate. Their observed compute and evidence costs produce an outcome-blind
forecast for the fixed main sample, remaining qualifications, reproductions
and final verification. Admission is about fitting the already frozen resource
envelope; neither one-second performance nor the operational score is a gate.

## Main sample and analysis

Freeze 400 fresh games: 200 independent initial-state seeds, each used for one
swapped-seat pair. The three operational and 200 main seeds occupy distinct
parts of a fresh deterministic namespace. Freeze complete initial states,
exact order, seed-registry check and statistical seed before model execution.
No outcome-dependent stopping, sample extension, replacement games or selecting
among candidates is permitted.

Score a win as1, draw as0.5 and loss as0. The primary unit of uncertainty is
the initial-state pair, not an individual game. Use the established50,000
PCG64 paired bootstrap, prospectively fixed seed20261101512, two-sided95%
interval and degenerate-interval safeguard. A useful head-to-head gain requires
all of: score at least55%, paired lower confidence bound above50%, and score
at least50% in each seat. These requirements apply both to the official full
sample and a conservative analysis assigning every administrative hard-guard
game to a candidate loss. Report full-sample administrative sensitivity,
all game counts, seats and uncertainty; do not silently drop timed-out games.

Correct moves exceeding one second are accepted. Report complete-response
latency distributions, over-one-second and over700ms counts, maxima and
research hard-guard events separately for both arms.700ms is descriptive.
These live timings use different encountered positions and are not controlled
measurements of implementation speed on identical inputs.

## Real-clock and failure policy

Retain the nine-second internal and ten-second complete-response research
guards for both players. Only an independently classified research hard-guard
failure is a game loss in the main sample. Preserve its raw evidence, end that
game, close the owned pair and qualify a fresh pair for the next scheduled
game. The failed decision is never retried. Operational hard-guard failures
stop qualification; they cannot be hidden by replacement games.

Infrastructure, legality, privacy, clock, verification, resource, unresolved
deadlock or2,000-ply safety-cap failures stop the attempt as incomplete. They
are not invented natural results or player losses. No active move clock is
reset, no partial game is replaced and no budget is extended. Preserve and
diagnose the failure promptly when observed. A technical correction must use
separate files and a separately reviewed recovery plan; this freeze contains
no automatic retry or recovery authorization.

## Verification and bounded persistence

Retain raw requests/responses with lossless compression, complete scientific
outputs, callback proofs, before/after states, legal observers, full clock and
placement receipts, and exact process identities. Validate every completed
move against the independent corrected dictionary ledger. Replay every saved
game independently to verify all transitions, terminal conditions, scores,
tie-breaks and winners. Resource accounting and full inventories remain part
of acceptance.

Reuse qualified resident workers within fixed ten-game main blocks, with a
separate six-game operational block. Close both at block boundaries and after
an administrative forfeit before the next fixed game. Each new pair gets full
startup qualification. This bounds raw streams without starting two models
for every normal game. There are normally82 live lifecycles and two fresh
replay lifecycles; the conservative prospective allowlist permits at most814
total, with at most two resident trees and one active request.

Select at most96 successful original moves without using score or timing:
the established48 main cells (variant × seat × absolute-ply phase × private
reserve category),12 operational cells (game × variant), and up to36 main
coverage cells (three pair-index thirds × variant × seat × absolute-ply phase).
Use the frozen hash ordering, deduplicate, and disclose empty cells. Two fresh
fully qualified workers reproduce the same variant's full scientific result,
action, trace and afterstate exactly. No cross-quota equality is expected.

Saved verification streams game records rather than retaining the full study
in RAM. An independent saved-only reader checks the sealed inventory, input
pins, all games, selected exact reproductions, statistics, worker request
accounting, placement, resources and owner exits. Final acceptance also
requires observing the actual outer guardian exit and exact process cleanup.
A completion file alone is not acceptance.

## Resource envelope

All caps start from one exclusive owning launch and retain both real UTC and
monotonic origins. Stage caps never extend the original clocks.

- Overall72hours; all model activity ends within60hours; saved-only finalization
  has at most12hours inside the original72hours. The independent reader has at
  most4hours inside that final stage.
- Stage caps: initial qualification4hours, operational2hours, main48hours,
  fresh-process replay4hours, final12hours, each clipped to the original
  model/global deadline. Admission reserves downstream verification resources.
- Initial available RAM4GiB stable60seconds, then2GiB between decisions;
  combined resident model/helper RSS at most2GiB. At most3hours cumulative
  RAM waiting, always charged to the original clock. Waiting occurs before
  publishing a decision, never by suspending an active decision clock.
- Total evidence32GiB, including8GiB reserved for finalization, so model-stage
  evidence is capped at24GiB. Require37GiB free disk initially and5GiB throughout.
  Forecast includes raw and compressed evidence and all worker qualifications.
- At most406games,2,000plies per game,96 reproductions and814worker lifecycles,
  with all work further limited by the clocks. Each worker has two startup
  requests. Per raw stream256MiB, request1MiB, complete response8MiB for both
  arms, prospectively enlarged to accommodate512 qualification outputs.
- No scheduled chat monitors, unbounded polling, new dataset collection,
  fitting, external-opponent testing, six-world work or alternative candidates.

Model-free regression checks and independent worker, controller and design
reviews precede the freeze. An independent review of the exact frozen inputs,
process absence and launch contract precedes the single launch.

## What follows this comparison

This is one additional search rung, not a requirement to exhaust search gains
before learning. A positive result would establish a stronger research player
and a possible stronger source of training targets; it would not prove those
targets improve a trained network. A negative or inconclusive result would
bound this particular head-to-head gain, not establish that search is globally
saturated. Either outcome informs a small corrected-rules self-play or teacher
training pilot, keeping network architecture fixed and assessing new weights
in fresh games at matched search budget. That pilot needs its own qualified
entrypoint, design and authorization. No1024 follow-up is automatic.

The project goal remains demonstrably superhuman two-player Splendor, with
one-second deployment an eventual practical goal. Beating related ancestors
is progress toward stronger play; it does not by itself establish superhuman
strength against independent strong players or expert humans.
