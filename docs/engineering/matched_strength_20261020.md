# Matched external strength screen — prospective freeze

Prepared 20 October 2026. This is a fixed **60-game** study of the delivered
four-candidate local planner against qualified SplendoRust, with **120 seconds
per move** and genuine timeout losses included. Preparation does not launch it.
The owner requested preparation and freezing, then asked for the recommended
sample size. We selected 30 fresh seed pairs, ten per fixed model, as a coarse
screen proportionate to the measured cost and remaining deadline risk. The
earlier conditional 402-game roadmap remains a separate possible investment.
There is no result-dependent sample extension or substitution.

## Entrants and question

The estimand is the local player's mean game score against this exact opponent
on these corrected two-player base rules, at the declared clock and hardware,
averaged equally over the three fixed fits. It includes delivery reliability.
It does not estimate human rank, universal strength, one-second deployment
feasibility, or the strength of the eight-candidate planner.

- Local numerical profile: `four_candidate128_corrected_live_operational_20261019_v1`.
  Four proposals, 128 shared worlds, frozen raw035 continuations, serial arithmetic,
  original ordering and ties; fits 161001, 161002 and 161003. Checkpoints and all
  transitive qualified inputs are bound in the freeze. The new versioned adapter
  changes schedule/context admission and evidence destinations only. Numerical
  functions are compared structurally to the accepted adapter, and a runtime
  transfer/privacy/full-replay gate remains mandatory.
- Opponent: `entity_onehot_puct6400_canonical_atomic_prereveal_cpu_120s_v1`.
  Qualified canonical atomic pre-reveal adapter, 6,400 search iterations,
  108-second internal budget and 120-second external allowance. Its selected
  profile SHA256 is `6f74c3b6d77707c42619801d1b1bef3fbf203d1ee6ff548afbe935169c257599`.
- Rules: `splendor_base_qualified_rules_20261018_v1`; catalogue SHA256
  `db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312`.
  Retain the corrected card costs, end-of-round finish and purchased-card tiebreak.

Use the qualified i5-13600KF topology, one performance logical CPU (CPU0), normal
priority, one numerical thread and no GPU. A fresh process handles every move;
no pondering, between-game cache or cross-seat learned state is introduced.
Both players receive only the lawful observer. Actual game seeds, private seed
entropy, hidden deck order and opponent blind-reserve identities are absent
from worker requests. Public search streams and referee randomness are separate.

## Fixed sample and clock

Games 0–59 form pairs `game//2`, assigned fits by `pair%3`. Each pair uses the
same initial shuffled world and opposite seats. The local seat is
`1-((pair+game%2)%2)`. Run each six-game block in order 0,2,4,1,3,5 relative to its
start; this balances model/seat exposure over time. Search randomness has its
own public game-specific stream, with no cross-leg state sharing.

Before launch, freeze a private nonce, a full-SHA256 seed for every pair and the
complete sealed schedule. Check these against all discovered historical seed
registries, schedules and manifests. Any unscanned file or collision blocks the
freeze. Previous development games, the six pilot games and the eighteen
operational games contribute **zero** observations. Do not open their matchup
totals or select fits using their results.

The trusted parent clock starts before request serialization and pipe write,
and stops at the trusted complete-response newline timestamp. It covers all
within-turn action handling, search, inference and required local trace delivery
(write, flush, fsync and hash). The completed response must arrive within 120
seconds. Parent legality/replay checks are outside this clock. Initialization
has a separate 60-second limit for each player; no unused time carries forward.
Elapsed enforcement/cleanup overhead never grants additional thinking time.

Timeouts, worker exits, malformed/wrong-request/illegal/unsolicited responses,
write errors and recognized internal budget exhaustion are scored losses.
Complete the paired second leg, even after a forfeit. Do not retry a timed-out
move, grant grace, replace its seed, drop the game or salvage a partial search.
No anytime fallback is introduced. Unclassified worker errors, initialization
failures, wrapper/work/observer mismatches, mapping or referee defects and audit
failures are preserved scientific/infrastructure holds, not invented results.
Nonterminal play at 400 plies is a hold, not a draw.

## Mandatory assurance before accepting results

Preparation runs model-free adapter, clock, rules, statistics, admission and
analyzer tests. After a separately authorized launch, before strength games,
execute twelve fixed transfer/new-context requests: three fits, two contexts,
and original/hidden-state-perturbed versions. Fully replay all delivered worlds;
require exact accepted-pilot transfer where applicable and six privacy-pair
invariances. A failure stops the attempt before games. The unchanged qualified
edge-case suites and exact opponent identity remain frozen evidence.

Apply [the verification policy](verification_policy.md) to every actual game:
independently reconstruct legal actions, transitions, observer restrictions,
termination, final scores/winner, seat/model/seed assignments and full schedule.
Check every delivered local proposal, selected action, all 128 world records
and continuation aggregates. All opponent delivery receipts and qualified work
contracts must pass their checks. No sampled replay can excuse a bad actual move.

At every completed local root, replay every offered candidate in eight worlds,
selected by the eight smallest canonical-JSON SHA256 priorities using the fixed
audit domain, seed **202610206401**, root identity and world index. Add full
128-world replay at the minimum-priority root in each observed fit × phase
(0–19/20–39/40+) × acting seat × private-reserve category cell, and each fit's
support-size 1/2/3 cell. Deduplicate overlaps. Record all 81 potential cells;
missing cells remain untested and never cause extra games. Manifest construction
does not inspect winners, values, timing or audit success. Preserve every failed
or partial trace. A mismatch stops acceptance; diagnosis beyond the cap requires
a separate bounded plan. Do not alter a player within this study.

A separate model-free analyzer checks the complete sample, receipts, manifest,
coverage, all source bindings and failure attribution before computing matchup
statistics. Its process and any authenticated Windows console helper must close
before final acceptance. Final acceptance needs a passed finalization validation
receipt and no `completion_hold.json`. The parent's live stdout/stderr logs and
its later exit receipt are explicitly excluded from immutable result bindings;
all scientific worker receipts, traces and analysis records remain bound.
Intermediate progress reports show counts and failures, not W/D/L or
fit-specific strength. An incomplete or held attempt has no accepted primary
estimate; any later descriptive partial disclosure must be labeled incomplete.

## Prespecified analysis

Score local wins 1, draws 0.5 and losses 0, including timeout and other genuine
player forfeits. Average the two games within each seed pair, then the ten pair
means within each fit, then the three fit means equally. The seed pair is the
resampling unit. The primary estimate includes all sixty scheduled games.

Use 50,000 stratified bootstrap replicates, independently resampling ten complete
pairs with replacement within each fit, PCG64 seed **202610206701**, fixed draw
order and linear 2.5th/97.5th percentiles. Freeze the implementation and NumPy
runtime provenance. Report score and interval, overall W/D/L, paired scores,
descriptive per-fit and seat summaries, timeout/other-forfeit counts, delivered
work and wall/CPU/initialization/replay costs. Do not select a best fit or use
unpaired game intervals. No secondary multiple-comparison claims are planned.

A pooled interval entirely above/below 0.5 supports a directional result for
this matchup; otherwise report uncertainty. This is an approximate percentile
bootstrap with only ten pairs per stratum. It is not an equivalence test and
does not establish a small advantage or a universal strength claim. A rough
worst-variance normal 95% half-width at thirty pairs is about 18 percentage
points; actual bootstrap width depends on paired results. There is no early
success/futility stopping and no change to existing research success criteria.

## Resources, stopping and later launch

Measured closed operational timing suggests roughly **40–50 serial hours** for
sixty games and prescribed checks, conditional on similar game lengths and host
performance. See the frozen preparation cost projection for its provenance and
assumptions. This is a scenario, not a completion guarantee.

- Global cap: **72 hours**, charged against both monotonic elapsed time and wall
  time from the original launch. Protect 18 hours for post-game verification.
- Final analyzer: at most 3,600 seconds, followed by a protected 600 seconds for
  finalization; both remain inside the global cap. Each replay request has 600
  seconds plus separate 60-second initialization.
- New attempt evidence cap: **8 GiB**, with a 32 MiB in-flight/final-receipt reserve
  and at least 5 GiB free disk. Reuse immutable sources by reference; no old
  evidence is deleted or copied wholesale.
- Worker/process-tree RSS ceiling: 2 GiB. RAM admission requires at least 2 GiB
  physical and 4 GiB commit headroom. Initial admission and recovery from a low
  reading require thirteen good readings ten seconds apart spanning 120 seconds.
  A normal healthy worker boundary needs one good reading.
- Wait only before workers: at most two hours per admission and six hours total,
  all charged to the 72-hour cap. Never pause or refund an active move clock.
  No automatic restart, deadline extension or resource-cap enlargement.

Practical caps deliberately fail closed; they cannot guarantee completion under
the formal maximum game length. Preserve all completed, partial and failed
evidence on a hold. A later authorized recovery must retain the exact sample,
players and all failure losses; it cannot silently reset budgets or replace data.

Preparation command is separate from execution. The frozen output directory is
`outputs/matched_strength_20261020_01/`; its `freeze.json` binds this protocol,
review, config, sealed seeds, tests, implementation, checks and accepted lineage.
After explicit launch authorization, run the frozen runner's `run` action once
with the exact SHA256 of `freeze.json` supplied as `--freeze-sha256`,
with hidden processes and the shared execution lease. It performs the mandatory
runtime gate, fixed games, prescribed audits and bounded independent completion.
No benchmark, training or larger independent-data study starts in preparation.
The larger-data study remains deferred until storage and its justification are
separately resolved.
