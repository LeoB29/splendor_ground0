# Human--Resolver Ceiling Calibration (`074`)

Frozen before implementing the interactive parallel wrapper, running its exactness
check, or playing any human game against the resolver. This is a CPU-only external
calibration. It performs no fitting, selection, tuning, checkpoint promotion, or
reuse of the ten existing human-versus-`035` games as training data.

## Question and disagreement with the inherited diagnosis

The review's structural premise is correct: the deployed resolver evaluates only
raw `035`'s top four root successors and uses stochastic `035` continuations.
Its stronger conclusion is not established. Restricted action support and
`035`-guided follow-through do not mathematically cap the resolver below a human
opponent; a local refinement can still beat that opponent. This calibration asks
the direct question: does the experienced owner score above, near, or below the
frozen public-belief resolver?

The operator is the repository owner who previously scored 8--2 against `035`.
The machine opponent is the exact frozen `049b`/`050`/`051` resolver around
`outputs/legal_action_scorer_035_legal_observation/`
`legal_action_attention_scorer_best.pt`, SHA-256
`4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`.
Search parameters, root and continuation seed formulas, canonical tie-breaking,
and fallback reasons remain literal `050`. Parallel execution is an execution
change only.

## Stage A: exact execution equivalence

Before the GUI may be relied on, compare the existing serial resolver with the
within-decision process-parallel path on three deterministic, public game states:
one each from early (turns 0--19), mid (20--49), and late (50+). Generate the
states from a fixed raw-`035` self-play trajectory independent of all human data.
For each state use the same engine seed, acting seat, decision index, checkpoint,
belief determinizations, and continuation seeds.

Require byte-identical canonical selected actions, resolver telemetry, complete
`TwoPlyResolution` records, and observation-hook records after stable JSON
serialization. Each side must either complete a search with exactly 192
continuations or produce the same frozen fallback. The primary engineering
endpoint is a Boolean conjunction over all three states. Any mismatch forbids
interactive use; diagnose it without playing human games. The serial path remains
available as the reference. Estimated cost is 3--6 CPU-minutes plus process-start
overhead. No statistical power calculation applies to an exact identity test.

## Stage B: fixed ten-game external calibration

After Stage A and the full test suite pass, play exactly ten completed games using
fresh engine seeds `800000--800009`, in order. The human seat alternates 0, 1, 0,
1, ... so each side starts five times. Do not inspect a partial record to change
strategy, seeds, seats, resolver parameters, worker count, or the stopping rule.
The human may play normally; no prescribed human strategy is imposed.

The sole primary endpoint is human score share over the ten games: win = 1,
draw = 0.5, loss = 0. Report the raw W-D-L, score share, final-score margins, and
an exact binomial interval (with draws reported rather than forced into the
binomial calculation). Resolver intervention dose is diagnostic but mandatory:
report completed searches, fallbacks by reason, switches from raw `035`, and
switches divided by resolver decisions, overall and by phase.

Every completed game is appended to `outputs/human_games/replays.jsonl` in the
existing replay-corpus schema and to `games_summary.jsonl`, with
`opponent_kind=resolver`, checkpoint identity, execution worker count, and final
resolver telemetry. Those rows remain held out from all fitting, selection, and
tuning. An unfinished GUI game is discarded and replayed at the same scheduled
seed and seat; a completed valid game is never replaced. Stop after ten valid
completed games. An illegal action, unexpected resolver exception, nonfinite
leaf, altered checkpoint/contract, or replay-recording failure pauses the panel
for integrity diagnosis and does not authorize a replacement seed.

## Expected effect, sizing, and interpretation

Treating score share as a win probability only for advance sizing, the prior
results imply about +241 Elo-equivalent for the human versus `035` and +139 for
the resolver versus `035`. Subtracting these estimates predicts +102 Elo, or
human score share about `0.643` against the resolver: a +14.3-point effect over
parity.

Ten games are not decision-grade for that effect. With no draws, the one-sided
5% exact-binomial rejection region is at least 8 wins (null tail 5.47%), and its
power at `p=0.643` is only 24.7%. At least 78 games (reject at 47 or more wins)
would be required for 80% power at the expected effect. The agreed sample of ten
is justified only as a cheap external triage measurement: 78 interactive games
would impose disproportionate operator time, while the result can cheaply expose
a grossly wrong ceiling ordering before a 20--30-hour resolver-distillation run.
No formal pass, promotion, or branch closure may be claimed from this panel.

Predeclared interpretation:

- 8--10 human wins (or score share at least 0.80 with draws) supports the claim
  that raw resolver distillation is unlikely by itself to reach this human and
  deprioritizes resolver cloning as an endpoint.
- 0--4 human wins (or score share at most 0.40) falsifies the working claim that
  the resolver ceiling is below this human; proceed to the separately
  preregistered from-scratch resolver-cloning pilot if otherwise warranted.
- Every other result is explicitly inconclusive. It licenses neither conclusion
  and may motivate a separately preregistered extension, never post-hoc pooling.

The intervention-dose expectation comes from `050`: the resolver switched 3,868
of 5,720 completed searches and was attempted on every resolver turn. The actual
dose in these ten games must be reported; no minimum dose is a hidden gate.

## Cost and what it buys

Within-decision parallelism is expected to reduce an eligible move from roughly
60 seconds serial to several seconds on 10 CPU workers. Ten games are expected to
contain about 310 resolver moves and therefore cost roughly 15--35 minutes of
resolver waiting plus human thinking time. This buys an external calibration of
the programme's largest current crux and reusable exact parallel infrastructure.
It does not buy a precise strength estimate. By comparison, the existing-data
resolver-cloning pilot costs about two CPU-hours and tests learnability, not the
teacher ceiling; fresh resolver data would cost 20--30 wall-hours; another
hand-written strategy bot has already failed twice on specification quality.

## Stage A result (2026-08-03)

The full suite passed `585/585` before evidence. On the frozen turn 0, 25, and 50
states, both paths completed a search with exactly 192 resolved continuations and
produced byte-identical selected actions, telemetry, hook records, and complete
resolution payloads. Serial/parallel times were `54.76/9.94`, `36.14/6.73`, and
`11.73/3.17` seconds: speedups `5.51x`, `5.37x`, and `3.70x`. Stage A passes.

Evidence:
`outputs/resolver_parallel_equivalence_074/result.json`, SHA-256
`78c1d503bc48bb41eb5d71aa58955085ced1368c8451e38402faf3730d9687ee`.
## Stage B result (2026-08-03)

All ten fixed seeds and alternating seats were completed and recorded. All
integrity checks passed: 556 contiguous replay rows, matching summaries, frozen
checkpoint and contract, exact 192-branch accounting, all 48,960 continuations
resolved, and zero hard caps, loop adjudications, or rollout fallbacks.

The human went **7-3-0**, score share `0.700`, exact Clopper--Pearson 95% interval
`[0.3475, 0.9333]`, with one-sided exact-binomial p=`0.171875` against parity.
Mean human score margin was `+2.8` (median `+2`, range `-5` to `+13`). The seat
split was 3-2 from seat 0 and 4-1 from seat 1.

Resolver dose was high: 179 switches in 278 resolver decisions (`64.39%`), or
179/255 completed searches (`70.20%`). By phase it switched 72/100 early
(`72.0%`), 95/149 mid (`63.76%`), and 12/29 late (`41.38%`). There were 23
frozen structural fallbacks: 17 fewer-than-four-root, five root-terminal, and one
fewer-than-four-opponent-action.

Formal preregistered interpretation:
`inconclusive_no_ceiling_or_branch_decision`. Seven wins fall in the frozen 5--7
region. This neither validates nor falsifies the claim that resolver strength is
capped below this human, authorizes no promotion or training, and must not be
pooled post hoc with an extension.

Post-hoc behavioral diagnostic, excluded from the primary decision: across 278
decisions each, the human made 62 visible reserves including 24 tier-3 (19 of 28
opening visible reserves); the resolver made 27 visible reserves, zero tier-3,
and 27 blind deck reserves (24 in the opening). The human made two blind deck
reserves. This independently reproduces the strategic-mode asymmetry, but does
not establish that it caused the 7-3 outcome.

Evidence:
`outputs/human_resolver_ceiling_calibration_074/result.json`, SHA-256
`e7a55f394bea87f3f4ad462b164d1e1c5323baf41e616eaa2c6299d9c73a415c`.
