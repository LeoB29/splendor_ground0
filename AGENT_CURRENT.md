# Agent Current State

Use this file first in a fresh chat. `AGENT_HANDOFF.md` is the long historical log.

## START HERE: handoff as of 2026-08-03

Champion is still frozen `035`. Nothing has been promoted. The last four
experiments were `069` (failed), `071` (powered null), `072` (failed), `073`
(invalid). Read in this order:

1. This section.
2. `docs/project_review_response_070.md` -- the current blocker diagnosis with all
   arithmetic. **Read this before proposing any direction.** It supersedes the
   older "Best Next Step" section far below, which is stale and kept only for
   provenance.
3. The four dated entries immediately below this section.

### The diagnosis in one paragraph

The blocker is not credit assignment, compute, or a missing stronger operator. It
is that nearly every learning experiment was capped at 1--4% changed decisions by
its own safety gates and then asked for +2.5 to +5 points of score share against a
league spanning about 90 Elo with +/-5.8 points of noise at 400 clusters. The one
high-dose operator (the public-belief resolver, ~62% of decisions) produced the
only large gain; the one high-dose null (rolling horizon `043`) is genuinely dead;
everything between is a low-dose null that was read as a method refutation. Two
further findings sharpen this: `069`'s entropy floor was arithmetically
unreachable from its own parent's starting value, so it could not have passed; and
the resolver takes its root candidates from `035`'s top four with `035`
continuations, so it **inherits `035`'s strategic blind spot** and cannot exceed
refinements of `035`'s existing mode.

### Standing rule for any new learner

Either change `>=15%` of decisions and test at `>=200` paired seeds, or declare the
expected effect size in advance and size the panel for `>=80%` power against it. Do
not cap movement below detectability and then ask for a large effect. Declare the
intervention **dose** as a reported quantity, as `071` does.

### Next steps, in priority order

1. **Human versus resolver, 10 games.** The owner has agreed to play. This fixes
   the ceiling question that gates the entire resolver branch: the resolver is
   about +139 Elo-equivalent over `035` while the owner's ten games imply about
   +241, so distilling the resolver perfectly might still not reach human level.
   **Blocked on engineering:** a resolver move currently costs ~60 s because the
   192 continuations run serially (see `scripts/analysis/resolver_move_cost.py`).
   Parallelise within the decision -- branch seeds come from
   `gameplay_continuation_seed` and cross-branch state is order-independent
   counters, so it should be exact; verify byte-identical decisions against the
   serial path first. Then add `--opponent resolver` to `run_gui.py`, preserving
   replay capture in the schema of `outputs/human_games/replays.jsonl` so the
   existing forensic audit tooling works on the new games. Estimate 1--2 hours.
2. **If the scripted-strategy line is resumed, use a search instrument, not
   hand-written rules.** `072` and `073` both failed on the author's rule
   specification rather than on the expert's strategy: `072` burned the gold that
   makes a `7X` card payable, `073` specified gold discipline so strictly it became
   unsatisfiable and never spent gold at all. Both failures were in *execution*
   (token sequencing, gold budgeting), which search handles and rules do not. The
   structurally different version is `ShallowSearchBot` with leaf evaluation biased
   toward holding and owning high-efficiency cards. **Do not preregister a `074` in
   the hand-written family**; see the `073` entry.
3. **From-scratch resolver behaviour cloning, gated behind step 1.** The only
   demonstrated stronger operator, and the from-scratch no-anchor variant has never
   been tried -- all three prior attempts warm-started from and anchored to `035`,
   which `052a` diagnosed as the failure mechanism. A staged protocol with sample
   sizes, gates and falsification conditions is sketched in section 7 of the review
   response. Stage 1 costs about 2 hours on existing data and can close the branch
   cheaply; Stage 2 costs 20--30 h wall.

### Do not do these

Full kill list with reasons in section 8 of the review response. The ones most
likely to be re-proposed by a fresh agent:

- Any recurrent scale or entropy-repaired re-run of the `069` family. It is closed
  on **surrogate exhaustion** (a 15.7x efficiency decay across four generations),
  not on the entropy gate. Relaxing the floor would answer nothing.
- The `071` resolver-arm ablation, even though `071`'s own decision rule authorizes
  it. Reasons in the `071` entry.
- A `073b` or `074` hand-written scripted teacher.
- GPU rental. No measured bottleneck is neural fitting; all of it is engine
  simulation and rollouts.
- Any BGA scraping or automated access.

### Reproducible derivations

The numbers behind the diagnosis are re-runnable rather than only asserted:

- `scripts/analysis/gate_power_069.py` -- bootstraps the real `066` clusters to
  `069`'s planned panel and reports the conjunction's power curve.
- `scripts/analysis/card_efficiency.py` -- the card-efficiency structure and the
  reserve-gating mechanic behind `072`/`073`.
- `scripts/analysis/resolver_move_cost.py` -- measured per-decision resolver cost.

### Held-out data

The ten human games in `outputs/human_games/replays.jsonl` remain held out. They
have been used for **diagnosis** (action mix, reserve tiers), which is what they
were recorded for, and never to fit, select or tune anything. Keep it that way:
they are the project's only external calibration. Do not build a manual annotation
workflow and do not train on them.

## Environment (read before running anything)

- The working interpreter on this machine is `C:\Users\leoba\anaconda3\python.exe`
  (Python 3.12.7, torch 2.4.1+cpu, pytest 7.4.4). The repository `.venv` has
  neither torch nor pytest and must not be used.
- `splendor_ai` is now installed editable into that interpreter
  (`python -m pip install -e . --no-build-isolation`), so `import splendor_ai`
  resolves from any working directory and **`PYTHONPATH` is no longer required**.
- Before that install, four CLI tests failed spuriously on this machine
  (`test_conversion_diagnostics`, `test_pgs_corpus_audit`,
  `test_search_value_diff`, `test_value_risk_audit`). They spawn
  `python run_*.py` subprocesses, which inherit environment variables but not
  pytest's in-process `pythonpath = ["src"]`. If those four ever fail again,
  suspect a broken install, not a code regression.
- Current full suite: `572 passed` (565 historical plus 7 new `071` contracts).

## 2026-08-03 Gold-Disciplined Teacher Is Invalid on a Self-Contradictory Gate

- `073` was the single re-elicitation `072` authorized, and its protocol declared
  itself terminal for the scripted-strategy line. It changed exactly two rules,
  both elicited from the expert: protect gold while holding a high-cost target,
  and reserve-and-clear rather than filling all three slots. Both are carried on a
  `ValueReserveRules` record defaulting to disabled, and `072` was re-run to a
  scratch directory and **reproduced exactly** after the refactor. Suite passed
  `585/585` before evidence.
- Formal decision: `scripted_value_teacher_073_invalid_no_strength_decision`.
  The failing gate is `reserve_and_clear_active`, which required
  `multi_target_hand_decisions == 0` while the same protocol text explicitly
  exempts the rule-3 endgame block from reserve-and-clear. **The gate contradicts
  the protocol it was written to enforce.** This is a specification defect of the
  same class as `041`'s literal-`completed` rule and `049`'s source contract, not
  a property of the candidate.
- Strength, recorded but formally void: **2-398-0**, score share `0.005`, CI95
  `[0.0000,0.0125]`, margin `-15.405`. That is *worse* than `072`'s 21-379.
- A second specification defect was not caught by any gate and is the reason for
  the regression. Gold discipline as literally written requires the gold still
  held after a purchase to be at least the target's colour shortfall. Early in a
  game against a `7X` target with no bonuses that shortfall is about 7, while only
  `GOLD_TOKENS = 5` exist, so the condition is unsatisfiable and gold is never
  spent at all: `29` gold spent across the panel versus `856` in `072`. The bot
  hoards gold instead of protecting it. Conversion fell to `7.4%` and token-taking
  rose to `88%` of decisions. The rule-3 block also fired `464` times versus
  `072`'s `39`, because a bot this far behind is almost always facing an opponent
  one card from 15.
- Correction to an interim claim: a single-game check on seed `760000` seat 0 was
  a 15-12 win and was reported as evidence the changes worked. The panel shows
  that game to be one of only **2 wins in 400**, and its own cluster splits 0.5
  because the seat-1 game lost 0-15. Per-cluster shares across the panel are 198
  at `0.0` and 2 at `0.5`. Do not generalise from single-seed smokes.
- **Do not preregister a `073b` or a `074` in this family.** The terminality clause
  was written precisely to stop an invalid-then-retry chain, which is the pattern
  that produced `065`-`069`. Two encodings have now failed on the author's
  specification rather than on the expert's strategy, so the binding constraint is
  rule-writing quality, not the hypothesis. If this line is resumed it must be
  with a structurally different instrument -- for example a search bot whose leaf
  evaluation is biased toward high-efficiency reserves, so search handles token
  sequencing and gold budgeting instead of hand-written rules.
- Nothing here is evidence about the expert's strategy. The independent card-table
  and reserve-gating analysis is untouched, and the ten human games remain held
  out and unused for fitting.
- Evidence: `outputs/scripted_value_teacher_073/result.json`, SHA-256
  `56e9766e73dec1e4dd671674bf230ab38d06950781c380eb6bcc2f513fdfcdfc`.
  Implementation: the `ValueReserveRules` record in
  `src/splendor_ai/bots/value_reserve_bot.py`,
  `run_scripted_value_teacher_073.py`, and the four `073` contracts in
  `tests/test_value_reserve_bot.py`.

## 2026-08-03 Scripted Value Teacher Fails, but the Encoding Was Unfaithful

- `072` tested whether the expert's stated strategy -- reserve high-efficiency
  cards early for a delayed points payoff, ignore the opponent, never reserve from
  the deck -- wins in this engine. Deterministic rule-based bot, no search, no
  neural component, no randomness. Protocol
  `docs/experiments/scripted_value_teacher_072.md`; suite passed `581/581` before
  evidence.
- Result: **21-379-0** over 200 fresh clusters / 400 games at seeds
  `750000--750199`. Paired score share `0.0525`, CI95 `[0.0325,0.0750]`; mean
  paired margin `-12.64`, CI95 `[-13.2475,-12.01]`. All ten integrity gates
  passed, including exact determinism on a first-cluster replay and zero deck
  reserves in 12,644 decisions. Formal decision:
  `scripted_value_teacher_weaker_close_scripted_strategy_line`.
- **Do not read this as evidence about the expert's strategy.** The preregistered
  faithfulness diagnostics show the rules did not reproduce what was described.
  Action mix was `79.7%` takes / `10.9%` visible reserves / `1.7%` reserved buys
  versus the expert's `51.3%` / `23.3%` / `18.8%`; absolute deviations `0.284`,
  `0.124`, `0.171`. Reserve conversion was `15.8%` versus the expert's `81.0%`.
  Reserve tier share was `82.8%` tier 3 versus the expert's `44.4%`.
- Mechanically the bot paralysed itself. It buys `2.9` cards per game versus the
  expert's `7`, of which `849/1,159` are tier-1. It reserves `3.45` per game
  versus `6.3` and converts `0.55` versus `5.1`. Mean game length `63.2` turns
  versus the expert's `53.2` in wins. There were 18 `forced_pass_deadlock`
  endings in 400 games (`4.5%`, against 1 in 2,000 for `071`), the signature of
  hoarding tokens to the 10-cap with nothing affordable.
- Two structural causes, both traceable to the frozen rules rather than to bugs.
  First, effective value alone selects targets, so the target is almost always a
  `7X` mono tier-3 whose deficit cannot close: the bank holds at most 4 of a
  colour, so it needs 3 gold every time. Second, rule 6 spends that gold on cheap
  discount-builders -- `856` gold across `1,159` purchases -- destroying the only
  route to the target. Rule ordering also places reserving above token
  accumulation, so a freed hand slot outranks progress.
- Per the protocol, do **not** tune the rules, thresholds, or the efficiency
  filter to rescue this. The protocol permits exactly one re-elicitation from the
  expert, only because the faithfulness diagnostics failed, and it must be
  separately preregistered. The two decisive open questions are gold discipline
  while holding a `7X` target, and whether reserving is paced against conversion
  rather than fired whenever a hand slot is free.
- The card-efficiency and reserve-gating analysis behind the protocol stands
  independently of this result: points per token is monotone in cost
  concentration, each colour X has exactly two tier-3 cards costing `7X` worth 4
  and 5 points, and `BANK_GEMS_PER_COLOR_2P = 4` with gold obtainable only by
  reserving. Tests assert both derived claims.
- Evidence: `outputs/scripted_value_teacher_072/result.json`, SHA-256
  `04ac12dcdb677250f92b85fecb90270a9c6b63986f0daa8a4433fc2402793f38`.
  Implementation: `src/splendor_ai/bots/value_reserve_bot.py`,
  `src/splendor_ai/eval/run_scripted_value_teacher.py`,
  `tests/test_value_reserve_bot.py`.

## 2026-08-03 Deck-Reserve Mask Is Neutral at a 7.3% Dose

- `071` tested one inference-time action-class mask on frozen `035`:
  `RESERVE_DECK` removed from the greedy candidate set, identical weights on both
  sides. No training, no new checkpoint. Protocol
  `docs/experiments/deck_reserve_mask_ablation_071.md`; the suite passed
  `572/572` before evidence.
- The hypothesis came from outside the local league, which is why it was worth
  testing at all: the experienced player reported never reserving from the deck
  and judged it near-always wrong, and 3 of the 12 resolved mistakes in the
  disjoint reserve-generalization audit were `RESERVE_DECK`->`RESERVE_VISIBLE`.
- 1,000 fresh paired clusters / 2,000 games at seeds `740000--740999`. There were
  1,999 ordinary completions and one retained `forced_pass_deadlock`. All nine
  integrity gates passed; `deck_only_fallback` was 0, so the mask never had to
  yield.
- The mask bound on `7.317%` of 63,424 candidate decisions: 4,641 switches,
  about 2.32 per game, substituting `RESERVE_VISIBLE` on 98.7% of them (4,582),
  plus 48 buys and 9 takes. This is a genuinely dosed intervention, unlike
  `005` (0.89%) or the recurrent updates (3--4%).
- Paired score share was `0.50575`, CI95 `[0.4850,0.5265]`, SE `0.0105`; paired
  margin `-0.031`, CI95 `[-0.3085,+0.2425]`. The panel resolves effects above
  about `+2.8` points, so this is a **powered null**, not an underpowered one.
- It also bounds the average deck->visible substitution at `<=0.10` score-margin
  points per changed decision, against the audit's `+0.836` on screened states.
  Both are true: selected suspicious states show a real effect, the population
  average is about zero. Do not cite the audit's 3/12 as evidence that the class
  is generally bad.
- Incidental engine fact: first-player advantage in `035` mirror play is about
  57%. The two reported per-seat shares (`0.578`/`0.4335`) are algebraically the
  same paired estimate restated and carry no independent seat information.
- Formal decision: `deck_reserve_mask_neutral_at_measured_dose`. Do not promote
  either configuration, do not tune the mask to a phase window or tier subset,
  and do not rent a GPU. Champion `035` remains frozen.
- **The protocol authorizes a resolver-arm ablation. Do not exercise that
  authorization.** The null is conditional on an `035`-family opponent. A blind
  reserve's cost is not knowing what you get, which only matters against an
  opponent planning a specific conversion; against one that plans none, the
  substitution is free. Re-running it on the resolver (9.4% rate) would spend
  about six hours to re-learn this inside the same closed ecosystem. The question
  needs an opponent that punishes the informational loss: human games, or the
  scripted teacher below.
- Note for gate design: one legitimate `forced_pass_deadlock` occurred in 2,000
  games. Had integrity gate 2 been written as "all games `completed`", the way
  `041` was, this experiment would have been formally void. The protocol admitted
  it because the engine contract does.
- Evidence: `outputs/deck_reserve_mask_ablation_071/result.json`, SHA-256
  `944958238df4b8b0218a03315e07291e5d200374a6ea46fac8fdd87201af2d11`.
  Implementation: `src/splendor_ai/bots/deck_reserve_mask.py`,
  `src/splendor_ai/eval/run_deck_reserve_mask_ablation.py`,
  `tests/test_deck_reserve_mask_ablation.py`.

## 2026-08-03 Independent Review `070`: Blocker Re-Diagnosis

Full derivations, tables and reviewer-error corrections are in
`docs/project_review_response_070.md`. Read that before disagreeing with any item
here. Advisory only: it changes no gate, passes no failed experiment, and
promotes nothing.

- **Re-diagnosed blocker.** Not credit assignment, compute, or a missing stronger
  operator. Every learning experiment was capped at 1--4% changed decisions by its
  own safety gates, then asked for +2.5 to +5 points of score share against a
  league spanning about 90 Elo with +/-5.8 points of noise at 400 clusters. The
  one high-dose operator (resolver, ~62%) produced the only large gain; the one
  high-dose null (rolling horizon `043`) is genuinely dead; everything between is
  a low-dose null that was read as a method refutation. `005` is the clearest
  case: 0.89% dose, 200 pairs, needed roughly 26,000 to detect its own predicted
  effect.
- **`069` could not have passed.** It initialised from `064` generation 2 at
  normalized entropy `0.32058` and planned three generations against a frozen
  `0.30` floor, while the two prior generations of the same architecture had each
  lost about `0.02`. The floor was inherited from `062`, where full-weight early
  decisions put entropy at `0.386`; `064`'s 4x early down-weighting consumed
  roughly a quarter of the headroom with no implied change in sharpness, and the
  threshold was never re-derived. The entropy bonus also does not regularise the
  gated quantity: weight `0.002`, applied to the temperature-1.0 **5%-uniform-mixed**
  behaviour distribution, **un-normalized**, while the gate measures the raw
  masked softmax **width-normalized**.
- **The real negative result in `069` is surrogate exhaustion**, and it is
  sufficient on its own. Validation surrogate fell `+0.003788` -> `+0.001396` ->
  `+0.000681` -> `+0.000393` across `064` g1/g2 and `069` g1/g2 while KL stayed at
  0.003--0.005; surrogate-per-KL fell `1.199` -> `0.365` -> `0.173` -> `0.0766`, a
  15.7x efficiency decay while per-generation data grew about 9x. Four panels
  (`064`/`065`/`066`/`068`) independently put the candidate at parity with epoch
  zero. **Close the outcome-only recurrent branch on this, not on the entropy
  gate**, and do not re-run it with a relaxed floor.
- **Some resolver gates were unachievable in principle.** Frozen `035` already
  agrees with the resolver's chosen action on `31.8%`/`32.7%`/`33.0%` of resolved
  train/validation/confirmation rows, while the resolver's own test-retest
  agreement (`055`) is `45.8%`. The learnable band is about 13 points. `054`
  demanded `40%` validation accuracy, i.e. 55% of that band on a first attempt
  with frozen features; `052` demanded `+5` points while also requiring
  fallback-anchor KL `<=0.02`. Treat those as ceiling artifacts, not clean
  learnability failures.
- **The resolver inherits `035`'s strategic blind spot, structurally.** Its four
  root candidates are `ranked[:4]`, i.e. `035`'s top four
  (`run_paired_public_belief_resolver_gameplay.py:250-255`), and its continuations
  are `035` at eps 0.10 / tau 0.50 / top-k 3. It cannot pick outside `035`'s top
  four nor see the payoff of a line `035` will not execute in rollout. Measured:
  `035` made **0** tier-3 reserves in 271 deterministic decisions and **2** early
  tier-3 visible reserves in 63,622 self-play decisions; the human made **18** in
  the opening alone. The resolver chose early tier-3 visible reserves on 3 of
  2,227 early decisions (`0.13%`) and early deck reserves on `19.8%` -- about 6x
  `035`'s deterministic rate, a strategy-fusion signature from averaging over
  determinizations. **So resolver distillation is capped at refining `035`'s
  existing mode**: resolver is about +139 Elo-equivalent over `035` while the ten
  human games imply about +241.
- **Champion provenance is worse than the packet states.** `035` is built on the
  295-wide legacy tensor the project has itself deprecated, and is only 48-52
  against `022`, so the legality repair bought no measurable strength. `061`--`069`
  froze every `035` parameter and bolted a GRU on top, and `063` found the
  residual learned almost entirely from frozen current-state/action features
  (95.84% of KL survived history ablation). A frozen encoder that never encoded
  history-relevant structure cannot host a history mechanism.
- **Gate-design reforms recommended.** Keep integrity as a hard conjunction;
  replace flat learning/strength conjunctions with one pre-declared primary
  endpoint plus diagnostics; convert hard per-stratum sign floors into
  collapse guardrails. Bootstrapping the real `066` clusters to `069`'s planned
  400-cluster panel gives the final conjunction 9.3% type-I error but only
  **43% power at its own declared +0.025 minimum**, with the per-stratum
  sub-gates as the dominant power sink.
- **Standing rule for future learners.** Either change `>=15%` of decisions and
  test at `>=200` paired seeds, or declare the expected effect size in advance and
  size the panel for `>=80%` power against it. No more capping movement below
  detectability and then asking for a large effect.
- **Ranked next directions.** (1) Build a coherent scripted teacher for the
  human's actual strategy -- take tokens toward the highest-value soon-affordable
  tier-2/3 card, reserve high-value visible cards, never reserve from the deck,
  convert -- and play it 200 paired games versus `035`. Free to run, no search. A
  win supplies an existence proof, the first genuinely independent league
  opponent, and a coherent teacher for the `021`/`023` recipe, the only supervised
  recipe that ever produced a large gain here. (2) Ten human games versus the
  resolver, about 3 s per move, to fix the ceiling. (3) From-scratch resolver
  behaviour cloning, **demoted** and gated behind (2); a protocol sketch with
  sample sizes and falsification conditions is in the response document.
- **Open, untested by `071`:** the tier-3 reserve absence. The two
  outcome-grounded audits cannot test it, because their continuations are
  `035`-guided and so cannot convert a high-tier reserve. Reported honestly in the
  response document: of 20 resolved mistakes across both audits, only 2
  transitioned to tier 3.

## 2026-08-03 Scaled Recurrent Succession Stops at Entropy Contract

- Experiment `069` was prospectively frozen at `60e3417`/`887f40e` and
  implemented at `1f2112a`. The clean pre-evidence suite passed `565/565`.
  It explicitly treated `068` as failed and asked a new scale-or-stop learning
  question; it did not retroactively authorize Stage C or promotion.
- Scale generation 1 collected exactly 2,880/2,880 fresh eligible games and
  90,290 actor decisions across six opponents and both seats, with exact
  192/48 train/validation games per cell. Every game completed normally. It
  passed every offline and parent-retention gate: raw validation KL
  `0.003935`, argmax movement `3.483%`, overall/midlate surrogate
  `+0.000681/+0.000706`, history-rotation TV `0.02574`, normalized validation
  entropy `0.30934`, and parent score share `0.495` with margin `+0.28`.
  Candidate SHA-256:
  `d82f65df319a130c2dded424e0d21249736a8b5dfe38c5cfb5c21053947a5566`.
- The first process was stopped only after generation 1 was atomically
  accepted, when incomplete generation-2 collection exposed that Python still
  retained the prior full corpus during tuple assignment. No generation-2
  checkpoint, record, metric, or gate existed. Commit `da2862a` explicitly
  releases completed corpus objects at generation boundaries; the corrected
  full suite again passed `565/565`. Resume verified generation 1 by parent,
  candidate, record, and protocol hashes and did not replay or retrain it.
- Scale generation 2 then collected exactly 3,360/3,360 eligible games and
  105,087 decisions across seven opponents and both seats, again with exact
  192/48 cell splits. There were 3,358 ordinary completions and two legitimate
  forced-pass deadlocks; both were retained. No seed was replaced or excluded,
  and every integrity/learning condition except one passed.
- Generation 2 had raw validation KL `0.005131`, movement `3.634%`, positive
  overall/midlate surrogate `+0.000393/+0.000244`, history TV `0.03604`, exact
  empty-history fallback and reload, and a favorable 100-game parent gate:
  `0.520` score share and `-0.10` mean paired margin. Its only failed gate was
  normalized validation entropy: `0.29426` versus the frozen `0.30` minimum.
  This is a close learning-contract failure, not integrity failure or observed
  gameplay regression, but it cannot be waived after evidence.
- The runner correctly stopped before generation 3, the 3,840-game final
  collection block, and the 1,600-game broad gate. It completed 6,240 fresh
  league games total. Formal decision:
  `recurrent_scale_learning_contract_failed_stop_outcome_only_branch`.
  Do not accept either `069` checkpoint, run the missing blocks, relax the
  entropy floor, promote over raw `035`, or repeat this outcome-only recurrent
  recipe. Champion `035` remains frozen. The next step must be a project-level
  direction reassessment, not another recurrent scale or confirmation run.
- Evidence: `outputs/recurrent_scale_or_stop_069/result.json`, SHA-256
  `0970851f468ffae65ac68846a8aba439f27b7815985915c0f69542c86e7c8f3f`.

## 2026-08-02 Fresh Seat-0 Confirmation Is Neutral but Misses CI Floor

- Experiment `068` was preregistered at `7716d12`, implemented at `7a580c0`,
  and run CPU-only after the full suite passed `558/558`. It performed no
  training, selection, adaptation, or seed replacement. Generation 2 and its
  epoch-zero control each played recurrent seat 0 on 100 fresh matched seeds
  against each of raw `035`, `016`, clean `023`, and shallow search: 400
  paired-seed clusters and 800 games.
- All 11 integrity gates passed. The opponent counts and fresh seed bands were
  exact, recurrent occupied seat 0 in every game, all 800 games completed
  normally, metrics recomputed exactly, and every checkpoint hash remained
  frozen. There were no illegal actions or excluded stalled games.
- The result was practically neutral: candidate score share was `0.60000`
  versus control `0.60125`, delta `-0.00125`. The opponent-stratified paired
  CI95 was `[-0.06000,+0.05625]`, and the score sign-flip p-value was `1.0`.
  Mean paired margin favored generation 2 by `+0.31`, CI95
  `[-0.49,+1.1075]`, with p=`0.450`.
- Opponent score deltas were `-0.03` against `016`, `+0.02` against clean
  `023`, `-0.03` against raw `035`, and `+0.04` against shallow. Every point
  estimate stayed above the `-0.05` practical floor and no opponent interval
  showed systematic collapse. Even/odd halves were `+0.0025/-0.0050`.
  Removing any opponent left the aggregate between `-0.01333` and `+0.00833`;
  removing any single seed left it between `-0.00376` and `+0.00125`.
- Seven of eight strength gates passed. The sole failure was the requirement
  that the score CI lower bound reach `-0.05`: it reached `-0.06`, missing by
  `0.01`. The score mean, margin mean and interval, opponent safeguards, and
  both split halves all passed. This does not establish weakness; it also does
  not establish the prospectively required noninferiority precision.
- Formal decision: `recurrent_seat0_confirmation_failed_no_scale`. Therefore
  do not retroactively pass `066`, promote generation 2, begin Stage C, rerun
  until favorable, or rent a GPU. Champion `035` remains frozen. Any further
  use of the disjoint accumulated panels must be a separately preregistered
  no-game synthesis or a genuinely new direction, not reinterpretation of
  this failed conjunction.
- Evidence: `outputs/recurrent_seat0_confirmation_068/result.json`, SHA-256
  `d4d366e6e77de3af9dba78f51b2341442198ca3752dd464361b1888e7d81d9b1`.

## 2026-08-02 Seat-0 Failure Does Not Replicate Across Panels

- Experiment `067` was preregistered at `9173449` and implemented at
  `7d6f09a`. The first invocation stopped before any bootstrap calculation or
  result file because the runner expected the `065` replay population under
  the wrong JSON key. Commit `e316768` repaired that schema-only binding and
  added a regression test; the corrected full suite passed `552/552` before
  evidence. The audit loaded no model, played no game, and performed no
  training.
- All integrity conditions passed. The audit exactly recomputed aggregate,
  opponent, seat, and alternating-half metrics from 40 disjoint `065` and 200
  disjoint `066` paired-seed clusters, with all 240 unique panel/opponent/seed
  keys and eight exact panel/opponent bootstrap strata.
- The `066` seat-0 miss was not stable when the panels were pooled. Seat-0
  score delta was `-0.03125`, inside the `-0.05` practical floor, with
  clustered CI95 `[-0.10625,+0.04375]`; its margin delta was `+0.1208`, CI95
  `[-0.95,+1.1917]`. It is neither proven noninferior nor systematically
  below the floor.
- Pooled seat 1 was stronger: score delta `+0.05833`, CI95
  `[-0.01875,+0.13542]`, which proves the frozen `-0.05` noninferiority bound;
  margin delta was `+0.80`, CI95 `[-0.2333,+1.8333]`. The pooled score
  interaction `seat1-seat0` was `+0.08958`, but CI95
  `[-0.02083,+0.20]` included zero, so there is no systematic overall seat
  interaction.
- Both point estimates reversed sign across panels: `065` seat 0/1 were
  `+0.10/-0.125`, while `066` was `-0.0575/+0.095`. The seat-0 panel
  difference CI included zero; the seat-1 panel difference did not, confirming
  cross-panel heterogeneity rather than a persistent disadvantage for one
  fixed seat. By opponent, only `016` seat 0 was below the floor in point
  estimate (`-0.075`), but its CI95 `[-0.25,+0.075]` was not systematic.
- Formal decision:
  `recurrent_seat_failure_not_replicated_authorize_targeted_confirmation_preregistration`.
  This does not pass `066`, promote generation 2, authorize Stage C, or relax
  the seat gate. It authorizes only a separately frozen fresh seat-0-focused
  matched confirmation. The most direct design is candidate versus epoch-zero
  control, both always occupying recurrent seat 0, on 100 new seeds per each
  of the same four opponents (800 games total), with opponent-stratified seed
  clustering and a primary seat-0 `-0.05` noninferiority gate. CPU remains
  appropriate; no GPU or retraining is justified.
- Evidence:
  `outputs/recurrent_seat_heterogeneity_audit_067/result.json`, SHA-256
  `03bb2d7386ab71d93302d02f4da266859c7e472b7facefeaa94f81c02c264662`.

## 2026-08-02 Larger Confirmation Is Positive Overall but Misses Seat Floor

- Experiment `066` was preregistered at `569e458`, implemented at `7841aff`,
  and run CPU-only after the full suite passed `546/546`. It performed no
  training or model selection. The panel used 50 fresh paired seeds against
  each of raw `035`, `016`, clean `023`, and shallow search, with candidate
  and epoch-zero control in both recurrent seats: 200 seed clusters and 800
  games total.
- All integrity gates passed. Every seed band and opponent block was exact,
  all 800 games ended validly, all score metrics recomputed exactly, checkpoint
  hashes were unchanged, and there were no illegal actions or replacements.
- Generation 2 was favorable overall. Candidate score share was `0.59875`
  versus control `0.58000`, delta `+0.01875`. The opponent-stratified paired
  CI95 was `[-0.03875,+0.07625]`, above the frozen `-0.05`
  noninferiority floor but not strong-positive evidence; sign-flip p=`0.554`.
  Mean paired margin delta was `+0.61`, CI95 `[-0.1575,+1.385]`, with
  p=`0.124`. Both aggregate point and margin gates passed.
- Every opponent was nonnegative: `016 +0.010`, clean `023 +0.015`, raw
  `035 +0.050`, and shallow `0.000`. All opponent-specific intervals included
  zero, but no point estimate breached the `-0.05` guardrail. Removing any
  one opponent left aggregate score delta positive (`+0.0083` to `+0.0250`).
  Even/odd seed halves were `+0.0575` and `-0.0200`, both above the frozen
  floor. Leave-one-seed-out aggregate deltas stayed positive.
- The only failed strength gate was recurrent-seat noninferiority. Seat 0
  score delta was `-0.0575`, only `0.0075` below its frozen `-0.05` floor;
  seat 1 was `+0.0950`. Seat-0 paired margin was nevertheless slightly
  positive (`+0.07`), while seat 1 was `+1.15`. Moreover, the earlier
  independent `065` panel showed the opposite score split (seat 0 `+0.10`,
  seat 1 `-0.125`), so the new seat asymmetry is not yet a stable directional
  finding. It still cannot be waived after evidence.
- Seven of eight frozen strength gates passed, but the conjunction did not.
  Formal decision: `recurrent_broad_confirmation_failed_no_scale`.
  Do not retroactively pass `066`, promote generation 2, replace champion
  `035`, start Stage C, or rent a GPU. The recurrent direction remains
  promising rather than disproved: the next justified step is a no-training,
  preregistered seat-stratified uncertainty/heterogeneity audit pooling the
  disjoint `065` and `066` panels. That audit should decide whether a bounded
  seat-targeted confirmation is warranted or the succession path remains too
  unstable to scale.
- Evidence:
  `outputs/recurrent_broad_confirmation_066/result.json`, SHA-256
  `dbbdedbb74b59ea5d0d6562eced9c08afdd43795107e2e3567e9dddb60ff162c`.

## 2026-08-02 Broad-Gate Failure Is Inconclusive, Not Systematic

- Experiment `065` was preregistered at `c26f207`, implemented at `af1aaaf`,
  and run CPU-only after the full suite passed `540/540`. It performed no
  training and used no fresh seeds. An instrumented wrapper replayed the exact
  160 final-gate games from `064` while evaluating frozen generation 2 and
  epoch-zero `035` on every recurrent decision.
- Integrity passed completely. All 160 endings, winners, scores, turn counts,
  termination reasons, and loop fields matched `064` exactly after normalizing
  only the tracing wrapper class name. All population score shares and deltas
  reproduced within `1e-12`; there were no new seeds and no model mutation.
- The apparent broad loss is far smaller than its uncertainty. The aggregate
  candidate-minus-control score-share delta remained `-0.0125`, but its
  opponent-stratified paired-seed bootstrap CI95 was
  `[-0.1375, +0.1125]`; the paired margin delta was `-0.2875`, CI95
  `[-1.8375, +1.2375]`. Sign-flip p-values were `1.0` and `0.734`.
- No opponent loss was systematic. Score-delta CI95 values were:
  `016 [-0.40,+0.15]`, clean `023 [-0.25,+0.15]`, raw
  `035 [-0.15,+0.30]`, and shallow `[-0.25,+0.25]`. Even/odd halves were
  `-0.025` and `0.0`; leave-one-opponent-out deltas ranged from `-0.05` to
  `+0.0167`. Seat 0 favored generation 2 by `+0.10`, while seat 1 favored
  epoch zero by `-0.125`; this is descriptive at the current sample size.
- Across 5,121 recurrent decisions in 160 trajectories, the two policies
  disagreed on 371 (`7.24%` decision-weighted, `7.33%` game-balanced), with
  game-balanced mean raw-softmax TV `0.0471`. Movement was not opponent
  concentrated: every opponent's game-balanced disagreement rate was about
  7--8%. It was phase concentrated: early `11.8%`, mid `5.8%`, late `3.7%`.
  Negative paired-result seeds had lower, not higher, movement than positive
  or tied seeds (`6%` versus about `8%`), so disagreement frequency does not
  explain the observed losses.
- Most greedy disagreements were fine-grained: 251/371 retained the same
  action type, and another 89 switched between visible and blind reserve.
  Only 31 changed across broader take/buy/reserve categories. This does not
  prove action equivalence, but it further cautions against interpreting the
  one-game aggregate score difference as a robust policy failure.
- Formal decision:
  `recurrent_broad_result_inconclusive_authorize_confirmation_preregistration`.
  This does not retroactively pass `064`; keep `035` as champion. It authorizes
  only a separately frozen, larger fresh paired confirmation of generation 2
  versus epoch zero across the same four opponents, with both seats and seed
  clustering. Do not train, promote, run Stage C, change the objective, or
  rent a GPU before that confirmation.
- Evidence:
  `outputs/recurrent_broad_failure_diagnosis_065/result.json`, SHA-256
  `616e0f6eb0ff7b968cfc0f9d60c51c2c186557c179c07a52586b978d784e0b4d`.

## 2026-07-26 History-Required Repair Composes Twice but Fails Broad Lift

- Experiment `064` was preregistered at `266edb0`, implemented at `3ab2545`,
  and run CPU-only after the clean full suite passed `532/532`. It replaced the
  shortcut-prone residual diagnosed in `063` with a multiplicative
  action/history interaction. Both the history projection and residual output
  are bias-free, so an empty transcript produces bit-exact raw `035` behavior
  even after training. Early turns received one-quarter of the per-decision
  weight used for turns 20+.
- Generation 1 collected exactly 320/320 fresh balanced eligible games and
  10,070 actor decisions, with 32 training and 8 validation games in every
  opponent-seat cell. It passed every offline gate: validation raw KL
  `0.003158`, argmax movement `3.45%`, surrogate `+0.003788`, mid/late
  surrogate `+0.004217`, and real-versus-rotated-history TV `0.014956`.
  Empty-history fallback remained exact. It passed its untouched 80-game
  immediate-parent gate at 55.0% score share and `+0.675` mean margin.
- Generation 2 added the accepted generation-1 snapshot and collected exactly
  400/400 fresh balanced eligible games and 12,602 decisions, again with 32/8
  games per cell. It also passed every offline gate: validation raw KL
  `0.003827`, movement `3.42%`, surrogate `+0.001396`, mid/late surrogate
  `+0.001679`, and history-rotation TV `0.022107`. Empty-history fallback was
  still exact. Its untouched parent gate passed exactly at the frozen 45.0%
  score-share floor, with mean margin `-0.7125`.
- The conditional 160-game broad gate was fully valid but did not show
  population improvement over epoch-zero `035`. Generation 2's score-share
  deltas were `+0.10` versus raw `035`, `0.00` versus shallow search, `-0.05`
  versus clean `023`, and `-0.10` versus `016`. Two opponents were
  nonnegative and the inclusive no-collapse floor passed, but aggregate delta
  was `-0.0125`, failing the preregistered strictly-positive requirement.
- The original report represented exact `-0.10` as
  `-0.10000000000000003`, incorrectly failing the inclusive no-collapse
  sub-gate. Commit `e6f51ec` added a `1e-12` comparison tolerance and the
  result was transparently reserialized with the original result hash in its
  correction note. Games, checkpoints, metrics, the failed aggregate gate,
  and the formal decision were unchanged.
- Formal decision:
  `history_required_repair_failed_diagnose_before_scale`. The repair solved
  the history-use and two-generation composition failures of `062`, which is
  meaningful progress, but it did not establish broad cumulative strength.
  Keep `035` as champion. Do not promote either `064` checkpoint, start the
  10,000-game Stage C run, relax the broad gate, or rent a GPU. The next
  justified step is a no-training diagnostic of the broad result: quantify
  paired-seed uncertainty and localize the `016`/`023` losses before deciding
  whether a larger confirmation gate or a league-objective change is warranted.
- Evidence:
  `outputs/history_required_recurrent_repair_064/result.json`, SHA-256
  `7a3094d37e6d37961b44f74029330c7083f5da3af7a9a03f8050b62bc59cf2e6`.
  Generation-1 and generation-2 checkpoint SHA-256 values are respectively
  `0c42c04df97114bf79147dfee80e8f296485b15742d6248c8f37ff7cee78756e`
  and `3ed0bd7e2dcda21646c7ac4f68e2d0ffcfe0bdacf0c151c453c52ea6a08fe09e`.

## 2026-07-25 Generation-2 Diagnosis: Noisy Credit, History Mostly Unused

- `063` was preregistered at `b3f0fb5`, implemented at `b762509`, and run only
  after the full suite passed `527/527`. It performed no training and no new
  gameplay. The exact generation-2 corpus was reconstructed from the original
  seeds: all 240 game summaries and the complete collection summary matched
  `062` exactly, and behavior reconstruction error remained below `1e-5`.
- The update strongly fit its 200 training games: game-balanced surrogate
  `+0.007162`, with 66.5% of games contributing positively. On the 40-game
  validation split it reversed to `-0.000659`, but the game-cluster CI95 was
  `[-0.007528, +0.005839]`. Alternating within-cell halves disagreed:
  `+0.000854` versus `-0.002172`. This is a real train/validation reversal but
  not evidence of a systematically negative population effect.
- The failure was localized. Early decisions had surrogate `-0.014283`, while
  mid and late were positive at `+0.005861` and `+0.005795`. The largest
  negative opponent/seat cells were clean `023` seat 0 (`-0.014801`) and raw
  `035` seat 1 (`-0.014424`), each based on only four validation games. Both
  actor seats were essentially equally weak overall, so this is not a global
  seat bug.
- The 17-23 parent gameplay result was also uncertain rather than decisive.
  Paired-seed bootstrap CI95 for candidate share was `[0.275, 0.575]`; paired
  margin CI95 was `[-2.475, +0.875]`. It still correctly failed the frozen
  `062` gate and cannot be retroactively accepted, but it does not establish
  systematic inferiority.
- The intended recurrent mechanism was barely used. Replacing every validation
  transcript with empty history retained 95.84% of real-history KL
  (`0.003283` versus `0.003426`); chosen-action log-ratio correlation was
  `0.9968`, and only 4.16% of movement disappeared. The residual learned
  almost entirely from frozen current-state/action features.
- Successive directions did not compose. Parameter-delta cosine was `+0.0348`
  overall and generation-1/generation-2 centered-logit cosine was `-0.0016`:
  essentially orthogonal, not a strong global inversion. Midgame logit cosine
  was specifically negative at `-0.1107`; early/late were mildly positive.
- Diagnosis: the second generation overfit high-variance whole-game credit,
  especially the weakest-credit early decisions, while taking an almost
  history-independent direction. The small validation/gameplay gates detected
  that unreliability but are too uncertain to claim true regression.
  Population self-play is not disproved; this exact state-heavy recurrent
  residual plus uniform whole-trajectory credit is.
- Formal decision:
  `generation2_failure_diagnostic_complete_no_scale_authorized`.
  Do not rerun `062`, enlarge it directly, relax its gates, or rent a GPU.
  Before another succession pilot, structurally require policy movement to
  pass through history and reduce early-decision credit dominance; use a fresh
  preregistered corpus and larger per-cell validation, not the `062` outcomes
  for selection.
- Evidence:
  `outputs/recurrent_generation2_diagnosis_063/result.json`, SHA-256
  `ee6b5275ff012081b7bce97637ccd64fd1bae25ae74e07c434f71915c6882a73`.

## 2026-07-25 Recurrent Stage B Accepts One Generation, Rejects the Second

- `062` was preregistered at `c0a9988` and implemented at `ced5257`; the full
  pre-evidence suite passed `523/523`. The first invocation exposed that the
  protocol had attached the legacy attention `023` hash to the intended clean
  canonical `023`. It stopped during opponent construction, before collection,
  training, or evidence. The exact historical canonical path/hash was repaired
  prospectively at `ad361ca`; focused checks passed before restarting from the
  original seeds.
- Generation 1 collected exactly 240/240 balanced valid games and 7,536 actor
  decisions, with no rejected attempts, cutoffs, stalls, or illegal actions.
  Its update passed every offline gate: validation raw KL `0.004530`, argmax
  movement `4.50%`, surrogate `+0.003656`, normalized entropy `0.3864`, exact
  behavior reconstruction and checkpoint reload. It then beat its epoch-zero
  recurrent/`035` parent 22-18 on the untouched 40-game gate: 55.0% score
  share and `+1.2` mean paired score margin. Generation 1 was accepted as an
  experimental historical snapshot, not promoted.
- Generation 2 added generation 1 to `035`, `016`, clean `023`, and legal
  shallow search. It again collected exactly 240/240 balanced valid games and
  7,489 decisions with no integrity failures. The update remained bounded and
  meaningful in size: raw KL `0.003426`, argmax movement `3.47%`, entropy
  `0.3683`, and exact reload.
- Generation 2 failed two independent quality gates. Its untouched validation
  surrogate was negative (`-0.000659`), and it lost the immediate-parent
  gameplay gate 17-23: 42.5% score share below the frozen 45% floor. Mean
  paired margin was `-0.8`, which passed the `-2.0` guardrail, but cannot
  override the two failures.
- The runner correctly stopped before generation 3 and the final static
  population gate. Formal decision:
  `recurrent_stage_b_generation_failed_diagnose_before_scale`.
  This is partial evidence that the recurrent population path can produce one
  retained local update, but not yet a reliable succession process. Do not
  run the 10,000-game Stage C block, promote generation 1, rerun generation 2
  on new seeds, relax the gates, or rent a GPU. The current blocker is
  cross-generation learning-signal reliability rather than engineering or
  throughput.
- Evidence is in `outputs/recurrent_league_stage_b_062/result.json`, SHA-256
  `2c4d245eb683d2008f84a3ac705c1fa684817e0fac3b7b000b07d7112d7eeb5d`.
  Accepted generation 1 SHA-256 is
  `e1805b4ff76cbe43f8ff19f3d38e94ef2ced87713109a451c9c249006945c485`.
  Rejected generation 2 SHA-256 is
  `b2d4f85b126b469eb70931a92cb9c808c44ceb743eb640803f97495530373d10`.

## 2026-07-25 Recurrent League Stage A Passes

- `061` was preregistered at `c68f3f0`, implemented at `603713f`, and run
  CPU-only after the corrected full suite passed `520/520`. The initial
  strategic checkpoint was `133d75f`.
- The new model freezes every `035` parameter and adds a legal public-history
  GRU plus an action-conditional residual whose final layer is initialized to
  exact zero. Match lifecycle hooks now reset bots at game start and publish
  every legal transition to both bots; bots without hooks are unchanged.
- All frozen Stage A integrity conditions passed. Across 510 retained decisions
  from seeds `610000--610007`, recurrent scores, values, and greedy choices
  exactly matched `035`. The second clean run reproduced every trace, ending,
  turn count, winner, and score exactly. All eight games completed normally.
- Full-history and event-by-event GRU states agreed to maximum absolute error
  `4.0233e-07`; padded-batch and single-row choices agreed; saving and reloading
  the checkpoint was exact. All 37 blind-reserve events exposed the tier but
  no card identity, and observer-relative encodings passed.
- Batched CPU inference measured 2,544 decisions/second versus 4,504 for the
  parent, a `1.7705x` cost ratio below the frozen `3.0x` feasibility ceiling.
  This is not yet a collection-time estimate, but establishes that recurrence
  is not an engineering or GPU blocker.
- Formal decision:
  `recurrent_stage_a_feasible_authorize_stage_b_preregistration`.
  No training occurred and this does not authorize the 10,000-game scale run
  or promotion. The next step is a separately frozen, small synchronous
  multi-generation Stage B learning pilot with historical league snapshots,
  parent-relative update gates, opponent/seat balancing, and complete
  trajectory score-margin/WDL targets. Keep the resolver out of collection.
- Evidence is in `outputs/recurrent_league_stage_a_061/result.json`, SHA-256
  `1b1469fca9b7abcc22525e6f57a2fcdf9091863efe80bbb5e862f6deb1665f08`.
  The epoch-zero checkpoint SHA-256 is
  `2d807231666420bb221049aa7e2a4a28358ef62618cf1c652ede67dd7249acef`.

## 2026-07-25 Strategic Reset to Recurrent Population Self-Play

- The post-`060` project review keeps the initial roadmap but identifies the
  exact stalled boundary: the legal simulator, baselines, and warm-start policy
  are established, while no reproducible search/action teacher has earned the
  transition from warm start to iterative self-play.
- Expert annotation was considered and rejected before implementation. The
  user played the ten recorded human games seriously and beat `035` 8-2, but
  does not know their population-relative strength, is not confident that
  isolated public positions admit reliable expert action labels, and cannot
  justify the time needed to annotate 100-150 states. Do not build a manual
  labeling workflow or treat these games as a training corpus.
- The next direction is a bounded history-aware population self-play program,
  not another resolver, PUCT, value, AWR, action-preference, or matched-rollout
  variant. Add a zero-initialized recurrent public-history residual so epoch
  zero reproduces `035`; train from complete on-policy league trajectories
  using the validated final-score-margin signal and WDL, with advantages
  balanced by opponent and seat.
- Unlike previous 36-120-game actor experiments and one-generation 2,048-group
  AWR updates, the planned pilot must use multiple synchronous generations,
  historical snapshots as league members, parent-relative rather than
  permanent-`035` update constraints, and broad population progression gates.
  This tests cumulative distribution-shift adaptation rather than a single
  local correction.
- Initial scale target is approximately 10,000 ordinary neural/baseline games.
  Measured `032` collection throughput projects about 28 minutes for comparable
  game simulation, but recurrent CPU training makes 3-6 hours the realistic
  no-resolver end-to-end estimate. Resolver games are far more expensive:
  keep the resolver out of training collection and reserve only 20-40 games
  for a final conditional evaluation.
- Stage A is authorized now: implement and run a CPU-only feasibility smoke for
  legal public-history construction, exact zero-residual reproduction of
  `035`, recurrent state lifecycle/checkpointing, batched trajectory handling,
  determinism, and measured throughput. It performs no policy improvement,
  promotion, or 10,000-game collection. GPU rental remains unnecessary until
  Stage A validates the pipeline and quantifies the actual bottleneck.

## 2026-07-25 Public-Belief Visit Targets Are Prior-Stable but Value-Unstable

- `060` prospectively froze the first small public-belief visit-search
  diagnostic at `a4e37ee` and implemented it at `500df1e`. Focused contracts
  passed `6/6`, then the complete suite passed `515/515` before evidence.
- It reused the exact outcome-blind `059` selection: 48 states from 17 games,
  balanced as 24 early, 12 mid, and 12 late. Each state received independent
  left/right searches with 12 information-legal determinizations, 192
  simulations, information-set-keyed opponent nodes, frozen `035` priors,
  and one matched score-margin continuation per simulation.
- Integrity passed completely. There were exactly 96 attempted searches;
  95 completed and 47/48 states had both arms complete. One midgame arm was
  substantively ineligible because a revisited opponent information set had
  a different canonical action set or prior. All 18,272 attempted
  continuations resolved, with zero hard caps and two legitimate
  loop-adjudicated endings. Root counts and 12-particle schedules were exact,
  and both first-state searches repeated identically.
- The visit distributions looked superficially stable: mean raw/collapsed
  left-right JSD was `0.02538`/`0.02533`, halfway-to-final JSD was `0.00978`,
  and early top-visit agreement was 83.33%. Search moved meaningfully from
  the parent prior: mean prior-to-search TV was `0.11833`, 25% of states
  changed the prior-top action on at least one arm, and 16.84% of completed
  arms changed it.
- That agreement was not supported by stable action values. Overall
  value-equivalent agreement was only 31.91%, mean Q Spearman `0.3370`,
  mean cross-regret `0.04610`, and p90 regret `0.11899`. Early was decisive:
  equivalence `12.5%`, Q Spearman `0.175`, mean regret `0.06379`, and p90
  regret `0.12419`, versus frozen requirements of 65%, `0.50`, `0.030`, and
  `0.075`. Mid equivalence was also only 36.36%; late alone was acceptable at
  66.67%.
- High exact top-visit agreement is therefore a prior-locking effect, not a
  reliable teacher. Independent-search JSD was only `0.00147` lower than
  average search-to-prior JSD, far short of the required `0.01` separation.
  Compared with frozen `059` `12x1`, early exact agreement rose 37.5 points,
  but equivalence fell 33.33 points and mean regret worsened by `0.02586`.
- Formal decision:
  `public_belief_visit_targets_unstable_pause_search_teacher_branch`.
  No full-search pilot, corpus collection, training, gameplay, promotion, or
  GPU rental is authorized. Do not enlarge this shallow visit search, tune
  PUCT constants, relax value-equivalence gates, or distill these visits.
  The next step is a project-level branch reassessment, not another
  search-teacher parameter experiment.
- Primary evaluation took 268.48 seconds wall time on ten CPU workers
  (2,416.34 aggregate process CPU seconds). Evidence is in
  `outputs/public_belief_visit_search_060/result.json`; SHA-256 is
  `8fcc6ea2a511e97171ea4835a130035fe69fbb3fd72891f54b99794ddc2c606c`.

## 2026-07-25 Belief Breadth Helps Values but Not Direct Action Labels

- `059` prospectively froze a compute-matched breadth audit at `e753b4e` and
  implemented it at `b609506`. Focused contracts passed, then the full suite
  passed `509/509` before evidence.
- It compared paired `3 determinizations x 4 continuations/leaf`, `6 x 2`,
  and `12 x 1` estimates, each exactly 192 continuations, on all 24 early
  states plus 12 mid and 12 late guardrails from the outcome-blind `055`
  selection. Belief particles were nested and the same twelve continuation
  RNG slots were paired across configurations.
- All 288 estimates and all 55,296 continuations completed, with zero
  structural fallback, hard caps, or loop adjudications. All three first-state
  left estimates repeated identically and every integrity condition passed.
- `12x1` was clearly the best allocation but not a stable direct teacher.
  On early states it improved exact agreement only from 41.67% to 45.83% and
  left equivalence unchanged at 45.83%; paired bootstrap intervals included
  no gain. Spearman improved from `0.124` to `0.318` but missed `0.50`.
  Mean regret fell from `0.06997` to `0.03793` (ratio `0.542`) with the paired
  regret-change CI entirely below zero, but still missed the `0.035` target;
  p90 was `0.07917` versus the `0.075` target.
- `6x2` was worse early: 20.83% exact, 25.0% equivalent, and `0.07865` mean
  regret. The benefit appears only when belief breadth reaches 12 and each
  leaf rollout is no longer repeated.
- `12x1` passed every structural and mid/late guardrail. Versus `3x4`, mid
  exact/equivalence rose from 8.33%/16.67% to 33.33%/33.33% and regret fell
  from `0.10330` to `0.04358`; late rose from 66.67%/75.0% to
  75.0%/83.33% and regret fell from `0.02708` to `0.00851`. This confirms that
  broader belief sampling improves value quality generally, even though a
  192-simulation shallow hard-choice target remains unstable.
- Formal decision:
  `belief_breadth_reallocation_insufficient_move_to_public_belief_visit_search`.
  Neither `12x1` nor `6x2` authorized a resolver pilot. No corpus, training,
  gameplay, promotion, or GPU rental occurred.
- Do not try `24x1`, relax the direct-action gate, or distill `12x1` scalar
  winners. The next justified experiment is a small preregistered
  public-belief visit-search diagnostic using the successful 12-particle
  allocation: select root/opponent branches adaptively with information-set-
  keyed UCB/PUCT, one rollout per simulation, and return complete root visit
  distributions. Compare independent visit distributions and their
  value-weighted action equivalence on the same balanced phases before any
  gameplay or training.
- Primary collection took 1,371.80 seconds wall time on ten CPU workers
  (9,209.50 aggregate process CPU seconds), plus three deterministic repeats.
  Evidence is in
  `outputs/compute_matched_belief_breadth_059/result.json`; SHA-256 is
  `14dad95d3e41cf2b2f108a44177fe237fa12d84deddad90e58ae6acaf7cc0d47`.

## 2026-07-25 Early Resolver Variance Is Belief-Sampling Dominated

- `058` prospectively froze an early-state crossed variance decomposition at
  `ecc2fd3` and implemented it at `17e9852`. Focused contracts passed, then
  the full suite passed `503/503` before evidence.
- It used all 24 outcome-blind early `055` states, exactly four in each
  seat x original-gap cell. Each state crossed four independent
  three-determinization belief batches with four paired continuation
  replicates, two continuations per leaf: exactly 36,864 primary
  continuations. All states/cells completed, every continuation resolved,
  there were zero hard caps or structural failures, and the first full
  1,536-continuation state repeated byte-identically. Every integrity
  condition passed.
- The hard-min two-way decomposition attributed 61.74% of early root-value
  variance to the belief-batch main effect, only 9.23% to continuation
  randomness, and 29.03% to interaction. Belief dominance held in both seats
  (`60.99%`/`62.73%`) and every gap stratum (`57.16%`-`67.42%`).
- Split stability makes the effect concrete. Changing belief halves after
  averaging continuation conditions produced only 20.83% exact and 25.0%
  value-equivalent agreement, Spearman `0.2183`, and mean regret `0.05994`.
  Changing continuation halves while averaging all belief batches produced
  62.5% exact and 75.0% equivalent agreement, Spearman `0.6028`, and mean
  regret `0.01029`. Belief-split regret was 5.83 times continuation-split
  regret.
- Opponent hard-min responses did switch heavily: every state/root switched,
  with mean modal deviation 75.52%. But a `tau=0.05` smooth-min did not cure
  the belief split: exact fell from 20.83% to 16.67%, equivalence from 25.0%
  to 16.67%, and regret slightly increased. Across both split axes,
  smooth-min exact gained only 4.17 points and regret ratio was `0.976`, so
  minimizer switching is primarily a symptom of changing belief samples, not
  the main cause.
- Formal decision:
  `belief_sampling_primary_design_broader_public_belief_search`. No corpus,
  training, gameplay, promotion, or GPU rental was authorized.
- Do not add more rollouts to the same three determinizations or tune the hard
  minimum. The next justified experiment is a compute-matched belief-breadth
  allocation audit at fixed 192-continuation budget: compare the current
  `3 determinizations x 4 continuations/leaf` with `6 x 2` and `12 x 1`,
  pooling all determinizations inside one information-set-consistent solve.
  Run independent paired estimates on balanced early states and require
  materially better early ordering/equivalence without mid/late regression.
  If `12 x 1` still fails, move to public-belief IS-MCTS/visit distributions
  rather than increasing direct-resolver budget.
- Primary collection took 1,412.04 seconds wall time on ten CPU workers
  (8,960.70 aggregate process CPU seconds), plus the deterministic repeat.
  Evidence is in
  `outputs/early_resolver_variance_decomposition_058/result.json`; SHA-256 is
  `9369c7a6ace8cacd0907fa4dd6d2dd72a48495aae3b09725cd5664d0ffa74a64`.

## 2026-07-25 Adaptive Resolver Allocation Failed

- `057` prospectively froze the adaptive-teacher audit at `6c1a0ea` and
  implemented it at `7f185bd`. Focused contracts passed, then the full suite
  passed `497/497` before evidence.
- It selected the 36 balanced `055` states not used by `056`, exactly two in
  every seat x phase x original-gap cell, spanning at least 15 games. Frozen
  A/B were reused and C-P used random namespaces disjoint from `056`. All 504
  new searches and all 96,768 continuations resolved with zero hard caps,
  loop adjudications, structural failures, or illegal actions. The first C
  estimate repeated byte-identically and every integrity condition passed.
- The preregistered confidence allocator used a mean 20.97 root units and
  1,006.67 continuations per arm, versus 32 units/1,536 continuations for
  uniform 8x: a 34.46% compute saving. It stopped on a stable confidence set
  in 59/72 arms. Mean set Jaccard was `0.8542` and diagnostic-distribution JSD
  `0.1135`, but those apparently stable sets were often broad: 34/72 arms
  still contained all four actions and only 5/72 contained one.
- Adaptive estimates achieved only 50.0% exact best agreement, 61.11%
  value-equivalent agreement, Spearman `0.4632`, mean cross-regret `0.02157`,
  and p90 regret `0.06615`. Relative to same-state uniform 8x, adaptivity lost
  11.11 exact-agreement points and 5.56 equivalence points, reduced Spearman
  by `0.1062`, and added `0.00919` mean regret. Saving compute by stopping on
  marginal root intervals therefore discarded strategically material
  evidence.
- Same-state uniform 8x was better but also failed: 61.11% exact, 66.67%
  equivalent, Spearman `0.5694`, and mean regret `0.01238`. Most importantly,
  it independently reproduced the early-game collapse: early exact/equivalent
  agreement was 33.33%/41.67%, versus 75.0%/83.33% mid and 75.0%/75.0% late.
  Adaptive early results were 33.33%/41.67%. Thus neither the allocator nor
  the second fresh uniform sample supports a teacher protocol.
- Formal decision: `adaptive_allocation_failed_redesign_search_teacher`.
  Neither point nor equivalence corpus generation was authorized. No
  training, gameplay, promotion, or GPU rental occurred.
- Do not tune the stopping threshold, train against these confidence sets, or
  pay for uniform 16x yet. The next justified experiment is a small,
  preregistered early-state variance-decomposition audit. Hold belief
  determinizations fixed while resampling continuations, then hold paired
  continuation conditions fixed while resampling belief determinizations,
  and record leaf/opponent-minimizer switching. This will distinguish rollout
  noise, hidden-belief sampling, and discontinuous minimax response selection.
  The result should determine whether the replacement teacher needs more
  belief breadth, better continuations, or a visit-distribution/soft-response
  search such as public-belief IS-MCTS. CPU remains appropriate.
- The new matrix took 2,018.26 seconds wall time on ten CPU workers
  (13,499.28 aggregate process CPU seconds). Evidence is in
  `outputs/adaptive_resolver_teacher_audit_057/result.json`; SHA-256 is
  `72d935d5b3e4430a362e365b4e782bcb04aca41c0c3c2ad84549f82e859a4754`.

## 2026-07-24 Resolver Budget Convergence Is Real but Uneven

- `056` prospectively froze a no-training 1x/2x/4x/8x convergence audit at
  `c70a1f7` and implemented it at `aabfb66`. The full suite passed `491/491`
  before evidence.
- The audit selected 36 outcome-blind fixed states, exactly two in every
  seat x phase x original-gap cell, from 15 games. It reused the two frozen
  `055` estimates and generated 14 new independent estimates per state:
  504 searches and 96,768 continuations. Every search and continuation
  resolved, with zero fallback, hard-cap, loop-adjudication, structural, or
  illegal-action events. The first new search repeated byte-identically.
- Averaging produced strong, monotonic convergence. From 1x to 2x/4x/8x,
  exact split-half best-action agreement rose from 44.10% to
  48.61%/56.94%/63.89%; equivalence within `0.025` regret rose from 44.44% to
  50.69%/59.72%/77.78%; mean Spearman rose from `0.2519` to
  `0.3864`/`0.4688`/`0.6065`; and mean cross-regret fell from `0.06220` to
  `0.03639`/`0.02062`/`0.00909`. Thus 8x reduced regret by 85.4% and gained
  19.79 percentage points of exact agreement.
- The aggregate 8x result was close to a usable teacher, but it still failed
  the preregistered exact-action thresholds: 63.89% agreement versus 70%,
  game-bootstrap lower bound 52.38% versus 60%, and Spearman `0.6065` versus
  `0.65`. More importantly, aggregate gains concealed an early-game collapse.
  At 8x, early/mid/late exact agreement was 25.0%/91.67%/75.0%,
  equivalence was 50.0%/91.67%/91.67%, and Spearman was
  `0.3860`/`0.6167`/`0.8167`. The strict phase floor therefore correctly
  prevented stable mid/late labels from masking unstable early labels.
- Formal decision:
  `budget_convergence_positive_but_8x_insufficient_redesign_teacher`.
  Convergence and regret-reduction gates passed, but neither an exact-action
  teacher nor the equivalence/uncertainty protocol was authorized because of
  the phase failure. No training, corpus collection, gameplay, promotion, or
  GPU rental occurred.
- Do not spend another uniform 16x budget. The next justified experiment is
  a preregistered fixed-state teacher-redesign diagnostic that retains the
  proven resolver payoff machinery but allocates continuation samples
  adaptively across root actions and exposes an uncertainty/equivalence set
  or stabilized visit distribution. It should compare compute-matched
  split-half stability against 4x/8x, with early phase as a primary gate.
  Only a stable and economical teacher would authorize corpus generation and
  distillation.
- The new matrix took 1,887.49 seconds wall time on ten CPU workers
  (aggregate process CPU 13,046.98 seconds). CPU remains appropriate for the
  next bounded diagnostic; a GPU would not accelerate the resolver sampling
  bottleneck.
- Evidence is in
  `outputs/resolver_budget_convergence_audit_056/result.json`; SHA-256 is
  `bc20cde5436d904a4e63a381a8173b000fcc5b66462d71194134bdab77dac081`.

## 2026-07-24 Resolver Target Repeatability Failure

- `055` prospectively froze a no-training repeatability audit at `70eaede`,
  clarified structural-fallback accounting at `8149992`, froze the cell
  coverage floor at `2a33e77`, and implemented it at `6cd6b5e`. The full suite
  passed `486/486` before evidence.
- It selected 72 unique outcome-blind `052` validation states, exactly four in
  every seat x phase x original-value-gap cell, spanning 18 games. Two
  completely independent information-legal resolver namespaces attempted
  every state. All 144 searches completed, all 27,648 continuations resolved,
  and there were zero fallback, hard-cap, loop-adjudication, or illegal-action
  events. The first fresh search repeated byte-identically.
- Fresh A/B values had strong coarse flattened correlation (`0.8175`) but poor
  within-state ordering: mean Spearman `0.3512`, exact best-action agreement
  `45.83%` with game-bootstrap CI95 `[33.33%,58.90%]`, and value MAE `0.1012`.
  Mean symmetric cross-regret was `0.0493` and p90 `0.1304`; only 47.22% of
  states were mutually value-equivalent within `0.025` regret, CI95
  `[35.06%,59.72%]`.
- Original `052` versus fresh A/B exact best agreement was even lower at
  37.50%/38.89%, with Spearman `0.1988`/`0.2560`. This is close to `054`'s
  34.98%/35.42% snapshot/history validation accuracy and strongly suggests
  those students were being judged against a largely irreproducible single
  search realization rather than simply failing to learn a stable label.
- Stability was phase-dependent: fresh exact agreement was 37.5% early,
  37.5% mid, and 62.5% late; symmetric regret was `0.0661`, `0.0681`, and
  `0.0138`. Even the original clear-gap stratum achieved only 54.17% fresh
  best agreement. Across original/A/B, only 19.44% of states had unanimous
  exact best and 22.22% were mutually value-equivalent.
- Formal decision: `resolver_targets_too_noisy_block_direct_distillation`.
  Both the exact-point and equivalence-target gates failed. This does not
  weaken the proven gameplay strength of resolver `051`; it means one
  192-continuation resolver call is not a stable supervised label,
  particularly in early and midgame.
- Do not collect a larger corpus of current-budget exact values, train a
  richer student against them, tune another head, or rent a GPU. If continuing
  the AlphaGo-style teacher branch, the next justified experiment is a
  separately preregistered search-budget convergence audit: average multiple
  independent resolver estimates at fixed states and measure whether
  ordering/regret stabilizes at effective 2x/4x/8x budgets. If stability does
  not rise economically, replace this shallow stochastic-minimax target with
  a search method that produces converged visit distributions rather than
  attempting further direct distillation.
- Evidence is in
  `outputs/resolver_target_repeatability_audit_055/result.json`; SHA-256 is
  `4aa3b54e77a4856c2c5cd1bb18b1a4088755bda7679bd52a7ce58627ba622714`.

## 2026-07-24 Public-History Resolver-Value Representation Failure

- `054` prospectively froze a matched snapshot-versus-public-history
  representation diagnostic at `1224ca6` and implemented it at `470629b`.
  The full suite passed `480/480` before evidence.
- Both arms embedded exact frozen `035` state/action/score features and trained
  the same 89,697 added parameters for exactly 20 epochs on the 4,397 resolved
  `052` training rows. The history arm alone received the validated
  information-legal chronological public-action encoding. Validation was
  reconstructed exactly once after both models and train metrics were fixed,
  using 1,135 resolved rows from the game-disjoint `052` validation split.
- Snapshot reached 41.62% train and 34.98% validation best-group accuracy;
  validation Spearman was `0.1315`, pairwise sign accuracy `55.51%`, and
  four-value MAE `0.17905`. It failed the 60% train, 40% validation, `0.25`
  Spearman, and 60% pairwise learnability gates.
- Public history improved in-sample fit but not generalization: 44.92% train
  and 35.42% validation best-group accuracy, validation Spearman `0.1170`,
  pairwise `54.88%`, and MAE `0.16778`. Its validation accuracy gain was only
  `+0.44` points, game-bootstrap CI95 `[-2.07,+3.01]`; disagreement capture
  improved only `+1.32` points and rank correlation became worse. Early,
  mid, and late validation accuracies were 31.45%, 32.53%, and 49.10%, with no
  catastrophic seat/outcome/phase stratum collapse.
- Formal decision:
  `resolver_targets_not_learnable_with_current_frozen_features`. Integrity
  passed completely. No old confirmation data was opened, no fresh simulation
  or gameplay ran, and no scaled corpus, promotion, or GPU rental is
  authorized. Public history is not the missing discriminator for this frozen
  representation: it fits some extra training detail but adds no stable
  held-out action ordering.
- Do not enlarge this frozen-feature head, add epochs, or collect a scaled
  distillation corpus from this result. If continuing the teacher branch, the
  next justified step is a separately preregistered, no-training repeatability
  audit of resolver action values/rankings on stratified frozen development
  states with independent continuation seeds. That would measure the
  learnable label ceiling before deciding between a richer end-to-end
  representation and abandoning direct resolver-value distillation.
- Evidence is in
  `outputs/resolver_public_history_representation_054/result.json`; SHA-256 is
  `2e5c062e3ed12688ba66a031c1aff368f34cf62d8f9cb21b7c9ed59cdd1a7945`.

## 2026-07-24 Frozen-Parent Resolver Residual Ranking Failure

- `053` prospectively froze one zero-initialized 24,769-parameter residual
  ranking proposal at `1b849b4`, clarified deterministic initialization at
  `4d04845`, and implemented it at `24d2ffb`. Every `035` parameter remained
  frozen; all `474/474` tests passed before evidence.
- Epoch zero reproduced `035` exactly. Across all 12 frozen epochs, no epoch
  met the three selection-anchor conditions, so no checkpoint was selected.
  The runner correctly stopped at `validation_residual_gate_failed`; it did not
  open old `052` confirmation, generate any fresh resolver game, or run
  gameplay.
- Epoch 1 was closest to anchor-safe: validation target-KL ratio improved to
  `0.7504`, 98.17% of disagreement gaps moved correctly, completed agreement
  retention was 98.65%, and fallback argmax retention was 99.07%. But fallback
  KL was `0.02978` versus the `0.02` ceiling, resolver-disagreement capture was
  only `0.65%` versus `15%`, overall resolver-agreement gain was exactly zero
  versus `+5` points, and action movement was only 1.53% versus 5%.
- Later epochs repeated `052`'s KL/argmax split despite the frozen parent and
  explicit anchor. Epoch 12 reduced complete-target KL to 51.30% of parent and
  moved 94.77% of disagreement gaps correctly, but captured only 4.05%,
  improved overall resolver agreement only `+0.264` points, retained only
  92.43% of completed agreement anchors, and produced fallback KL `0.08452`.
  No epoch approached the ranking gate while preserving the anchor.
- Formal interpretation: a small action-specific correction can learn the
  direction and soft shape of resolver supervision, but frozen `035`
  representations do not let it localize large decisive corrections without
  moving non-target states. Do not tune ranking/anchor weights, add epochs,
  loosen gates, or sweep residual capacity from this result. `053` closes the
  cheap residual repair proposed by `052a`; it does not invalidate the strong
  resolver or prove that scaled richer-model distillation is impossible. Raw
  `035` remains fast neural champion and the public-belief resolver remains
  the proven stronger search agent.
- Evidence is in
  `outputs/resolver_residual_ranking_distillation_053/result.json`; SHA-256 is
  `7610c56789cf3491cc889e4f9689957bf06a769fb906f5314731903c933339ef`.

## 2026-07-24 Resolver Distillation Disagreement Audit

- `052a` prospectively froze a post-hoc, no-training audit at `62f1636`,
  implemented it at `a1c1091`, and passed all `467/467` tests before evidence.
  It reproduced every frozen artifact/model hash and every row/model alignment
  check. Result SHA-256 is
  `226d28a4cd9667ef27eacfc1afccf8cac628d07b30006e9dc179ca282b2eba57`.
- On confirmation, the resolver made 731 raw switches among 1,093 resolved
  rows; all 731 were canonically distinct. The blended target retained the
  resolver action for 458 switches, retained the parent for 264, and preferred
  a third action for 9. Thus the 0.75/0.25 blend suppressed 36.1% of raw
  resolver switches, but target conflict is not the main explanation for the
  failed target-top gate.
- Among the 467 confirmation blended-target disagreements, the student moved
  the preferred-versus-parent logit gap in the correct direction on 87.79%
  (game-bootstrap CI95 84.55--90.97%). Mean gap improved from `-2.1826` to
  `-1.0588`, delta `+1.1238` (CI95 `+1.0151,+1.2349`), yet only 8.57% crossed
  zero and only 7.71% captured the preferred argmax (CI95 5.33--10.11%).
- This is not a validation/generalization collapse: preferred-action capture
  was only 5.39% in-sample, 4.07% on validation, and 7.71% on confirmation,
  with correct-direction movement 87.33%, 90.63%, and 87.79%. Both
  confirmation seed halves agreed.
- Parent margins explain the KL/argmax split. 268/467 confirmation target
  disagreements began at gap `<=-1` and captured only 0.4%, whereas the 15
  rows within `-0.05` captured 53.3%. Training broadly compressed gaps rather
  than learning enough action-specific correction to reverse large rankings.
  On fallback rows it raised entropy by `+0.724` nats while producing KL
  `0.186`, consistent with harmful global flattening.
- Formal interpretation: `052`'s global distribution cross-entropy learned
  the direction and soft shape of resolver targets but mostly temperature-
  flattened `035`; it neither transferred decisive rankings nor preserved the
  parent anchor. More epochs or a higher global learning rate are not the
  justified next move. A future prospective experiment should first test a
  zero-initialized, parent-frozen action-specific residual/ranking correction
  on existing development data, with explicit fallback/non-target anchoring
  and a completely fresh confirmation corpus only after validation success.
  No training, gameplay, promotion, new collection, or GPU is authorized by
  `052a`.

## 2026-07-22 Resolver Self-Play Distillation Offline Failure

- `052` protocol/implementation commits are `886a2e2` and `eb4fd2c`; all
  `456/456` tests passed before evidence. The deterministic smoke and complete
  80/20/20 resolver self-play train/validation/confirmation collections passed
  every integrity, privacy, clean-game, branch-accounting, and coverage gate,
  yielding 7,422 acting-player-public decision rows.
- The one frozen warm-start student reduced confirmation resolved-target KL
  from `1.064529` to `0.528818` and stayed within the 5--50% action-movement
  band, but target-top agreement improved only `+0.6404` percentage points
  versus the required `+5`, and fallback-anchor KL was `0.186057` versus the
  required `<=0.02`.
- Formal decision: `offline_distillation_gate_failed`. The student is rejected
  and no paired gameplay ran. Raw `035` remains the fast neural champion; the
  resolver remains the proven stronger full-game teacher. Evidence is under
  `outputs/resolver_selfplay_distillation_052`; result SHA-256 is
  `60458c1bea4328b64f33ad8238529d074a25d1034fa2845e0b339deee3827153`.

## 2026-07-21 Broad Public-Belief Resolver Confirmation Pass

- `051` froze three independent 50-pair broad blocks before gameplay. Protocol,
  prospective 023-hash correction, and implementation commits are `e687649`,
  `7f1e70b`, and `7f26e54`; all `450/450` tests passed before evidence.
- Resolver `035` passed decisively against every frozen opponent:
  - gate-`030` shallow: `74-25-1`, score share `74.5%`, CI95
    `[67.0%,82.0%]`, mean margin `+4.39`, seat shares `76%/73%`;
  - historical `016`: `74-25-1`, share `74.5%`, CI95
    `[65.5%,83.0%]`, margin `+4.06`, seats `74%/75%`;
  - loop-safe canonical `023`: `79-21-0`, share `79.0%`, CI95
    `[70.0%,87.0%]`, margin `+4.46`, seats `84%/74%`.
- All 300 games completed ordinarily with zero invalid termination, timeout,
  stall, or hard cap. All `1,655,616` continuations resolved with exact
  192-branch accounting; only 26 used loop-safe adjudication (`0.00157%`). The
  resolver completed 8,623 searches and switched 5,927 actions, with activity
  in every phase and every opponent block. Every frozen gate passed.
- Formal decision: authorize only a separately preregistered
  resolver-every-move teacher-corpus and distillation experiment. `051` itself
  authorizes no training, corpus generation, neural promotion, or GPU rental.
  The next protocol must store complete root action/value targets from wins and
  losses, alternate fresh synchronous collection with supervised updates, and
  retain raw `035` as the paired promotion control.
- Evidence is in `outputs/broad_public_belief_resolver_confirmation_051`.
  `result.json` SHA-256 is
  `1a731e5edf3c63cffb2b46d8adaebba9fc50b81e490367bb0c8448ef6a885ed4`.

## 2026-07-21 Paired Public-Belief Resolver Gameplay Pass

- `050` deployed the exact `049b` discovery resolver on every structurally
  eligible candidate turn against a fresh raw legal `035`, with paired seats
  over seeds `500100--500199`. Protocol and implementation commits are
  `a3e2aee` and `0588f4e`; the full suite passed `446/446` before evidence.
- All 200 games completed ordinarily with zero invalid termination, timeout,
  stall, hard cap, or model loop fallback. All `1,098,240` continuation
  branches resolved; six loop-safe continuation adjudications were only
  `0.000546%`, well below the frozen `0.1%` ceiling.
- The resolver completed 5,720/6,190 attempted decision searches and changed
  raw `035`'s action 3,868 times. It searched and switched in every phase;
  the 470 structural fallbacks used only preregistered legal reasons.
- Resolver `035` beat raw `035` by `138-62-0`: paired score share `69.0%`,
  bootstrap CI95 `[62.5%,75.0%]`, and mean final-score margin `+3.595`, CI95
  `[+2.70,+4.48]`. Seat shares were `73%` and `65%`. Every integrity,
  activity, strength, confidence, margin, and seat gate passed.
- Formal decision: authorize a separately preregistered broad confirmation
  against fresh gate-`030` information-legal shallow search, historical legal
  scorer `016`, and loop-safe canonical policy `023`. Do not train, distill,
  generate self-play, or rent a GPU before that broad gate. Raw `035` remains
  the checkpoint champion until the broad confirmation passes.
- Evidence is in `outputs/paired_public_belief_resolver_gameplay_050`.
  `result.json` SHA-256 is
  `68060ffc5270b320a185f618c016f76892ae4dbc972623cd70be34cc63dff195`;
  `direct_vs_035.json` SHA-256 is
  `46e141fb8f146842b28ccda3c409ae18209e84f5eca86c5c4083e0a5f919dcd2`.

## 2026-07-20 Eligibility-Safe Bounded Resolver Replication Pass

- `049b` prospectively repaired only `049`'s source contract. Protocol and
  implementation checkpoints are `688d1a0` and `c0c39ce`; the full suite passed
  `441/441` before evidence. All resolver widths, belief semantics, leaf policy,
  budgets, solver rules, and numerical gates remained frozen from `049`.
- Six disjoint seed bands produced exactly 32 valid candidate-bearing `035`
  mirror games per seat/phase cell: 192 attempts, 192 admissions, zero invalid
  terminal, zero missing-phase skip. Selection yielded 48 unique roots, exactly
  eight per cell, with zero structural failure.
- All 18,432 continuations resolved naturally with zero hard cap, loop
  adjudication, fallback, or illegal action. Privacy/action structure passed,
  and a from-zero repeat of the first eight roots was byte-identical.
- Leaf discovery/evaluation correlation was `0.91803`, MAE `0.07371`, and sign
  agreement `92.95%` over 2,127 eligible leaves. Four-action resolved values
  correlated at `0.97532`; best actions agreed on `83.33%` of roots.
- The resolver changed 34/48 roots. Effect signs agreed on 30/32 eligible roots
  (`93.75%`), with 28/48 roots resolved. Untouched mean gain over raw `035` was
  `+0.09601` normalized margin, bootstrap CI95
  `[+0.05842,+0.13498]`. Both seats and all phases were positive; every frozen
  gate passed.
- Formal decision: `049b` authorizes a separately preregistered fresh paired
  resolver-versus-raw-`035` gameplay gate for this exact resolver. It does not
  authorize training or promotion. The next protocol must freeze when resolving
  is invoked, legal fallbacks for structurally ineligible/terminal states,
  matched paired seeds/seats, compute budgets, integrity rules, and strength
  thresholds before gameplay. CPU remains appropriate; no GPU is justified.
  Raw legal `035` remains champion until the gameplay gate passes.
- Evidence is in `outputs/bounded_public_belief_resolving_049b`; result SHA-256
  is `1a539d66b4596e074076319b96f4e0114fd3ce930fb2dc2cc646c973b3a813a7`.

## 2026-07-19 Bounded Public-Belief Resolving Diagnostic Result

- `049` implemented the fixed two-ply Bayesian resolver over three legal
  determinizations, four root actions, and information-set-consistent top-four
  opponent responses. Protocol/implementation checkpoints are `4eb9287` and
  `89fe06e`; all 438 tests passed before evidence.
- All 192 fresh `035` mirror games completed normally. One clean game, seed
  `490083`, ended at turn 50 before assigned seat 1 acted in the frozen late
  (`50+`) phase. It therefore lacked an assigned-phase candidate and failed the
  literal source gate requiring all 192 games to contain one. The outcome-blind
  selector still had enough alternatives and produced 48 unique roots with
  exact eight-state balance in all six seat/phase cells, but `049` cannot be
  retroactively passed.
- Every substantive resolver gate passed. All 18,432 branches resolved naturally
  with zero fallback, adjudication, hard cap, or illegal action. Privacy,
  hidden-order invariance, opponent information-set action stability, and the
  repeated first eight roots passed.
- Discovery/evaluation leaf correlation was `0.9400`, MAE `0.06688`, and sign
  agreement `93.88%` over 2,123 eligible leaves. Four-action resolved-value
  correlation was `0.97486`; best-action agreement was `75%`.
- The resolver changed 38/48 roots. Effect sign agreement was `91.43%` over 35
  eligible roots, with 32/48 resolved. Untouched mean discovery-selected gain
  over raw `035` was `+0.07769` normalized margin, CI95
  `[+0.05104,+0.10408]`. Every seat and phase was positive; all target-quality
  gates passed with room.
- Formal decision: no gameplay or training from `049`. This is an operational
  source-contract failure, not negative resolver evidence. Next preregister one
  fresh `049b` replication with eligibility-safe preallocated seed bands that
  collect exactly 32 candidate-bearing games per seat/phase cell, new 48 roots,
  and a new continuation namespace while freezing every resolver/leaf parameter
  and all target thresholds. A complete replication pass may authorize the
  resolver gameplay gate. No GPU is needed; raw legal `035` remains champion.
- Evidence is in `outputs/bounded_public_belief_resolving_049`; result SHA-256
  is `e7b4742bf04b4f1e0448610847494b41c6558fd2d8aa832e088edf2b14c47cbf`.

## 2026-07-19 Outcome-Sampling CFR Feasibility Result

- `048` implemented an unclipped log-space outcome-sampling regret estimator,
  an explicitly finite full-game learning contract, and deterministic bounded
  advantage memory. Protocol/implementation checkpoints are `d127a4c` and
  `d34ef24`; all 434 tests passed before evidence.
- The estimator is correct on reduced Splendor. Both 200,000-sample fixed-
  uniform audits covered 100% of eligible information sets; mean component
  errors were `0.001851/0.000709`, maxima `0.012526/0.004920`, and earliest-
  information-set maxima `0.003568/0.002707`, all below frozen bounds.
- All three adaptive 100,000-iteration runs passed with `98.54--99.51%` exact
  information-set coverage, zero illegal/unstable sets, low normalized regret,
  and exactly zero current-profile exploitability. Seed `480046` repeated byte-
  identically. This validates the estimator math, not a deployable current
  policy.
- Full-game integrity also passed. All 256 fresh trajectories ended through
  ordinary completion—zero altered-game adjudications—and emitted 7,915 finite
  aligned rows: 2,560 early, 3,840 mid, 1,515 late, with 1--470 legal actions.
  Privacy, exact checkpoint hash, first-16 replay, capacity-512 reservoir repeat,
  and 9.70 MiB serialization all passed.
- The sole failed gate was decisive importance-weight variance. Overall ESS
  ratio was `0.00016263` versus `0.01`; early/mid/late ratios were
  `0.001575/0.001380/0.000850`. P99 absolute weight was `2.92e22` versus `1e6`,
  maximum log weight `72.124` versus `log(1e12)=27.631`, and both split-half
  tail-stability limits narrowly failed. Correct full-game outcome sampling is
  therefore statistically unusable without forbidden clipping/bias.
- Do not add samples, rent a GPU, train on the reservoir, or tune exploration
  after this result. `047` ruled out full terminal external enumeration; `048`
  rules out single-trajectory terminal importance weighting. The next CFR
  branch must be separately preregistered bounded subgame/continual resolving
  with an independently validated, information-legal leaf contract. Start with
  a CPU target-quality diagnostic, not neural fitting or gameplay. Raw legal
  `035` remains champion; BGA remains unused.
- Evidence is in `outputs/outcome_sampling_cfr_feasibility_048`; result SHA-256
  is `69c4a976fa121178269a31145c3aa645a7696177155bbab57ec02e35ddf478ff`.

## 2026-07-19 Full-Game Deep CFR CPU Feasibility Result

- `047` implemented chronological information-legal public-history tensors,
  variable legal-action targets, deterministic Algorithm-R reservoirs, and a
  production-engine terminal external-traversal probe. The preregistration is
  `c944b4f`; all 429 tests pass.
- Representation passed completely. Eight fresh `023` mirror games at seeds
  `470000--470007` all completed normally, yielding 498 rows: 160 early, 240
  mid, and 98 late. Legal-action counts were 2--186. The frozen width-64 `023`
  hash/config loaded exactly and all 63 CPU batches were finite and aligned.
- All privacy and invariance checks passed: hidden-deck permutations were
  invisible, blind reserves hid identity from the opponent and history until
  purchase, cosmetic state/action permutations were stable, and privileged
  entities were rejected. Both capacity-128 reservoirs repeated exactly,
  round-tripped losslessly, and occupied 4,653,242 bytes combined.
- Standard full terminal external traversal failed the frozen feasibility gate.
  Attempts `(470100,p0)`, `(470101,p0)`, and `(470101,p1)` reached the 25,000-
  node cap at depths 130, 170, and 262. `(470100,p1)` reached the invalid
  400-turn limit after 9,929 nodes. No attempt completed its entire tree; all
  correctly emitted zero eligible training rows, with zero illegal actions.
  The repeated first attempt was byte-identical excluding elapsed time.
- This is not a representation, privacy, memory, checkpoint, or CPU-speed
  failure. It is the expected mismatch between enumerating every traverser
  action and Splendor's long, occasionally cyclic full-game tree. Neural fitting
  and GPU rental are not authorized; more external-traversal budget would not
  repair the finite-game/branching contract.
- Next preregister a bounded full-game outcome-sampling CFR feasibility stage.
  It must define an explicit finite-game loop utility before evidence, use
  full-support behavior probabilities and unbiased importance weighting, prove
  correctness against exact `046`, and gate full-game target variance/ESS before
  any neural fit. Do not add a learned value bootstrap. Raw legal `035` remains
  champion; BGA remains unused.
- Evidence is in `outputs/full_game_deep_cfr_feasibility_047`; result SHA-256 is
  `7e59f4b8d91c2b2c3be2c593820706cc4167987aa482469b93570abd9e8dfd44`.

## 2026-07-19 MCCFR Average-Policy Repair

- `046` corrected only the external sampled average-policy rule: accumulate the
  sampled opponent's strategy during the other player's regret pass. It kept
  legacy `044` reproducible and produced identical regrets/current policies for
  matched seeds. Protocol/implementation checkpoints are `e20f249` and
  `368df81`; all 423 pre-evidence tests passed.
- Formal result: pass all ten gates. Five 50,000-iteration seeds had corrected
  average exploitability `0.000336231–0.000470736`, median `0.000383431`, versus
  limits `0.020/0.010`. Worst one-player incentive was `0.000618184` versus
  `0.040`; exact reference exploitability was `0.000177429`.
- Coverage was `95.63–99.03%`, regrets/integrity passed, exact/corrected
  cross-play differed by at most `0.000283`, and filling missing sets changed
  exploitability by zero. Replacing every sub-100-observation strategy with
  exact changed it by at most `0.00006302`, proving rare states no longer carry
  the result. Seed `440046` repeated byte-identically.
- This resolves `045`: CFR regret learning was sound; the legacy average
  accumulator was wrong. A separately preregistered full-game neural Deep CFR
  feasibility pilot is now authorized. Start with CPU traversal/shape/memory
  smoke; rent a GPU only for neural fitting when justified. No gameplay or
  promotion is yet authorized, and raw legal `035` remains champion.
- Evidence is under `outputs/mccfr_average_policy_repair_046`; result SHA-256 is
  `bc025be45cbb8afd1c907f4e74b587a3647f2414ac11e5d5aba55b2223622449`.
  BGA remains a future authorized-data option; none was used.

## 2026-07-18 CFR Strategic-Equivalence Diagnostic

- `045` reconstructed the frozen `044` policies with exact matching hashes and
  added deterministic information-set-consistent best response plus
  reach/value-weighted action analysis. Protocol/implementation checkpoints are
  `d038a47` and `6ac8b4e`; all 419 pre-evidence tests passed.
- Formal result: fail two of ten gates. Exact CFR exploitability was only
  `0.000177429`, but the external stored average policy was `0.293314`, failing
  both its `0.020` absolute gate and `0.015` excess-over-exact gate. Player 0's
  best response achieved value `0.558514` against a profile whose self-play
  value was nearly zero.
- The seven `044` high-TV discrepancies were all harmless: conservative total
  reach-weighted loss was `0.0000171397` for player 0 and zero for player 1.
  Cross-play with exact remained within `0.000523`. Those tests passed because
  equilibrium-like play does not seek the rare vulnerabilities.
- Post-result localization showed the defect is sampled average-policy quality
  in information sets with fewer than 100 observations. Filling missing sets
  changed nothing; replacing only sub-100-observation averages with exact
  reduced exploitability to `0.000401`. The final regret-matched sampled policy
  itself measured `0.0` exploitability on this reduced game.
- Do not abandon CFR and do not scale yet. Preregister a tabular repair of
  reach/sampling-correct average-strategy accumulation (or Deep CFR-style
  strategy memory), replicate across multiple fixed seeds, and require exact
  best-response robustness in rare-observation bands. No GPU until it passes.
- Evidence is under `outputs/cfr_strategic_equivalence_045`; result SHA-256 is
  `528e0b0582d2c39c752e88d0cfb5c82188fba510fa0caba8dd915b281a691701`.
  Raw legal `035` remains champion; no BGA data was used.

## 2026-07-18 Reduced Splendor CFR Viability Gate

- `044` implemented a history-aware information-legal key, a finite four-action
  production-engine Splendor duel, exact alternating CFR, and seeded external-
  sampling MCCFR. Protocol/implementation/cache checkpoints are `c506cba`,
  `6406c10`, and `e072cd9`. All 415 pre-evidence tests passed.
- The evidence run completed normally and formally failed two of thirteen
  gates. Exact CFR converged deterministically at 10,000 iterations: normalized
  positive regret was `0.0000292944` / `0.00000373672`, more than 99% below the
  iteration-100 values. Privacy, stable action sets, and legality all passed.
- Seeded 50,000-iteration external MCCFR was byte-deterministic, covered
  `96.1165%` of exact information sets, had regret `0.000137200` /
  `0.00000775862`, and matched the exact root strategy at TV `0.000211297`.
- Deeper raw policy agreement failed over 30 sufficiently observed shared
  information sets: mean TV `0.1503728` narrowly exceeded `0.15`, while p90 TV
  `0.6666667` exceeded `0.30`. Several large differences were exact-uniform
  versus sampled-concentrated policies, consistent with but not proving
  strategically interchangeable actions/non-unique equilibria.
- Honor the failed gate: do not start full-game neural Deep CFR or rent a GPU.
  Next preregister exact best-response exploitability plus reach/value-weighted
  action-equivalence diagnostics on the same reduced game. This distinguishes
  harmless policy non-uniqueness from a sampled-solver defect without tuning
  `044` after its result.
- Evidence is in `outputs/deep_cfr_viability_044/result.json`; SHA-256 is
  `6b7edee724b98cc149d8bbc1826a75fa7592e246a7d85ea384e703734fbaf632`.
  BGA replays remain a future authorized-data-dependent warm-start/validation
  possibility; none were collected. Raw legal `035` remains champion.

## 2026-07-18 Final Rolling-Horizon Robustness Gate

- `041` and `042` used identical gameplay but produced conflicting 100-pair
  estimates: 58.5% and 48.0% against raw `035`. Direct replay checks reproduced
  representative seeds, checkpoint hashes and budgets match, and the effective
  sample is only 100 paired units per block. A larger final estimate is
  justified to resolve seed variance without tuning the planner.
- `rolling_horizon_robustness_043` is frozen in
  `docs/experiments/rolling_horizon_robustness_043.md`. It runs five fresh,
  non-contiguous 100-pair blocks, 500 pairs / 1,000 games total, without interim
  outcome inspection or early stopping. `041` and `042` are forbidden from the
  promotion calculation.
- The frozen direct gate requires complete integrity and exact budget in every
  block, aggregate share at least 55%, aggregate paired CI95 lower bound above
  50%, positive aggregate margin, and share above 50% in at least four of five
  blocks. Only a full pass unlocks fresh shallow/`016`/`023` broad gates.
- The candidate, opponent, evaluator, search parameters, seed derivations, and
  clean forced-deadlock semantics are unchanged. CPU only; `035` remains
  champion unless every authorized gate passes.
- Result: definitive near-parity rejection. All 409 pre-evidence tests passed.
  The five blocks completed all 500 pairs / 1,000 games with zero invalid
  terminals, deadlocks, timeouts, or stalls and exact budget: 31,954 candidate
  decisions and 1,022,528 sequence evaluations.
- Aggregate result was 505-494-1, share 50.55% with paired CI95 [47.75%,
  53.40%], and paired margin +0.211 with CI95 [-0.141, +0.561]. Share and
  confidence gates failed; only the positive point-margin strength gate passed.
- Block shares were A 47.00%, B 52.50%, C 49.25%, D 50.00%, E 54.00%. Only
  two of five were strictly above 50%, failing the consistency gate. Candidate
  share was 53.40% from seat 0 and 47.70% from seat 1, validating the paired
  design rather than a promotion claim.
- Broad blocks were correctly not run. Historical `041` + `042` + `043` is
  718-681-1 / 51.32% descriptively, but is forbidden from promotion pooling.
  Treat BMRH as near parity and close it permanently; do not add games, tune,
  or generate teacher data. Raw legal `035` remains champion.
- Result hashes: `direct_aggregate.json`
  `a18f8d5e494b2422e23dcbaf8a778d261fc856f30df959392d054d6bd8308e8a`;
  `gate_final.json`
  `aeb56fa318b43ad9913a3cef728a3d281c12991c0793d9b0e5bd397b7b4a43f2`.

## 2026-07-18 Deadlock-Validated Rolling-Horizon Replication

- A post-`041` audit reconstructed seed `403078`: turns 28 and 29 each had the
  complete legal set `[PASS]`; all colored bank stacks were empty, both players
  held ten colored tokens and three reserves, and neither could buy. This was a
  genuine forced deadlock, not a loop or cutoff.
- More importantly, the repository had already frozen and tested
  `forced_pass_deadlock` as a valid clean terminal in DAgger `026`/`027`.
  `041`'s literal-`completed` integrity rule contradicted that established
  contract. Keep `041` formally rejected, but do not interpret it as negative
  strength evidence.
- The corrected fresh replication is preregistered as
  `deadlock_validated_rolling_horizon_042` in
  `docs/experiments/deadlock_validated_rolling_horizon_042.md`. It changes no
  planner or strength parameter. It accepts exactly `completed` and
  `forced_pass_deadlock` as clean terminals, preserving the engine's existing
  score/card-count adjudication, and rejects every cutoff or stall.
- `042` uses 100 entirely fresh paired seeds against raw `035`; only a full
  pass unlocks fresh shallow/`016`/`023` blocks. No `041` game is reused.
- Result: clean rejection on strength. All 405 pre-evidence tests passed. The
  100-pair/200-game direct block had 200 ordinary completions, zero valid
  deadlocks, invalid terminations, timeouts, or stalls, and exact budget
  accounting: 6,422 candidate decisions and 205,504 sequence evaluations.
- Score-delta BMRH went 96-104-0 versus raw `035`, paired share 48.0% with CI95
  [41.0%, 55.0%], and paired margin -0.43 with CI95 [-1.26, +0.39]. It failed
  the frozen 55% share, lower-CI-above-50%, and positive-margin gates. Candidate
  results by seat were 46-54 from seat 0 and 50-50 from seat 1.
- The conditional broad blocks were correctly not run. The contrast with
  `041`'s 117-83 positive block means the apparent gain did not replicate on
  fresh seeds; do not pool the blocks for promotion or tune this planner after
  the result. Close rolling horizon and keep raw legal `035` champion.
- Result hashes: `direct_vs_035.json`
  `aff56aa33a4f2e6e4d11190b9432b9c4f51062d91b023c77d76a887640895314`;
  `gate_final.json`
  `e8e97d2e0a01e865dd3f22b945869f67d3d63d4850eddb5c55c47af44f0bd5ac`.

## 2026-07-18 Frozen Score-Delta Rolling-Horizon Gate

- `040` rejected the fixed event evaluator but left one clean question: its
  unchanged score-delta BMRH control won 45-15 and has never faced raw `035`.
- `score_delta_rolling_horizon_041` is frozen in
  `docs/experiments/score_delta_rolling_horizon_041.md`. It changes no `040`
  planner parameter and runs 100 fresh paired seeds / 200 games against raw
  `035` under strict integrity, exact-budget, 55% share, lower-CI-above-50%,
  and positive-margin gates.
- Only a complete direct pass unlocks fresh 50-pair blocks against gate-`030`
  shallow search, `016`, and loop-safe clean `023`, each under the same strength
  conditions. Failure closes the rolling-horizon branch; success authorizes
  only a separately frozen teacher-corpus loop. `035` remains champion.
- Result: formal rejection on the sole integrity gate. All 401 pre-evidence
  tests passed. The full 100-pair/200-game direct block otherwise showed clear
  strength: score-delta BMRH went 117-83-0 versus raw `035`, paired share 58.5%
  with CI95 [52.0%, 65.0%], and paired margin +0.91 with CI95 [+0.16, +1.64].
  Expected-count, exact-budget, share, confidence, and margin gates passed.
- Seed `403078`, candidate seat 0, terminated at turn 30 through
  `forced_pass_deadlock` with score 2-0 and candidate adjudicated winner. It was
  neither a stall nor timeout, but it is one non-`completed` game and the frozen
  contract required zero. No broad game was generated.
- Keep `035` champion and do not repair, tune, or promote this operator after
  the result. Close rolling horizon as the current branch while retaining the
  important scoped fact that it passed every statistical strength gate and
  failed only the forced-pass terminal-integrity condition.
- Result hashes: `direct_vs_035.json`
  `3ece75beb2ad19d2b5eeadaa7321b684d8e7ac1ec1de388b34a79ec3af968465`;
  `gate_final.json`
  `c90daf97251c0621e16416e783a2ff3cdf3a966e66c9a394043059171b398423`.

## 2026-07-17 Event-Value Rolling-Horizon Diagnostic

- The hierarchical option direction closed at `039`; do not tune it.
- The next independently justified algorithmic direction is frozen as
  `event_value_rolling_horizon_040` in
  `docs/experiments/event_value_rolling_horizon_040.md`.
- `040` is a non-training CPU diagnostic of information-legal branching-
  mutation rolling-horizon planning. It evolves two root-player decisions,
  uses frozen legal `035` for intervening opponent moves, and compares a fixed
  published five-event evaluator against an otherwise identical score-delta
  control.
- No checkpoint promotion, self-play, parameter tuning, or GPU use is
  authorized. A four-game operational smoke precedes a frozen 60-game paired
  evaluator gate; a conditional 60-game raw-`035` gate runs only if the event
  evaluator first beats its score-delta control.
- Result: reject the fixed published event evaluator. All 397 pre-evidence tests
  passed. The four-game smoke passed every operational gate in 43.5 seconds.
  The frozen 60-game control completed with zero failures, but event-value lost
  15-45-0 to score-delta: paired score share 25.0%, CI95 [15.0%, 35.0%], and
  mean paired margin -3.9833, CI95 [-5.3000, -2.6667]. Only the zero-failure
  control gate passed; the three strength gates failed.
- The conditional raw-`035` block was correctly not run. Do not tune the event
  weights, horizon, or budget after this result. Scope the rejection to the
  fixed five-event evaluator: the score-delta rolling-horizon control itself
  was much stronger in this matched block, but has not been tested against raw
  `035` and is not authorized as a teacher or promotion candidate.
- Exact result hashes: `control.json`
  `a28c793f822881bdf736bbb0194ef2dbfb55bd25f2a4dabd406108d15fe61eea`;
  `gate_final.json`
  `60f57de76973d5faf9b80df11f5460985ad11056f86d1125bfd23225f5de3608`.

## 2026-07-17 Outcome-Blind Plan Persistence Diagnostic

- A repository audit showed that the proposed high-budget search-oracle test
  had already been completed as the 320-state s64-versus-s512 all-legal matched-
  return ablation. Do not repeat or extend that rejected budget sweep.
- The next diagnostic is instead `outcome_blind_plan_persistence_038`, frozen in
  `docs/experiments/outcome_blind_plan_persistence_038.md`. It generates 256
  fresh greedy-035 games in two game-disjoint, exactly opponent/seat-balanced
  halves, strips hidden deck order and opponent-private reserves, and never
  stores or reads outcomes.
- For each qualifying token take, it asks whether the chosen legal take advances
  the first already-visible/self-reserved card bought or visibly reserved in the
  next four own turns more than the alternative legal takes from the exact same
  state. Both halves, multi-take episodes, opponent/seat cells, and phases must
  pass frozen persistence gates. A pass authorizes only a separate hierarchy
  design; failure closes hierarchy. No training or promotion is allowed.
- Result: `038` passed every frozen gate on a contract-clean corpus of 256
  completed games and 8,059 actor turns. Discovery had 630 anchors: game-cluster
  progress advantage 0.7031, CI95 [0.6638, 0.7433], and progress-rank percentile
  0.7895, CI95 [0.7772, 0.8019]. Confirmation independently had 648 anchors:
  advantage 0.7002, CI95 [0.6596, 0.7402], and rank percentile 0.7879, CI95
  [0.7751, 0.8004].
- Multi-take persistence also replicated: 58 discovery episodes at advantage
  0.6568, CI95 [0.5770, 0.7371], and 65 confirmation episodes at 0.5863, CI95
  [0.4947, 0.6709]. Every opponent/seat cell and all four observed phases had
  positive mean advantage; zero games stalled or timed out.
- Scope the conclusion carefully. Most anchors target the immediately following
  own-turn purchase (1,136/1,278); 126 span two own turns and 16 span three.
  The evidence establishes robust card-acquisition motifs, including replicated
  two-take episodes, but not general long-horizon, noble, denial, or hidden-card
  planning. It authorizes a separately frozen option-representation predictive
  probe, not policy training or gameplay. Exact evidence is under
  `outputs/plan_persistence_diagnostic_038`.
- The authorized predictive experiment is `option_predictive_probe_039`, frozen
  in `docs/experiments/option_predictive_probe_039.md`. It generates 512 new
  outcome-blind balanced games with 192/64/256 game-disjoint
  train/validation/confirmation splits. Both identical probe heads use the
  frozen `023` canonical transformer and contextual candidate-card entities;
  the state-only head receives an all-zero option vector while the conditioned
  head receives the frozen embedding of the selected take. Only probe heads
  train. Confirmation is not loaded until both validation-selected epochs are
  fixed. A full incremental overall, multi-turn, and no-collapse pass only
  authorizes supervised hierarchy reconstruction, not gameplay.
- Result: `039` rejects and closes the hierarchical option direction. The fresh
  privacy-clean corpus contained 512 completed games, 16,201 actor turns, 934
  train examples, 352 validation examples, and 1,216 untouched confirmation
  examples. Both heads independently selected epoch 22; the option head was
  already worse on validation NLL (2.0841 versus 2.0635).
- On confirmation, state-only NLL/top1/MRR were 1.9784 / 32.48% / 0.5184;
  option-conditioned results were worse at 1.9952 / 31.58% / 0.5140. The
  game-cluster NLL delta was +0.01790, CI95 [+0.00132, +0.03451], establishing
  significant regression rather than merely an unresolved gain. Top1 delta was
  -1.20 points, CI95 [-2.46, +0.07], and MRR delta -0.00529, CI95
  [-0.01335, +0.00269].
- The 113 multi-turn confirmation examples also had worse NLL (2.3070 versus
  2.2784). Their raw top1 and MRR rose only 2.65 and 1.28 points respectively,
  with unresolved clustered intervals and MRR below the frozen +2-point floor.
  Nine gates failed, including the no-stratum-collapse gate.
- Interpretation: `038` measured a real behavioral motif, but not an incremental
  latent decision variable. Given the complete public state and contextual card
  entities, the selected take does not add predictive information robustly; it
  mostly reflects cost structure already present in state. Do not tune `039`,
  train a hierarchical gameplay model, or spend GPU on this direction.

## 2026-07-17 Start-Role Value v2 Result and Direction Change

- Final focused value experiment `037` completed under its frozen contract and
  was rejected at validation. It selected epoch 1 of 6, but improved the frozen
  selection score only from 3.403912 to 3.396863 (0.207%, below the required
  0.5%). Later epochs were worse, so this was not an undertraining result.
- Relative to the unchanged v1 reliability-ramp baseline, aggregate MAE slightly
  worsened from 0.283440 to 0.283790. Starting-player MAE worsened 1.12%
  (0.273370 to 0.276430); second-player MAE improved only 0.81% (0.293601 to
  0.291216), below the required 1% improvement in both seats. Correlation
  nonregression passed for both seats, and every other original offline,
  ordering, calibration, and successor-consistency gate passed.
- The runner therefore generated no fresh confirmation corpus and ran no PUCT
  games. There is no v2 deployment checkpoint and `035` remains champion. Exact
  evidence is in
  `outputs/start_role_public_belief_value_v2_037/{validation_selection,gate_final}.json`.
- This closes the current public-belief value/PUCT bootstrap as preregistered.
  Do not tune `037`, relax its gates, add epochs, or spend GPU on the same
  objective. The next project direction is hierarchical plans/options: first
  establish that public-state multi-turn intent is stable and action-relevant,
  then train or search over it only if that diagnostic passes.

## 2026-07-17 External AlphaGo Lesson and Revised Next Experiment

- The useful lesson from Eric Jang's account of rebuilding AlphaGo is not
  "add more MCTS." AlphaGo is supervised policy iteration around a search
  operator that is already demonstrably stronger than the current network.
  Search is run on every encountered state, including games later lost, and the
  complete soft visit distribution is distilled. Collection and training are
  kept synchronous enough that the targets remain close to the current policy.
- A value model must be grounded before search can be assumed to improve play.
  With inaccurate leaves or too few simulations, MCTS can be worse than the raw
  policy. Warm-starting near competent behavior matters more than random-start
  purity, while architecture is secondary to a valid target and improvement
  operator.
- Splendor is not Go: hidden deck/refill information makes naive determinization
  biased. Any analogue must use acting-player-visible public belief features and
  information-legal search. The project has not yet established search as a
  stronger operator than `035`; prior PUCT and one-shot rollout results therefore
  do not justify mass search distillation.
- The materially new fact is scale. Legal value `029` used only 240 games, while
  the fresh balanced `032`+`033` leagues provide 4,096 games and 129,482
  acting-035 turns. This justifies one serious scaled public-belief value
  bootstrap before moving to hierarchical plan modeling.
- Frozen experiment `036` is documented in
  `docs/experiments/scaled_public_belief_value_036.md`. It uses the original
  032/033 group-disjoint train/validation boundaries, exact
  generation/opponent/seat/phase training balance, the compatible promoted `023`
  entity representation, categorical score-margin and WDL heads, calibration
  and ordering selection, and a completely fresh 512-game confirmation corpus.
  Only a full offline pass unlocks learned-value PUCT64 versus heuristic-leaf
  PUCT64, then raw `035`. Only a direct search pass authorizes synchronous
  search-every-move visit-distribution collection. `035` remains champion.
- Result: `036` is a strong offline near-pass but is rejected under its frozen
  contract. Epoch 3 trained from 113,248 positions and was checked on 16,164
  positions from 512 completely fresh, exactly balanced games with zero
  failures. Macro MAE improved 20.1% (0.28768 vs 0.36015), ordering improved to
  66.38% from 61.47%, macro correlation to 0.446 from 0.299, midgame ordering
  improved by 7.5-14.2 points, and WDL ECE10 was 4.51%. However turn-0-14 MAE
  regressed 6.7% versus the heuristic, beyond the frozen 5% allowance, and the
  value-best successor was in `035`'s top three only 48.05% of the time versus
  the 50% requirement. All other gates passed. Do not relax these thresholds or
  run PUCT post hoc; no gameplay ran and `035` remains champion. Evidence is in
  `outputs/scaled_public_belief_value_036/{selection,offline_gate,gate_final}.json`.
  Checkpoint SHA256 is
  `537b3100980d18093042bf377a009da11a90f4fa63ad4beeee5c9cd5b95c0fa6`.
- The authorized follow-up is `reliability_blend_value_036b`, frozen in
  `docs/experiments/reliability_blend_value_036b.md`. It does not retrain or
  relax `036`. It selects among exactly 12 phase/confidence blends on existing
  validation data, requiring every original offline gate there, then uses a new
  512-game confirmation block at seeds 381000-381127. Only a full fresh pass
  unlocks reliability-value PUCT64 versus heuristic-leaf PUCT64 and then raw
  `035`. Until those conditional gates pass, `035` remains champion.
- Result: `036b` selected the phase ramp `(0, 0.5, 0.75, 1.0)` with no WDL
  confidence cutoff. It passed every original offline gate on a second fresh,
  exactly balanced 512-game corpus: macro MAE improved 21.2%, ordering improved
  4.47 points, successor top-three consistency reached 58.98%, and there were
  zero failures. This unlocked the matched PUCT A/B, where reliability-value
  PUCT64 went 108-92 against heuristic-leaf PUCT64 over 200 games: 54.0% score
  share, CI95 [48%, 60%], mean margin +0.595, zero failures. It missed both the
  frozen 55% share floor and CI-lower-above-50% gate. Reject at the search A/B;
  do not run raw `035`, do not generate search-every-move targets, and keep `035`
  champion. Evidence is under `outputs/reliability_blend_value_036b`.
- Post-`036b` seat audit found the directional gain was asymmetric: reliability-value
  search scored 61% from seat 0 (+1.60 margin) but only 47% from seat 1 (-0.41),
  while the two seed halves were 53% and 55%. A concrete representation issue
  was then confirmed: `canonical_public_belief_key` records whether the observer
  is the starting player, but the v1 entity tensor does not. The final authorized
  value experiment is `start_role_public_belief_value_v2_037`, frozen in
  `docs/experiments/start_role_public_belief_value_v2_037.md`. It preserves v1,
  warm-starts exactly from `036`, adds a zero-initialized explicit start-role
  embedding, requires separate improvement for both seats, and adds a per-seat
  gameplay floor. Failure closes value/PUCT and moves to hierarchical plans.

## Goal

Build a strong 2-player Splendor AI with exact rules, legal observations/actions, structured legal-action scorers, search/PGS diagnostics, and disciplined promotion gates.

## Current Active Baselines

- Overall strongest benchmark: `outputs/legal_action_scorer_035_legal_observation/legal_action_attention_scorer_best.pt`;
  gate `030` did not displace it head-to-head.
- Clean-line operational search parent: information-legal `ShallowSearchBot` depth 1,
  one determinization, branching 6 with buy/reserve/take caps 4/2/2, seeds
  23/29; promoted by operational gate `030`.
- Active legal-observation prior: `outputs/legal_action_scorer_035_legal_observation/legal_action_attention_scorer_best.pt`
- Frozen historical strength baseline: `outputs/legal_action_scorer_022_attention_distill/legal_action_attention_scorer_best.pt`
- Conservative strong baseline: `outputs/legal_action_scorer_016_policy_guided_mix/legal_action_scorer_best.pt`
- Rejected diagnostic candidate: `outputs/legal_action_scorer_028_early_pgs_distill/legal_action_attention_scorer_best.pt`
- Rejected constrained candidate: `outputs/legal_action_scorer_030_constrained_pref/legal_action_attention_scorer_best.pt`
- Rejected trajectory candidate: `outputs/legal_action_scorer_031_trajectory_ppo/legal_action_attention_scorer.pt`
- Rejected group-relative candidate: `outputs/legal_action_scorer_032_group_relative_ppo/legal_action_attention_scorer.pt`
- Rejected clean LOO actor candidate: `outputs/legal_action_scorer_040_clean_rloo_actor/legal_action_attention_scorer.pt`
- Rejected clean LOO magnitude correction: `outputs/legal_action_scorer_041_clean_rloo_actor_lr1e5/legal_action_attention_scorer.pt`
- Rejected reserve counterfactual calibration candidate: `outputs/legal_action_scorer_042_reserve_counterfactual/legal_action_attention_scorer.pt`
- Rejected reserve hard-target head diagnostic: `outputs/legal_action_scorer_043_reserve_hardtop_head/legal_action_attention_scorer.pt`
- Rejected reserve hard-target full-model diagnostic: `outputs/legal_action_scorer_044_reserve_hardtop_fullmodel/legal_action_attention_scorer.pt`
- Rejected rollout-credit candidate: `outputs/legal_action_scorer_034_rollout_credit_anchored/legal_action_attention_scorer.pt`
- Current legal late-game value candidate, not promoted: `outputs/value_model_009_puct035_legal_late40/value_model.pt`
- Rejected canonical public-belief value candidate: `outputs/canonical_entity_public_value_029/canonical_entity_public_value_width64_best.pt`
- Historical contaminated value model, diagnostics only: `outputs/value_model_006_late40_outcome_currentgen/value_model_best.pt`
- Promoted clean entity actor parent: `outputs/canonical_entity_coherent_023/canonical_entity_action_scorer_coherent_width64_best.pt`
- Previous clean entity parent, retained as historical control: `outputs/canonical_entity_scale_004/canonical_entity_action_scorer_best.pt`
- Current clean actor-critic smoke candidate, not promoted: `outputs/canonical_entity_actor_005_smoke/canonical_entity_actor.pt`
- Rejected clean actor-critic no-movement generation: `outputs/canonical_entity_actor_006_seed90000/canonical_entity_actor.pt`
- Current clean actor-critic movement candidate, not promoted: `outputs/canonical_entity_actor_007_kl_target/canonical_entity_actor.pt`
- Rejected independent KL-target replication: `outputs/canonical_entity_actor_008_kl_target_seed110000/canonical_entity_actor.pt`
- Rejected balanced 120-game actor-critic candidate: `outputs/canonical_entity_actor_009_balanced120/canonical_entity_actor.pt`
- Rejected frontier-curriculum actor-critic candidate: `outputs/canonical_entity_actor_010_frontier120/canonical_entity_actor.pt`
- Rejected matched score-margin actor candidate: `outputs/canonical_entity_return_ab_013/canonical_entity_actor_score_margin.pt`
- Rejected dense temporal-credit candidate: `outputs/canonical_entity_temporal_credit_ab_014/canonical_entity_actor_dense_score_change.pt`
- Rejected late auxiliary-Q diagnostic: `outputs/canonical_entity_action_value_019/canonical_entity_action_value_scorer.pt`
- Rejected scaled supervised entity candidates: `outputs/canonical_entity_supervised_scale_020/canonical_entity_action_scorer_width64_best.pt` and `canonical_entity_action_scorer_width128_best.pt`
- Successful shallow-actor-only diagnostic, not promoted: `outputs/canonical_entity_shallow_actor_021/canonical_entity_action_scorer_shallow_actor_width64_best.pt`
- Promoted coherent entity parent: `outputs/canonical_entity_coherent_023/canonical_entity_action_scorer_coherent_width64_best.pt`
- Rejected on-policy DAgger candidate: `outputs/canonical_entity_dagger_026/canonical_entity_action_scorer_dagger_width64_best.pt`

## Current Conclusions

- Foundation reset is underway. `canonical_public_belief_key` is now the
  versioned exact observer-relative contract for new search/training work: it
  preserves public card identities (including purchased cards), preserves the
  public tier of a blind reserve without exposing its identity, and ignores
  incidental board/reserve/noble/deck ordering. The 295-wide tensor remains
  frozen exclusively for legacy checkpoints; do not extend new work through it.
- PUCT information-state keys and edges now use canonical public-belief and
  successor-action keys. Equivalent take/return encodings with the same net
  state transition share one edge and are scored through one representative
  action. This repairs a search branching artifact but is not an
  engine-strength promotion; the next task is a clean entity/action model on
  this contract.
- A clean `CanonicalEntityActionScorerModel` is now available for that next
  training line. It consumes variable-length canonical state entities and
  legal-action entities directly, uses no positional embeddings, and has
  permutation-equivalence and legal-hidden-reserve tests. Checkpoint inference
  recognizes `canonical_entity_action_scorer`; it is randomly initialized and
  no historical checkpoint or historical 295-vector is used to initialize it.
  The first clean CPU pilot is `outputs/canonical_entity_pilot_003`: eight
  fresh Greedy-vs-depth1-information-legal-ShallowSearch games, 518 snapshot
  positions, zero stalls/timeouts, and no learned checkpoint in collection.
  A randomly initialized 32-hidden model trained for three epochs achieved
  held-out policy loss 3.773 versus a uniform 3.846 (60 positions from one
  held-out game). This passes a representation/learning viability gate only;
  it is far too small for promotion or strength comparison. The greedy-only
  pilot `002` was rejected (7/20 timeouts), and depth2/two-determinization
  search was rejected as a CPU corpus configuration after exceeding 15 minutes
  for 30 games without completing. Scale the verified depth1/one-
  determinization corpus to a game-held-out block before considering a GPU.
- The scaled clean block is `outputs/canonical_entity_scale_004`: 50 fresh
  rule-based legal games, 3,180 positions, zero stalls/timeouts, and 8
  game-held-out validation games. The random 32-hidden entity model trained
  four epochs reached aggregate held-out policy loss 2.953 versus uniform
  3.406; all eight held-out games beat their own uniform baseline. Its first
  strength readout is deliberately non-promotional: 15-5 versus Greedy, 5-15
  versus the legal depth1 shallow-search teacher, and 3-17 versus legal `035`
  on 20 games each (the `035` block used paired seeds). It is a viable clean
  behavioral apprentice, not a champion. Do not scale pure imitation alone.
  `entity_scale_004` is now frozen as the clean actor parent.
- The clean supervised scaling experiment `canonical_entity_supervised_scale_020`
  closes the simple data/width-scaling hypothesis. It generated 500 fresh
  Greedy-vs-legal-depth1 games (31,954 positions) with fixed 400/50/50
  game-heldout splits, then trained random 64- and 128-wide one-layer entity
  models for eight epochs. Width128 was selected by validation and passed every
  untouched offline gate, lowering confirmation loss from `004`'s 3.0488 to
  2.2567, raising top-one accuracy from 20.9% to 34.2%, and beating `004` on
  all 50 confirmation games' mean losses. Despite that, it lost the mandatory
  paired parent match 13-47. Source-stratified confirmation shows it improved
  one-step imitation on both Greedy and shallow-search actions, so the inversion
  is a trajectory/distribution-shift failure rather than merely fitting the
  weak half of the labels. Reject both candidates, keep `004` frozen, and do
  not promote larger mixed-behavior imitation from this corpus. The next clean
  diagnostic should isolate a coherent teacher policy: train a preregistered
  shallow-actor-only ablation on the existing corpus before paying to generate
  shallow-vs-shallow games.
- The bounded shallow-actor-only ablation `canonical_entity_shallow_actor_021`
  succeeds. It reused `020`'s exact 400/50/50 games, retained only actions
  selected by `ShallowSearchBot`, and trained the preregistered random-init
  width64 model for eight epochs. On 1,616 untouched shallow confirmation
  positions it lowered `004` loss from 2.0122 to 1.4635, raised top-one accuracy
  from 37.1% to 50.2%, improved every held-out game's mean loss, and passed all
  offline gates. It then beat frozen `004` 40-19-1 over 60 paired-seat games,
  a 67.5% score share above the fixed 60% threshold. This validates coherent
  teacher filtering and falsifies mixed-policy labels as the key `020` failure.
  `021` is not yet a promoted parent; its gate only authorized coherent corpus
  generation. The new `canonical_entity_shallow_selfplay_corpus_022` contains
  500 depth1-vs-depth1 games, 31,080 positions, zero failures, and frozen
  400/50/50 seed splits created before training. The next experiment should
  train one width64 model from scratch on `022`, then require the same offline,
  paired-parent, and broad strong-opponent gates before promotion.
- `canonical_entity_coherent_023` validates the coherent supervised direction
  but is not promoted because its final loop-integrity gate fails. A fresh
  width64 model trained from scratch on `022`; validation selected epoch 7 and
  every untouched offline gate passed. It beat `004` 44-16, then materially
  improved matched broad score share versus Greedy (95.0% vs 83.75%), legal
  shallow depth1 (55.0% vs 17.5%), `016` (40.0% vs 15.0%), and `035` (41.25%
  vs 5.0%). The original Greedy control contained one `004` repetition cutoff,
  so a preregistered one-shot fresh replacement was run rather than pooled. In
  that replacement, `004` was clean at 70.0%, while `023` scored 92.5% but had
  one repetition cutoff. Every strength gate passes, but Greedy integrity is
  false and there is no second replacement. Keep `004` frozen. Do not retrain
  the model: the next required work is loop-safe inference for
  `CanonicalEntityActionScorerBot`, followed by fresh safety/promotion gates.
- The first clean actor-critic smoke generation is
  `outputs/canonical_entity_actor_005_smoke`. Its actor sees only canonical
  public entities; a separate training-only critic sees hidden deck cards and
  opponent blind-reserve identities. The 12-game rollout league included the
  frozen parent, Greedy, legal shallow depth 1, `016`, `035`, and PUCT035 s8.
  Full-support sampling, exact stored behavior probabilities, GAE(lambda 0.95),
  clipped ratios, entropy, KL, and paired parent gates are implemented. All
  smoke gates passed, including zero rollout failures, behavior reconstruction
  error 6.9e-7, max ratio 1.016, mean KL 2.0e-5, and normalized entropy 0.841.
  The candidate beat its parent 12-8 in 20 paired games. This validates the
  learning mechanism but is too small for promotion; keep `004` frozen and
  require independent larger generations plus broad league confirmation.
- The independent larger generation `entity_actor_006_seed90000` resolves the
  smoke ambiguity. It used 72 fresh games (12 per league opponent), 2,364 actor
  decisions, and the unchanged `005` optimizer settings. The update remained
  clean but was extremely small: mean KL 6.88e-5, max ratio 1.037, normalized
  entropy 0.834, and policy loss -5.0e-6. It scored 46-54 against frozen `004`
  over 100 paired games. Same-seed controls show essentially no broad-policy
  movement: `006`/`004` scored 7/8 wins versus `016`, 2/1 versus `035`, and
  54/55 versus Greedy on 60 games each. `006` scored 9-51 versus legal shallow
  depth 1, the same 15% share as `004`'s prior 5-15 result. Reject `006` and do
  not repeat the identical one-epoch, 1e-5 actor update. Before more expensive
  games, add an effective-movement gate and test a KL-targeted update schedule
  that is materially larger while retaining the frozen-parent safety bound.
- `entity_actor_007_kl_target` implements that movement gate and KL-targeted
  schedule. On 36 fresh league games, actor learning rate 3e-5 reached mean KL
  0.00241 in the first epoch and stopped automatically. Mean probability TV was
  0.0271, 4.93% of rollout-state argmaxes changed, centered-logit movement was
  0.0424, and relative parameter movement was 0.000654; max ratio remained
  1.194 and entropy 0.842. The initial 60-game parent gate scored 37-22-1, but
  an independent 100-game paired block scored only 52-48. Keep `004` frozen and
  retain `007` as proof that bounded effective movement is possible, not as a
  promoted parent. Do not launch expensive broad/search confirmation until the
  same recipe produces a repeatable gain from another independent rollout seed.
- The exact independent replication `entity_actor_008_kl_target_seed110000`
  fails that requirement. It used the same 36-game league and hyperparameters
  on disjoint seeds, passed every generic safety/movement gate, but scored only
  24-34-2 against `004` in its 60-game paired progression block. It therefore
  did not receive a 100-game or broad/search gate. Despite similar final KL
  (0.00228) and probability TV (0.0252), it required five epochs instead of one,
  moved parameters almost three times farther, flipped 8.81% of rollout-state
  argmaxes, and had weaker advantage/log-probability correlation. Reject `008`:
  the 36-game credit sample is not stable enough for this recipe, and target KL
  alone does not make independently trained policy directions reliable.
- The rollout scheduler used by `005`-`008` also aliased opponent and actor seat:
  with six opponents and alternating seats, every opponent always occupied one
  fixed actor seat. This is repaired and tested. `009` used 120 games with
  exactly 10 games in every opponent-seat cell, plus explicit stratum/outcome
  telemetry and a real 60% parent progression gate. It still failed 26-34
  against `004`. The balanced corpus reveals a deeper credit problem: all 80
  games against shallow search, `016`, `035`, and PUCT were losses. Terminal
  win/loss returns have no outcome variation in those strata, and naive
  opponent/seat normalization would manufacture relative signal from all-loss
  groups. Reject `009`; keep the scheduler/gate repair, and change league
  eligibility or the return signal before another actor update.
- `entity_actor_010_frontier120` tested the safer league change: 120 balanced
  games using only `004` and Greedy, with strong opponents held out. Every one
  of the four opponent-seat cells contained 30 games; seat outcomes were
  exactly matched at 26-34, and both opponents produced wins and losses. The
  update nevertheless had weak advantage/log-probability correlation (+0.053)
  and scored 33-26-1 against `004`, only 55.8% versus the preregistered 60%
  progression threshold. Three repetition cutoffs also exceeded the ceiling of
  one. Reject `010` and skip confirmation/strong-opponent games. Curriculum
  selection removes outcome degeneracy but does not solve temporal credit;
  validate a denser return signal before another policy update.
- Final score margin is now validated as that denser signal without performing
  any actor update. A 120-game calibration block at seeds 140000-140119 showed
  that raw margin preserves direction and adds variation in every all-loss
  cell, but `clip(margin / 15)` saturated 20.8% of games and was rejected. A
  fresh 120-game confirmation at seeds 150000-150119 using
  `clip(margin / 20, -1, 1)` passed every preregistered gate: 31 distinct target
  values versus two outcomes, outcome correlation 0.870, sign alignment 98.3%,
  zero saturation, and useful margin variation in all six all-loss cells. The
  `/20` target is eligible for a future critic/actor experiment; no candidate
  checkpoint was created and `004` remains frozen.
- Matched return-target A/B `entity_return_ab_013` reused one fresh 120-game
  balanced full-league corpus for binary-outcome and score-margin branches with
  identical initial weights, critic-validation games, and minibatch order.
  Score margin improved every preregistered credit diagnostic: held-out critic
  skill 0.742 versus 0.662, critic correlation 0.866 versus 0.816, and
  advantage/log-probability correlation 0.066 versus 0.031. It also beat the
  binary candidate 33-27 with no cutoffs. Against frozen `004` it scored the
  same 33-27, only 55% versus the fixed 60% progression threshold. Reject the
  policy candidate and skip confirmation/strong gates, while retaining the
  matched evidence that score margin is the superior credit target.
- Matched temporal-credit A/B `entity_temporal_credit_ab_014` compared the
  validated terminal score-margin target with causal actor-interval margin
  changes on one fresh shared 120-game balanced corpus. Dense rewards were
  genuinely less sparse (28.14% versus 3.12% nonzero), but held-out critic
  skill fell from 0.674 to 0.583, critic correlation fell from 0.828 to 0.768,
  advantage/log-probability correlation fell slightly, and raw advantage
  standard deviation increased from 0.161 to 0.181. Reject dense potential
  shaping and skip all games. It assigns both the actor's action and the
  opponent response to one interval and adds transition noise; the next credit
  experiment should be action-conditional/counterfactual rather than another
  temporal reward redistribution.
- Fresh entity counterfactual-credit diagnostic `015` evaluated 18
  game-disjoint, phase-balanced `004` states with five candidate actions and
  eight matched continuations per action. Whole-action rankings replicated
  (mean split Spearman 0.568, CI95 [0.329, 0.777]), all states were informative,
  and there were zero continuation cutoffs. The sampled action's advantage did
  not replicate overall: half-to-half correlation 0.154, CI95
  [-0.354, 0.783], so the fixed gate failed and no policy update is eligible.
  The failure is phase-localized: mid/late correlation was 0.905/0.842 with
  100% eligible sign agreement, versus early correlation -0.321 and 33% sign
  agreement. Treat mid/late-only credit as a new hypothesis requiring a fresh
  preregistered confirmation; do not select phases or train from `015` itself.
- Fresh fixed-baseline counterfactual confirmation `016` used new seeds, 30
  game-disjoint states, action-independent policy reference sets, and 24
  continuations per candidate. Late credit passed every fixed phase gate.
  Mid credit passed six of seven but split-half top-one agreement was 5/12,
  below the fixed 6/12 requirement; both phases were required. Do not weaken or
  rerun the threshold and do not train. The continuous mid signal was strong
  (rank CI95 [0.207, 0.766], chosen-advantage correlation CI95
  [0.320, 0.958]), so this is a disciplined near-miss rather than evidence that
  counterfactual credit is absent. Frozen `004` remains the parent.
- The scaled late-only program reached a clear stopping condition. Invalid
  corpus `017` was rejected by its own phase-selection gate after a zero-target
  selector bug chose early states; the bug is fixed and regression-tested.
  Corrected `018` contains 240 game-disjoint late states and 30,624 terminal
  continuations with very strong replicated action signal. Nevertheless the
  single preregistered auxiliary Q model `019` was worse than frozen `004` on
  validation and untouched confirmation for top-one, regret, and Spearman.
  No actor update was eligible. Close this counterfactual-to-Q/actor direction;
  do not tune the Q loss, capacity, or epoch against these held-out results.
- The previous observation encoded every opponent reserved-card identity. This was
  illegal for cards reserved blind from a deck. Reserve provenance is now tracked:
  visible reserves remain public, while deck reserves are visible only to their
  owner. Old snapshots deserialize conservatively as unknown/private.
- `022` remains the frozen historical strength baseline, but it was trained under
  the leaked observation contract. Do not use it as the prior for legal MCTS or
  new training.
- `016`, `022`, and `value006` all store 295-wide inputs, as does `035`; shape
  does not identify legality because masking preserved tensor width. Artifact
  chronology and training provenance establish that the first three predate the
  repair. At inference the fixed encoder masks them, but their learned feature
  semantics remain contaminated.
- `035` is the first corrected-observation prior. It was initialized from `022`
  and retrained on the original six-corpus mix. Historical snapshots cannot
  recover which reserves were public, so `035` is conservatively under-informed
  on those rows.
- `035` vs `022` under the fixed encoder is 48-52 over 100 games. Treat them as
  tied in strength; use `035` for legal inference, while recognizing that its
  initialization and historical teacher labels are not clean-from-scratch.
- Historical shallow search and PGS consumed the true hidden-deck order during
  simulated refills. Search, PGS, benchmark, and corpus construction now support
  observer-relative determinization, and generic benchmark/corpus search
  defaults are information-legal. Oracle search requires an explicit legacy
  flag where available.
- Evaluation-only information-set PUCT is implemented with a fresh
  determinization per simulation, acting-player observation keys, legal `035`
  priors, bounded heuristic leaves, optional checkpoint value leaves, and paired
  benchmark telemetry.
- PUCT now supports variable-length batched prior/value inference, Q-preserving
  virtual visits, detailed phase timing, and bounded information-state tree
  retention. Cross-move tree reuse is opt-in because it improved reuse rate but
  has not passed a direct strength A/B.
- Profiling removed two concrete bottlenecks from legal-action feature encoding:
  pre-action affordability is computed once per action set, and post-action
  affordability is memoized by resulting token holdings. Batched and individual
  feature/model outputs are covered by equality tests.
- Do not start MCTS self-play training yet. At 64 simulations, PUCT is tied within
  wide confidence intervals against `035`, plain search, and PGS016, and trails
  frozen `022` directionally. Doubling to 128 simulations did not improve the
  readout and exceeded one second per searched move.
- `value_model_006` is superseded for live MCTS because its training observations
  predate the legality repair. Its historical 9-11 leaf A/B remains diagnostic
  only.
- New legal PUCT outcome models `007`-`009` do not yet pass the leaf gate.
  Unrestricted value007 was unstable across simulation budgets; the legal
  late-turn value009 tied heuristic leaves 10-9-1 at s64. Keep value009 as a
  diagnostic candidate and do not use it for self-play target generation.
- Fresh 100-game legal strength confirmations show PUCT035 is competitive but
  still not a clean engine for self-play target generation:
  - s64 vs policy035: 50-50, score share 0.50, CI95 [0.43, 0.57];
  - s128 vs policy035: 54-46, score share 0.54, CI95 [0.46, 0.62];
  - s64 vs legal shallow search d2/d4: 52-48, score share 0.52, CI95 [0.41, 0.63];
  - s64 vs legal PGS035 d2/d4: 59-41, score share 0.59, CI95 [0.51, 0.68].
  This beats legal PGS on the tested block, but does not clearly beat the
  underlying policy or plain legal search.
- Late-only n-step PUCT leaves are now the first promising leaf/backup-quality
  improvement. With depth 2, width 1, candidate limit 4, and turn >= 40 gating,
  PUCT035 s64 scored 120-80 over two 100-game paired blocks vs policy035, score
  share 0.60, CI95 [0.54, 0.66], mean margin +0.905. Plain heuristic-leaf PUCT
  on the same seed blocks scored 104-96, score share 0.52, CI95 [0.465, 0.575],
  mean margin 0.0. Treat this as a candidate engine improvement that deserves
  search/PGS confirmation before self-play target generation.
- Search/PGS confirmation was mixed. On the same legal shallow-search seed block
  where plain PUCT was 52-48, late-n-step PUCT scored 62-37-1, score share
  0.625, CI95 [0.52, 0.725]. On the same legal PGS035 seed block where plain
  PUCT was 59-41, late-n-step PUCT scored only 52-48, score share 0.52, CI95
  [0.44, 0.60]. Do not promote late-n-step PUCT to self-play engine yet.
- The optimized engine can now run 128 simulations in approximately the old
  64-simulation time budget, but the added search still does not beat its legal
  prior. On the same 10 paired seeds, sequential s128 scored 9-11 against
  policy035 and corrected batch-8 s128 scored 7-13. Performance engineering is
  no longer the primary blocker; leaf/backup signal quality is.
- Do not promote `028`.
- Do not continue value-guided live PGS right now.
- Do not continue early token/reserve PGS imitation as the main recipe.
- `028` improved PGS agreement but regressed against search.
- The regression is not late seat1, not loop/fallback related, and not simply worse immediate search agreement.
- The likely issue is conversion efficiency: `028` turns early decisions into fewer total buys, fewer scoring buys, and fewer bought points than `022`.
- A conversion diagnostic now confirms that `028` shifts the 10 regression seeds toward more reserved buys and fewer visible/scoring buys; do not train again until this pattern is interpreted.
- `029` tested a conservative filter of the `028` PGS corpus. It partially recovered vs search and conversion, but lost badly to both `022` and `016`; do not promote it.
- `030` tested the strongest defensible version of this line: high-margin conversion-filtered pairwise corrections, explicit `022` regression anchors, low learning rate, one epoch, and pre-benchmark drift gates.
- `030` passed drift gates and looked promising in the first 100 search games, but failed fresh 200-game direct/search confirmation and still regressed conversion on the original 10 seeds.
- Retain `022` and stop supervised PGS-imitation/patch variants. Small top-action drift can still produce large trajectory drift, so another `031` weight/filter sweep is not justified.
- Genuine stochastic trajectory training is now implemented. `031` used signed terminal returns and `032` used matched-seed group-relative returns.
- Both trajectory candidates stayed close to `022`, but both failed the sensitive-seed conversion gate. Do not promote either.
- Do not respond by sweeping PPO learning rates or KL weights. The remaining problem is temporal/action credit assignment, not optimizer stability.
- Complete-rollout action credit is now implemented and tested. `033` showed that target-state KL does not constrain global drift; `034` added broad anchor KL and passed drift/target gates.
- `034` still failed conversion badly. Complete rollouts from a sparse root set inherit bias from the continuation policy and are not sufficient as direct policy targets.

## Latest Important Results

### Legal Observation Repair and `035`

Implemented reserve provenance across state transitions, replay snapshots,
diagnostics, and observation encoding. New tests cover:

- own blind reserves remaining visible;
- opponent blind reserves being hidden;
- opponent visible reserves remaining public;
- old snapshots defaulting to private/unknown provenance;
- determinization invariance to the simulator's true hidden-deck order.

`035` training:

- Parent: `022`.
- Data: the original `022` six-corpus mix.
- Two epochs, batch size 256, learning rate `1e-5`.
- Best epoch: 2.
- Validation loss: 1.0684.
- Validation accuracy: 65.84%.

### Evaluation-Only PUCT Gate

Implementation:

- `src/splendor_ai/bots/puct_bot.py`
- `src/splendor_ai/eval/run_puct_benchmark.py`
- `run_puct_benchmark.py`
- `tests/test_puct_bot.py`
- `tests/test_run_puct_benchmark.py`

The runner uses the same seed twice with seats swapped and reports paired
bootstrap confidence intervals. At 64 simulations PUCT delivered about 110-113
simulations/second and 0.56-0.58 seconds per searched move on CPU.

Initial 10-pair elimination results:

- PUCT035 s64 vs policy035: 10-9-1, score share 0.525,
  CI95 [0.35, 0.70].
- PUCT035 s64 vs plain search: 11-9, score share 0.55,
  CI95 [0.35, 0.75].
- PUCT035 s64 vs PGS016: 11-9, score share 0.55,
  CI95 [0.35, 0.75].
- PUCT035 s64 vs frozen policy022: 7-13, score share 0.35,
  CI95 [0.15, 0.55].
- PUCT035 s128 vs policy035: 11-9, score share 0.55,
  CI95 [0.35, 0.75], but 1.16 seconds per searched move.
- PUCT035 + value006 vs heuristic-leaf PUCT at s32: 9-11,
  score share 0.45, CI95 [0.25, 0.65].

Artifacts are under `outputs/benchmarks/puct035_*.json`.

Interpretation: PUCT is operational and competitive, but it has not passed the
engine-strength gate. The performance work recommended by this first gate is
documented below. Do not generate self-play targets from this version.

### Batched PUCT and Profiling

Implemented:

- variable-length batched policy scoring with legal-action padding masks;
- batched value-model evaluation;
- batch leaf collection with Q-preserving virtual visits;
- optional cross-move information-state tree retention with new-game/seat reset;
- phase telemetry for determinization, traversal, inference, batch calls, and
  retained-root reuse;
- exact action-set feature caching for repeated state-level and token-level
  affordability calculations.

Profile changes on an initial 128-simulation search:

- prior launches: 128 sequential calls to about 18 batch calls at batch size 8;
- model forward time fell to roughly 8% of profiled runtime;
- remaining dominant cost is legal-action feature construction, followed by
  legal-action generation/tree traversal;
- complete-game throughput is approximately 224-299 simulations/second, compared
  with 110-113 before this work;
- 128 simulations now average roughly 0.43-0.57 seconds per searched move,
  compared with 1.16 seconds before optimization.

Fresh paired elimination results:

- batched s128 vs frozen policy022, seed 41200: 14-6, score share 0.70,
  CI95 [0.55, 0.85];
- batched s128 with reuse vs policy035, seed 41300: 5-15;
- batched s128 without reuse on the same seeds: 6-14;
- sequential optimized s128 without reuse on the same seeds: 9-11;
- batch-4 s128 without reuse on the same seeds: 8-12;
- Q-preserving batch-8 s128 without reuse on the same seeds: 7-13.

The 022 result is not sufficient to promote the engine because the corrected
legal policy035 is the relevant prior baseline, and PUCT fails to add value
against it. Do not spend more compute tuning batch size on 20-game readouts.
Keep batch size 8 for throughput experiments, keep tree reuse opt-in, and address
leaf/backup quality before repeating larger strength gates.

### Information-Legal Search and Value Re-baseline

Audit:

- `016`, `022`, `035`, and `value006` all declare `observation_size=295`; this
  cannot distinguish leaked and masked encoders.
- `016`, `022`, and `value006` predate the masking repair.
- `035` is the corrected masked fine-tune and remains the legal prior.
- Shallow search and PGS previously called `env.step` against the simulator's
  true hidden deck, making historical search teachers and opponents mild
  deck-order oracles.

Implemented:

- shared `determinize_for_observer` hidden-card sampling;
- information-legal shallow search/PGS that average action values across
  determinizations;
- true-deck-order and opponent-private-reserve invariance tests;
- legal search defaults in generic benchmark and replay-corpus runners;
- PUCT replay-corpus support with manifests and full state snapshots;
- turn-gated PUCT value leaves.

Legal PUCT outcome corpora:

- `data/corpus_puct035_legal_selfplay_036`: 100 games at s64, 6,370 states;
- `data/corpus_puct035_legal_selfplay_037`: 200 games at s32, 12,822 states;
- total: 300 games, 19,192 states, zero stalls/timeouts.

Value experiments:

- `value007`: value006 fine-tuned on the first 100 games. Whole-corpus fit
  improved, but game-held-out performance was weak and unrestricted s64 leaves
  lost 7-13 to heuristic leaves.
- `value008`: trained from scratch on all 300 legal games. It strongly overfit
  trajectories; early-state terminal outcomes remain weak labels.
- `value009`: trained from scratch on turn 40+ states only. Final checkpoint
  reached held-out late-state MSE 0.1685, sign accuracy 0.576, and Pearson 0.354.
  With heuristic leaves before turn 40 and value009 afterward, the direct s64
  result was 10-9-1, score share 0.525, CI95 [0.375, 0.675].

Clean elimination benchmarks:

- PUCT035 s64 vs legal two-determinization shallow search: 8-12.
- PUCT035 s64 vs legal two-determinization PGS035: 8-12.
- Policy035 vs policy022 under the fixed encoder: 48-52.

Conclusion: the contamination recommendations were meaningful and are now
enforced in code. The legal value data improved offline late-state prediction,
but not enough to improve PUCT reliably. Do not run a 100-200 game value
confirmation or start policy self-play training from these leaves.

### Clean Legal Strength Table and Rollout Leaf Probe

Fresh 100-game paired-seat confirmations:

- `outputs/benchmarks/puct035_s64_vs_policy035_paired100_seed44600.json`:
  PUCT035 s64 vs policy035 was 50-50, score share 0.50, CI95 [0.43, 0.57],
  mean margin +0.04.
- `outputs/benchmarks/puct035_s128_vs_policy035_paired100_seed44900.json`:
  PUCT035 s128 vs policy035 was 54-46, score share 0.54, CI95 [0.46, 0.62],
  mean margin +0.21.
- `outputs/benchmarks/puct035_s64_vs_legal_search_d2d4_paired100_seed44700.json`:
  PUCT035 s64 vs information-legal shallow search depth 2 with 4
  determinizations was 52-48, score share 0.52, CI95 [0.41, 0.63], mean margin
  -0.06.
- `outputs/benchmarks/puct035_s64_vs_legal_pgs035_d2d4_paired100_seed44800.json`:
  PUCT035 s64 vs information-legal PGS035 depth 2 with 4 determinizations was
  59-41, score share 0.59, CI95 [0.51, 0.68], mean margin +1.39.

Interpretation: PUCT035 is a useful legal search opponent and may now be better
than legal PGS035, but it still fails the most important engine gate: clearly
improving on the policy prior it is supposed to strengthen. Do not start
AlphaZero-style policy self-play from this engine yet.

Implemented a gated rollout-leaf evaluator in PUCT:

- `PuctConfig.rollout_leaf_count`, `rollout_leaf_max_depth`,
  `rollout_leaf_top_k`, `rollout_leaf_temperature`, and `rollout_leaf_policy`.
- Benchmark/replay CLIs expose and record the rollout-leaf config.
- `rollout_leaf_policy=random` is the default rollout mode when rollouts are
  enabled; `heuristic` remains available for diagnostics.

Rollout-leaf probes:

- Heuristic-guided rollout leaves at s64, one rollout, depth 8 timed out before
  finishing 20 games in one hour. This is too expensive for current CPU gates.
- Random rollout leaves at s64, one rollout, depth 8 vs policy035 on seed 45100
  scored 9-11 and took 223 seconds.
- Plain heuristic leaves on the same seed block scored 10-10 and took 100
  seconds.

Conclusion: keep rollout leaves as an experimental option, but do not promote
random rollout leaves or spend a larger gate on them. Naive heuristic rollouts
need major optimization before they are viable.

### Late N-Step Leaf Candidate

Implemented a separate optimized n-step leaf evaluator in PUCT:

- `PuctConfig.n_step_leaf_depth`, `n_step_leaf_width`,
  `n_step_leaf_candidate_limit`, `n_step_leaf_min_turn_index`, and
  `n_step_leaf_max_turn_index`.
- Value checkpoints still take precedence over n-step leaves; n-step leaves take
  precedence over rollout leaves; otherwise PUCT falls back to the original
  heuristic leaf.
- The evaluator prunes legal actions before scoring, follows a narrow
  heuristic-guided line/beam, and is exposed in benchmark/replay CLI metadata.

Early probes:

- Depth 2, width 1, candidate limit 4 from all turns on seed 45200: 13-7 vs
  policy035, identical to plain PUCT on the same seeds, but about 3x slower.
- Depth 1, width 1, candidate limit 4 from all turns on seed 45300: 10-10,
  identical to plain PUCT on the same seeds, but about 2x slower.
- Per-move leaf-value memoization produced no useful cache hits; leaf states are
  mostly unique under the current traversal.

Useful candidate:

- Late-only n-step, depth 2, width 1, candidate limit 4, turn >= 40:
  - seed 45300, 20 games: 13-7 vs policy035, compared with 10-10 for plain PUCT
    on the same seeds;
  - seed 45400, 100 games: 56-44, score share 0.56, CI95 [0.47, 0.65], mean
    margin +0.43;
  - same seed 45400 plain PUCT: 49-51, score share 0.49, CI95 [0.42, 0.57],
    mean margin -0.56;
  - seed 45500, 100 games: 64-36, score share 0.64, CI95 [0.56, 0.72], mean
    margin +1.38;
  - same seed 45500 plain PUCT: 55-45, score share 0.55, CI95 [0.48, 0.63],
    mean margin +0.56.

Combined 200-game policy gate:

- Late n-step PUCT035 s64: 120-80, score share 0.60, CI95 [0.54, 0.66], mean
  margin +0.905.
- Plain PUCT035 s64 on the same seeds: 104-96, score share 0.52, CI95
  [0.465, 0.575], mean margin 0.0.

Interpretation: this is the first leaf-quality change with a credible positive
policy-gate signal. It is not yet enough to start self-play: confirm against
legal shallow search and legal PGS first, because the earlier clean table showed
plain PUCT was only tied with legal search and merely ahead of legal PGS.

### Late N-Step Search/PGS Confirmation

Confirmed the active late-n-step candidate against the two legal search
baselines:

- `outputs/benchmarks/puct035_nstepd2w1c4_late40_s64_vs_legal_search_d2d4_paired100_seed44700.json`:
  late-n-step PUCT035 s64 vs information-legal shallow search depth 2 with 4
  determinizations was 62-37-1, score share 0.625, CI95 [0.52, 0.725], mean
  margin +1.62, margin CI95 [0.09, 3.09].
- Same seed/block plain baseline:
  `outputs/benchmarks/puct035_s64_vs_legal_search_d2d4_paired100_seed44700.json`
  was 52-48, score share 0.52, CI95 [0.41, 0.63], mean margin -0.06.
- `outputs/benchmarks/puct035_nstepd2w1c4_late40_s64_vs_legal_pgs035_d2d4_paired100_seed44800.json`:
  late-n-step PUCT035 s64 vs information-legal PGS035 depth 2 with 4
  determinizations and prior weight 3000 was 52-48, score share 0.52, CI95
  [0.44, 0.60], mean margin +0.60, margin CI95 [-0.53, 1.78].
- Same seed/block plain baseline:
  `outputs/benchmarks/puct035_s64_vs_legal_pgs035_d2d4_paired100_seed44800.json`
  was 59-41, score share 0.59, CI95 [0.51, 0.68], mean margin +1.39.

Interpretation: late-n-step is not a simple promotion. It clearly improves the
legal shallow-search matchup and the policy035 gate, but it weakens the legal
PGS matchup relative to plain PUCT. Keep it as the active diagnostic candidate,
but do not generate self-play/root-visit targets from it until the PGS regression
is understood or a hybrid leaf rule passes all three gates.

### Hybrid N-Step Probes

Implemented extra optional n-step controls:

- `n_step_leaf_min_max_score`: only enable n-step leaves when either player has
  reached at least this score.
- `n_step_leaf_weight`: blend n-step value with the original heuristic leaf
  value; default 1.0 preserves previous behavior.

The goal was to keep the late-n-step policy/search gains while restoring the
plain-PUCT PGS matchup. Probes used the same 20-game seed blocks as the earlier
diagnostics:

- Turn >= 50 vs legal PGS035: 10-10.
- Turn >= 60 vs legal PGS035: 13-7, but the 100-game confirmation was only
  52-47-1, still weaker than plain PUCT's 59-41 on the same block.
- Turn >= 60 vs policy035: 11-9.
- Turn >= 60 vs legal shallow search: 12-8.
- Turn >= 40 and max score >= 10 vs legal PGS035: 9-11.
- Turn >= 40 and max score >= 13 vs legal PGS035: 12-8, but policy035 and legal
  shallow search probes were both 10-10.
- Turn >= 40, weight 0.5 vs policy035: 12-8; vs legal PGS035: 9-11.
- Turn >= 40, weight 0.25 vs policy035: 11-9; vs legal PGS035: 10-10.

Conclusion: simple turn gating, score gating, and scalar blending do not yet
produce a robust hybrid. The PGS regression is not just excessive late-game
usage; it appears tied to which root/trajectory contexts n-step changes. Do not
run a larger hybrid gate from these settings.

### PUCT Root-Decision Diff Audit

Implemented:

- `PuctSearchStats` now records the selected root action and per-action visits,
  Q, and prior.
- `run_puct_decision_diff.py` evaluates plain and candidate PUCT on the exact
  same roots while separately playing matched plain-driven and candidate-driven
  trajectories against an information-legal opponent.
- The runner skips the plain search exactly when the candidate performed zero
  n-step evaluations, records determinization variance and root rankings for
  changed choices, and summarizes action transitions and outcome flips.

Initial s64 audit against legal PGS035 used three fresh paired seeds, six seat
conditions, and twelve full games:

- 381 comparable PUCT roots; 26 changed, a 6.8% change rate; all changes were
  late-game roots.
- Outcomes: candidate improved two conditions, regressed one, and left three
  unchanged. Mean candidate-minus-plain score-margin delta was +0.33. This is a
  diagnostic sample, not a strength estimate.
- 15/26 changes were `BUY_VISIBLE -> BUY_VISIBLE`; the issue is mostly card
  selection within an action type, not a broad reserve/take phase error.
- In 23/26 changed roots, n-step selected an action with lower policy prior than
  plain PUCT's choice. The selected action's Q rose by +0.20 on average under
  n-step.
- In the one regressed condition, all three changed roots overrode a higher-prior
  choice and the selected-action Q uplift averaged +0.38. The earliest changed
  candidate-trajectory root, turn 43, moved a tier-1 visible buy from prior
  0.698 to prior 0.077 because n-step raised its Q from -0.434 to +0.014.

Interpretation: the PGS regression is consistent with narrow n-step producing
occasional optimistic low-prior outliers, not with n-step being broadly harmful.
Do not add another turn/score gate. The next candidate should robustify the leaf
delta itself, for example by clipping n-step's deviation from the baseline
heuristic or aggregating multiple plausible short lines, then use the diff audit
before any full strength gate.

### Clipped N-Step Leaf Candidate

Implemented `n_step_leaf_delta_clip` across PUCT, benchmark metadata, replay
corpus configuration, and the root-decision audit. When configured, the leaf
value is:

`heuristic + weight * clamp(n_step - heuristic, -clip, +clip)`

The default is `None`, preserving the previous engine exactly. Telemetry records
raw/applied absolute leaf deltas and the number of clipped evaluations.

Selected one threshold, `0.20`, from the root audit: it is just above the mean
Q uplift on improved changed roots (+0.18) and below the regressed-root mean
(+0.38). Twenty-game paired-seat probes reused the first ten seed pairs of each
historical 100-game block:

- vs policy035, seed 45400: clipped 12-8, score share 0.60, mean margin +1.75;
  plain was 11-9/+0.15 and uncapped late n-step was 13-7/+1.35.
- vs legal shallow search d2/d4, seed 44700: clipped 15-5, score share 0.75,
  mean margin +4.30; plain was 13-7/+1.75 and uncapped was 18-1-1/+6.05.
- vs legal PGS035 d2/d4, seed 44800: clipped 12-8, score share 0.60, mean margin
  +3.20; plain was 12-8/+1.65 and uncapped was 9-11/-0.35.

Across the three probes, clipping activated on 39-41% of evaluated n-step
leaves. Mean absolute raw delta was 0.206-0.214 and mean applied absolute delta
was 0.107-0.109. The cap therefore materially changed the signal rather than
passing through almost all leaves.

Interpretation: `clip=0.20` is the first simple hybrid to preserve a positive
profile on policy, legal search, and legal PGS slices. The samples are only 20
games each with wide intervals, and the seed blocks are reused diagnostics; do
not promote yet. Give this exact fixed configuration fresh paired confirmation
before considering self-play targets. Runtime remains expensive because the
n-step line is evaluated before clipping.

Fresh 100-game paired confirmations froze the exact configuration and used
previously unused seed blocks:

- vs policy035, seed 46200: 58-42, score share 0.58, paired CI95
  [0.49, 0.67], mean margin +1.34, margin CI95 [+0.27, +2.41]. There were no
  draws, timeouts, or non-completed games; PUCT wins were balanced by seat
  (30 as seat 0, 28 as seat 1).
- vs legal PGS035 d2/d4, seed 46300: 50-49-1, score share 0.505, paired CI95
  [0.405, 0.605], mean margin +0.90, margin CI95 [-0.47, +2.25]. There were no
  timeouts or non-completed games; PUCT wins were balanced by seat (26 as seat
  0, 24 as seat 1).
- Clipping remained stable rather than seed-specific: 41.2% of evaluated
  n-step leaves were clipped against policy and 41.8% against PGS. Mean raw
  absolute deltas were 0.216/0.221 and applied deltas were 0.109/0.109.

Interpretation: the candidate has a credible positive margin signal against its
policy prior, but it is tied with legal PGS on the fresh confirmation. The PGS
repair observed on the reused 20-game slice did not establish a fresh advantage.
Do not promote to self-play/root-visit target generation. The legal-search
confirmation was correctly skipped because it was conditional on both policy
and PGS passing. Do not sweep the clip threshold.

### `028` Training

`028` was trained from `022` using broad anchors plus focused early token/reserve PGS distillation:

- Focused corpus: `data/corpus_pgs016_early_token_reserve_028/replays.jsonl`
- Rows: 2,053
- PGS-changed rows: 789
- Best checkpoint: `outputs/legal_action_scorer_028_early_pgs_distill/legal_action_attention_scorer_best.pt`

Diagnostics improved:

- all-phase PGS agreement:
  - `016`: 71.50%
  - `022`: 70.67%
  - `028`: 72.00%
- early PGS agreement:
  - `022`: 60.48%
  - `028`: 63.71%

But game strength did not improve:

- `028` vs `022`: 24-25-1
- `028` vs `016`: 25-25-0
- vs search on seed block 28200:
  - `022`: 18-11-1
  - `028`: 12-18-0
  - `016`: 12-17-1

### Regression Audit

Differential seeds where `022` beat search and `028` lost search:

`28200, 28204, 28206, 28207, 28208, 28212, 28215, 28218, 28220, 28227`

Artifacts:

- `data/corpus_regression_022_vs_search_seed28200_snapshots/`
- `data/corpus_regression_028_vs_search_seed28200_snapshots/`
- `outputs/analysis/policy_diff_022_to_028_on_022_search_wins_diffseeds.json`
- `outputs/analysis/policy_diff_022_to_028_on_028_search_losses_diffseeds.json`
- `outputs/analysis/policy_diff_022_028_search_regression_derived_summary.json`

Main regression signal on those 10 seeds:

- First buy turn: both average 7.5
- First scoring buy: `028` slightly earlier, 32.1 vs `022` 34.5
- But conversion is much worse:
  - `022`: 21.1 buys, 7.4 scoring buys, 16.3 bought points
  - `028`: 18.8 buys, 5.2 scoring buys, 9.6 bought points

### Conversion Diagnostic

New tooling:

- `src/splendor_ai/training/run_conversion_diagnostics.py`
- root wrapper: `run_conversion_diagnostics.py`
- tests: `tests/test_conversion_diagnostics.py`

Artifacts:

- all rows in the snapshot corpora: `outputs/analysis/conversion_diagnostics_022_028_diffseeds.json`
- exact 10 regression seeds: `outputs/analysis/conversion_diagnostics_022_028_diff10.json`

Command for the exact 10-seed report:

```powershell
$env:PYTHONPATH='src'; python run_conversion_diagnostics.py --replay-path data/corpus_regression_022_vs_search_seed28200_snapshots/replays.jsonl --label 022 --replay-path data/corpus_regression_028_vs_search_seed28200_snapshots/replays.jsonl --label 028 --output-path outputs/analysis/conversion_diagnostics_022_028_diff10.json --model-turns-only --game-seed-filter 28200 28204 28206 28207 28208 28212 28215 28218 28220 28227 --early-turn-max 20 --window-turns 20 35 50 --log-every 0
```

Exact 10-seed model-seat readout:

- `022`: win rate 1.0, final score 16.3, score margin +4.7, 21.1 buys, 15.7 visible buys, 5.4 reserved buys, 6.5 scoring buys, 11.5 bought card points, 7.4 reserves, 2.0 non-converting reserves.
- `028`: win rate 0.0, final score 9.6, score margin -6.0, 18.8 buys, 12.4 visible buys, 6.4 reserved buys, 4.6 scoring buys, 6.9 bought card points, 8.1 reserves, 1.7 non-converting reserves.
- Paired delta `028 - 022`: -2.3 total buys, -3.3 visible buys, +1.0 reserved buys, -1.9 scoring buys, -4.6 bought card points, -2.1 noble points, -6.7 final score, -10.7 score margin.
- First buy is still equal at 7.5; first scoring buy is earlier for `028` (32.5 vs 34.9), so the problem is not delayed first conversion.
- Biggest changed early action-type buckets by count are mostly negative:
  - `RESERVE_VISIBLE -> RESERVE_DECK`: count 11, -7.27 bought card points, -3.18 buys, -3.09 scoring buys.
  - `RESERVE_VISIBLE -> RESERVE_VISIBLE`: count 9, -7.0 bought card points, -2.89 buys, -2.67 scoring buys.
  - `BUY_RESERVED -> BUY_RESERVED`: count 9, -4.89 bought card points, -2.11 buys, -1.67 scoring buys.
  - `TAKE_TOKENS -> TAKE_TOKENS`: count 10, -4.1 bought card points, -0.5 buys, -1.5 scoring buys.

Interpretation: `028` is not mainly missing affordable point-card opportunities in the logged states; it has fewer missed affordable point cards. The stronger signal is a reserve/deck-reserve and reserved-buy path that converts into fewer visible buys, fewer scoring buys, fewer card points, and fewer noble points.

### `029` Conservative PGS Filter

New tooling:

- `src/splendor_ai/training/run_pgs_corpus_audit.py`
- root wrapper: `run_pgs_corpus_audit.py`
- tests: `tests/test_pgs_corpus_audit.py`

Audit artifact:

- `outputs/analysis/pgs_corpus_audit_028.json`

Filtered corpus:

- `data/corpus_pgs016_early_token_reserve_029_conservative/replays.jsonl`
- `data/corpus_pgs016_early_token_reserve_029_conservative/summary.json`

Audit result on the 2,053-row `028` focused corpus:

- 789 changed rows, 1,264 unchanged anchors.
- 288 risky early changed rows.
- 284 early reserve-to-reserve changes.
- 160 early selected blind deck reserves.
- 130 early `RESERVE_VISIBLE -> RESERVE_DECK` changes.
- Changed-row search margin median: 34.
- Risky-row search margin median: 21.
- Conservative profile output:
  - 1,893 rows written;
  - 160 rows dropped;
  - 128 rows downweighted to 0.3 of their original weight.

`029` training:

- Parent: `outputs/legal_action_scorer_022_attention_distill/legal_action_attention_scorer_best.pt`
- Candidate: `outputs/legal_action_scorer_029_conservative_early_pgs/legal_action_attention_scorer_best.pt`
- Same broad corpus mix as `028`.
- Replaced the duplicated `028` focused corpus with the duplicated conservative filtered corpus.
- Best epoch: 3.

Action-level result on the original `028` focused corpus:

- `022` stored-teacher match: 52.75%.
- `028` stored-teacher match: 56.65%.
- `029` stored-teacher match: 57.09%.

Promotion gates:

- `029` vs `022`, 50 games seed 28300: 18-32.
- `029` vs `016`, 50 games seed 28400: 21-29.
- `029` vs search, 30 games seed 28200: 15-14-1.
  - Better than `028` at 12-18-0.
  - Worse than `022` at 18-11-1.

Artifacts:

- `outputs/benchmarks/legal_action_scorer_029_vs_022_seed28300_50.json`
- `outputs/benchmarks/legal_action_scorer_029_vs_016_seed28400_50.json`
- `outputs/benchmarks/legal_action_scorer_029_vs_search_seed28200_30.json`
- `data/corpus_regression_029_vs_search_seed28200_snapshots/`
- `outputs/analysis/conversion_diagnostics_022_029_diff10.json`
- `outputs/analysis/policy_diff_022_to_029_on_pgs028_corpus.json`
- `outputs/analysis/policy_diff_028_to_029_on_pgs028_corpus.json`

Exact 10-seed conversion comparison:

- `029`: win rate 0.3, final score 11.4, 19.2 buys, 12.9 visible buys, 6.3 reserved buys, 5.4 scoring buys, 8.1 bought card points.
- `029 - 022`: -1.9 buys, -2.8 visible buys, +0.9 reserved buys, -1.1 scoring buys, -3.4 bought card points, -1.5 noble points, -4.9 final score.
- This is a partial recovery from `028`, but the same reserve-heavy conversion defect remains.

### `030` Constrained Preference Experiment

New reusable tooling:

- `src/splendor_ai/training/run_constrained_preference_corpus.py`
  - selects only changed PGS rows where the stored search advantage over `022` is at least 50;
  - rejects blind deck reserves and token corrections that replace buys;
  - only retains scoring buys, visible scoring reserves replacing blind reserves, or token choices that improve distance to a visible scoring card;
  - extracts exact `022` model-turn anchors from the 10 differential seeds.
- `src/splendor_ai/training/run_policy_drift.py`
  - reports checkpoint top-action agreement by phase, prior action type, and transition on arbitrary snapshot corpora.
- Harness config: `configs/candidate_harness_030_existing.json`
- Harness result: `outputs/harness/candidate_harness_030_existing_summary.json`

Corpus:

- Preferences: `data/corpus_030_constrained/preference_rows.jsonl`
- Anti-regression anchors: `data/corpus_030_constrained/anti_regression_022_rows.jsonl`
- Summary: `data/corpus_030_constrained/summary.json`
- 2,053 PGS rows produced 49 preference corrections:
  - 45 `TAKE_TOKENS -> TAKE_TOKENS`;
  - 4 `RESERVE_DECK -> RESERVE_VISIBLE`.
- Extracted 340 `022` anchors from the 10 differential seeds, weighted 2.0.

Training:

- Parent: `022`.
- Exact sanitized `022` training mix plus the 340 anchors and 49 preference rows.
- Preference rows were pairwise-only: no CE or search-distillation contribution.
- One CPU epoch, learning rate `1e-5`, pairwise loss weight `0.05`.
- Candidate: `outputs/legal_action_scorer_030_constrained_pref/legal_action_attention_scorer_best.pt`

Pre-benchmark diagnostics:

- Broad 2,053-row PGS slice top-action agreement with `022`: 95.76%.
- Protected 340-row regression-anchor agreement: 98.24%.
- On the 49 selected corrections, `030` changed only three actions and adopted two preferred actions.
- All configured diagnostic gates passed.

Benchmarks:

- vs `022`, seed 28500, 100 games: 50-50-0.
- vs `016`, seed 28600, 100 games: 56-43-1.
- vs `022`, fresh confirmation seed 28900, 200 games: 94-105-1.
- Combined direct vs `022`: 144-155-1 over 300 games.
- Paired search seed 28700, 100 games:
  - `030`: 50-50-0, average margin +0.14;
  - `022`: 43-57-0, average margin -0.86.
- Paired search confirmation seed 29100, 200 games:
  - `030`: 83-115-2, average margin -0.645;
  - `022`: 86-112-2, average margin -0.305.
- Across all 300 paired search seeds, the `030 - 022` score-margin delta was only +0.107 with 95% CI approximately `[-0.521, +0.734]`; 63 seeds improved, 172 tied, and 65 worsened.

Conversion confirmation:

- Snapshot corpus: `data/corpus_regression_030_vs_search_seed28200_snapshots/`
- Report: `outputs/analysis/conversion_diagnostics_022_030_diff10.json`
- On the original 10 differential seeds:
  - `030`: win rate 0.6, final score 12.4, 19.1 buys, 13.5 visible buys, 5.3 scoring buys, 7.9 bought card points.
  - `030 - 022`: -0.4 win rate, -3.9 final score, -3.8 score margin, -2.0 buys, -2.2 visible buys, -1.2 scoring buys, -3.6 bought card points.
  - Reserve count was not inflated (`7.2` vs `7.4`), but first scoring buy moved later (`39.1` vs `34.9`).
- Interpretation: filtering removed the obvious reserve pathology, but very small policy drift still compounds into weaker trajectories. This invalidates the assumption that anchor-state top-action retention is enough to make PGS patching safe.

### `031` / `032` Trajectory Policy Improvement

New reusable infrastructure:

- `LegalActionAttentionSamplingBot`
  - top-k epsilon exploration around an attention checkpoint;
  - records the exact behavior probability for every selected action.
- Replay export now preserves behavior probability/log-probability and exploration metadata.
- `run_trajectory_advantage_corpus.py`
  - builds signed outcome/score-margin trajectory returns.
- `run_group_relative_trajectory_corpus.py`
  - centers returns across repeated stochastic rollouts of the same board seed, player role, and opponent.
- `run_trajectory_policy_gradient.py`
  - clipped PPO ratio against recorded behavior probabilities;
  - frozen-parent KL regularization;
  - entropy regularization and gradient clipping.

Verification:

- End-to-end smoke behavior-log-probability reconstruction error: `6.3e-8`.
- Full suite: 179 passed.

`031` plain trajectory PPO:

- Data:
  - 400 stochastic `022` self-play games;
  - 100 stochastic `022` vs `016` games;
  - 60 stochastic `022` vs search games;
  - 30,589 policy actions across 560 completed games.
- Training:
  - one epoch from `022`, learning rate `3e-6`;
  - PPO clip `0.1`, parent KL weight `1.0`;
  - mean KL `3.9e-5`, clip fraction `0.17%`.
- Drift:
  - broad PGS slice agreement 97.27%;
  - protected regression agreement 99.12%;
  - on-policy trajectory agreement 99.08%.
- Initial gates:
  - vs `022`: 47-53;
  - vs `016`: 48-52;
  - paired 100-seed search: `031` 50-50, `022` 49-51.
- Conversion gate on the original 10 seeds:
  - `031`: win rate 0.4, final score 12.0, 18.9 buys, 5.5 scoring buys, 8.7 bought card points.
  - `031 - 022`: -4.3 final score, -5.9 margin, -2.2 buys, -1.0 scoring buys, -2.8 bought card points.
- Result: reject before larger confirmation.

`032` group-relative PPO:

- Data:
  - three stochastic rollouts for every matched condition;
  - 100 self-play seeds, 50 `016` seeds, and 20 search seeds;
  - 270 groups, all size 3; 269 groups had return variation;
  - 25,836 action rows from 807 informative trajectories.
- Training used the same settings as `031`; only target construction changed.
- Drift:
  - broad PGS slice agreement 96.88%;
  - protected regression agreement 99.12%;
  - grouped-trajectory agreement 98.84%.
- Conversion gate:
  - `032`: win rate 0.6, final score 12.9, 19.2 buys, 5.9 scoring buys, 9.3 bought card points.
  - `032 - 022`: -3.4 final score, -3.4 margin, -1.9 buys, -0.6 scoring buys, -2.2 bought card points.
- Group-relative targets improved over `031`, but still failed the conversion thresholds.
- Harness: `configs/candidate_harness_032_existing.json`
- Result: diagnostic failure; broad benchmarks correctly skipped.

### `033` / `034` Complete-Rollout Action Credit

New tooling:

- `run_rollout_action_credit.py`
  - phase-balanced snapshot sampling;
  - parent-policy top actions plus strongest available buy candidates;
  - complete stochastic game rollouts after forcing each root action;
  - common rollout seeds across root actions;
  - paired action-advantage uncertainty;
  - adaptive refinement: initially changed decisions receive 16 additional rollouts, for 24 total per action.
- `run_rollout_credit_training.py`
  - soft rollout-return action distributions;
  - frozen-parent KL on target states;
  - optional separate broad anchor stream with parent KL.

Rollout corpus:

- Search-derived states: `data/corpus_rollout_credit_033_search/`
- Self-play states: `data/corpus_rollout_credit_033_selfplay/`
- 90 states evaluated; 68 passed the minimum action-return range.
- 62 initially changed states received adaptive refinement.
- Final rollout best differed from `022` on 46 written states.
- 37 changed states had positive paired 95% lower bounds.
- All confident `BUY_* -> TAKE_TOKENS` changes involved zero-point tier-1 buys, not scoring buys.

`033` target-only KL:

- Improved rollout-distribution loss but did not improve rollout-best top-action match.
- Broad top-action agreement fell to 91.09%; protected agreement fell to 95.29%.
- Rejected before game evaluation.

`034` broad-anchor KL:

- Parent: `022`.
- Training: three epochs, learning rate `1e-5`, target-state KL weight 2.0, broad-anchor KL weight 5.0.
- Rollout-best match improved from 32.35% to 33.82%.
- Broad PGS-slice agreement: 96.83%.
- Protected regression agreement: 98.53%.
- Passed target-uptake and drift gates.
- Conversion on the original 10 seeds:
  - `034`: win rate 0.4, final score 12.3, 19.3 buys, 5.6 scoring buys, 8.1 bought card points.
  - `034 - 022`: -4.0 final score, -5.6 margin, -1.8 buys, -0.9 scoring buys, -3.4 bought card points.
- Harness: `configs/candidate_harness_034_existing.json`
- Result: conversion diagnostic failure; broad benchmarks skipped.

Interpretation:

- Complete rollout outcomes are better grounded than shallow-search heuristic labels.
- They are still conditional on the current stochastic continuation policy and sparse root candidate set.
- Small rollout-target changes continue to alter downstream trajectories in ways that the local return estimates do not generalize.
- Do not tune `034`; move to tree-search visit distributions with repeated state evaluation and policy/value backup.

### Generation-1 Root-Visit Pilot

Implemented the first legal search-distillation pilot without changing the
active baseline:

- PUCT root telemetry now exports prior entropy, normalized entropy, top-1
  probability, effective branching, visit entropy, visited-action count, and a
  normalized root-visit distribution for every replayed search action.
- Replay/dataset/training now preserve those distributions and apply a direct
  masked cross-entropy root-visit objective. `run_legal_action_supervised.py`
  exposes `--root-visit-loss-weight`.
- `run_benchmark.py` now supports `--paired-seeds` for direct
  checkpoint-vs-checkpoint gates: each seed is played twice with seats reversed.

Data:

- Broad legal policy data: 2,000 completed stochastic games, 127,336 steps:
  1,000 policy035 self-play, 500 policy035 vs historical `016`, and 500
  policy035 vs historical `022`.
- Search data: 300 completed clipped-PUCT035 self-play games, 19,030 steps,
  all with normalized root-visit targets, zero stalls, and zero timeouts.
- The clipped corpus used the frozen candidate configuration: s64, legal `035`
  priors, depth-2/width-1 late n-step leaves, candidate limit 4,
  `min_turn_index=40`, and `delta_clip=0.20`.

Policy `036`:

- `outputs/legal_action_scorer_036_generation1_root_visit_pilot/`
- Initialized from legal `035`; two epochs, learning rate `1e-5`, game-level
  validation split, ordinary policy/value loss plus root-visit loss weight 2.0.
- 146,366 total snapshot rows; best validation epoch 2, loss 0.8826 and legal
  action accuracy 90.99%.
- It is not promoted. Fresh paired 100-game gates, all legal:
  - vs policy035: 48-51-1; policy036 score share 0.485, CI95 [0.40, 0.57].
  - vs legal PGS035: 45-54-1; share 0.455, CI95 [0.375, 0.54].
  - vs clipped PUCT035: 41-59; share 0.41, CI95 [0.33, 0.49], mean paired
    margin -1.49, CI95 [-2.63, -0.38].
  - vs plain PUCT035: 45-55; share 0.45, CI95 [0.37, 0.53].
- Clipped PUCT telemetry in that held-out gate: mean root prior effective
  actions 4.41, visit effective actions 3.55, and visited actions 5.34. The
  exported targets therefore contain a real but narrow search preference, not
  a degenerate one-action label.

Value `010`:

- `outputs/value_model_010_generation1_mixed_wdl/`
- Trained on the same mixed corpus with margin regression and WDL loss weight
  0.5. Best validation epoch was 1: MSE 0.2103, outcome-sign accuracy 59.24%,
  Pearson 0.2661.
- Keep it diagnostic only. It has not earned a learned-leaf A/B or live search
  integration.

Interpretation:

- The corpus/export/training/gating infrastructure works end to end, but one
  root-visit target per visited PUCT state plus broad behavior cloning did not
  turn the policy into a stronger standalone agent.
- Do not promote `036`, do not replace `035`, and do not use `010` in live
  PUCT/PGS. This is a bounded negative result, not evidence to discard legal
  PUCT or the root-visit data path.

### `036` Root-Visit Target-Uptake Audit

Implemented `run_root_visit_target_audit.py` and
`src/splendor_ai/training/run_root_visit_target_audit.py`. The audit recreates
the exact mixed-corpus, game-level split used by `036`, isolates the PUCT source
on either side of that split, and reports game-bootstrap CIs for target entropy,
cross-entropy/KL, top-action agreement, model drift, phase, and target action
family. Artifacts:

- `outputs/diagnostics/root_visit_target_uptake_036_vs035_train.json`
- `outputs/diagnostics/root_visit_target_uptake_036_vs035_heldout.json`

Results:

- The held-out PUCT slice is 28 games / 1,816 rows. Targets are not degenerate:
  3.52 effective actions, 5.32 visited actions, 21.07 legal actions, and target
  top probability 0.639.
- They are nevertheless weak as policy-improvement labels. Legal `035` already
  matches the PUCT visit argmax on 93.85% of held-out rows and has target KL
  0.0547. The tree mostly refines probabilities among actions `035` already
  ranks first, rather than supplying new decisions.
- `036` did not learn the complete visit distribution. On the training PUCT
  rows, it improves CE by only 0.0017 nats. On held-out rows it regresses by
  0.0022 nats (CI95 [-0.0009, 0.0052]) and target KL rises from 0.0547 to
  0.0569.
- This is not simple inactivity or ordinary overfit. `036` substantially moves
  the distribution (candidate-to-035 policy KL 0.0561) and raises probability
  on the target top action from 0.601 to 0.686, but it lowers policy entropy
  from 1.180 to 0.850 and loses the target visit tail.
- The distortion is stable in train and validation: it improves buy-visible and
  buy-reserved visit fit, but worsens early-phase fit and, especially,
  token-taking (+0.0392 held-out CE) and reserve-deck (+0.0121) targets.
- The direct source of that collapse is measured: 98.73% of held-out PUCT replay
  behavior actions equal the visit argmax. The ordinary one-hot policy loss on
  PUCT rows therefore conflicts with their soft visit target; the PUCT source
  is only about 13% of the mixed corpus, so broad one-hot behavior cloning also
  dominates the intended calibration signal.

Conclusion:

- `036` is neither a clean full-distribution uptake nor evidence that more of
  the same PUCT games are needed. The current objective is structurally biased
  toward an overconfident argmax approximation, and its target has too little
  top-action novelty over `035` to justify a broad policy replacement.

### `037` Conflict-Free Root-Visit Correction

Implemented `run_root_visit_correction.py` and
`src/splendor_ai/training/run_root_visit_correction.py`. This runner recreates
the exact `036` split, trains only the 272-game PUCT training partition against
soft root visits, and takes a separate batch from the 127,336-row broad legal
stream for frozen-`035` parent KL. It deliberately applies no one-hot replay CE
or value loss to the PUCT target rows.

- Candidate: `outputs/legal_action_scorer_037_root_visit_conflict_free/`
- Fixed recipe: parent `035`, two epochs, learning rate `1e-5`, root target
  batches 128, anchor batches 128, anchor KL weight 2.0, game split seed 36.
- Best validation epoch: 1. The internal held-out root CE was 1.01862 and
  entropy 1.13033. The second epoch did not improve validation CE.
- Independent held-out uptake audit:
  `outputs/diagnostics/root_visit_target_uptake_037_vs035_heldout.json`.
  `037` improves visit CE from 1.0252 (`035`) to 1.0156, delta -0.0096 with
  CI95 [-0.0107, -0.0084], and lowers target KL from 0.0547 to 0.0451.
- The correction is broad rather than a buy-only artifact: every phase and each
  target action family improves CE, including early positions (-0.0128),
  reserve deck (-0.0178), and token taking (-0.0047). Its policy entropy is
  1.1265, close to `035`'s 1.1804 and far above `036`'s collapsed 0.8498.
- It fails the decision-novelty gate. `037` changes `035`'s top action on only
  0.86% of held-out rows (CI95 [0.48%, 1.28%]); target-top agreement is 93.74%
  versus `035`'s 93.85%. The improved distribution fit therefore does not
  create a plausible standalone policy-strength change.
- Do not run any game benchmark or promote `037`. The requested pre-game gate
  did its job: correct soft-target uptake is possible, but the present PUCT
  visit targets have too little decision novelty to be a useful policy-imitation
  engine.

### Counterfactual Leaf-Quality Audit

Implemented `run_leaf_quality_audit.py` and
`src/splendor_ai/training/run_leaf_quality_audit.py`. It samples late-game PUCT
replay states, applies the same observer-relative determinization and
continuation seeds to every legal candidate action, and compares root-player
heuristic, exact clipped n-step, value009 scalar/outcome, and value010
scalar/outcome signals against matched policy035 continuation returns.

- Primary confirmation artifact:
  `outputs/diagnostics/leaf_quality_audit_puct035_late_counterfactual_confirm60.json`.
- 60 phase-stratified states were evaluated from the existing 300-game clipped
  PUCT corpus; 46 had a rollout action-return range of at least 0.05 and were
  retained for state-level ranking metrics. Each candidate used three legal
  determinizations and ten common-seed continuations per determinization.
- Clipped n-step is the only clear leaf candidate: mean rollout regret 0.266
  versus heuristic 0.370, delta -0.105 with CI95 [-0.216, -0.016]. Its mean
  Spearman action-ranking correlation is 0.168, CI95 [0.014, 0.318], versus the
  heuristic's -0.112. Top-one accuracy is directionally higher (30.4% versus
  23.9%) but not independently decisive.
- No learned value signal earns another training or live-leaf experiment.
  Value010 scalar has a small regret improvement but no robust rank-correlation
  or top-one improvement; value009 is weaker, particularly after turn 60.
  Do not train another value model from these data.

Fresh evaluation-only strength gate for the audited clipped rule:

- New runner support: `run_puct_benchmark.py --opponent-plain-puct` disables
  n-step/rollout leaves only for the PUCT opponent, allowing a fair ablation.
- s64 clipped PUCT035 vs policy035: 57-42-1, paired share 0.575, CI95
  [0.51, 0.645]; mean margin +0.83, CI95 [-0.02, +1.64].
- s64 clipped PUCT035 vs legal PGS035: 56-43-1, share 0.565, CI95
  [0.48, 0.65]; margin +1.08, CI95 [-0.28, +2.38].
- s64 clipped PUCT035 vs independently configured plain PUCT035: 56-43-1,
  share 0.565, CI95 [0.485, 0.645]; margin +0.65, CI95 [-0.39, +1.72].
- All three 100-game paired blocks completed with zero timeouts.

Conclusion:

- The audit rehabilitates clip `0.20` as the best current leaf/backup rule and
  explains why it can beat policy035. It does not establish a self-play engine:
  the fresh legal-PGS and plain-PUCT intervals still include a tie. Keep it
  evaluation-only and do not generate self-play targets or train another policy
  from it.

### Final Clipped-PUCT Promotion Gate

Completed the preregistered fresh 200-game paired confirmations without any
configuration changes. Both use legal `035` priors, s64, `c_puct=1.5`,
batch-8 inference, no reuse, and the frozen late depth-2/width-1/candidate-4
clip-0.20 rule. Artifacts:

- `outputs/benchmarks/puct035_clip020_finalconfirm_vs_legal_pgs035_paired200.json`
- `outputs/benchmarks/puct035_clip020_finalconfirm_vs_plainpuct035_paired200.json`

Results:

- vs legal PGS035: 100-98-2, paired score share 0.505, CI95 [0.435, 0.575],
  mean margin +0.755, CI95 [-0.11, +1.65].
- vs independently configured plain PUCT035: 104-96, paired share 0.52, CI95
  [0.46, 0.58], mean margin +0.51, CI95 [-0.19, +1.235].
- All 400 games completed normally with zero timeouts.

Decision:

- Do not promote clipped n-step PUCT to a self-play engine. It fails the
  declared rule requiring both legal-PGS and plain-PUCT lower score-share bounds
  to exceed 0.50. The counterfactual leaf advantage is real but does not survive
  full-game confirmation with enough certainty.

### Counterfactual Action-Value Pilot

Implemented a separate action-conditioned Q pipeline rather than changing the
policy trainer:

- `run_leaf_quality_audit.py` can now export its informative matched-return
  rows as a legal-action replay corpus.
- `run_action_value_training.py` and
  `src/splendor_ai/training/run_action_value_training.py` train a fresh score
  head on counterfactual action returns with Smooth L1 and pairwise ranking
  losses. The shared legal `035` encoders are initialized from the frozen prior;
  no one-hot policy CE, root-visit CE, or terminal WDL target is used.
- Held-out evaluation compares Q against the heuristic on top-one selection,
  regret, per-state Spearman ranking correlation, and disagreement with the
  `035` prior. The offline acceptance rule requires both a negative upper
  confidence bound for Q-minus-heuristic regret and a positive lower confidence
  bound for Q-minus-heuristic Spearman before any PUCT integration is allowed.

Corpus and audit:

- `outputs/diagnostics/action_value_corpus_001_audit.json` evaluated 300
  sampled late PUCT states from the existing clipped-PUCT corpus. Each state
  used six legal candidate actions selected from the frozen `035` policy, three
  legal determinizations, and eight matched policy035 continuations per
  action/determinization.
- 206 states had a counterfactual return range of at least 0.05 and were
  exported to `data/corpus_action_value_late_puct035_001/replays.jsonl`.
  They cover 127 games and turns 40--73.
- This larger audit reproduces the useful leaf diagnostic: clipped n-step
  regret is 0.257 versus the heuristic's 0.312, delta -0.055 with CI95
  [-0.110, -0.004]. This validates the return target quality but does not
  promote clipped PUCT, whose final full-game gate remains closed.

Q-model result:

- Candidate: `outputs/action_value_scorer_001_counterfactual_late/`.
- Game-level held-out split: 159 train rows and 47 validation rows from 25
  games. Best epoch 29 of 30.
- Q top-one rate is directionally higher (38.3% versus 36.2%) but inconclusive,
  delta +2.1 points, CI95 [-17.0, +21.3]. Its mean regret is worse: 0.341
  versus 0.297, delta +0.044, CI95 [-0.073, +0.163]. Spearman is likewise
  inconclusive: +0.037, CI95 [-0.182, +0.257].
- Q changes the `035` top action on 53.2% of held-out states, but this is not
  useful demonstrated novelty because the regret gate fails.
- Decision: do not integrate this Q checkpoint into PUCT and do not run PUCT
  strength games. The pilot is a clean negative/inconclusive result, not
  evidence for another action-Q hyperparameter sweep.

### Residual PUCT Error Attribution Audit

Implemented `run_residual_error_audit.py` and
`src/splendor_ai/training/run_residual_error_audit.py`. This is a diagnostic
only: it does not modify a checkpoint or run a promotion game gate. For every
sampled root state, it evaluates **every legal action** under common
determinizations and continuation seeds, then compares the matched-return
oracle with the `035` prior top action and the recorded clipped-PUCT root-visit
choice. It reports policy top-six coverage separately from the PUCT attribution,
because root PUCT itself expands all legal actions.

- Artifact: `outputs/diagnostics/residual_error_audit_puct035_clip020_late80.json`.
- The fixed, phase-stratified sample contains 80 late states: 27 turns 40--49,
  27 turns 50--59, and 26 turn 60+. Each legal action used three legal
  determinizations and six common-seed policy035 continuations per
  determinization.
- Only 13/80 states have a statistically resolved best root action under the
  conservative paired criterion (best-vs-runner-up mean margin at least 0.05
  and greater than 1.96 paired standard errors). The other 67 are explicitly
  labelled unresolved; do not use them to claim a search failure.
- Of the 13 resolved states, PUCT matches the pooled matched-return oracle once
  and chooses the wrong `035` top-prior action 11 times. The remaining error is
  one backup/search miss where PUCT diverges from both the prior and the
  oracle. Thus PUCT equals the `035` prior on 12/13 resolved states, while the
  prior equals the oracle on only 1/13. This is strong diagnostic evidence of
  root-prior dominance, but still a small resolved sample rather than a
  promotion-quality estimate.
- The policy top-six candidate set omits the oracle on 8/13 resolved states
  (61.5%, CI95 [30.8%, 84.6%]). This independently explains why an action-Q
  rule restricted to that candidate set is not a credible next fix.
- The per-determinization best action changes on 11/13 resolved states (84.6%,
  CI95 [61.5%, 100%]). This is expected information-set variation, not an error
  label: PUCT should optimize the mean over determinizations, and should not be
  simplified to one determinization per move.

Decision:

- Do not change the leaf, train another Q/value model, or run a PUCT strength
  gate from this result. The audit rules those directions out more clearly.
- The next testable hypothesis is narrowly about root control: can an
  evaluation-only PUCT with a less dominant root prior recover more of the
  already resolved matched-return actions without degrading the rest of the
  pool?

### Root-Prior Control Ablation

Implemented a root-only prior control in `PuctConfig`:
`root_prior_mix=1.0` preserves the frozen neural prior, `0.0` makes only the
root prior uniform, and intermediate values linearly mix the two. Child-node
priors are unchanged. Added `run_root_prior_control_ablation.py` and
`src/splendor_ai/training/run_root_prior_control_ablation.py`, which re-runs
legal PUCT on the 80 residual-audit snapshots and scores selected actions using
the already computed matched-return table. It generates no new continuations
and runs no games.

- Artifact:
  `outputs/diagnostics/root_prior_control_ablation_puct035_clip020_late80.json`.
- Fixed variants: frozen root prior (`1.0`), 50/50 root mix (`0.5`), and uniform
  root prior (`0.0`). All retain s64, `c_puct=1.5`, FPU 0, batch-8 inference,
  and the frozen late clip-0.20 leaf rule.
- The freshly rerun frozen control matches the source corpus root choice on
  92.5% of all 80 states, validating the replay setup.
- The 50/50 mix changes only 8.8% of choices and fails both offline gates:
  resolved regret delta -0.038, CI95 [-0.107, 0.000], and oracle-agreement
  delta +7.7 points, CI95 [0.0, +23.1].
- Uniform root control changes 56.3% of choices, increases root visit effective
  actions from 2.64 to 19.07, and improves resolved mean regret from 0.926 to
  0.568. Its paired regret delta is -0.358, CI95 [-0.681, -0.073]. Oracle
  agreement rises from 1/13 (7.7%) to 4/13 (30.8%), delta +23.1 points, but its
  CI95 is [0.0, +46.2] and therefore does not establish a strictly positive
  improvement.

Decision:

- Do not run continuation or paired game evaluations yet. The predeclared gate
  required both a negative regret upper bound and a positive oracle-agreement
  lower bound; uniform clears the first only.
- Uniform root control is the first result since clip-0.20 with a robustly
  favorable offline regret signal. The next justified work is a preregistered,
  larger all-legal matched-continuation confirmation of **only** frozen versus
  uniform root control. It should target enough resolved states to decide the
  agreement gate; do not add another mix value or change child priors.

### Disjoint Uniform Root-Control Confirmation

Completed the preregistered confirmation using
`outputs/diagnostics/residual_error_audit_puct035_clip020_late320_confirm.json`
and
`outputs/diagnostics/root_prior_control_ablation_puct035_clip020_late320_confirm.json`.

- The new all-legal audit sampled 320 phase-stratified states (107 turns
  40--49, 107 turns 50--59, 106 turn 60+), with the original 80 states
  explicitly excluded. The artifacts record all 80 exclusions and zero state
  overlap.
- It retains exactly the fixed protocol: three legal determinizations, six
  common-seed policy035 continuations per legal action/determination, the same
  paired resolution rule, and s64 clipped-PUCT root selections.
- 32/320 states have a resolved matched-return root action. This is fewer than
  the rough planning estimate but more than twice the exploratory resolved
  sample; the predeclared gate is evaluated on the actual resolved pool.
- The frozen rerun reproduces the source root choice on 97.8% of all states.
  On resolved states it has mean regret 0.598 and oracle agreement 31.25%.
- Uniform root control changes 60.9% of all choices but is worse: mean regret
  0.660, paired delta +0.063, CI95 [-0.120, +0.241], and oracle agreement
  28.13%, delta -3.13 points, CI95 [-18.75, +12.5]. It fails both the regret
  and agreement gates.

Decision:

- Retire uniform root control and the root-prior-mixing line. The 80-state
  apparent gain was an underpowered exploratory result and does not replicate
  on the disjoint confirmation.
- Do not run PUCT continuation or paired game tests, add another mix setting,
  change child priors, or increase simulations from this result.

### Strategic Review Guide and CPU High-Budget Gate

The following principles from the external strategic review are retained as
future guidance, with the listed constraints:

1. Training-time search may use a larger budget than the final playing agent;
   the one-second online target does not itself rule out an offline search
   operator.
2. A larger-budget operator must first demonstrate action-quality improvement
   and then paired-game strength before it generates self-play targets. Search
   budget alone is not evidence that its visits contain useful policy novelty.
3. Keep value work dormant until a stronger, demonstrably useful search corpus
   exists. Playout-cap randomization, forced-playout target pruning, and opening
   exploration are downstream corpus mechanisms, not substitutes for an engine
   strength gate.
4. Preserve paired promotion gates and the distinction between diagnostics and
   vetoes. CPU remains the default while experiments are realistically feasible;
   revisit CUDA only when corpus scale or a redesigned batched search path makes
   inference a material runtime bottleneck.

Implemented `run_puct_budget_ablation.py` and
`src/splendor_ai/training/run_puct_budget_ablation.py` to test that high-budget
hypothesis on CPU without new continuations. It reuses the 320-state all-legal
matched-return audit and compares only frozen-prior s64 and s512 PUCT, with the
same clip-0.20 leaf rule, `c_puct=1.5`, batch-8 inference, and root prior mix
1.0.

- Artifact:
  `outputs/diagnostics/puct035_clip020_cpu_s512_action_quality_late320.json`.
- CPU performance is practical for this gate: s64 averages 0.533 seconds/root
  and 268 simulations/second; s512 averages 4.158 seconds/root and 320
  simulations/second. The complete 320-state comparison took about 25 minutes.
- On the 32 resolved states, s64 has mean regret 0.598 and oracle agreement
  31.25%. s512 is slightly worse: regret 0.609, delta +0.012 with CI95
  [-0.101, +0.120], and oracle agreement 28.13%, delta -3.13 points with CI95
  [-9.38, 0.0]. It changes only 14.7% of root choices.

Decision:

- Do not run s512 paired games or generate a high-budget self-play corpus.
  s512 fails both preregistered action-quality gates on the existing
  all-legal labels.
- The project has now tested the budget-change hypothesis directly. Do not
  reinterpret the rejected s64 line as merely CPU-starved, and do not implement
  KataGo-style target machinery on this engine until a new, independently
  justified search operator exists.

### Mid-Game Headroom Audit

Implemented `run_midgame_headroom_audit.py` and
`src/splendor_ai/training/run_midgame_headroom_audit.py`. It is a resumable,
two-stage all-legal matched-return audit: a cheap screen of 20 states in each
of turns 15--24, 25--32, and 33--40, followed by the full three-determinization,
six-continuation refinement of the ten highest provisional-range states per
band. A checkpoint JSON is written after every completed state so interrupted
CPU runs resume without changing the fixed sample or seeds.

- Artifact: `outputs/diagnostics/midgame_headroom_puct035_two_stage_001.json`.
- Screening covers 60 phase-stratified states; refinement covers 30 states,
  ten in each band. Raw action return ranges are large (band means 1.29--1.51),
  but this is not the relevant confidence measure.
- After full refinement, only 3/30 states have a best-versus-runner-up margin
  above both 0.05 and 1.96 paired standard errors. The resolved states occur
  once in turns 25--32 and twice in turns 33--40; none occur in turns 15--24.
  The policy and recorded s64 PUCT choice miss the oracle on each of those
  three states, but three resolved examples do not establish phase-level
  headroom or support an action-quality promotion gate.

Decision:

- Do not implement or run the policy-guided terminal-rollout leaf pilot yet.
  The proposed sequence required meaningful mid-game resolved headroom first;
  this audit finds high raw variance, not enough reliable action separation.
- The strategic-review claim that mid-game likely supplies substantially more
  usable headroom than late game is not supported by this first controlled
  screen. Outcome-grounded leaves remain a conceptually distinct future
  evaluator class, but are not justified for a live PUCT pilot from these data.

## Best Next Step (SUPERSEDED 2026-08-03 -- provenance only)

**Do not act on this section.** It predates the `070` independent review, `071`,
`072` and `073`. The current direction is the "Next steps" list in the START HERE
section at the top of this file, with reasoning in
`docs/project_review_response_070.md`. The PUCT/value closures recorded below still
stand and are reflected in the review's kill list.

Retain `022` as the frozen historical baseline and `035` as the active legal
prior. Keep clip `0.20` frozen as the best tested PUCT leaf diagnostic, but do
not call it a promotion-ready self-play engine.

Recommended next direction:

1. Stop root-only PUCT visit imitation, terminal WDL value training, and this
   first direct action-Q recipe. `037`, the value models, and action-value 001
   do not justify further same-family sweeps.
2. Stop optimizing clipped n-step leaves. The final 200-game gate has answered
   the promotion question; do not add simulations, sweep clip settings, or
   generate self-play from this engine.
3. Close the current PUCT-improvement line. Root visits, learned value leaves,
   candidate-limited action Q, clipped-leaf promotion, root-prior control, and
   frozen-prior s512 have each received an information-legal offline or paired
   confirmation and do not justify further local sweeps. The initial mid-game
   screen also does not justify an outcome-rollout leaf pilot. Retain `035` as
   the active legal policy and `022` as the historical baseline. Any future
   improvement proposal must start from a new independently justified
   algorithmic direction, not another PUCT parameter, budget, or leaf-target
   variation.
4. Keep all future direct gates paired-seed and information-legal. The
   10-game conversion diagnostic remains diagnostic only, never a standalone
   promotion veto.

The human calibration and clean actor experiment below supersede the earlier
claim that no independently justified non-PUCT direction was available. The
game-level actor family remains conceptually open, but candidate `040` is a
measured near no-op and must not receive strength games.

## Independent ES Gate (Retired)

The PUCT line remains closed. A separate, explicitly gated game-outcome ES
experiment was implemented and evaluated; it does not justify an iterative ES
run.

- New `src/splendor_ai/eval/parallel_paired.py` and
  `run_parallel_paired_benchmark.py` run legal attention checkpoints in
  parallel over seed/seat pairs. Every completed pair is atomically persisted
  and the runner reports paired score share, final margin, and the ES utility
  `0.75 * outcome + 0.25 * tanh(margin / 5)`. It retries atomic progress-file
  replacement briefly on Windows file-lock races. It passed a real `035` versus
  itself one-pair smoke test at
  `outputs/benchmarks/parallel_paired_smoke_035_self.json`.
- New `src/splendor_ai/training/es_adapter.py` constrains ES to a deterministic
  rank-4, fixed-A adapter over four matrices in final attention layer 1. This
  is 3,584 trainable B parameters, not the 98k score head.
- New `src/splendor_ai/training/run_es_stage0.py` and `run_es_stage0.py`
  implement the corrected test: behavior-only sigma calibration on stored
  states; 16 antithetic directions with paired-game fitness; one aggregate
  normalized update; then a fresh paired-seed gate. It does not select the
  best individual direction. A calibration veto now prevents it from spending
  fitness games if its perturbations do not change at least 1% of sampled legal
  actions. Stage 0 otherwise passes only if its fresh bounded-utility lower
  confidence bound exceeds zero with no stall/timeout.
- The clean clone trained from random attention weights on 63,622 legal `035`
  self-play states for three policy-CE-only epochs. Its best held-out action
  accuracy was 86.1%. It tied `035` over a fresh 100-pair gate (46.75% share,
  CI95 [40.25%, 53.25%]) and therefore cleared only the viability prerequisite.
- The first ES Stage 0 used sub-behavioral sigmas `0.25` and `0.75`; both
  changed 0% of sampled actions, so its game result is invalid and ignored.
  A behavior-only scan found mean action-change rates of 1.2%, 5.1%, and 19.1%
  at sigmas 3, 10, and 30. The corrected Stage 0 selected sigma 30 and passed
  against its clean-clone parent: fresh 100-pair utility +0.141, CI95
  [+0.0058, +0.272], with 57.25% score share and no failures.
- This local pass did **not** transfer. The preregistered fresh 200-pair bridge
  against frozen legal `035` is at
  `outputs/benchmarks/es_stage0_cleanclone038_002_aggregate_vs_policy035_paired400.json`:
  score share 49.375%, CI95 [44.375%, 54.375%]; bounded utility -0.0091,
  CI95 [-0.1025, +0.0854]; zero stalls/timeouts. Do not run ES iterations,
  create an ES corpus, sweep sigma/rank, or run a final PGS gate. This adapter
  direction is retired because its only demonstrated gain was parent-relative.

## Human Calibration And Clean LOO Actor

The GUI replay is adequate for a first human calibration. It stores every
pre-action legal set, public observation, full state snapshot, chosen action,
seat, checkpoint metadata, outcome, and final scores. The ten completed games
in `outputs/human_games/replays.jsonl` are balanced across seats. The human won
8/10, 4/5 from each seat, with mean final-score margin +5.6. This is a small,
non-random human sample and not a statistical bot benchmark, but it is strong
evidence that legal `035` is not yet near the practical human-strength goal.
These rows remain held out from all training and promotion gates.

A fresh legal actor-only group-relative experiment was implemented from the
clean-from-random clone `038`, with no critic and no human data:

- `run_group_relative_trajectory_corpus.py` now supports true leave-one-out
  group baselines. Each trajectory is compared with the other three runs of
  the same environment seed and role, not with a baseline containing its own
  outcome.
- New `run_group_relative_signal_audit.py` verifies complete group structure,
  informative return variation, stochastic decision uptake, exact behavior
  probability reconstruction from the frozen parent, and clean termination
  before training.
- The first epsilon-0.20 pilot was rejected before training because only 1.91%
  of written rows were non-greedy versus the preregistered 2.00% floor. The
  threshold was not relaxed. One fixed correction at epsilon 0.30 used fresh
  seeds 81100--81199 and four independent sampling seeds.
- Accepted corpus:
  `data/corpus_trajectory_040_clean_rloo_vs035_e030_grouped/trajectory_advantages.jsonl`.
  It contains 100 complete four-trajectory groups, 86 informative groups, and
  11,035 actor rows. Non-greedy uptake is 2.43%; behavior reconstruction MAE is
  `9.1e-8`. The full audit passes at
  `outputs/analysis/group_relative_signal_audit_040_clean_vs035_e030.json`.
- Exactly one actor epoch was trained from clean clone `038`, with learning
  rate `3e-6`, PPO clip 0.10, KL weight 1.0, and no critic. Training behavior
  reconstruction MAE is `1.1e-7`, mean ratio 1.000006, clip fraction 0.0033,
  and mean exploration KL `5.1e-5`.
- New `run_actor_movement_audit.py` measures greedy action changes and both raw
  and behavior-policy KL on independent 5,000-row samples from the accepted
  on-policy stream and broad legal `035` self-play. Its bounds were fixed
  before inspection: on-policy movement >=1%, broad movement >=0.5%, both
  <=10%, and mean exploration KL <=0.01.
- Candidate `040` fails the movement gate at
  `outputs/analysis/actor_movement_audit_040_clean_rloo.json`: 20/5,000
  on-policy decisions change (0.40%) and 23/5,000 broad decisions change
  (0.46%). Mean exploration KL is only about `5.9e-5` on both streams. This is
  under-movement, not damaging drift.

Decision:

- Reject candidate `040` before games. Do not run the conditional paired gates
  versus `035` or legal PGS and do not reinterpret unplayed gates as ties.
- Do not regenerate trajectories or alter the LOO target; that signal audit
  passed. If this family gets one correction, change only update magnitude from
  clean clone `038` using the same accepted corpus, with a preregistered single
  setting and the same movement/KL gate. Do not sweep learning rate/KL or relax
  the movement floor after observing `040`.
- Only a correction that clears the movement audit earns fresh paired games
  against `035`, followed conditionally by legal PGS. Conversion diagnostics
  remain descriptive and human replays remain held out.

The one allowed magnitude correction has now been run. Candidate `041` reused
the exact accepted corpus, clean `038` initialization, seed, one-epoch schedule,
PPO clip, KL regularization, entropy term, and audit bounds. The only change
from `040` was learning rate `3e-6` -> `1e-5`.

- Training artifact:
  `outputs/legal_action_scorer_041_clean_rloo_actor_lr1e5/legal_action_attention_scorer.pt`.
- Movement artifact:
  `outputs/analysis/actor_movement_audit_041_clean_rloo_lr1e5.json`.
  It passes all fixed gates: 84/5,000 changed decisions (1.68%) on both the
  on-policy and broad streams, with broad mean exploration KL `2.17e-4`.
- Fresh direct gate:
  `outputs/benchmarks/actor041_clean_rloo_lr1e5_vs_policy035_paired400.json`.
  Over 200 paired seeds / 400 games, `041` has score share 49.125%, CI95
  [44.5%, 53.75%], mean outcome -0.0175, bounded utility -0.0071 with CI95
  [-0.0960, +0.0825], and zero stalls/timeouts.

Decision:

- `041` solved the under-movement defect in `040`, but failed the preregistered
  direct superiority gate against legal `035`. Treat it as tied, not promoted.
- Do not run the conditional legal-PGS gate. It was downstream of a positive
  `035` lower confidence bound and that condition was not met.
- Close this single-update clean LOO recipe. Do not sweep learning rate, KL,
  epochs, exploration, or movement thresholds on the same corpus. Any future
  game-outcome actor proposal needs a materially different credit/update design
  and a new justification, not another magnitude correction.

## Held-Out Human-Game Forensic Audit

Implemented `run_human_game_forensic_audit.py` and
`src/splendor_ai/training/run_human_game_forensic_audit.py`. The audit keeps all
human rows held out from training and performs two distinct stages:

- Every `035` turn is reconstructed and classified for robust forced wins,
  immediate opponent-winning replies and available blocks, same-target
  immediate dominance, available scoring conversions, reserve outcomes, and
  repeated non-buy streaks. Tactical claims must agree across eight
  observer-relative hidden-card determinizations.
- Only suspicious states receive expensive outcome-grounded evaluation. The
  recorded action is compared with the best available action from every legal
  action type plus tactical, scoring, dominance, and policy-top alternatives.
  Each action receives 3 determinizations x 8 common-seed policy-guided terminal
  continuations. A mistake resolves only when paired return delta is at least
  0.10, exceeds 1.96 standard errors, and is positive in every determinization.

Final artifacts:

- Machine-readable audit:
  `outputs/diagnostics/human_game_forensic_policy035_full_v2.json`.
- Concise review:
  `outputs/diagnostics/human_game_forensic_policy035_full_v2.md`.
- Crash-resumable state-level progress:
  `outputs/diagnostics/human_game_forensic_policy035_full_v2.progress.json`.

Results:

- All 271 `035` turns across the ten games reconstruct exactly; recorded and
  reloaded policy-top actions agree 100%.
- There are zero robust missed forced wins, zero failures to block an immediate
  opponent win, zero same-target dominance cases, and zero turns where `035`
  declined an already-affordable scoring purchase. The replay does not support
  an endgame tactical-failure diagnosis or the proposed tactical override.
- Screening flags 29/271 turns: 26 reserve suspicions and three four-turn
  non-buy streaks. All 29 receive matched terminal continuations with complete
  action-class coverage and 24 samples per action.
- Eight states resolve as high-confidence mistakes. All eight occur in games
  the human won; seven are early and one mid-game. Every chosen action is a
  reserve. Better action transitions are four different visible reserves,
  three token takes, and one reserved-card purchase.
- Seven resolved chosen reserves were never converted; one reserved an already
  affordable zero-point card. Mean paired continuation improvement is +0.621.
  The policy nevertheless favors the worse chosen actions by a mean +2.709
  score units over the better alternatives. This is a large ranking error, not
  a near-tie at the policy head.

Decision:

- Do not build an endgame tactical wrapper from these games. It would address
  none of the eight resolved errors.
- Classify the observed gap as early reserve/target planning, not immediate
  conversion and not terminal tactics. The relevant credit horizon is whole
  trajectory: four errors prefer a different reserve target and four prefer a
  different action class whose benefit only appears downstream.
- Do not train on these ten human games. First run the same outcome-grounded
  action-class audit on a disjoint sample of early `035` reserve states from
  non-human legal games. This tests whether the pattern is a general policy
  defect or specific to openings the human exploited.
- If the reserve-target error generalizes, use those non-human matched outcomes
  to design a reserve-focused multi-generation/league learner with broad
  opponent coverage. If it does not, the next learner needs explicit opponent
  diversity rather than a generic reserve correction. In either case, do not
  return to a single local update against frozen `035`.

## Non-Human Reserve Generalization Audit

Implemented `run_reserve_generalization_audit.py` and
`src/splendor_ai/training/run_reserve_generalization_audit.py`. The experiment
was fixed before outcome inspection:

- Source corpus:
  `data/corpus_generation1_policy035_selfplay_038/replays.jsonl`, containing
  1,000 non-human legal-observation policy035 self-play games at seeds
  47000-47999. Human replay rows are explicitly rejected.
- The runner screened all 20,000 turns 0-19 and audited only states where the
  frozen deterministic policy035 argmax itself was a reserve. This prevents
  the source corpus's epsilon exploration from manufacturing policy errors.
- Sixty states from 60 unique games were sampled without looking at rollout
  outcomes: 15 each from turns 0-4, 5-9, 10-14, and 15-19.
- Every candidate action received 3 determinizations x 8 paired terminal
  continuations. Selection and completed state outcomes are crash-resumable,
  and the raw matched returns are retained for later analysis.
- The preregistered generalization gate required complete continuations, at
  least six strict per-state resolutions, a Wilson 95% rate lower bound above
  3%, resolved states in at least two turn bands, and mean resolved paired
  delta at least +0.20.

Final artifacts:

- `outputs/diagnostics/reserve_generalization_policy035_full.json`
- `outputs/diagnostics/reserve_generalization_policy035_full.md`
- `outputs/diagnostics/reserve_generalization_policy035_full.progress.json`

Results:

- The preregistered gate passes: 12/60 resolved mistakes (20.0%, Wilson 95% CI
  11.8%-31.8%), present in all four bands. The rates are 2/15, 3/15, 6/15,
  and 1/15 respectively, peaking at 40% on turns 10-14.
- Mean paired return improvement on resolved states is +0.836. Policy035 ranks
  the worse chosen reserve above the rollout-preferred alternative by a mean
  1.976 score units.
- Ten of the 12 resolutions remain significant under a post-audit Bonferroni
  robustness check correcting for selection of the best candidate alternative.
- Nine better alternatives are another reserve action and three are token
  takes; none is an immediate buy. The defect is mainly reserve target/tier
  selection, not a blanket tendency to reserve when another action class is
  obviously preferable.
- The recorded stochastic corpus action matches the audited deterministic
  policy action on 85% of all selected states and 11/12 resolved states. The
  result is therefore not driven by exploratory source actions.

Decision:

- The human reserve-planning diagnosis generalizes to disjoint non-human
  trajectories. Do not build a blanket reserve penalty or an endgame tactical
  override; both would target the wrong failure mode.
- Keep these 60 states as an audit/validation block. The next justified learning
  line is a fresh, larger reserve-choice corpus from disjoint games, weighted
  toward turns 5-14 and preserving comparisons among reserve targets as well as
  reserve-versus-take choices.
- Use that signal inside an iterative multi-generation learner evaluated against
  a broad league, not another one-shot correction against frozen `035`. Require
  fresh held-out reserve-ranking improvement before any paired game promotion
  gates, then test against `035`, legal PGS, and at least one held-out league
  opponent.

### Reserve Counterfactual Calibration Candidate `042`

The completed 500-state `035`-vs-`016` corpus was compiled by
`run_reserve_counterfactual_corpus.py` into a frozen, game-disjoint training
contract at `data/corpus_reserve_counterfactual_042`:

- 400 train games and 100 internal held-out games, with 100/25 games in each
  of the turn bands 0-4, 5-9, 10-14, and 15-19.
- The earlier 60-state policy035 self-play audit is a separate external check
  and is never optimized on.
- A separate 5,000-row legal policy035 parent-KL anchor stream was sampled from
  a different `035`-vs-`022` league block. It supplies no action targets.

`run_reserve_counterfactual_training.py` trained exactly one candidate from
legal `035`. It used the soft matched terminal-return distribution over every
evaluated candidate action, no one-hot replay CE, no behavior cloning, and
only the final score head (98,817 parameters) was trainable. State/action
encoders and action-set attention were frozen. The fixed schedule was four
epochs at learning rate `1e-5`, target temperature 0.35, parent-KL weight 2.0,
and CPU execution.

Artifacts:

- Candidate and metrics:
  `outputs/legal_action_scorer_042_reserve_counterfactual/`.
- Fixed uptake/drift gate:
  `outputs/legal_action_scorer_042_reserve_counterfactual/reserve_counterfactual_uptake_audit.json`.

Results:

- Soft return calibration improves decisively on the 100 held-out games:
  counterfactual CE delta `042 - 035` is -0.1246, CI95 [-0.1436, -0.1071].
  It also improves on the held-out 60-state external audit: -0.1128, CI95
  [-0.1334, -0.0940].
- Broad drift is bounded: 15/2,000 changed top actions (0.75%) and mean parent
  KL 0.00423.
- It fails the preregistered decisive-action gate. Held-out top-one return
  accuracy is only 20% versus `035`'s 18%, delta +2 points with CI95 [0, +5].
  Spearman improvement is likewise inconclusive (+0.005, CI95 [-0.01, +0.02]).
  The external 60-state best-action accuracy is unchanged at 20%.

Decision:

- Reject `042` before games. Soft target calibration alone is not a promotion
  criterion and must not be mistaken for improved play.
- Do not tune its epoch count, learning rate, KL weight, target temperature, or
  score-head scope. That would be a response to its failed top-action gate.
- The 500-state corpus remains valuable evidence that the policy has a real
  early reserve-target defect, but this specific soft-distribution imitation
  objective does not convert that evidence into better discrete decisions.

### Reserve Objective/Capacity Diagnostic

The bounded follow-up tested exactly two offline-only explanations for `042`'s
failure on the same frozen 400/100/60 split and 5,000-row parent-KL anchors.
Neither candidate was eligible for games.

- `043` changes only the target objective: exact highest-return candidate CE
  instead of the soft return distribution, while retaining the 98,817-parameter
  score-head-only scope and all `042` optimizer settings.
- `044` keeps that hard top target but unfreezes the full 666,114-parameter
  model. The corpus, four epochs, learning rate, anchor-KL weight, and gates
  remain unchanged.

The comparison artifact is
`outputs/diagnostics/reserve_counterfactual_objective_capacity_diagnostic.json`
and its concise Markdown report is `.md`.

Results:

- `043` does not improve top-one accuracy even on the 400 training games
  (-0.5 points, CI95 [-1.25, 0]) or the 100 held-out games (0 points). It
  therefore rejects soft-target dilution as the explanation.
- `044` improves held-out distributional CE by -0.560, but its training top-one
  gain is inconclusive (+2.5 points, CI95 [-0.25, +5.75]) and held-out top-one
  accuracy regresses by 3 points, CI95 [-10, +4]. Its broad parent KL is 0.0586,
  far above the fixed 0.01 ceiling.

Decision:

- Close this one-step matched-return policy-imitation family: soft versus hard
  target sharpness and score-head versus full-model capacity do not produce a
  generalizable discrete-action improvement under bounded drift.
- Do not run paired games, generate another corpus, or tune epochs, learning
  rate, KL, target temperature, or scope within this family.
- Any next learner must change the credit or decision process itself, rather
  than trying to fit the same offline return rankings more aggressively.

### Clean Entity Actor-Critic Smoke `005`

Implementation:

- `src/splendor_ai/training/entity_actor_critic.py`
- `src/splendor_ai/training/run_entity_actor_critic.py`
- `run_entity_actor_critic.py`
- `tests/test_entity_actor_critic.py`

The public actor checkpoint is initialized from frozen `entity_scale_004` and
retains the 13-type public entity contract. The separate privileged critic has
two additional entity types for actual hidden-deck cards and opponent blind
reserves; privileged state is never passed to the actor. Every canonical legal
successor has positive behavior probability through a 5% uniform mixture.
Rollouts retain the exact old action distribution and use terminal score outcome
with GAE/TD-lambda credit, clipped PPO updates, and explicit KL, entropy,
importance-ratio, exploration, action-coverage, and rollout-integrity gates.

CPU smoke artifacts are under `outputs/canonical_entity_actor_005_smoke`:

- 12 fresh league games, 375 actor decisions, zero stalls/timeouts;
- opponents evenly split across `entity004`, Greedy, legal shallow depth 1,
  `016`, `035`, and PUCT035 s8;
- minimum behavior probability 0.000734, five action types, 304/375 sampled
  actions different from the actor argmax, and rollout entropy 0.841;
- behavior log-probability reconstruction error 6.89e-7, max post-update
  importance ratio 1.0158, mean KL 2.04e-5, and post-update entropy 0.841;
- paired no-regression smoke: candidate 12, frozen parent 8, no draws or
  timeouts; every preregistered runner gate passed.

Decision:

- The architecture, legality boundary, stochastic rollout path, credit update,
  and conservative gates work end to end on CPU. A GPU is not justified by
  this workload yet.
- Do not promote `005` from a 20-game readout. The independent larger `006`
  generation below supersedes its apparent 12-8 edge.

### Clean Entity Actor-Critic Generation `006`

`outputs/canonical_entity_actor_006_seed90000` was trained from immutable `004`
with the same conservative hyperparameters as `005`, but on 72 fresh league
games at seed 90000 (12 each against `004`, Greedy, legal shallow depth 1,
`016`, `035`, and PUCT035 s8). Its 2,364 actor decisions had five action types,
minimum behavior probability 0.000541, 1,818 non-argmax samples, and zero
stalls/timeouts. All implementation gates passed.

Update diagnostics:

- behavior reconstruction error 7.84e-7;
- max importance ratio 1.0375 and mean ratio 0.99987;
- mean KL 6.88e-5 and normalized entropy 0.8335;
- policy loss -5.03e-6 and critic loss 0.1296.

Fresh strength gates:

- paired `006` versus frozen `004`, 100 games: 46-54, one timed-out game;
- same seeds versus `016`, 60 games: `006` 7-53 and `004` 8-52, with one
  timeout in each block;
- same seeds versus legal `035`, 60 games: `006` 2-58 and `004` 1-59;
- same seeds versus Greedy, 60 games: `006` 54-6 and `004` 55-5;
- `006` versus legal shallow depth 1, 60 games: 9-51, matching the 15% score
  share of `004`'s earlier 5-15 block. All rule-opponent games completed with
  no stalls/timeouts or loop fallbacks.

Artifacts:

- `outputs/canonical_entity_actor_006_seed90000/actor_critic_metrics.json`
- `outputs/benchmarks/entity_actor006_vs016_seed92000_g60.json`
- `outputs/benchmarks/entity004_vs016_seed92000_g60.json`
- `outputs/benchmarks/entity_actor006_vs035_seed93000_g60.json`
- `outputs/benchmarks/entity004_vs035_seed93000_g60.json`
- `outputs/benchmarks/entity_actor006_rules_seed94000_g60.json`
- `outputs/benchmarks/entity004_greedy_seed94000_g60.json`

Decision:

- Reject `006` as a no-movement generation and retain `004` as the frozen
  parent. The `005` 12-8 result was small-sample noise, not replicated progress.
- Do not spend another full league/search gate on the identical update recipe.
  Add an offline effective-movement gate first: report gradient norm, actor
  parameter/logit movement, advantage-action covariance, and target-state KL.
  Then test a KL-targeted multi-epoch or higher-learning-rate update that aims
  for measurable but bounded movement before launching games.
- CPU remains appropriate. The expensive portion is serial simulation/search,
  not the small actor/critic optimization; a GPU would not fix this blocker.

### KL-Targeted Entity Actor-Critic Generation `007`

The actor-critic implementation now records pre-clipping gradient norms,
parameter movement, centered-logit movement, probability total variation,
argmax changes, per-epoch rollout-state KL, advantage/log-probability
covariance, and mean raw advantage by sampled action type. Actor training can
run for multiple epochs but stops after the first epoch whose exact rollout-
state KL reaches `target_kl`. The runner can skip parent games unless all
offline movement and safety gates pass.

`outputs/canonical_entity_actor_007_kl_target` used 36 fresh league games at
seed 100000, six maximum actor epochs, actor learning rate 3e-5, target KL
0.002, minimum effective KL 0.0005, centered-logit movement floor 0.005, and
hard KL ceiling 0.01. The six league opponents were evenly represented.

Results:

- 1,156 actor steps, zero rollout stalls/timeouts, and five action types;
- target reached after one epoch: mean KL 0.002415, max importance ratio 1.194,
  normalized entropy 0.842, and policy loss -0.01356;
- first-batch/mean/max pre-clipping gradient norms 0.937/1.355/6.339;
- parameter delta L2 0.02473, relative delta 0.000654;
- mean/max centered-logit absolute movement 0.04244/0.2562;
- mean/max probability TV 0.02714/0.07068 and 4.93% argmax changes;
- advantage/log-probability correlation +0.254. Mean raw advantages were
  +0.104 for visible buys, +0.070 for blind reserves, -0.0075 for reserved
  buys, -0.0225 for visible reserves, and -0.0961 for token takes.

Strength:

- initial paired parent gate, 60 games at seed 101000: 37-22-1 for `007`, with
  one repetition cutoff;
- independent paired confirmation, 100 games at seed 102000: 52-48 for `007`,
  with two repetition cutoffs;
- aggregate readout 89-70-1, score share 0.559, but the larger independent block
  does not establish a repeatable advantage.

Decision:

- The no-movement blocker is fixed: the higher learning rate creates measurable
  policy movement and the KL stop bounds it automatically. This is a successful
  learning-mechanism experiment.
- Do not promote `007`; retain frozen `004`. Do not run the expensive broad or
  shallow-search gates after a 52-48 independent confirmation.
- The next eligible experiment is the same fixed KL-target recipe on another
  disjoint rollout seed. Require another independently trained candidate to
  clear a preregistered parent threshold before broad-league confirmation.
- CPU remains sufficient; optimization and diagnostics completed quickly
  relative to rollout/search simulation.

Artifacts:

- `outputs/canonical_entity_actor_007_kl_target/actor_critic_metrics.json`
- `outputs/benchmarks/entity_actor007_vs004_seed102000_g100.json`

### Independent KL-Target Replication `008`

`outputs/canonical_entity_actor_008_kl_target_seed110000` repeated the exact
`007` training recipe from frozen `004` on rollout seeds 110000-110035. Before
the run, progression required at least 36/60 parent score points, bounded
offline diagnostics, and at most one cutoff. A fresh 100-game confirmation was
eligible only after that gate, and broad/search evaluation required at least
55/100 in confirmation.

Results:

- 36 clean league games and 1,169 actor steps; zero rollout stalls/timeouts;
- final KL 0.002282, max ratio 1.208, entropy 0.816, and probability TV 0.0252;
- KL by epoch: 0.000107, 0.000200, 0.000951, 0.001476, 0.002282; the target was
  reached after five epochs rather than `007`'s single epoch;
- parameter delta L2 0.07163 and relative delta 0.001894, nearly three times
  `007`'s relative movement despite similar final behavior-policy KL;
- centered-logit movement 0.06323 and 8.81% argmax changes, versus 0.04244 and
  4.93% for `007`;
- advantage/log-probability correlation +0.167, weaker than `007`'s +0.254;
- paired parent progression, 60 games at seed 111000: 24-34-2, no cutoffs.

Decision:

- Reject `008` and stop at the preregistered progression gate. The 100-game
  confirmation and broad/search gates were correctly skipped.
- The KL/movement mechanism works mechanically but the learned direction does
  not replicate across 36-game credit samples. Do not repeat another 36-game
  seed or tune learning rate/target KL against these two outcomes.
- The next learner experiment should reduce credit variance before updating:
  use a materially larger balanced rollout block and/or opponent/seat-stratified
  advantage normalization, then require both held-out movement bounds and the
  same parent progression gate. Frozen `004` remains the parent.

Artifact:

- `outputs/canonical_entity_actor_008_kl_target_seed110000/actor_critic_metrics.json`

### Balanced 120-Game Actor-Critic Generation `009`

Audit and implementation:

- The prior scheduler selected opponent with `game_index % 6` and actor seat
  with `game_index % 2`. Because six is even, each opponent was permanently
  aliased to one actor seat in generations `005`-`008`.
- The scheduler now cycles through every `(opponent, actor_seat)` pair before
  repeating. Rollout metrics include counts and outcomes by opponent and seat,
  plus minimum/maximum stratum size.
- The runner now has enforceable minimum parent score-share and maximum cutoff
  arguments. Its `all_gates_passed` result can represent a real progression
  threshold rather than the historical 40% no-regression smoke floor.
- Tests assert exact two-seat coverage for multiple opponents. The current full
  suite is 281 passing after matched return-branch coverage below.

`outputs/canonical_entity_actor_009_balanced120` used rollout seeds
120000-120119, the fixed `007/008` KL-target optimizer settings, and exactly 10
games in each of the 12 opponent-seat cells. Progression required 60% score
share in 60 paired parent games and at most one cutoff.

Rollout evidence:

- 120 games, 3,812 actor steps, zero stalls/timeouts;
- actor seat 0: 14-46; actor seat 1: 9-51;
- versus `004`: 6-14; versus Greedy: 17-3;
- versus legal shallow depth 1, `016`, `035`, and PUCT035 s8: 0-20 against
  every opponent, totaling 0-80 across the four strong strata.

Update diagnostics:

- one epoch reached KL 0.003173; max ratio 1.285 and entropy 0.817;
- probability TV 0.03063, relative parameter delta 0.001007, centered-logit
  movement 0.05051, and 6.93% rollout-state argmax changes;
- advantage/log-probability correlation fell to +0.090, below `007` (+0.254)
  and `008` (+0.167).

Strength and decision:

- Paired parent progression at seed 121000: `009` 26, frozen `004` 34, no
  draws or cutoffs. The 60% gate failed; confirmation and broad gates were
  correctly skipped.
- Reject `009` and retain `004`. Larger and seat-balanced sampling did not make
  sparse terminal win/loss credit reliable.
- Do not apply opponent/seat-stratified normalization to these data: eight
  opponent-seat cells among the four strong opponents contain only losses, so
  normalization would create relative gradients without positive outcome
  evidence.
- Before another actor update, change the training league to opponents within
  the actor's learning frontier (non-degenerate win rates), or replace binary
  terminal outcome with a validated denser return such as final score margin.
  Keep `016`, `035`, and PUCT as evaluation opponents until the clean actor can
  occasionally beat them.

Artifact:

- `outputs/canonical_entity_actor_009_balanced120/actor_critic_metrics.json`

### Frontier-Curriculum Actor-Critic Generation `010`

The runner now supports a provenance-recorded `frontier` league profile. It
trains only against frozen `004` and Greedy; legal shallow search, `016`, `035`,
and PUCT remain evaluation-only until the actor can produce non-degenerate
outcomes against them. The historical `full` profile remains available.

`outputs/canonical_entity_actor_010_frontier120` used rollout seeds
130000-130119, the fixed KL-target optimizer, and 30 games in each of the four
`(004/Greedy) x (seat0/seat1)` cells. Progression again required 60% score share
in 60 paired parent games and at most one cutoff.

Rollout evidence:

- 120 games, 4,070 actor steps, zero stalls/timeouts;
- actor seat 0 and actor seat 1 both scored exactly 26-34;
- versus `004`: 9-51; versus Greedy: 43-17. Both opponent groups contained
  positive and negative outcomes, removing `009`'s all-loss strata.

Update diagnostics:

- one epoch reached KL 0.003153; max ratio 1.282 and entropy 0.773;
- probability TV 0.02907, relative parameter delta 0.001173, centered-logit
  movement 0.05151, and 5.01% rollout-state argmax changes;
- advantage/log-probability correlation remained weak at +0.053, below `009`'s
  +0.090 despite the non-degenerate curriculum.

Strength and decision:

- Paired parent progression at seed 131000: `010` 33, frozen `004` 26, one
  draw; score share 0.558. Three repetition cutoffs occurred.
- Reject `010`: it misses the preregistered 0.60 score-share threshold and the
  cutoff ceiling of one. The independent 100-game confirmation and strong-
  opponent gates were correctly skipped.
- The frontier curriculum improves the directional game readout relative to
  `009`, but does not establish repeatable strength and introduces loop risk.
  Keep `004` frozen.
- Outcome degeneracy is no longer the main blocker. Before another policy
  update, validate a denser credit target—preferably final score margin—against
  terminal win/loss on frozen rollout trajectories. Do not tune the curriculum,
  learning rate, or KL target against `010`'s near-miss.

Artifact:

- `outputs/canonical_entity_actor_010_frontier120/actor_critic_metrics.json`

### Final Score-Margin Return Validation `011`-`012`

Implementation:

- Actor rollouts now retain opponent label, raw final score margin, and a
  bounded normalized score-margin return in addition to terminal outcome.
- `entity_return_validation.py` extracts one record per game and reports global
  and opponent-seat-stratified target density, standard deviation, outcome
  alignment, and saturation.
- `run_entity_return_validation.py` runs the full balanced league without an
  optimizer or actor update. It gates recording completeness, rollout
  cleanliness, exact opponent-seat balance, sign alignment, density,
  saturation, and variation in every all-loss cell.

Calibration `011`:

- Artifact: `outputs/diagnostics/entity_score_margin_validation_011.json`.
- 120 fresh games at seeds 140000-140119, exactly 10 per opponent-seat cell,
  zero stalls/timeouts.
- Raw score margin had 30 distinct values, correlated 0.883 with outcome,
  preserved outcome direction in 98.3% of games, and added meaningful
  variation in all six all-loss cells.
- The initially proposed `clip(raw / 15, -1, 1)` saturated 20.8% of games,
  exceeding the fixed 10% ceiling. Reject the `/15` scale while retaining the
  raw-margin hypothesis.

Independent scale confirmation `012`:

- Artifact:
  `outputs/diagnostics/entity_score_margin_validation_012_scale20_confirm.json`.
- 120 disjoint games at seeds 150000-150119, again exactly 10 per cell with zero
  stalls/timeouts. Frozen `004` was not updated.
- `clip(raw / 20, -1, 1)` produced 31 distinct values versus two binary outcome
  values, correlation 0.870, 98.3% sign alignment, and zero saturation.
- All six all-loss cells passed the preregistered density requirement. Each had
  4-7 unique margins and normalized standard deviation 0.056-0.138, above the
  0.03 floor.
- Two non-draw games ended with tied scores and were decided by Splendor's card-
  count tiebreak, explaining the 98.3% rather than 100% sign alignment; this
  remains above the fixed 95% floor.

Decision:

- Validate normalized final score margin with scale 20 as a denser return
  candidate. This decision generalizes across two disjoint balanced 120-game
  blocks and specifically resolves binary outcome degeneracy in all-loss cells.
- This is return validation only. No actor or critic candidate was trained, no
  game-strength claim follows, and frozen `004` remains the parent.
- The next eligible learning experiment may substitute the validated score-
  margin return for binary terminal outcome while keeping the corrected
  opponent-seat scheduler, KL/movement gates, and strict parent progression
  thresholds fixed. Do not tune the scale against downstream game results.

### Matched Return-Target A/B `013`

Implementation:

- Actor-critic training can now reuse one collected rollout corpus through
  `train_actor_critic_from_steps`; branch-local advantages/value targets are
  cloned while immutable states, legal actions, and behavior probabilities are
  shared.
- `ActorCriticConfig.return_mode` selects `outcome` or `score_margin` without
  changing collection, actor initialization, critic initialization, shuffle
  seeds, optimizer settings, or KL gates.
- A game-disjoint critic split excludes every fifth game from warmup and reports
  held-out MSE, constant-baseline MSE, normalized skill, and correlation before
  GAE. Both branches use the exact same 759 held-out actor states.
- `run_entity_return_ab.py` requires score margin to improve held-out critic
  skill, critic correlation, and absolute advantage/action correlation before
  allowing any games. Initial game gates require 55% versus the binary control
  and 60% versus frozen `004`, with at most one cutoff.

Experiment:

- Artifact: `outputs/canonical_entity_return_ab_013/return_ab_metrics.json`.
- One 120-game balanced full-league corpus at seeds 160000-160119, 3,869 actor
  steps, exactly 10 games per opponent-seat cell, zero rollout stalls/timeouts.
- Both branches started from frozen `004`, used actor learning rate 3e-5, target
  KL 0.002, five critic warmup epochs, and otherwise identical schedules.

Matched offline results:

- binary outcome critic: held-out MSE 0.2414 versus constant 0.7137, normalized
  skill 0.6617, correlation 0.8157;
- score-margin critic: held-out MSE 0.0566 versus constant 0.2194, normalized
  skill 0.7422, correlation 0.8660;
- score-margin advantage/log-probability correlation 0.0665 versus binary
  0.0314;
- score margin moved less at the policy level despite better credit: KL 0.00403
  versus 0.00660, probability TV 0.0346 versus 0.0433, and argmax changes 5.04%
  versus 6.18%. Both branches remained inside all safety bounds.

Initial games and decision:

- score-margin treatment versus matched binary control, paired 60 games at seed
  161000: 33-27, no draws or cutoffs; the fixed 55% treatment gate passes
  exactly.
- score-margin treatment versus frozen `004`, paired 60 games at seed 162000:
  33-27, no draws or cutoffs; score share 0.55 fails the fixed 0.60 progression
  threshold.
- Reject the score-margin policy candidate and do not run independent
  confirmation or strong-opponent gates. Frozen `004` remains the parent.
- Retain score margin as the superior credit target: it wins the controlled
  offline comparison and the direct matched-candidate game gate. The remaining
  blocker is converting that improved credit into a repeatable parent gain, not
  target density or critic learnability. Do not tune `/20`, learning rate, or KL
  against this near-miss.

### Matched Temporal-Credit A/B `014`

Implementation:

- `ActorCriticConfig.return_mode="score_margin_delta"` trains the critic on
  remaining normalized final margin and gives each actor transition the change
  in normalized score margin from its current decision state to the next actor
  decision state, or to the terminal state on the final step.
- The per-transition rewards telescope exactly to final normalized margin minus
  the margin already present at the first actor decision. This is causal,
  preserves the `/20` objective scale, and avoids giving the policy credit for
  points scored before it acted.
- Update telemetry now reports nonzero temporal-reward fraction and mean
  absolute temporal reward. `run_entity_temporal_credit_ab.py` reuses one
  rollout corpus and identical initial weights, validation games, optimizer
  settings, shuffle seeds, and KL gates for terminal-margin control and dense
  treatment.
- Offline gates require both branches to be safe, dense rewards to be at least
  1.5x less sparse, and dense treatment to improve held-out critic skill,
  critic correlation, and absolute advantage/log-probability correlation.
  Only then are fixed 55% treatment-control and 60% frozen-parent game gates
  eligible.

Experiment:

- Artifact:
  `outputs/canonical_entity_temporal_credit_ab_014/temporal_credit_ab_metrics.json`.
- One 120-game balanced full-league corpus at seeds 170000-170119, 3,845 actor
  steps, exactly 10 games per opponent-seat cell, zero stalls/timeouts, and
  752 actor states from every fifth game held out from critic warmup.
- Both branches started from frozen `004` and retained the `013` update recipe:
  actor learning rate 3e-5, target KL 0.002, five critic warmup epochs, six
  maximum actor epochs, and identical deterministic schedules.

Matched results and decision:

- Terminal-margin control: 3.12% nonzero temporal rewards, mean absolute reward
  0.0161, held-out critic MSE 0.0733 versus constant 0.2248, skill 0.6738,
  correlation 0.8282, advantage standard deviation 0.1611, and
  advantage/log-probability correlation 0.0460.
- Dense score-change treatment: 28.14% nonzero temporal rewards, mean absolute
  reward 0.0271, held-out critic MSE 0.0816 versus constant 0.1957, skill
  0.5832, correlation 0.7679, advantage standard deviation 0.1813, and
  advantage/log-probability correlation 0.0443.
- Both policy updates were safe and comparable: terminal/dense KL
  0.00565/0.00539, max ratio 1.336/1.331, centered-logit movement
  0.0702/0.0691, and one actor epoch each after the target-KL stop.
- Dense rewards passed the sparsity gate by roughly 9x but failed all three
  learning-quality gates. The runner correctly skipped treatment-control and
  frozen-parent games; no game-strength claim is made.
- Reject `014`; frozen `004` remains the parent. Do not sweep the dense reward
  scale, learning rate, or KL. Potential-based redistribution made the critic
  target harder and increased credit variance. The next justified direction is
  action-conditional/counterfactual advantage estimation that separates the
  selected action from opponent-response variance.

### Entity Action-Conditional Credit Diagnostic `015`

Implementation and fixed gates:

- `run_entity_counterfactual_credit.py` collects fresh stochastic `004`
  trajectories against the balanced six-opponent league without changing any
  checkpoint. It selects one state per game, equally across turns 0-19, 20-39,
  and 40+, so state-level uncertainty is game-disjoint.
- Each state compares the parent top three actions, the behavior-sampled action
  when different, and action-type diversity up to five candidates. Every
  candidate receives the same observer-relative hidden-card determinizations
  and continuation seeds under stochastic `004` play.
- Four continuations per determinization are split 2/2, producing independent
  four-return halves per action. Diagnostics cover full action-ranking
  Spearman/top-one replication, behavior-action centered-advantage correlation
  and sign replication, return range, parent regret, and cutoff rate.
- Before collection, acceptance required clean balanced source games, 18 unique
  games with six states per phase, at most 1% continuation cutoffs, at least
  50% informative states, split top-one agreement at least 50%, positive lower
  confidence bounds for both mean split Spearman and chosen-advantage
  correlation, at least five sign-eligible states with 60% sign agreement, and
  nondegenerate positive and negative advantage tails.

Experiment:

- Artifact: `outputs/diagnostics/entity_counterfactual_credit_015.json`.
- Source seeds 180000-180035 produced 36 games and 1,136 actor states, exactly
  three games in each opponent-seat cell, with zero stalls/timeouts. The actor
  went 7-29; outcomes were not used for state selection.
- The deterministic selection contains 18 different games, six states per
  early/mid/late phase. All states have five candidate actions. Two legal
  determinizations and four continuations per determinization produced 720
  complete terminal continuations and zero cutoffs.

Results and decision:

- All 18 states exceeded the fixed 0.05 action-return range; mean range was
  0.273. Split top-one agreement was 10/18 (55.6%), and mean split Spearman was
  0.568 with CI95 [0.329, 0.777]. The action-ranking replication gate passes.
- Behavior-action advantages were nondegenerate: standard deviation 0.090,
  27.8% positive, 72.2% negative, with 15 sign-eligible states and 73.3% sign
  agreement. However, half-to-half magnitude correlation was only 0.154 with
  CI95 [-0.354, 0.783]. The preregistered chosen-advantage gate fails.
- The parent top action differed from the full-return top action on 12/18
  states; mean parent regret was 0.0868. Only three discovery-half best actions
  were independently resolved on the validation half. This is headroom
  evidence, not a safe policy target.
- Post-hoc phase inspection localizes the instability: chosen-advantage
  correlation is -0.321 early, 0.905 mid, and 0.842 late; eligible sign
  agreement is 33%, 100%, and 100%, respectively. This cannot retroactively
  change the aggregate gate.
- Do not train from `015` and do not perform a full-game counterfactual policy
  update. Frozen `004` remains the parent. A fresh, preregistered mid/late-only
  confirmation with more independent continuations is scientifically eligible;
  it must reproduce both phases separately before any matched policy update.

### Fixed-Baseline Mid/Late Confirmation `016`

Implementation and preregistration:

- `run_entity_counterfactual_confirmation.py` preserves `015`'s original
  candidate-mean estimator for reproducibility and selects a new fixed
  policy-reference mode. The five reference actions depend only on frozen
  parent scores and action-type coverage. A sampled behavior action outside
  that set is evaluated as a sixth action but cannot change the baseline.
- The baseline is the softmax-policy-weighted return of the fixed reference
  set. Fresh selection targets were fixed at six early negative-control states,
  12 mid states, and 12 late states, all from different games.
- Each action received three observer-relative determinizations and eight
  matched continuations per determinization. The 12/12 independent halves
  tripled per-action evidence over `015`.
- Mid and late were gated independently. Each required at least 50% informative
  states, 50% split top-one agreement, positive lower 95% bounds for split
  Spearman and chosen-advantage correlation, at least six sign-eligible states
  with 70% agreement, at least 10% in both advantage tails, advantage standard
  deviation at least 0.03, and at least two discovery-best actions independently
  resolved on the validation half. Both phases had to pass; early results could
  not affect the decision.

Experiment:

- Artifact: `outputs/diagnostics/entity_counterfactual_confirmation_016.json`.
- Source seeds 190000-190047 produced 48 games and 1,563 actor states, exactly
  four games in every opponent-seat cell, with zero source stalls/timeouts.
  The actor went 9-39; outcomes were not used for selection.
- The 30 selected states exactly matched the fixed phase counts and unique-game
  requirement. Candidate counts averaged 5.67 early and 5.33 mid/late because
  off-reference sampled actions were retained.
- The run produced 3,888 continuations. Twelve mid continuations cut off,
  0.31% overall, below the fixed 1% ceiling; all others reached terminal state.

Phase results:

- Early negative control: all 6 informative, top-one agreement 3/6, split
  Spearman 0.603 with CI95 [0.384, 0.805], chosen-advantage correlation 0.821
  with CI95 [-0.228, 1.000], sign agreement 5/6, and one independently resolved
  discovery-best action. Early remained underpowered/inconclusive as intended.
- Mid: 10/12 informative, top-one agreement 5/12, split Spearman 0.529 with
  CI95 [0.207, 0.766], chosen-advantage correlation 0.831 with CI95
  [0.320, 0.958], sign agreement 7/9, advantage standard deviation 0.0666, and
  four independently resolved discovery-best actions. Every criterion except
  the fixed 6/12 top-one threshold passed.
- Late: 11/12 informative, top-one agreement 7/12, split Spearman 0.812 with
  CI95 [0.645, 0.933], chosen-advantage correlation 0.670 with CI95
  [0.437, 0.867], sign agreement 6/6, advantage standard deviation 0.0308, and
  two independently resolved discovery-best actions. Every late gate passed.

Decision:

- The confirmation fails because mid top-one replication is 5/12 rather than
  the fixed 6/12 and both phases were required. Do not weaken the discrete gate,
  rerun fresh seeds to chase one state, or train a late-only candidate after
  observing the result.
- No actor or critic checkpoint was trained and no games were run. Frozen `004`
  remains the parent. The continuous counterfactual signal is real, especially
  late, but has not satisfied the complete contract for a safe policy update.

### Scaled Late Counterfactual Corpus/Q Gate `017`-`019`

Implementation:

- `run_entity_late_counterfactual_corpus.py` generates a fixed late-only corpus
  with one state per source game, action-independent policy reference sets,
  immutable game-level train/validation/confirmation splits, atomic JSONL
  progress, restart support, and parallel state evaluation. Corpus quality
  gates must pass before Q training can load it.
- `run_entity_action_value_training.py` initializes the canonical entity model
  from frozen `004`, deterministically resets its score head, and trains one
  auxiliary Q model with Smooth L1 action-return loss plus pairwise ranking.
  The validation-selected epoch is evaluated once on untouched confirmation.
  Both splits must beat parent regret and Spearman with strict confidence bounds
  and improve top-one direction before any actor update becomes reachable.

Selector failure and repair:

- First artifact `outputs/diagnostics/entity_late_counterfactual_corpus_017.json`
  completed 240 states and 32,208 continuations, but its phase gate failed.
  A zero early target was not skipped by the shared selector, so all 240 states
  were early rather than late. No Q training consumed `017`.
- The selector now explicitly skips zero-target phases, with a regression test
  requiring `(0 early, 0 mid, 9 late)` to return only turn-40+ states. `017` is
  retained only as evidence that the quality gate caught the integration bug.

Corrected corpus `018`:

- Artifact: `outputs/diagnostics/entity_late_counterfactual_corpus_018.json`;
  progress: the adjacent `.progress.jsonl` file.
- Seeds 200000-200239 produced 240 clean balanced games, 7,714 actor states,
  exactly 20 games per opponent-seat cell, and zero source stalls/timeouts.
- Selection contains 240 unique turn-40+ games and fixed splits of 160 train,
  40 validation, and 40 untouched confirmation states.
- Three determinizations x eight continuations per evaluated action produced
  30,624 terminal continuations with zero cutoffs. 209/240 states passed the
  0.05 action-range floor; mean range was 0.1540.
- Replication is strong at scale: split top-one agreement 73.75%; mean split
  Spearman 0.8159, CI95 [0.7796, 0.8482]; chosen-advantage correlation 0.9124,
  CI95 [0.8747, 0.9419]; sign agreement 92.2%; 54 independently resolved best
  actions. Every corpus gate passes.

Auxiliary Q `019` and stopping decision:

- Artifact: `outputs/canonical_entity_action_value_019/entity_action_value_metrics.json`.
  The validation-selected checkpoint is training-only and must not be used as
  a bot or actor.
- Best epoch was epoch 1. On validation, Q versus parent top-one was
  22.5% versus 25.0%; mean regret 0.0886 versus 0.0724, delta +0.0161 with
  CI95 [-0.0224, +0.0623]; mean Spearman -0.0317 versus +0.0619, delta -0.0937
  with CI95 [-0.3563, +0.1782]. All three validation criteria fail.
- On untouched confirmation, Q versus parent top-one was 20.0% versus 22.5%;
  mean regret 0.0899 versus 0.0703, delta +0.0196 with CI95
  [-0.0058, +0.0496]; mean Spearman -0.0491 versus +0.0163, delta -0.0654 with
  CI95 [-0.2910, +0.1679]. All three confirmation criteria fail.
- Training loss continued falling while validation regret worsened after epoch
  1, confirming failed generalization rather than insufficient optimization.
  The auxiliary Q gate therefore blocks the actor-update stage; no actor was
  trained and no games were run.
- Close the current counterfactual-to-Q/actor learning direction. The raw late
  action signal is real, but this model does not generalize it across games.
  Do not tune epochs, loss weights, capacity, or split selection on `018/019`.
  Frozen `004` remains the parent.

### Clean Supervised Foundation Scale `020`

Implementation and contract:

- `run_entity_supervised_scale.py` and
  `src/splendor_ai/training/run_entity_supervised_scale.py` implement resumable,
  parallel clean corpus collection, atomically completed per-game files, fixed
  game-disjoint train/validation/confirmation splits, random-init matched-width
  training, detailed rank metrics, paired per-game bootstrap gates, and a
  confirmation holdout that is not used for model selection.
- The two preregistered candidates preserve the one-layer `004` architecture
  and optimizer while scaling widths to 64/64/96 (98,658 parameters) and
  128/128/192 (377,538 parameters). Both trained eight epochs at learning rate
  3e-4, batch size 128; best epochs were selected by validation policy loss.
- Focused pipeline tests passed before the full run. Commit `fc01c47` is the
  pre-experiment workspace checkpoint.

Corpus and offline results:

- Artifact directory: `outputs/canonical_entity_supervised_scale_020`.
  Seeds 220000-220499 produced 500 fresh information-legal
  Greedy-vs-depth1/one-determinization games, 31,954 positions, 246/254 seat
  wins, and zero stalls, timeouts, or loop fallbacks. Splits contain 25,500
  train, 3,222 validation, and 3,232 untouched confirmation positions from
  400/50/50 disjoint games.
- Validation loss was 2.2372 for width64 and 2.2154 for width128, so width128
  was fixed before confirmation. On confirmation, frozen `004` versus width128
  loss was 3.0488 versus 2.2567; top-one accuracy 20.85% versus 34.19%; top-three
  37.84% versus 53.77%; and mean reciprocal rank 0.3374 versus 0.4853.
- Width128 had lower loss on all 50 confirmation games. Its paired mean loss
  delta was -0.7939 with bootstrap CI95 [-0.8428, -0.7442]. Every fixed offline
  gate passed, including the 5% relative-loss floor, rank improvement, strictly
  negative paired CI, 65% per-game floor, and uniform-policy control.
- Post-hoc source stratification is in `teacher_stratified_confirmation.json`.
  On 1,616 Greedy action positions, `004` versus width128 loss was 4.0853 versus
  3.0038 and MRR 0.1355 versus 0.3239. On 1,616 shallow-search positions, loss
  was 2.0122 versus 1.5097 and MRR 0.5393 versus 0.6467. The larger candidate
  did not improve aggregate offline metrics merely by fitting one source.

Mandatory parent gate and decision:

- `parent_match_60.json` used 30 fresh paired seeds (230000-230029), both seats
  per seed, and zero stalls/timeouts. Width128 lost to frozen `004` by 13-47,
  only 21.7% score share versus the required 60%. The strong-opponent gate was
  therefore not run.
- This is a decisive one-step-imitation/closed-loop inversion. Better held-out
  action prediction from the mixed Greedy/shallow trajectory corpus does not
  yield a coherent deterministic policy when the model controls every turn.
  Reject width64 and width128 for play and retain `004` as frozen parent.
- Do not tune width, epochs, or mixed-corpus loss against this result. The next
  bounded test should use the existing split but retain only actions selected
  by `ShallowSearchBot`, train one width64 random-init model, and require the
  same untouched offline and paired-parent gates. Only a successful ablation
  justifies the more expensive coherent shallow-vs-shallow corpus generation.

### Shallow-Actor Ablation `021` and Coherent Corpus `022`

Implementation:

- `run_entity_shallow_actor_ablation.py` filters replay positions by the acting
  bot recorded in each row, refuses any seed split that differs from `020`'s
  frozen `game_splits.json`, trains exactly one random-init width64 candidate,
  and uses the same validation selection and untouched confirmation gates as
  `020`. Entity training now supports per-epoch progress callbacks.
- The clean scale collector now accepts `greedy_shallow` or `shallow_shallow`
  pairings and a `--generate-only` mode. Generate-only collection persists the
  400/50/50 game split before any future model training or evaluation.
- Pre-experiment implementation checkpoints are `3a6cae0` for the shallow-only
  learning gate and `20f5a6b` for coherent shallow self-play collection.

Shallow-only ablation `021`:

- Source is `outputs/canonical_entity_supervised_scale_020/replays.jsonl` with
  the exact `020` seed split. Filtering retained 12,750 train, 1,611 validation,
  and 1,616 confirmation actions selected by `ShallowSearchBot`.
- The 64/64/96 one-layer model used the same initialization seed, batch size
  128, learning rate 3e-4, and eight epochs as width64 in `020`. Validation loss
  improved every epoch from 1.9490 to 1.4083; epoch 8 was selected.
- On untouched confirmation, frozen `004` versus `021` policy loss was 2.0122
  versus 1.4635, top-one accuracy 37.13% versus 50.25%, top-three 63.80% versus
  71.35%, and MRR 0.5393 versus 0.6390. `021` improved all 50 game means; its
  paired loss delta was -0.5501 with CI95 [-0.6001, -0.5027]. All offline gates
  passed.
- The mandatory parent match used 30 fresh paired seeds 240000-240029. `021`
  beat frozen `004` 40-19-1 with zero stalls/timeouts, a 67.5% score share above
  the fixed 60% gate. This is a successful causal ablation: removing mixed
  Greedy labels repairs the offline/closed-loop inversion seen in `020`.
- Do not promote `021` yet. Its preregistered purpose was to decide whether a
  coherent teacher corpus was justified; it passed that decision gate.

Coherent shallow self-play corpus `022`:

- Artifact: `outputs/canonical_entity_shallow_selfplay_corpus_022`. Seeds
  250000-250499 produced 500 fresh games with identical information-legal
  depth1/one-determinization shallow teachers in both seats, 31,080 positions,
  312/185 seat wins and 3 draws, zero stalls/timeouts, and zero loop fallbacks.
- `game_splits.json` was frozen before training: 400 train, 50 validation, and
  50 untouched confirmation games using split seed 20260716. No model has been
  trained on `022` and no holdout has been evaluated.
- Next: train one random-init width64 model on only the 400 `022` train games,
  select its epoch on the 50 validation games, evaluate confirmation once, and
  require >=60% paired score share versus `004` before broad Greedy, legal
  shallow-search, `035`, and other strong-opponent confirmation. Only that full
  gate can promote a new clean parent.

### Coherent Entity Promotion Candidate `023`

Training and offline gates:

- `run_entity_coherent_supervised.py` trains exactly one random-init width64
  model after verifying `022`'s coherent teacher metadata, zero-failure source,
  and frozen 400/50/50 seed split. Implementation checkpoint: `cc89154`.
- `outputs/canonical_entity_coherent_023` used 24,862 train, 3,090 validation,
  and 3,128 untouched confirmation positions. Validation loss improved through
  epoch 7 to 1.1733 and slightly regressed at epoch 8 to 1.1754, so epoch 7 was
  selected without using confirmation.
- On confirmation, frozen `004` versus `023` policy loss was 1.8068 versus
  1.1801, top-one accuracy 41.75% versus 57.00%, top-three 70.49% versus 79.19%,
  and MRR 0.5877 versus 0.7018. `023` improved all 50 game means; paired loss
  delta was -0.6269 with CI95 [-0.6540, -0.5994]. Every offline gate passed.
- The mandatory paired-parent block used seeds 260000-260029. `023` beat `004`
  44-16 with zero stalls/timeouts, a 73.3% score share above the fixed 60% gate.

Preregistered broad gate and result:

- `run_entity_broad_promotion_gate.py`, checkpoint `93c66d1`, froze 40 games
  per candidate/opponent and same-seed `004` control. Required candidate floors
  were 80% Greedy, 30% legal shallow, 20% `016`, and 20% `035`; required
  candidate-minus-control deltas were -5, +10, +5, and +5 percentage points.
  Any stall/timeout in candidate or control invalidated that opponent block.
- Candidate/control score shares were: Greedy 95.0%/83.75%; legal shallow
  55.0%/17.5%; `016` 40.0%/15.0%; and `035` 41.25%/5.0%. Every absolute and
  matched strength threshold passed, as did non-Greedy integrity.
- The original Greedy block was integrity-invalid because `004` repeated a
  state four times at seed 270022, turn 33, score 0-0. Before rerunning, commit
  `0036276` preregistered exactly one fresh replacement block, replacing rather
  than pooling the original. Seeds 274000-274039 gave clean `004` share 70.0%
  and `023` share 92.5%, but `023` repeated at seed 274002, turn 29, score 1-0.
  Both incidents ended by `repetition_cutoff`; neither entity bot recorded a
  loop-fallback trigger because `CanonicalEntityActionScorerBot` has no loop
  fallback implementation.
- Final artifact `broad_promotion_gate_final.json` has every strength gate true
  and only `greedy_integrity` false. Honor the preregistered stop: do not run a
  second replacement and do not promote `023`. Frozen `004` remains parent.
- Next work is operational, not another learning run: implement and test a
  canonical-successor-aware loop fallback for entity checkpoint inference,
  then rerun fresh parent and broad gates because the deployed bot behavior has
  changed. The strength evidence makes this fix worthwhile; GPU is irrelevant.

### Loop-Safe Entity Promotion Preregistration `024`

- `CanonicalEntityActionScorerBot` now applies the configured inference-only
  loop fallback to canonical successor representatives. Its cycle key is the
  canonical public-belief key with absolute turn number reduced to parity; on a
  repeated state it first prefers a close-scoring buy, then the highest-scoring
  least-used canonical successor. This permits a deterministic escape even
  when no buy is legal. Benchmark construction now passes the CLI fallback
  configuration to entity bots.
- Focused tests pass (22/22), the full suite passes (306/306), and the exact
  former candidate failure at seed 274002 now completes 18-0 at turn 72 with
  exactly one fallback instead of reaching the repetition cutoff at turn 29.
  This replay is a regression smoke only, not promotion evidence.
- Freeze the following fresh gate before viewing results. Use unchanged
  fallback defaults (state visits 2, own no-progress actions 6, buy-score gap
  8), `max_turns=400`, `repetition_limit=4`, `no_progress_limit=60`, CPU, and
  the unchanged `023` and `004` checkpoint weights.
- Parent gate: 60 paired-seat games (30 paired seeds 280000-280029), `023`
  versus frozen `004`; require candidate score share at least 60% and zero
  stalls/timeouts for either bot.
- Broad gate: 40 games per candidate/control-opponent block with matched fresh
  seeds: Greedy 281000-281039, legal shallow depth1 282000-282039, `016`
  paired seeds 283000-283019, and `035` paired seeds 284000-284019. Retain the
  v1 candidate floors of 80%, 30%, 20%, and 20%, and candidate-minus-`004`
  deltas of -5, +10, +5, and +5 percentage points respectively. Every block
  requires zero stalls/timeouts.
- Promotion requires both the parent gate and every broad strength/integrity
  gate. There is no replacement block or result-contingent rerun in `024`.
- Result: all preregistered gates pass. The paired parent block finished 48-12
  for `023` (80.0% score share versus the 60% floor), with zero failures; `023`
  triggered no fallbacks and `004` triggered five. Fresh candidate/control
  broad shares were 92.5%/90.0% versus Greedy, 37.5%/12.5% versus shallow,
  32.5%/10.0% versus `016`, and 40.0%/11.25% versus `035`. All 320 broad games
  completed without a stall or timeout; every absolute, matched-control, and
  integrity gate passed. The candidate needed zero fallbacks in the fresh
  broad blocks; `004` used four versus `016` and two versus shallow.
- Promote `023` as the clean entity actor parent. Keep `004` as a historical
  control, not the active parent. Final decision artifact:
  `outputs/canonical_entity_loop_safe_024/promotion_gate_final.json`.

### Preregistered On-Policy DAgger Iteration `025`

- `run_entity_dagger.py` implements one bounded dataset-aggregation iteration
  around promoted `023`. It generates exactly 240 fresh games at seeds
  290000-290239: 60 each against `023`, legal shallow depth1, `016`, and `035`.
  Non-self opponents alternate the parent across seats. The frozen 192/24/24
  train/validation/confirmation split is stratified to contain 48/6/6 games
  from each league profile.
- Every state is relabeled by the same information-legal shallow-depth1,
  one-determinization teacher used by the successful coherent line. Rollout
  actions and provenance remain stored separately. This prevents mixed-policy
  labels while moving training support onto states visited by `023` and its
  broad league. Rollout failures invalidate the corpus; fallback triggers are
  recorded but remain valid deployed-policy behavior.
- Train exactly one width64 candidate initialized from `023`, four epochs at
  learning rate `3e-5`, weight decay `1e-4`, batch size 128, policy loss only.
  Each epoch has exact 50/50 exposure to frozen old `022` train positions and
  new `025` train positions; validation selection uses the same source balance.
  No width, learning-rate, epoch, or source-weight sweep is allowed.
- Before games, untouched new confirmation must improve loss by at least 5%,
  top-one accuracy by at least 3 points, MRR, paired-loss CI95, and at least 65%
  of game means. Untouched old `022` confirmation may regress by at most 2% in
  loss, 2 points top-one, and 1 point MRR. Any failed offline condition stops
  the experiment before game gates.
- If eligible, require a 60-game paired-seat match against `023` at seeds
  300000-300029 with at least 60% candidate score share and zero failures.
  Then run fresh 40-game candidate/control blocks versus Greedy (301000), legal
  shallow depth1 (302000), `016` paired (303000), and `035` paired (304000).
  Candidate absolute floors remain 80%, 30%, 20%, and 20%; required deltas
  versus matched `023` are -5, +5, -5, and -5 points respectively. All blocks
  require zero failures. There is no replacement or result-contingent rerun.
- Implementation smoke only: 12 balanced games completed without failure and
  shallow labels disagreed with rollout actions on 39.4% of 734 states. A
  one-epoch mechanics smoke produced all expected artifacts; its four-game
  holdouts are not evidence and cannot affect the frozen `025` decision.
- Result: reject `025` before training. All 240 games were generated, but seed
  290015 in the `035` profile reached the repetition cutoff at turn 45. Audit
  showed that both players had only `PASS`; neither `023`, `035`, nor any
  successor fallback had an alternative action. Per the frozen integrity rule,
  no `025` row is eligible for training and there is no replacement game.

### Forced-Pass Repair and Fresh DAgger Block `026`

- Replay collection and match evaluation now end two consecutive forced passes
  as a clean adjudicated `forced_pass_deadlock`. This is not a model fallback:
  it handles the engine state where both players have exactly one legal action,
  `PASS`, and further play cannot change the state. The exact failed seed now
  ends cleanly at turn 41 after recording the two forced passes. Focused tests
  pass 22/22 and the full suite passes 312/312.
- Freeze a completely fresh DAgger block at seeds 291000-291239 in
  `outputs/canonical_entity_dagger_026`; reuse none of `025`. All other `025`
  league balance, 192/24/24 stratified split, teacher, old/new 50/50 exposure,
  four-epoch optimizer, offline confirmation gates, and promotion gates remain
  unchanged. Stalls and true timeouts still invalidate `026`; an adjudicated
  `forced_pass_deadlock` is a valid clean terminal.
- Result: `026` generated 240/240 clean games, 60 per profile, with 15,158
  states and zero stalls, timeouts, or fallback triggers. The shallow teacher
  disagreed with rollout actions on 5,819 states (38.39%). Frozen split sizes
  were 12,124/1,510/1,524 new positions; old `022` contributed
  24,862/3,090/3,128 positions.
- The single candidate trained all four epochs and epoch 4 was selected by
  validation. On untouched new confirmation it improved loss by 2.80%, top-one
  by 1.25 points, and MRR by 1.12 points; it improved 79.17% of game means and
  paired loss CI95 was [-0.0486, -0.0156]. On untouched old confirmation it
  also improved loss by 2.89%, top-one by 1.53 points, and MRR by 0.96 points.
- Reject `026` before games: the frozen new-confirmation thresholds required
  at least 5% loss improvement and +3 top-one points, and both fail. All other
  uptake, significance, game-breadth, and old-retention gates pass. Do not tune
  epochs, learning rate, or source weights on these confirmation results; no
  parent or broad matches are authorized. `023` remains the promoted parent.

### Preregistered Non-Promotional DAgger Diagnostic `027`

- Purpose: determine whether rejected candidate `026`'s small but significant
  imitation gains correspond to direct playing strength. This is diagnostic
  only and cannot retroactively pass `026`, trigger broad gates, or promote it.
- Play exactly 100 direct games as 50 paired-seat seeds, 310000-310049, with
  `026` as candidate and promoted `023` as baseline. Both use
  `canonical_entity_action_scorer` on CPU with deployed loop fallback,
  `max_turns=400`, `repetition_limit=4`, and `no_progress_limit=60`.
- Write the raw match to
  `outputs/canonical_entity_dagger_diagnostic_027/paired_026_vs_023.json`.
  Any true timeout or stall invalidates the diagnostic. A clean adjudicated
  `forced_pass_deadlock` remains valid under the frozen engine contract.
- Analyze candidate game score as 1 for a win, 0.5 for a draw, and 0 for a
  loss. Average the two games for each seed before inference so seat-paired
  games are the independent units. Report overall score share and a fixed-seed
  percentile bootstrap CI95 over the 50 pair means (200,000 resamples,
  analysis RNG seed 27023).
- Interpret a candidate score share of at least 60% with pair-aware CI95 lower
  bound above 50% as convincing positive evidence. The symmetric condition is
  convincing negative evidence. Everything else is inconclusive/roughly even.
  Regardless of outcome, `023` remains the promoted parent. Positive evidence
  justifies one independently preregistered fresh DAgger replication; even or
  negative evidence closes DAgger as the primary improvement direction.
- Result: all 100 games completed with zero stalls, timeouts, forced-pass
  deadlocks, or fallback triggers. `026` finished 48-52 with no draws, for a
  48% score share. Across the 50 paired seeds it swept 9, split 30, and lost
  both games on 11; the frozen pair-aware bootstrap CI95 is [39%, 57%]. This is
  inconclusive/roughly even under the preregistered rule and provides no game
  evidence that the offline improvement made `026` stronger. Close DAgger as
  the primary direction, retain `026` only as a rejected diagnostic artifact,
  and keep `023` promoted. Raw and analyzed results are in
  `outputs/canonical_entity_dagger_diagnostic_027/paired_026_vs_023.json` and
  `paired_analysis.json`.

### Preregistered Canonical Search-Lift Gate `028`

- `CheckpointLegalActionPriorScorer` already supported canonical-entity
  checkpoints. The missing evaluation path is now repaired:
  `run_puct_benchmark.py` can construct a raw
  `canonical_entity_action_scorer` policy opponent and records stalls,
  timeouts, termination reasons, and fallback telemetry. Canonical prior
  batching, live PUCT inference, and CLI selection have explicit tests. The
  focused suite passes 23/23 and the full suite passes 315/315.
- A two-game mechanics-only smoke at seed 319900 tied 1-1 with zero failures.
  PUCT64 averaged 0.194 search seconds per move and 330 simulations/second on
  CPU. This seed is disjoint from the gate and the smoke is not evidence.
- Freeze exactly 100 games as 50 paired-seat seeds 320000-320049 in
  `outputs/canonical_entity_search_lift_028/puct64_vs_parent023_paired100.json`.
  Both agents use promoted `023` (SHA256
  `29EF7C13C5553EC88C8BCB98F7378C925F0E600E2FCF3D944DF5926EFED7A070`):
  the candidate is information-set PUCT and the control is deployed greedy
  canonical-entity inference with loop fallback.
- Freeze PUCT at 64 simulations, c_puct 1.5, prior temperature 1.0, FPU
  reduction 0.1, full learned-prior mix, maximum depth 80, heuristic scale
  4000, inference batches of 8, no tree reuse, and no learned/n-step/rollout
  leaf. Run on CPU with `max_turns=400`, repetition limit 4, and no-progress
  limit 60. No parameter or budget sweep is allowed after seeing the result.
- Require zero stalls and true timeouts. Forced-pass deadlock remains a valid
  adjudicated terminal under the engine contract. The lift gate passes only
  if PUCT score share is at least 60% and the built-in 20,000-resample
  pair-aware bootstrap CI95 lower bound is above 50%.
- A pass authorizes preregistration of one search-policy/root-visit corpus and
  an iterative expert-iteration candidate. A failure authorizes no training
  corpus and closes fixed-heuristic canonical PUCT as the next improvement
  direction; it does not change `023`'s promoted status.
- Result: all 100 games and 50 seed pairs completed with zero stalls, timeouts,
  forced-pass deadlocks, or fallback triggers. PUCT64 beat raw `023` 56-44,
  sweeping 15 pairs, splitting 26, and losing both seats on 9. Its score share
  was 56% with pair-aware bootstrap CI95 [47%, 65%]; mean final score margin
  was +0.74 with CI95 [-0.69, +2.13]. The direction is encouraging but neither
  the frozen 60% share threshold nor the lower-bound-above-50% threshold
  passes. Reject `028` before corpus generation: do not tune PUCT parameters,
  sweep budgets, or train root visits from this result. Fixed-heuristic
  canonical PUCT is closed as the next improvement line and `023` remains
  promoted. Raw and decision artifacts are
  `outputs/canonical_entity_search_lift_028/puct64_vs_parent023_paired100.json`
  and `gate_decision.json`.

### Preregistered Public-Belief Value Gate `029`

- Purpose: test a genuinely different leaf-evaluator class after fixed-
  heuristic PUCT64's encouraging but inconclusive 56% result. Do not sweep
  PUCT parameters, budgets, or heuristic rules. No new rollout generation is
  permitted before the offline value gate passes.
- Source only the 240 clean league games in
  `outputs/canonical_entity_dagger_026/replays.jsonl`. Preserve its frozen
  192/24/24 game-disjoint train/validation/confirmation splits exactly; every
  split remains opponent-profile balanced at 48/6/6 games for each of parent,
  shallow, `016`, and `035`. Training cycles the eight profile x acting-seat
  position strata with exactly equal frequency per epoch.
- Train exactly one separate `CanonicalEntityValueModel` with the public
  width64 entity encoder shape from promoted `023`. Initialize only public
  entity embeddings/input/encoder from `023`; initialize a fresh value head,
  then allow the entire value model to train. Inputs use only
  `canonical_public_belief_v1`; hidden deck order and opponent blind-reserve
  identity are never encoded and have invariance tests.
- Target `clip((acting player final score - opponent final score) / 20, -1, 1)`.
  Train eight CPU epochs with MSE, AdamW learning rate `3e-4`, weight decay
  `1e-4`, batch size 128, gradient clip 1.0, and seed 20290716. Select once by
  validation macro MAE across the eight opponent/seat strata. No architecture,
  target scale, learning-rate, epoch, or initialization sweep is allowed.
- Compare the selected model against the exact deployed PUCT heuristic
  `legal_heuristic_value(scale=4000)` on the untouched 24-game confirmation
  split. Report aggregate, opponent, acting-seat, and phase (0-14, 15-29,
  30-44, 45+) MSE/MAE/correlation/sign accuracy. Ordering is measured on
  per-game, per-actor, per-phase mean predictions within matched
  opponent/seat/phase strata, avoiding position-count pseudoreplication.
- Offline eligibility requires all six frozen gates: at least 10% macro
  opponent/seat MAE improvement; paired-game MAE bootstrap CI95 wholly below
  zero (20,000 resamples); ordering concordance at least 55%; concordance at
  least 3 points above the heuristic; higher macro ordering correlation; and
  no opponent or phase MAE regression above 10%. Any failure stops before
  gameplay and forbids tuning on confirmation.
- If and only if offline eligible, run a fresh 60-game/30-pair learned-leaf
  PUCT64 versus heuristic-leaf PUCT64 A/B at seeds 330000-330029. Freeze the
  full `028` PUCT64 configuration for both; only the candidate leaf evaluator
  differs. Require zero stalls/timeouts, at least 60% learned-leaf score share,
  and pair-aware CI95 lower bound above 50%.
- If and only if that A/B passes, run learned-leaf PUCT64 versus raw deployed
  `023` for 100 games/50 paired seeds 331000-331049 with the same integrity,
  60% share, and lower-bound-above-50% gates. There are no replacement runs.
  Only both gameplay passes authorize a separately preregistered expert-
  iteration corpus; neither result directly promotes a model.
- Implementation adds public-state batching, a dedicated public entity value
  checkpoint, canonical value loading in `CheckpointValueEvaluator`, the
  balanced trainer/evaluator, and six focused contract tests. Focused tests
  pass 32/32 and the full suite passes 321/321 before training.
- Result: the single CPU run completed all eight epochs in 79 seconds and
  selected epoch 7. On 1,524 untouched confirmation positions, macro
  opponent/seat MAE improved 18.58% (0.3721 to 0.3030). Paired per-game MAE
  difference was -0.0685, CI95 [-0.1178, -0.0191], and the candidate improved
  16/24 games. Every opponent and phase remained within the 10% regression
  ceiling, so the error and breadth gates pass.
- Reject `029` before gameplay because all ordering gates fail. Candidate
  concordance is 52.31% versus the heuristic's 61.92%, below both the absolute
  55% floor and the required +3-point improvement. Macro stratum correlation
  is 0.116 versus 0.236. A post-gate dispersion diagnostic confirms
  regression toward the mean: target/candidate/heuristic standard deviations
  are 0.357/0.236/0.423, while candidate and heuristic slopes on target are
  0.212 and 0.429. Lower average error therefore did not preserve the leaf
  ranking search needs.
- Do not run the seeds 330000 or 331000 PUCT gates, generate root targets, or
  tune loss, target scale, epochs, capacity, or initialization on confirmation.
  Retain the checkpoint as a rejected calibration diagnostic only; `023`
  remains promoted. Full metrics and the frozen decision are in
  `outputs/canonical_entity_public_value_029/public_value_metrics.json` and
  `gate_decision.json`.

### Preregistered Shallow-Search Operational Gate `030`

- Purpose: stop compressing the only teacher with demonstrated headroom and
  test that teacher itself as an operational agent. This is a deployment-class
  search candidate, not a neural checkpoint promotion and not permission to
  tune search after seeing results.
- Freeze the exact coherent teacher: information-legal `ShallowSearchBot`,
  depth 1, one determinization, total branching 6, buy/reserve/take caps
  4/2/2, search seed 23, and determinization seed 29. Use CPU,
  `max_turns=400`, repetition limit 4, and no-progress limit 60.
- First run exactly 100 games as 50 paired-seat seeds 340000-340049 against
  promoted `023` (SHA256
  `29EF7C13C5553EC88C8BCB98F7378C925F0E600E2FCF3D944DF5926EFED7A070`).
  Require zero stalls/true timeouts, at least 60% shallow score share, and a
  20,000-resample pair-aware CI95 lower bound above 50%. Any failure stops the
  experiment before broad blocks.
- Only after a parent pass, compare shallow and matched `023` controls on the
  same 30 paired seeds/60 games per opponent: Greedy at 341000-341029, `016`
  at 342000-342029, and `035` at 343000-343029. The checkpoint hashes are
  `876AC23A005D367AC48B3619BB7CDCB6E0230A3E9E97943D7EECDC6D07D1B056`
  and `4422A82B593340EECD02AF3A0941F24F0318CC7453E6A0739FD9F524539AF2CD`.
- Broad absolute shallow floors are 80% versus Greedy and 25% versus each
  strong learned opponent. Required matched-control deltas are at least -5
  points versus Greedy and +5 points versus both `016` and `035`. Candidate
  and control blocks must all have zero stalls/true timeouts. There are no
  replacement blocks or result-contingent threshold changes.
- A complete pass designates the exact shallow configuration as the
  clean-line operational search parent while retaining `023` as the promoted fast
  neural parent. It then authorizes a separately preregistered selective-search
  hybrid. Failure leaves `023` operational and sends the project to a decision
  about scaled multi-generation self-play rather than another local search or
  imitation adjustment.
- `run_shallow_operational_gate.py` implements pair-atomic resumable parallel
  execution, fixed checkpoint identities, role/seat-aware pair statistics,
  conditional broad execution, and a final decision artifact under
  `outputs/shallow_operational_030`. Four focused gate tests pass; the relevant
  focused suite passes 17/17 and the full suite passes 325/325. A disjoint
  seed-339900 one-pair mechanics smoke tied 1-1 with zero failures; it is not
  evidence and did not run broad blocks.
- Result: pass all parent and broad gates over 460 clean games with zero
  stalls, true timeouts, forced-pass deadlocks, or fallback triggers. Against
  fast neural parent `023`, shallow won 61-39 for 61% score share, pair-aware
  CI95 [52%, 70%], and mean final margin +1.22, CI95 [+0.02, +2.42].
- In matched 60-game broad blocks, shallow and `023` both scored 98.33% versus
  Greedy. Shallow scored 40.0% versus `016` compared with `023`'s 30.0%, a
  +10-point delta, and 38.33% versus `035` compared with 21.67%, a +16.67-point
  delta. Every absolute, matched-control, and integrity gate passes.
- Designate the exact shallow configuration as the clean-line operational search parent
  while retaining `023` as the promoted fast neural parent. This is not a new
  checkpoint and does not reopen parameter or budget sweeps. The successful
  gate authorizes one separately preregistered selective-search hybrid whose
  purpose is to retain shallow strength while reducing average inference cost.
  Raw block artifacts, final gates, and the concise decision are under
  `outputs/shallow_operational_030/`, especially `gate_final.json` and
  `operational_decision.json`.

### Preregistered Selective-Search Hybrid Gate `031`

- Purpose: retain most of the exact shallow teacher's clean-line strength while
  reducing its average search use. This does not claim that shallow or the
  hybrid is stronger than overall benchmark `035`.
- The candidate uses promoted `023` on ordinary confident turns and invokes
  the exact gate-`030` shallow configuration when either (a) the top-two
  canonical-successor logits differ by at most `0.5`, or (b) either player has
  at least 12 points or an available buy can reach the 15-point threshold.
  The rule is frozen before gameplay; no result-contingent threshold changes
  or router sweep are authorized.
- Router selection used only the frozen `026` validation games. Across 1,510
  positions, the rule routes 46.8% to search and covers 75.2% of `023`'s
  disagreements with shallow, raising teacher agreement from 55.7% to 89.0%
  under the offline routing counterfactual. Confirmation games were not used.
- Stage 1 uses 50 paired-seat seeds 344000-344049 against shallow. Require zero
  stalls/timeouts, hybrid score share at least 45%, pair-aware CI95 lower bound
  at least 35%, both neural and search turns, and search on at most 60% of
  hybrid turns. Failure stops the experiment.
- Stage 2 conditionally uses 50 paired-seat seeds 345000-345049 against `023`.
  Require zero stalls/timeouts, at least 55% hybrid score share, CI95 lower
  bound above 45%, and the same at-most-60% search fraction. Failure stops
  before broad blocks.
- Stage 3 conditionally compares hybrid and matched shallow controls on 30
  paired seeds/60 games each versus `016` at 346000-346029 and `035` at
  347000-347029. Each candidate/control block must have zero failures; hybrid
  must score at least 25%, trail matched shallow by no more than 5 points, and
  search on at most 60% of its turns.
- `SelectiveSearchHybridBot` preserves canonical-successor policy scoring and
  loop tracking. `run_selective_search_gate.py` provides pair-atomic resumable
  execution, telemetry, checkpoint hashes, staged gates, and a frozen final
  decision under `outputs/selective_search_hybrid_031`. CPU remains the correct
  execution target because this experiment is search- and engine-bound.
- Result: reject the hybrid after the conditional second stage. Stage 1 passed:
  hybrid beat shallow 57-42 with one draw over 100 games, for 57.5% score share
  and pair-aware CI95 [49%, 66%]. Mean final margin was +0.81, CI95
  [-0.35, +1.99]. It searched 1,460 of 3,093 turns (47.20%), with zero
  stalls, timeouts, deadlocks, or loop fallbacks.
- Stage 2 failed: hybrid tied raw `023` 50-50 over 100 games, for 50% score
  share and CI95 [42%, 58%], below both the frozen 55% mean and above-45% CI
  gates. Mean final margin was +0.25, CI95 [-0.91, +1.42]. Search use remained
  within budget at 1,449 of 3,148 turns (46.03%), with zero integrity failures.
- Do not promote this router or tune its threshold on these results. The
  conditional `016`/`035` blocks were correctly skipped, so seeds 346000-346029
  and 347000-347029 remain unused. The result shows a useful speed reduction
  and favorable direct interaction with shallow, but no gameplay improvement
  over the fast policy it wraps. Frozen metrics are in
  `outputs/selective_search_hybrid_031/gate_final.json`.

### Preregistered League Self-Play Generation `032`

- Purpose: make the first genuinely iterative improvement attempt anchored on
  overall champion `035`. This is not another `023`/shallow router, PUCT
  variation, one-step reserve imitation, PPO magnitude correction, or use of
  human games.
- Before outcomes, calibrate exploration on 5,000 deterministic rows sampled
  from the existing 47k `035` self-play corpus. The fixed candidate grid is
  `(epsilon, temperature, top_k) = (0.30,0.50,3), (0.40,1.00,3),
  (0.50,1.00,3), (0.50,1.50,5), (0.60,1.50,5)`. Select the setting closest to
  15% expected non-greedy actions among settings inside the frozen 10%-20%
  band. This calibration never reads game outcomes; failure stops generation.
- Collect 512 environment-seed groups at 350000-350511, four stochastic `035`
  trajectories per group (2,048 games). Groups are exactly balanced at 64 in
  each of the eight `(opponent, actor seat)` cells for league opponents `035`,
  `016`, exact gate-030 shallow, and clean fast parent `023`. Within a group,
  environment seed, opponent, and actor seat are shared; only actor sampling
  seed changes. Human rows are forbidden. Any stall/timeout or sampling rate
  outside 10%-20% stops before training.
- Freeze groups 0-447 for training and 448-511 for validation. The update uses
  only leave-one-out group-ranked `tanh(final score margin / 20)` returns.
  Signal gates require all 512 four-member groups, at least 75% informative
  groups, exact behavior-probability reconstruction, clean termination, and
  at least 10% observed non-greedy actor decisions.
- Train exactly one full `035` attention candidate for one epoch with
  advantage-weighted regression: learning rate `1e-5`, batch 256, advantage
  temperature 0.25, maximum weight 10, parent raw-policy KL weight 1.0,
  entropy weight 0.001, weight decay `1e-4`, and gradient norm 1. This replaces
  signed clipped PPO with exponentially weighted supervised likelihood over
  complete better-return trajectories; it is not eligible for a parameter
  sweep on downstream results.
- Before games, require lower validation AWR policy loss than frozen `035`,
  1%-10% top-action movement on both 5,000 on-policy and broad rows, mean raw
  KL at most 0.01 on both streams, and behavior reconstruction MAE at most
  `1e-5`. Failure rejects the candidate before strength games.
- A passing offline candidate gets one fresh direct gate against `035`: 200
  paired-seat seeds 351000-351199 (400 games), zero integrity failures, at
  least 55% candidate score share, and pair-aware CI95 lower bound above 50%.
  Failure rejects Generation 1 and does not authorize a retry on these data.
- Only a direct pass unlocks matched candidate and frozen-`035` control blocks,
  50 paired seeds each versus `016` at 352000, shallow at 353000, and `023` at
  354000. Every block must be clean; candidate share must be at least 25% and
  trail the matched `035` control by no more than 5 points. A full pass promotes
  the candidate and authorizes Generation 2; otherwise `035` remains champion.
- `run_league_generation_032.py` and
  `src/splendor_ai/training/run_league_generation.py` implement the staged,
  hash-bound, resumable contract. Training remains small; CPU game simulation
  dominates, so no GPU rental is justified for Generation 1.
- Result: reject Generation 1 at the preregistered movement/KL gate, before
  gameplay. Behavior-only calibration selected epsilon 0.50, temperature 1.0,
  top-3 with 15.257% expected non-greedy probability. The 2,048-game corpus
  then observed 9,738 non-greedy actions in 64,852 actor turns (15.016%), exact
  64-group balance in every opponent/seat cell, zero stalls/timeouts, and one
  adjudicated forced-pass deadlock. All 512 groups were informative; mean
  group return range was 0.6275 and behavior reconstruction MAE was `1.18e-7`.
- The single AWR update improved game-disjoint validation policy loss from
  0.72098 to 0.61229. Greedy movement was healthy at 5.04% on-policy and 4.98%
  on broad rows, and behavior reconstruction remained exact. However, mean raw
  policy KL was 0.04128 on-policy and 0.04084 broad, exceeding the frozen 0.01
  ceiling by more than 4x. The lower exploration-policy KL (~0.0058) does not
  override the explicitly preregistered raw-policy safety gate.
- Do not run the direct `035` gate, broad gates, Generation 2, or a learning
  rate/KL/temperature correction on this corpus. Seeds 351000-351199 and all
  352000-354049 broad seeds remain unused. Retain the clean balanced corpus as
  evidence and a possible source for a separately justified future method;
  candidate `032` is rejected and `035` remains overall champion. Frozen
  results are under `outputs/league_generation_032/gate_final.json`, with the
  concise gate evidence in `movement_gate.json`.

### Preregistered Trust-Region League Generation `033`

- Purpose: retain `032`'s independently replicated AWR proposal recipe while
  enforcing the failed raw-KL constraint structurally. This is not permission
  to tune `032`, reuse its training rows, or choose an interpolation from game
  results. Frozen `035` remains the parent and overall champion.
- Generate a completely fresh 512-group/2,048-game corpus at environment seeds
  355000-355511 and sampling seeds beginning 33000000. Preserve the exact
  eight opponent/seat cells and 448/64 game-disjoint train/validation split
  from `032`. Use the already behavior-calibrated epsilon 0.50, temperature
  1.0, top-3 sampler; require 10%-20% observed non-greedy actions, zero
  stalls/timeouts, all four trajectories per group, at least 75% informative
  groups, and exact behavior reconstruction. No file under
  `data/league_generation_032` is an eligible training source.
- Train exactly one unrestricted proposal with the frozen `032` AWR recipe:
  one full-model epoch, learning rate `1e-5`, batch 256, advantage temperature
  0.25, maximum weight 10, KL weight 1.0, entropy weight 0.001, score-margin
  scale 20, and leave-one-out four-trajectory group returns.
- Construct the deployable candidate by linear parameter interpolation
  `theta = theta035 + alpha * (theta_proposal - theta035)`. Select `alpha`
  without gameplay using exactly 12 deterministic bisection steps on all fresh
  validation rows. Choose the largest tested `alpha` whose validation mean raw
  policy KL is at most 0.008. If the unrestricted proposal is already below
  0.008, use `alpha=1`. Do not select on validation loss or action movement.
- Before games, require positive alpha, projected validation raw KL at most
  0.008001, at least 3% validation AWR policy-loss improvement over `035`,
  1%-10% greedy action movement on both 5,000-row on-policy and broad streams,
  mean raw policy KL at most 0.01 on both streams, and behavior reconstruction
  MAE at most `1e-5`. Failure rejects `033` without an alpha, optimizer, or
  threshold correction.
- A complete offline pass unlocks 200 paired-seat seeds 356000-356199 against
  `035` (400 games). Require zero failures, score share at least 55%, and
  pair-aware CI95 lower bound above 50%. Only that pass unlocks matched
  candidate and frozen-`035` control blocks of 50 paired seeds versus `016` at
  357000, exact shallow at 358000, and `023` at 359000. Broad blocks require
  zero failures, at least 25% candidate share, and no more than a 5-point
  regression from matched `035`.
- `run_trust_region_generation_033.py`,
  `run_trust_region_generation.py`, and `run_awr_trust_region.py` implement the
  hash-bound, resumable protocol and parameter projection. CPU remains
  appropriate because replay reconstruction and game simulation dominate.
- Result: reject `033` at the untouched direct `035` gate; do not run the
  conditional broad gates or retune alpha on these data. The fresh corpus
  completed all 2,048 games with exact opponent/seat balance, zero stalls or
  timeouts, 64,630 actor turns, a 14.875% non-greedy rate, and all 512 groups
  informative. Behavior reconstruction MAE was `1.13e-7`.
- The frozen unrestricted proposal improved game-disjoint validation policy
  loss from 0.71479 to 0.60470 but had raw KL 0.04081. Twelve-step structural
  projection selected `alpha=0.435791015625`, yielding validation raw KL
  0.007992 and policy loss 0.65919, a 7.78% improvement. Offline movement gates
  all passed: greedy action movement was 3.06% on-policy and 3.60% broad, with
  mean raw KL 0.00831 and 0.00815 respectively.
- In the preregistered 200 paired seeds/400 games versus `035`, the projected
  candidate scored 46.875% share (paired CI95 43.50%-50.25%), mean outcome
  -6.25%, and mean final-score margin -0.2525. All games completed normally,
  but both strength thresholds failed. Seeds 357000-359049 remain unused;
  candidate `033` is rejected and `035` remains overall champion. This result
  separates the blocker from update magnitude: enforcing the KL constraint
  succeeded, but the AWR direction did not translate offline improvement into
  stronger play. Frozen evidence is in
  `outputs/trust_region_generation_033/gate_final.json`.

### Preregistered Policy-Delta Causal Audit `034`

- Purpose: explain `033`'s failed full-game gate by measuring the local causal
  value of only the actions it changes relative to frozen `035`. This is a
  diagnostic, not another AWR update, checkpoint promotion, router, or
  permission to select actions from their evaluated returns. No training or
  gameplay promotion is reachable from this runner.
- Generate 640 completely fresh deterministic-`035` source games at seeds
  360000-360639, exactly 80 in each of the eight opponent/actor-seat cells for
  opponents `035`, `016`, exact gate-030 shallow, and promoted clean `023`.
  Store snapshot-enabled replays and require exact balance, zero stalls, and
  zero timeouts. The candidate never acts during source collection.
- Compare frozen `035` with projected `033` on every acting-`035` source state.
  Selection may read only seed, turn, actor seat, and the pre-action snapshot;
  it must not read winner, final scores, final value, or any other outcome.
  Exclude raw-action changes that share the same canonical public successor.
  Deterministically select 80 disagreements from unique source games, exactly
  10 per opponent/seat cell. Phase is measured rather than forced because an
  outcome-free planning subset showed that canonical disagreements are sparse
  and uneven by phase; report early/mid/late counts and results explicitly.
- At every selected state compare exactly two fixed actions: the deployed
  `035` top action and the projected `033` top action. Use three independent
  observer-relative legal determinizations and eight matched continuation seeds
  per determinization/action. Both branches use the same frozen `035`
  continuation policy with epsilon 0.1, temperature 0.5, top-3 sampling,
  maximum 120 additional turns, and normalized final score margin `/20`.
  Require continuation cutoffs at most 1% for an interpretable result.
- The unique source game/state is the statistical unit. Report candidate-minus-
  parent mean return and a 20,000-resample CI95, split-half correlation and
  sign agreement, resolved-state frequency, candidate/parent resolved wins,
  and phase, opponent, seat, and opponent/seat/phase breakdowns. A state is
  resolved only when absolute paired mean is at least 0.025 and exceeds 1.96
  paired standard errors.
- Call the action signal stable only if continuation integrity passes, the
  split-half correlation CI95 lower bound is positive, at least 20 states have
  half-effects of magnitude 0.01 with at least 65% sign agreement, and at least
  10% of states are resolved. Stable aggregate CI95 wholly below zero supports
  local harm and closes AWR; stable CI95 wholly above zero supports a separate
  compounding/distribution-shift hypothesis. Any other outcome is unresolved
  and also closes AWR as the primary direction. None of these diagnoses trains
  or promotes a model.
- `run_policy_delta_causal_audit_034.py` and
  `src/splendor_ai/training/run_policy_delta_causal_audit.py` implement the
  hash-bound, outcome-blind, pair-matched, resumable CPU contract. The fixed
  continuation seeds begin at 34000000. A separate eight-game/two-state
  mechanics smoke on throwaway seeds completed all 96 continuations and is
  ineligible evidence.
- Result: the complete diagnostic is valid and closes AWR as the primary
  improvement direction. All 640 source games completed normally with exact
  80-game opponent/seat balance, 20,365 acting-`035` turns, and zero stalls,
  timeouts, or fallbacks. Outcome-blind scanning found 351 distinct canonical-
  successor disagreements and selected 80 unique games with exact 10-state
  opponent/seat balance; phase coverage was 35 early, 29 mid, and 16 late.
- All 3,840 continuation branches terminated with zero cutoffs. The action-
  conditional measurements are highly reproducible: split-half correlation
  was 0.8669 with CI95 [0.7885, 0.9204], 67 states had sign-eligible half
  effects with 85.07% agreement, and 40/80 states met the fixed resolved rule.
  Every preregistered signal-stability condition passes.
- The `033` direction nevertheless has no aggregate local advantage. Mean
  candidate-minus-`035` return was -0.01109, equivalent to -0.2219 final-score
  points, with CI95 [-0.9891, +0.5391] points. Resolved states split exactly
  20 favoring `033` and 20 favoring `035`; raw state signs were 45.0% versus
  51.25%, with 3.75% ties. Every phase, opponent, and seat aggregate CI spans
  zero. The frozen diagnosis is `local_effect_unresolved_close_awr`; no model
  was trained or promoted and `035` remains overall champion.
- Interpretation: `033`'s failed game gate is not repaired by claiming that
  its changed actions are locally superior, nor does this audit establish
  systematic local harm. Instead, it shows that matched action-level returns
  are stable and frequently resolvable while trajectory AWR is approximately
  unaligned with them. A separately preregistered, fresh, game-disjoint direct
  action-preference corpus is now the strongest justified learning direction;
  do not train on these 80 diagnostic states. Frozen results are in
  `outputs/policy_delta_causal_audit_034/audit_final.json` and the concise
  decision is in `operational_decision.json`.

### Preregistered Action-Preference Generation 001

The next learning experiment is frozen before any new source games or matched
continuations are generated. It tests direct action-level supervision rather
than another trajectory-AWR update.

- Generate 2,880 fresh `035` league games at seeds 361000-363879, exactly
  balanced across `035`, `016`, shallow, and `023` opponents and both actor
  seats. Each opponent/seat cell is also assigned evenly to early, mid, and
  late selection phases.
- Select exactly one state per game without reading outcomes: the state in its
  assigned phase with the smallest `035` logit gap between its top two
  distinct canonical successors. Retain the 96 most uncertain games per each
  of 24 opponent/seat/phase strata, then freeze game-disjoint splits of 64/16/16
  per stratum: 1,536 train, 384 validation, and 384 untouched confirmation.
- Evaluate both actions with identical information-legal determinizations and
  matched continuation seeds: 3 determinizations x 8 continuations per action.
  Require complete continuations, stable split-half action signal, at least
  450/100/100 resolved train/validation/confirmation states, and both label
  directions in every split before training.
- Train one full attention scorer initialized from `035` for 8 epochs with
  direct pairwise softplus loss, AdamW at 3e-5, and a frozen-parent KL anchor.
  Select the epoch only on validation pairwise loss, then deterministically
  interpolate its parameters toward `035` until validation raw KL is at most
  0.004. Confirmation labels are first read only after both choices are fixed.
- Offline passage requires at least 3% pairwise-loss improvement on validation
  and confirmation, a confirmation paired-loss CI wholly below zero, at least
  +5 percentage points confirmation accuracy, at least 20% accuracy on states
  where the challenger is correct, and at least 90% retention where `035` is
  correct. Movement must be 1-5% on fresh on-policy and broad streams with mean
  raw KL at most 0.008.
- Only then run an untouched 200-pair/400-game gate against `035` requiring at
  least 55% score share and CI95 lower bound above 50%, followed conditionally
  by matched broad gates against `016`, shallow, and `023`. `035` remains the
  champion unless every gate passes. CPU is appropriate; GPU would only reduce
  the short training stage, not the dominant rollout cost.
- Implementation is in `run_action_preference_corpus.py`,
  `run_action_preference_training.py`, and
  `run_action_preference_generation.py`. The runner is resumable at source
  games, evaluated states, training completion, and gameplay gates. The shared
  movement audit now masks padded action rows before argmax comparisons.
- Pre-evidence verification: 347 tests passed in 147.69 seconds, including four
  new action-preference contracts.
- Provenance correction: the initial 4-worker run completed 1,866 source games
  before its first restart; an inspection command had queried the wrong nested
  directory and initially reported zero. The count was discovered from runner
  progress only, without inspecting outcomes. The operational worker count was
  then raised to 10 for this 10-core/20-thread machine. A second pause at 2,395
  completed source games froze this worker history explicitly in the protocol.
  Worker count cannot affect the fixed per-game/per-continuation seeds; all
  selection, split, continuation, threshold, and promotion rules are unchanged.
- Final result (2026-07-17): all 2,880 source games completed with
  exact 360-game balance in every opponent/seat cell, 91,340 acting-`035`
  turns, zero stalls, zero timeouts, and zero loop fallbacks. Outcome-blind
  selection passed with exactly 2,304 unique games and frozen 1,536/384/384
  splits; all 24 strata have their exact allocation and no source game lacked
  its assigned phase. All 2,304 states and 110,592 action branches were
  evaluated. Signal quality was strong: split-half correlation 0.8332, 87.63%
  sign agreement across 1,899 eligible states, and 1,057 resolved states
  (696/172/189 train/validation/confirmation) with approximately balanced
  preference directions in every split. Aggregate top-two return was neutral:
  -0.0095 score points with CI95 [-0.1336, +0.1140]. The corpus nevertheless
  failed its strict operational gate because one confirmation-state candidate
  continuation entered a score-stable repetition loop: 1 cutoff among 110,592
  branches. No training occurred and `035` remains champion.

### Preregistered Action-Preference Loop Repair 002

Generation 002 repairs the sole operational failure without changing or
discarding Generation 001 evidence.

- Reuse exactly the frozen 2,304 outcome-blind source states, game-disjoint
  splits, action pairs, and opponent/seat/phase allocation. Previous
  continuation outcomes are forbidden; reevaluate every state using fresh
  continuation seed namespace 36000000.
- Normal actions use the unchanged stochastic `035` continuation policy. Only
  at the exact repetition/no-progress boundary where the old evaluator would
  stop, select the highest-scored canonical buy if available; otherwise select
  the highest-scored least-used distinct canonical successor. Record every
  fallback trigger.
- If 64 fallback actions cannot escape a score-stable loop, resolve it by its
  limiting score margin rather than treating it as missing data. Require zero
  unresolved continuations, zero 400-turn hard caps, and no more than 0.1% loop
  adjudications, in addition to every frozen Generation 001 signal, resolved-
  count, label-balance, training, movement, direct, and broad gate.
- The known failed branch was used only as a pre-protocol operational test. A
  plain canonical-diversification fallback and a buy-prioritized variant both
  remained score-stable for 400 turns; the fixed 64-trigger adjudication
  resolved it after 98 turns. This diagnostic uses the old seed and is
  ineligible learning evidence.
- Implementation is isolated in `run_action_preference_repair.py`; outputs go
  to `outputs/action_preference_generation_002_loop_repair`. No repaired
  continuation outcome may be generated before this protocol is tested and
  checkpointed.
- Pre-evidence verification: the final full suite passed with 350 tests in 148.98
  seconds, including canonical buy priority, least-used canonical-successor
  diversification, bounded loop-adjudication contracts, and literal post-model-
  selection confirmation loading.
- Final result: the repaired corpus passed every gate with all 2,304 states,
  zero cutoffs, zero hard caps, and—under fresh seeds—zero fallback or loop-
  adjudication triggers. It produced 1,071 resolved states: 709 train, 163
  validation, and 199 confirmation, with balanced preference directions.
  Split-half correlation was 0.8323 and eligible sign agreement was 86.92%.
- The single full-model pairwise proposal selected epoch 4, then projected to
  alpha 0.884765625 for validation raw KL 0.003998. It failed the offline gate:
  validation loss improved only 0.22% versus the required 3%; confirmation
  loss worsened by 0.00157 with CI95 [-0.00811, +0.01084]; accuracy improved
  only 1.0 point versus the required 5; and parent-win retention was 74.23%
  versus the required 90%. No movement or gameplay gates ran. Reject the
  candidate, keep `035`, and do not tune full-model learning rate or epochs.

### Preregistered Frozen-Representation Learnability Probe 003

Generation 002 supplied only 709 resolved training labels for a multi-million-
parameter update, while training loss improved continuously and validation
stayed flat. Probe 003 tests the bounded alternative hypothesis that the labels
are predictable but require a much lower-capacity learner.

- Freeze every `035` parameter. For each resolved parent/challenger pair,
  extract only information-legal frozen features: state hidden vector,
  challenger-minus-parent contextual action vector, pair-mean contextual
  vector, frozen challenger-minus-parent logit, and legal turn-phase one-hot.
  Opponent identity is excluded from features.
- Use only Generation 002's 709 training rows for model selection. Select one
  L2 penalty from `[1e-4,1e-3,1e-2,1e-1,1,10]` by minimum mean log loss across
  five deterministic folds stratified by label, phase, opponent, and seat.
  Fit one linear logistic head on all training rows after selection.
- Parse the 163 validation labels exactly once after selection and fitting.
  Confirmation is path-recorded but never opened. Require balanced accuracy at
  least 55%, bootstrap AUC CI95 lower bound above 0.5, at least 2% log-loss
  improvement over the train-prevalence intercept, train/validation balanced-
  accuracy gap at most 10 points, and at least 45% balanced accuracy in every
  phase/opponent/seat validation bucket with at least ten rows.
- This probe is diagnostic only: no gameplay or promotion is allowed. A full
  pass authorizes a fresh, larger disjoint preference corpus and a separately
  preregistered selective residual reranker. Failure closes action-preference
  learning as the primary direction. CPU is sufficient.
- Pre-evidence verification (2026-07-17): the focused preference contracts
  passed 9/9 and the complete suite passed 352 tests in 100.65 seconds. Commit
  this protocol and runner before launching the probe or reading validation.
- Final result: the probe ran from preregistration commit `f97664a` and failed
  decisively. Train-only CV selected the strongest penalty, L2 `10.0`; its
  mean held-out log loss was 0.68817, balanced accuracy 50.03%, and AUC 0.50236.
  The final 516-feature head reached training AUC 0.68857 but only 50.0%
  balanced accuracy. On the once-opened 163-row validation split it achieved
  50.0% balanced accuracy, AUC 0.46612 with bootstrap CI95
  [0.37683, 0.55631], and log loss 0.70022 versus the 0.69812 prevalence
  baseline, a -0.30% relative improvement. The sample-count, generalization-
  gap, breakdown-floor, and confirmation-untouched gates passed; the minimum
  balanced-accuracy, AUC, and log-loss gates failed. Confirmation was never
  opened, no gameplay ran, and no checkpoint is promotable.
- Decision: close direct action-preference learning as the primary direction;
  do not rescue it with another capacity, learning-rate, or corpus-size sweep.
  `035` remains overall champion. Frozen evidence is in
  `outputs/action_preference_learnability_probe_003/probe_final.json`; its
  SHA256 is `5fd1c5f6a2fe3a1d01f4338b2bf9322a423b9f0d36355d3fd3dcf347e85645a9`.

### Preregistered Selective Matched-Rollout Gate `004`

The negative learnability result closes compression of the action-preference
signal, not the direct signal itself. Generation 002's 0.8323 split-half
correlation, 86.92% eligible sign agreement, and 1,071 resolved states justify
one distinct search operator that consumes matched terminal returns directly.

- Freeze legal `035` as both the normal action policy and stochastic
  continuation policy. Search only when its top two distinct canonical public
  successors have raw logit gap at most 0.5, and permit at most two searched
  decisions per game. Every other decision is unchanged `035` argmax.
- Development may read only Generation 002 train and validation rows; its
  confirmation split is forbidden. Select one positive switch threshold from
  `[0.025, 0.05, 0.075, 0.10]` using symmetric train-half evaluation: maximum
  held-out-half mean gain among thresholds with 5%-30% switch rate, at least 20
  switches, and positive mean gain; ties choose the larger threshold. Require
  the fixed threshold on validation to retain 5%-30% coverage, at least 20
  switches, at least 60% positive evaluation-half effects, and bootstrap mean
  gain CI95 wholly above zero. Failure stops before new simulation.
- Only a development pass generates 960 fresh `035` league games at seeds
  372000-372959, exactly balanced over `035`, `016`, gate-030 shallow, and
  clean `023` opponents, both actor seats, and assigned early/mid/late phase.
  Outcome-blind selection retains the 20 most uncertain eligible unique games
  per opponent/seat/phase stratum: 480 states total.
- At each fresh state compare only the two frozen actions with three legal
  determinizations and eight matched continuations per action. The selector
  sees the first four continuations per determinization; the other four are an
  independent evaluation half. Continuations use stochastic `035` at epsilon
  0.1, temperature 0.5, top-3, score-margin `/20`, the loop-safe 400-turn
  contract, and fresh seed namespace 38000000.
- Fresh action quality must have every branch resolved, zero hard caps, at most
  0.1% loop adjudications, 5%-30% switches, at least 20 switches, at least 60%
  positive evaluation effects, bootstrap mean evaluation gain CI95 wholly
  above zero, and phase/opponent/seat breadth: at least three switches and mean
  switched gain no worse than -0.025 in every bucket.
- Only a full fresh pass unlocks 200 paired seeds/400 games versus raw `035` at
  seeds 374000-374199. Require zero game failures, at least 55% score share,
  pair-bootstrap CI95 lower bound above 50%, both searched and switched turns,
  search on at most 10% of candidate turns, zero rollout hard caps, and at most
  0.1% rollout adjudications. A pass establishes a stronger non-learning
  teacher and authorizes a separately preregistered iterative corpus; it does
  not promote a neural checkpoint. A failure closes this direct-rollout
  operator. CPU remains appropriate because environment simulation dominates.
- Implementation is isolated in `run_selective_rollout_gate.py` and
  `src/splendor_ai/eval/run_selective_rollout_gate.py`. No development metrics,
  fresh games, continuation outcomes, or gameplay may be generated until the
  protocol, tests, and full-suite verification are checkpointed.
- Pre-evidence verification (2026-07-17): the focused rollout/preference suite
  passed 14/14 and the complete suite passed 357 tests in 142.63 seconds.
- Final result: reject `004` at the fresh action-quality gate and do not run
  gameplay. Train-only selection chose threshold 0.075. On 384 development
  validation rows it switched 26.04% of half-orientations and gained +0.04004
  normalized return with CI95 [+0.03158, +0.04893]; 91.0% of switched actions
  were positive on the opposite half. The development gate passed completely.
- All 960 fresh games completed normally with exact 120-game opponent/seat
  balance, 30,389 actor turns, and zero stalls, timeouts, or fallbacks.
  Outcome-blind selection produced exactly 480 unique states, 20 in every
  opponent/seat/phase stratum, all within the frozen 0.5 routing gap.
- All 23,040 fresh continuation branches resolved naturally with zero
  fallbacks, adjudications, or hard caps. Independent evaluation strongly
  confirms action quality: 147 switched states gained +0.14158 on average,
  CI95 [+0.12234, +0.16088], and 90.48% were positive. Including unchanged
  states, mean gain was +0.04336, CI95 [+0.03526, +0.05183]. Every phase,
  opponent, and seat breadth gate passed.
- The sole failed condition is the preregistered maximum switch rate:
  147/480 = 30.625%, three states above the 30% ceiling. This is a strict
  near-boundary operational rejection, not a reversal of the direct signal.
  Do not relax the ceiling, select threshold 0.10 after seeing this result, or
  run seeds 374000-374199. Under the frozen decision rule, close this exact
  direct-rollout operator and keep `035` overall champion. The evidence is in
  `outputs/selective_matched_rollout_004/fresh_action_quality.json` (SHA256
  `51e1f7282504463a4ace4fa52142b70a63663325617750a6b4a10b3e4bbc67e6`)
  and `gate_final.json` (SHA256
  `6a00bd0ab5d89ccf96208b9940fbfdd0d1f7795ddc2255a18464c49a5102c12d`).

### Preregistered One-Shot Matched-Rollout Gate `005`

`004` formally closes its uncertainty-enriched two-search contract, but its
independent action-quality result is the strongest positive decision signal in
the project. `005` tests a deployment-realistic, structurally bounded operator;
it is not permission to relax `004`, select threshold 0.10, or reuse any `004`
source state, continuation, or gameplay seed.

- Freeze threshold 0.075, raw-logit routing gap 0.5, three information-legal
  determinizations, four discovery continuations per determinization/action,
  stochastic `035` continuation at epsilon 0.1/temperature 0.5/top-3, score
  margin `/20`, and the loop-safe 400-turn continuation contract. No parameter
  selection occurs in `005`.
- Assign every game one search phase—early, mid, or late—by environment seed
  modulo three. Permit exactly one comparison: the first acting-`035` state in
  the assigned phase whose top two distinct canonical successors have gap at
  most 0.5. If none exists, the game remains raw `035`. This is online-causal;
  it never chooses a later, more uncertain state after seeing the future.
- Generate 960 fresh league games at seeds 375000-375959, exactly balanced over
  `035`, `016`, gate-030 shallow, clean `023`, both actor seats, and assigned
  phase. Retain the first 20 eligible unique games in each of the 24
  opponent/seat/phase strata, without outcomes: 480 confirmation states.
- Evaluate each state with the fixed discovery half and a separate four-
  continuation evaluation half under fresh continuation namespace 40000000.
  Require every branch resolved, zero hard caps, at most 0.1% adjudications,
  at least 20 switches, bootstrap mean evaluation gain CI95 wholly above zero,
  at least 60% positive switched effects, and phase/opponent/seat breadth with
  at least three switches and mean switched gain at least -0.025 per bucket.
- Measure capacity over every acting-`035` source turn, not the enriched 480-
  state confirmation set. The eligible one-shot search fraction must be at most
  5%; after multiplying by the independently measured switch probability, the
  projected changed-turn fraction must be at most 3%.
- Only a complete fresh pass unlocks 200 paired seeds/400 games against raw
  `035` at seeds 377000-377199. Each candidate game uses the same seed-based
  phase schedule and at most one comparison. Require zero failures, at least
  55% score share, pair-bootstrap CI95 lower bound above 50%, actual searches
  in every phase, at least one switch, search on at most 5% and switching on at
  most 3% of candidate turns, zero rollout hard caps, and at most 0.1% rollout
  adjudications. A pass establishes a stronger non-learning teacher but does
  not promote a neural model. Any failure closes direct matched-rollout search
  and sends the project to the architectural-reset branch.
- Implementation is isolated in `run_one_shot_rollout_gate.py` and
  `src/splendor_ai/eval/run_one_shot_rollout_gate.py`. No `005` source game,
  continuation, or gameplay result may be generated before this protocol,
  focused contracts, and the complete suite are checkpointed.
- Pre-evidence verification (2026-07-17): the combined focused rollout suite
  passed 17/17 and the complete suite passed 360 tests in 140.61 seconds.
- Final result: `005` passed every fresh offline gate and failed the final
  direct `035` gameplay gate. All 960 source games completed normally with
  exact 120-game opponent/seat balance, 30,422 actor turns, and zero stalls,
  timeouts, or fallbacks. The online-causal first-trigger rule found 920
  eligible games and implied a 3.024% search-turn rate. Selection produced the
  exact 480-state, 20-per-stratum confirmation block.
- All 23,040 confirmation branches resolved naturally with zero fallbacks,
  adjudications, or hard caps. The fixed 0.075 rule switched 145/480 states.
  Mean independent evaluation gain including unchanged states was +0.04592,
  CI95 [+0.03711, +0.05521]; switched actions gained +0.15201, CI95
  [+0.13037, +0.17382], and 91.72% were positive. Every phase, opponent, seat,
  integrity, breadth, and capacity gate passed. Projected changed-turn rate was
  only 0.914%.
- In the preregistered 200 paired seeds/400 games versus raw `035`, the one-shot
  candidate went 189-206-5: score share 47.875%, pair-bootstrap CI95
  [44.625%, 51.125%], and mean final-score margin -0.0825, CI95
  [-0.5125, +0.34]. All 400 games completed normally. It searched 384/12,864
  candidate turns (2.985%) and changed 115 (0.894%), with every phase covered
  and zero rollout fallbacks, adjudications, or hard caps. Only the 55% score-
  share and lower-bound-above-50% gates failed.
- Decision: reject `005`, keep `035` overall champion, and close direct matched-
  terminal-rollout search as an improvement line. Two independent fresh state
  blocks show that its local estimates are reproducible, but the intervention
  does not translate to trajectory strength; do not tune its threshold, phase
  schedule, rollout count, continuation policy, or routing budget. The next
  project branch must be the architectural reset, not another rollout-search
  repair. Frozen evidence is `fresh_action_quality.json` (SHA256
  `6a83255ca81f1d137f55d221afb14eeb39fa93412e294786f001cb170abccae5`),
  `direct_vs_035.json` (SHA256
  `e694e4dec1e766e3766007c6538dba3f03937d76514adb1d1d1531f6f546e01c`),
  and `gate_final.json` (SHA256
  `0b18b647ea920786088329970bc2f7928c44481cb7a72c0b67b9b4be8e7e8e90`).

### Completed Overnight Reserve Corpus

A 500-state matched-return corpus job launched at 2026-07-14 01:28 local and
completed at 04:37 through `run_reserve_training_500_overnight.ps1`. Its pass
and the downstream rejected `042`-`044` experiments are documented in the
reserve sections above; it is no longer an active process.

- Source: `data/corpus_generation1_policy035_vs016_038_retry/replays.jsonl`,
  seeds 48000-48499. These 500 games are disjoint from the 60-state validation
  source at seeds 47000-47999 and add opponent diversity through baseline 016.
- Selection completed successfully: all 5,000 policy035 turns 0-19 were
  screened; 500 deterministic policy035 reserve states were selected, 125 per
  turn band and exactly one from each of 500 unique games.
- Evaluation is fixed at 3 determinizations x 8 paired terminal continuations
  per candidate. Human rows are forbidden and selection never uses outcomes.
- Progress and source snapshots:
  `data/corpus_reserve_choice_policy035_vs016_500/reserve_matched_returns.progress.json`.
- Final outputs, when complete:
  `data/corpus_reserve_choice_policy035_vs016_500/reserve_matched_returns.json`
  and `.md`.
- Logs and PID are in the same directory as `overnight.stdout.log`,
  `overnight.stderr.log`, and `overnight.pid`. At launch the wrapper PID was
  45760 and Python PID was 46652; treat those as stale if the job has completed.
- The wrapper creates a temporary Windows system-awake request and releases it
  on exit. It is crash-resumable after every completed state. Do not launch a
  second copy while the recorded process is active. If interrupted, rerun the
  same wrapper; cached selection and completed states will be reused.

## Useful Commands

Full test command (no `PYTHONPATH` needed since the editable install; see the
Environment section at the top of this file):

```powershell
& "C:\Users\leoba\anaconda3\python.exe" -m pytest -q --basetemp .codex_pytest_tmp_all
```

Deck-reserve mask ablation `071` (reproduces the frozen panel exactly):

```powershell
& "C:\Users\leoba\anaconda3\python.exe" run_deck_reserve_mask_ablation.py --workers 10
```

Latest full suite: `572 passed` in about 203 s on 2026-08-03.

Historical full test status (kept for provenance; the counts below predate the
`071` contracts and the editable install):

- 360 passed after legal hidden-information search, PUCT batching, root-visit
  replay/training telemetry, paired direct checkpoint gates, the generation-1
  pilot, the held-out uptake audit, conflict-free root-visit correction,
  counterfactual leaf-quality audit, action-value corpus export/training,
  residual PUCT error attribution, disjoint root-prior confirmation,
  CPU high-budget action-quality gating, resumable mid-game headroom auditing,
  plain-PUCT ablation support, true leave-one-out trajectory construction,
  pre-training signal gating, actor movement/KL auditing, and held-out
  human-game forensic action-quality auditing, and disjoint non-human reserve
  generalization auditing with resumable matched terminal continuations, and
  reserve counterfactual corpus compilation, constrained training, uptake
  gating, fixed objective/capacity diagnosis, canonical entity modeling, the
  public-actor/privileged-critic smoke generation, opponent-seat balancing,
  frontier-curriculum selection, score-margin return validation, and matched
  return-mode actor/critic branching with game-disjoint critic diagnostics,
  causal telescoping dense score-change credit with matched A/B gates, and
  fresh game-disjoint entity counterfactual-credit stability diagnostics, and
  fixed-reference mid/late counterfactual confirmation gates, resumable
  parallel late counterfactual collection, auxiliary entity-Q gating, and
  resumable three-way-heldout clean entity foundation scaling, actor-source
  filtering, and coherent shallow-self-play corpus contracts.
  Coherent single-width promotion training, matched broad strength gates, and
  one-shot integrity replacement contracts are also covered. Fresh balanced
  AWR league collection and deterministic parameter-space trust-region
  projection contracts for `032`/`033` are covered as well, together with the
  outcome-blind, unique-game, matched-continuation causal audit contract for
  `034`.

## Files To Read Next

- `AGENT_CURRENT.md`
- `docs/project_review_response_070.md` for the current blocker diagnosis,
  gate-power arithmetic, closure classification, kill list, and the deferred
  resolver-BC protocol sketch
- `docs/experiments/deck_reserve_mask_ablation_071.md` and
  `src/splendor_ai/bots/deck_reserve_mask.py` as the template for a cheap,
  properly dosed, inference-only ablation
- `docs/experiments/scripted_value_teacher_072.md` and `..._073.md` for the
  faithfulness-diagnostic pattern and the terminality clause that stopped an
  invalid-then-retry chain
- `scripts/analysis/` for the re-runnable derivations behind the diagnosis
- `AGENT_HANDOFF.md` only if historical detail is needed
- `src/splendor_ai/training/run_policy_diff_against_teacher.py`
- `src/splendor_ai/training/run_policy_search_disagreement.py`
- `src/splendor_ai/training/run_conversion_diagnostics.py`
- `src/splendor_ai/training/run_pgs_corpus_audit.py`
- `src/splendor_ai/training/run_constrained_preference_corpus.py`
- `src/splendor_ai/training/run_policy_drift.py`
- `src/splendor_ai/training/run_candidate_harness.py`
- `src/splendor_ai/training/run_trajectory_advantage_corpus.py`
- `src/splendor_ai/training/run_group_relative_trajectory_corpus.py`
- `src/splendor_ai/training/run_trajectory_policy_gradient.py`
- `src/splendor_ai/training/run_rollout_action_credit.py`
- `src/splendor_ai/training/run_rollout_credit_training.py`
- `src/splendor_ai/training/improve_replay_actions.py`
- `src/splendor_ai/training/run_legal_action_supervised.py`
- `src/splendor_ai/bots/search_bot.py`
- `src/splendor_ai/bots/puct_bot.py`
- `src/splendor_ai/eval/run_puct_benchmark.py`
- `src/splendor_ai/training/run_root_visit_target_audit.py`
- `src/splendor_ai/training/run_root_visit_correction.py`
- `src/splendor_ai/training/run_leaf_quality_audit.py`
- `src/splendor_ai/training/run_action_value_training.py`
- `src/splendor_ai/training/run_residual_error_audit.py`
- `src/splendor_ai/training/run_root_prior_control_ablation.py`
- `src/splendor_ai/training/run_puct_budget_ablation.py`
- `src/splendor_ai/training/run_midgame_headroom_audit.py`
- `src/splendor_ai/training/run_group_relative_signal_audit.py`
- `src/splendor_ai/training/run_actor_movement_audit.py`
- `src/splendor_ai/training/run_human_game_forensic_audit.py`
- `src/splendor_ai/training/run_reserve_generalization_audit.py`
- `src/splendor_ai/training/entity_actor_critic.py`
- `src/splendor_ai/training/run_entity_actor_critic.py`
- `src/splendor_ai/training/entity_return_validation.py`
- `src/splendor_ai/training/run_entity_return_validation.py`
- `src/splendor_ai/training/run_entity_return_ab.py`
- `src/splendor_ai/training/run_entity_temporal_credit_ab.py`
- `src/splendor_ai/training/run_entity_counterfactual_credit.py`
- `src/splendor_ai/training/run_entity_counterfactual_confirmation.py`
- `src/splendor_ai/training/run_entity_late_counterfactual_corpus.py`
- `src/splendor_ai/training/run_entity_action_value_training.py`
- `src/splendor_ai/training/run_entity_supervised_scale.py`
- `src/splendor_ai/training/run_entity_shallow_actor_ablation.py`
- `src/splendor_ai/training/run_entity_coherent_supervised.py`
- `src/splendor_ai/training/run_entity_broad_promotion_gate.py`
