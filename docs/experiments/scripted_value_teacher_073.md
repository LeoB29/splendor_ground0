# Scripted Value Teacher, Gold-Disciplined and Reserve-and-Clear (`073`)

Frozen before implementing either rule change, running any game, or inspecting any
outcome. No training, no checkpoint, no neural component.

## Authorization and terminality

`072` returned `scripted_value_teacher_weaker_close_scripted_strategy_line` at
21-379, but its preregistered faithfulness diagnostics failed: the encoded rules
did not reproduce the strategy the expert described. `072`'s protocol permits
**exactly one** re-elicitation on that basis, separately preregistered. This is it.

**`073` is terminal for this line.** If it fails, close the scripted-strategy
direction regardless of what the diagnostics say. Do not preregister a `074` in
this family, do not sweep thresholds, and do not re-elicit again. That rule is
frozen here so a third attempt cannot be justified after seeing this result.

## What `072` established, and what it did not

Established: the encoded bot paralyses itself. Versus the expert's recorded mix it
took tokens on `79.7%` of decisions against `51.3%`, bought from reserve on `1.7%`
against `18.8%`, converted `15.8%` of reserves against `81.0%`, and bought `2.9`
cards per game against about `7`. It ended `4.5%` of games in
`forced_pass_deadlock` against `071`'s 1 in 2,000, the signature of hoarding to
the ten-token cap.

Not established: anything about whether the expert's strategy wins. `072` tested
an unfaithful encoding.

Mechanical causes identified in `072`, both in the frozen rules rather than in
bugs:

1. Targets are chosen by effective value alone, so the target is nearly always a
   `7X` mono tier-3. `BANK_GEMS_PER_COLOR_2P = 4`, so that card always needs at
   least 3 gold.
2. Rule 6 spent exactly that gold on cheap discount-builders: `856` gold across
   `1,159` purchases, of which `849` were tier-1.

## The two changes, both elicited

Asked directly, the expert answered: gold is **protected** when holding a
high-cost target, and reserving is **reserve-and-clear** rather than keeping all
three slots filled. `073` encodes those two answers and changes nothing else.

**Change 1 -- gold discipline.** A purchase may spend gold only if at least one
holds:

- it reaches 15 points, or
- the card being bought *is* the current best hand target, or
- after spending, the gold still held is at least the best hand target's colour
  shortfall, where colour shortfall is
  `sum over colours of max(0, cost - bonus - colour tokens held)`.

Otherwise the purchase is skipped and a later rule fires. This makes the
`4 tokens + 3 gold = 7` route to a `7X` card survivable, which `072` destroyed.

**Change 2 -- reserve and clear.** Rule 7 fires only when **no** card in hand
passes the efficiency filter. The bot therefore carries at most one live
high-value target and must convert it before reserving the next. The endgame block
in rule 3 is unaffected and may still use a free slot.

Everything else is frozen exactly as in `072`: the `0.40` efficiency filter, the
effective-value ranking, rule order, the ten-rule structure, the deterministic
variant preference, no deck reserves, no noble targeting, and no opponent model
outside rule 3.

Implementation constraint: the two changes are carried on a frozen
`ValueReserveRules` record defaulting to both disabled, so `072` remains
bit-reproducible from the same code.

## Panel

200 fresh paired seeds, `760000`--`760199`, each played twice with the teacher in
seat 0 then seat 1: **400 games**. The band is disjoint from every prior
experiment, including `072`'s `750000--750199`. Engine limits are the established
`max_turns=400`, `repetition_limit=4`, `no_progress_limit=60`. Valid terminals are
exactly `completed` and `forced_pass_deadlock`. No game may be replaced or
excluded on the basis of its outcome. Paired-cluster bootstrap, 20,000 draws,
seed `760073`.

## Declared power

Per-pair score-share standard deviation is about `0.32` (`043`), so 200 clusters
give a standard error near `0.023`. This resolves roughly `+6.4` points at 80%
power. Declared coarse in advance: the purpose is to separate clearly stronger,
clearly weaker, and competitive.

## Integrity gates

All required: exactly 200 clusters and 400 games with seeds used once per seat;
every game `completed` or `forced_pass_deadlock` with zero stalls and zero
timeouts; zero illegal actions and no null action; zero `RESERVE_DECK`; exact
determinism on a first-cluster replay; `035` hash equal to
`4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`; and both rule
changes observably active, meaning at least one purchase was skipped for gold
discipline and the count of decisions holding two or more filter-passing hand
cards is zero.

## Strength and decisions

Primary metric: teacher paired score share against raw legal `035`.

- Integrity failure: `scripted_value_teacher_073_invalid_no_strength_decision`.
- CI95 lower bound above `0.50`:
  `scripted_value_teacher_073_stronger_authorize_bc_teacher_preregistration`.
  Authorizes only a separately preregistered behaviour-cloning corpus and student
  with its own untouched gameplay gate. Does not promote the teacher.
- CI95 upper bound below `0.50`:
  `scripted_value_teacher_073_weaker_close_scripted_strategy_line_permanently`.
- Interval straddling `0.50`:
  `scripted_value_teacher_073_competitive_close_line_report_only`. Note this
  differs from `072`: a straddling interval now closes the line rather than
  authorizing a larger panel, because `073` is terminal.

A result materially better than `072` but still below `0.50` is **not** a partial
success and authorizes nothing. It would show the two elicited changes matter
without showing the strategy wins.

## Reported diagnostics, gated on nothing

Same as `072` and on the same terms -- reported to interpret the result, never a
gate, never a tuning target: action mix and reserve tier share against the
expert's recorded `51.3/23.3/18.8/6.6` and `14/21/28`, reserve conversion rate
against `81.0%`, purchases by tier, nobles claimed, mean game length, gold spent
per purchase, and the count of purchases skipped by gold discipline. Comparison
against the expert's recorded mix is a faithfulness check on an independently
derived rule set; the ten human games are not used to fit, select or tune
anything and remain held out.

## Prohibitions

Do not tune rules, thresholds, or the filter after seeing this result. Do not
promote the teacher. Do not pool this panel with `072` or any future panel. Do not
preregister a further re-elicitation. Do not rent a GPU; there is no neural
component.

## Evidence

`outputs/scripted_value_teacher_073/result.json`, SHA-256 recorded in
`AGENT_CURRENT.md` after the run.
