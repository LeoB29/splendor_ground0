# AlphaZero Programme Lessons After 102

Date recorded: 2026-08-15

## Why this document exists

This is the controlling strategic retrospective after the first convincing
multi-generation AlphaZero result. Read it before proposing more one-step
corrective experiments or returning to `035`-derived teachers.

## What 102 established

The clean generation-2 to generation-5 rolling lineage produced measurable
compound improvement under a prospectively fixed 200-game paired endpoint
test:

- searched gen5 vs searched gen2: 116-1-83, 58.25%, margin +1.385, paired CI95
  `[51.5%,64.75%]`, seats 62%/54.5%;
- raw gen5 vs raw gen2: 110-1-89, 55.25%, margin +0.995, paired CI95
  `[48%,62.25%]`, seats 55.5%/55%;
- gen5 search vs gen5 raw: 129-0-71, 64.5%, margin +1.865, paired CI95
  `[58%,71%]`, seats 67%/62%.

All 600 games completed normally. The controlling decision is
`endpoint_compounding_with_raw_transfer_and_healthy_search`. This establishes
internal lineage improvement, not superiority to `035` or the research
champion, and it did not authorize installation. Exact generation 2 remains
installed while exact generation 5 is the successful sandbox endpoint.

## Why the loop appeared difficult at first

1. **The initial dose was too small.** The 090 pilot used only 32--64 games and
   32--64 simulations per move. The successful lineage used 256 games per
   generation and 256 simulations per move. A from-scratch network trained on
   sparse terminal outcomes needs substantially more clean search data.
2. **Search was not initially a dependable policy-improvement operator.** The
   64-simulation configurations did not reliably improve the raw network. The
   094 calibration established 256 simulations and `c_puct=2.25` as a strong
   operator (64% over generation-2 raw on 100 confirmation games). AlphaZero
   cannot compound when its search teacher is not stronger than its student.
3. **Small arenas were interpreted too literally.** The fixed 16-game
   intermediate diagnostics contained apparent regressions (including 6-10
   and 7-9 searched results), while the adequately sized endpoint was 116-1-83.
   These small panels are capable of detecting catastrophic breakage but not
   modest strength movement.
4. **Value learning and replay were initially unstable.** Small fresh corpora,
   sparse WDL labels, and single-step evaluation made ordinary value variance
   look like a structural failure. More games, a rolling two-corpus replay
   window, early stopping, and multiple generations produced a healthier
   trajectory. Absolute validation losses from different rolling splits must
   not be compared as though they share one dataset.
5. **We expected every generation to look monotonically stronger.** A useful
   learning trajectory can contain locally flat or negative measurements.
   Advancement inside an engineering sandbox should depend on data/checkpoint
   validity and basic teacher health, while accumulated strength is judged at
   a sufficiently powered endpoint. This does not mean ignoring genuine
   corruption or propagating an invalid corpus.
6. **Ordinary engineering failures mattered.** One forced-pass deadlock game
   gave an inadmissible label in 099, and the host crashed during 101.
   Outcome-blind sanitization, exact artifact hashes, atomic per-game progress,
   and exact-prefix resumption were necessary to keep the evidence trustworthy.
7. **Mechanism diagnostics were sometimes mistaken for outcomes.** Declining
   search disagreement can mean policy uptake, not search collapse. A temporary
   increase in `RESERVE_DECK` selection was worth auditing but did not establish
   that the architecture was wrong; the rate subsequently receded. Gameplay
   evidence must arbitrate such ambiguous diagnostics.

## Durable methodological lessons

- AlphaZero is a feedback system, not an automatic self-play command:
  `network -> stronger search -> clean targets -> improved network -> stronger search`.
- Demonstrate that search improves the current raw network before treating its
  visits as a high-quality teacher.
- Use adequate search and self-play dose from the beginning of a scaled run.
- Retain a bounded recent replay window and compare parent/child losses only on
  the same held-out split.
- Use tiny arenas as health monitors, not promotion or architecture verdicts.
- Use fresh paired seats and at least 100 pairs for important endpoint claims;
  report paired-bootstrap uncertainty and seat-specific results.
- Do not require noisy adjacent-generation results to be monotonic. Evaluate a
  predeclared multi-generation endpoint before redesigning a healthy loop.
- Keep abnormal-game exclusion outcome-blind and never train on invalid
  adjudicated labels.
- Preserve atomic progress and exact-prefix resume for every expensive panel.
- Keep blind deck reservation legal but private and monitored. It is a narrow
  imperfect-information exception, not a reason to organize the whole system
  around belief machinery.
- Keep `035` as an external thermometer. Do not use it as the AlphaZero prior,
  teacher, candidate generator, continuation policy, encoder, or target.
- Separate three questions: internal loop improvement, absolute strength versus
  `035`, and formal champion promotion. Evidence for one is not evidence for
  all three.

## What the apparent simplicity of AlphaGo does not imply

Go is strategically much larger, but it offers a clean deterministic
perfect-information interface, a fixed action grid, useful symmetries, and a
research programme with enormous compute and mature infrastructure. Splendor
is smaller strategically but has variable legal action objects, stochastic
card revelation, a private reserve exception, and sparse terminal labels.
Retrospective presentations also compress extensive calibration and engineering
into a simple algorithmic diagram. The difficulty observed here is therefore
not evidence that the AlphaZero direction is inappropriate.

## `035` uncertainty resolved by 103

The pre-103 best guess was that searched play at approximately the gen5 level
would score 20--30% against `035`, based on a weak old 0-8 gen2-raw anchor and
later internal gains. Direct 103 evidence now controls: searched gen8 scored
53-0-147 (26.5%, paired CI95 `[20.5%,33%]`) against raw legal `035`; raw gen8
scored 24-1-175 (12.25%, CI `[8%,17%]`). Search materially narrows the gap, but
the current AlphaZero lineage remains decisively below `035` in absolute
strength. This does not negate two powered internal compounding results.

## Next experiment decision: 103 scaled continuation

An initial suggestion was to benchmark gen5 against `035` immediately. The
owner correctly questioned its decision value: the likely result was a clear
loss, after which the programme would continue self-play anyway. Measurement
alone would delay the action already favored by the evidence.

That suggestion is superseded. Frozen experiment 103 advances the unchanged
successful recipe from gen5 through gen8, then evaluates gen8 against gen5 on
a powered endpoint. Only an internal endpoint pass opens the direct gen8 versus
`035` benchmark. This obtains an absolute anchor when it can change the next
decision and avoids repeatedly testing every generation against `035`.

Protocol: `docs/engineering/alphazero_scaled_continuation_103.md`.
