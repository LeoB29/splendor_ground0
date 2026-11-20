# AlphaZero Strategy After 103

Date recorded: 2026-08-16

## Controlling evidence

Two consecutive, prospectively evaluated three-generation blocks compounded
searched strength:

| Block | Powered searched endpoint | Paired CI95 |
|---|---:|---:|
| gen2 -> gen5 | 116-1-83, 58.25% | [51.5%, 64.75%] |
| gen5 -> gen8 | 118-0-82, 59.0% | [52.0%, 66.0%] |

This is replication of the learning dynamic, not merely two favorable
adjacent-generation panels. In each block, the improved endpoint became the
parent for another unchanged rolling self-play sequence and improvement
continued.

Raw transfer was smaller but strikingly consistent: raw gen5/gen2 scored
55.25%, and raw gen8/gen5 scored 55.5% overall (the latter missed its second-seat
safeguard by 0.5 percentage points). Search remains the stronger component:
gen8 search/raw scored 71.25%.

Absolute strength remains low. Direct 103 evidence measured searched gen8 at
26.5% against raw legal `035` and raw gen8 at 12.25%. These facts coexist:
the loop is reliably improving from a weak starting point, while the mature
external baseline remains decisively stronger.

## Interpretation

- The central architecture/search/self-play loop is no longer the leading
  uncertainty. It has passed twice at powered multi-generation endpoints.
- There is no evidence of searched-strength saturation yet: the two block gains
  are almost identical (58.25% and 59%).
- The network does not absorb all of search's strength, but full absorption is
  not required for an AlphaZero agent whose deployed policy includes search.
- Do not infer that every intermediate generation is monotonic; the small
  health panels remain noisy and non-gating.
- Do not redesign the representation, objective, replay window, or search
  operator merely because gen8 loses to `035`. That external gap was expected
  and the unchanged recipe is still producing replicated gains.
- A rough Bradley-Terry illustration puts each searched block near +60 Elo and
  gen8 about -177 Elo relative to `035`. If gains remained stable, approximately
  three more blocks could reach parity. This is planning intuition only;
  Splendor can be non-transitive and improvement can saturate.

## Recommended next step: 104

Run one more unchanged three-generation continuation, gen8 -> gen9 -> gen10 ->
gen11:

- 256 clean self-play games per parent generation;
- the confirmed 256-simulation / `c_puct=2.25` search operator;
- the same rolling two-clean-corpus replay window;
- the same stabilized training recipe and outcome-blind sanitation;
- small fixed diagnostics retained only as health observations;
- a powered 100-pair / 200-game searched gen11-versus-gen8 endpoint, plus raw
  transfer and gen11-search-versus-raw diagnostics;
- only after an internal searched pass, a fresh 200-game searched gen11 versus
  raw `035` benchmark and a cheap raw diagnostic;
- no automatic installation or champion promotion.

The value of 104 is no longer to prove the loop can work. It tests whether the
observed approximately +60-Elo-per-block slope persists for a third block and
whether that internal gain transfers to the external `035` matchup. A rough,
non-gating expectation is a searched gen11 `035` share in the mid-30% range.

Do not launch multiple further blocks blindly. Review 104 before deciding
whether to continue the unchanged recipe toward gen14/gen17, increase model or
data scale, or prioritize batched parallel self-play engineering.

