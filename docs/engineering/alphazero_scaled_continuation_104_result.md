# AlphaZero Scaled Continuation 104 Result

Date completed: 2026-08-18

## Outcome

Experiment 104 completed gen8 -> gen9 -> gen10 -> gen11 in 26.49 hours. All
three 256-game self-play corpora completed normally and all three child
checkpoints passed the frozen training-validity contract. Stderr was empty.

The powered searched endpoint narrowly missed its prospective practical gate:

| Panel | Result | Share | Margin | Seats | Paired CI95 |
|---|---:|---:|---:|---:|---:|
| searched gen11 vs searched gen8 | 109-0-91 | 54.5% | +0.355 | 57% / 52% | [47.5%, 61.5%] |
| raw gen11 vs raw gen8 | 97-2-101 | 49.0% | +0.085 | 56.5% / 41.5% | [42.5%, 55.75%] |
| gen11 search vs gen11 raw | 137-3-60 | 69.25% | +2.885 | 83% / 55.5% | [63%, 75.25%] |

The primary searched panel missed the 55% threshold by one win. Its positive
margin and both seat safeguards passed, but its interval included both no gain
and a gain comparable to earlier blocks. The raw endpoint was flat and also
contained one `forced_pass_deadlock`; this abnormal raw diagnostic did not
affect the all-normal primary searched evidence. Search remained a powerful
improvement operator over the final raw policy.

The prospective conditional `035` benchmark did not open. Nothing was
installed or promoted. The exact generation-2 installation was retained.

## Interpretation

The frozen decision is
`generation_11_endpoint_not_compounded_035_skipped`; the internal diagnosis is
`endpoint_practically_flat_diagnose_before_scaling`.

This means a third block was not confirmed. It does not prove that the block
made no progress: 54.5% is one game below the engineering gate and the sampling
interval is wide. At the same time, raw gen11's 49% result and the final 69.25%
search-over-raw result make incomplete search-target absorption a live concern.

The self-play summaries also show rising blind deck reservation, correctly
labelled by the checkpoint that generated each corpus: gen7 2.98%, gen8 4.68%,
gen9 5.76%, and gen10 6.34% of positions. This is a diagnostic signal, not a
causal verdict or a reason to remove the legal action.

## Preserved artifacts

- Raw result SHA-256:
  `63a51a8b98d0850fbd39430f3fe8469046b8fcc2b5d3961f25bba69c4b3e90ad`.
- Final sandbox gen11 SHA-256:
  `c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0`.
- No follow-on scaling should begin before an independent pooled replication
  and an existing-data plateau audit distinguish noise, forgetting, target
  absorption, and action-family drift.

