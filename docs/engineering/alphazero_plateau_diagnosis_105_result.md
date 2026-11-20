# AlphaZero Plateau Diagnosis and Pooled Replication 105 Result

Date completed: 2026-08-18

## Outcome

Experiment 105 completed in 4.48 hours with empty stderr. Its existing-data
audit sealed over 3,928 common network rows, 64 deterministically reconstructed
states, 256 fixed root searches, and 64 complete-game reserve traces. The fresh
searched gen11/gen8 replication completed all 200 games normally:

- fresh replication: 111-1-88, 55.75%, margin +0.80, seats 60.5%/51%, paired
  CI95 `[48.25%,63.25%]`;
- prospectively pooled 104+105 result: 220-1-179, 55.125%, margin +0.5775,
  seats 58.75%/51.5%, paired CI95 `[50%,60.125%]`.

The pooled panel passed every frozen practical criterion. The interval touches
50%, so this is a narrow positive result rather than overwhelming evidence.
The controlling decision is
`pooled_compounding_confirmed_review_035_no_install`. No model was installed.

## Offline attribution

The common-state matrix showed a recency/forgetting pattern. Relative to gen8,
gen11's objective loss was 0.0185 worse on the old gen7 corpus but 0.0656 better
on the newest gen10 corpus. Across gen7 -> gen10 corpora, gen11 moved from worse
old-target fit to materially better new-target fit. Gen11 search still changed
25% of raw actions on the fixed common states, consistent with the independent
69.25% gen11-search/gen11-raw gameplay result in 104.

This makes a two-corpus replay horizon the leading efficiency concern if later
scaling flattens again. It does not overturn the pooled gameplay result or prove
that capacity is irrelevant. The reserve follow-through sample did not support
a simple causal claim that blind reservation explains the plateau.

## Preserved artifacts and next step

- Raw result SHA-256:
  `71c2cb3564468059b7d82906d5e18f6fd8f4c144475c4e353dcc66156616a401`.
- Offline audit:
  `outputs/alphazero_plateau_diagnosis_105/offline_audit.json`.
- The next permitted measurement is a separately frozen exact gen11 versus raw
  legal `035` benchmark. It remains an external thermometer, never a teacher.
- No further multi-generation block should be launched until that external
  measurement and the replay-forgetting evidence are reviewed together.

