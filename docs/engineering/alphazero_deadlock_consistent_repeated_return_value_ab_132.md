# AlphaZero Deadlock-Consistent Repeated-Return Value A/B 132

Date frozen: 2026-08-30

## Purpose

Experiment 131 stopped correctly under its frozen rule after a continuation
ended in `forced_pass_deadlock`. Exact diagnosis reproduced two independent
cases: state 48, world 2 / policy 1 at turn 33, and state 294, world 1 /
policy 1 at turn 30. Both began from valid ordinary states. In the state-48
case all colored bank stacks were empty, both players held three reserved
cards, neither could buy, and each complete legal set was exactly `[PASS]`.
After both passes the canonical state signature repeated exactly.

The engine, replay, and ordinary gameplay contract already treats two sole
passes as a clean finite `forced_pass_deadlock`, adjudicated by prestige score
and then fewest purchased cards. AlphaZero search previously knew only the
engine's `state.terminal` flag, so it searched through the pass cycle until its
depth limit instead of backing up the adjudicated outcome. Experiment 132
repairs that mismatch and repeats the causal value-target test from scratch.
It does not salvage or reuse any 131 continuation.

## Frozen lineage and inputs

- failed experiment-131 result SHA-256:
  `dd0e0cd2cec25c5ba1fbf52e25bd81d75f3c548b9706c4dd056642343301d9dc`,
  with decision
  `repeated_return_value_integrity_failed_no_strength_conclusion`;
- experiment-130 result SHA-256:
  `cc884522a98a6b45105d28a8d878bd31267ad6504f444b14cb9d1dfd0df4d417`;
- exact generation-11 checkpoint SHA-256:
  `c97e7e06910639c0edf61fdd878fd665e390298790830e0e792061d9107354d0`;
- retained installed generation-2 checkpoint SHA-256:
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`;
- generation-2, -5, -8, and -11 replay hashes respectively:
  `855b9ae3cb4e05da7b26b759ce8fa4e62c8c44aea4a0dcee2da0e01a1790de56`,
  `0f9393fb020cf01c4d06f829c03e7426913dcc29d6afde49091cb9def57772e1`,
  `96f0d346743f30bd29e329f4bf3e2a7a361e778391c6d6e5833ca2a06af9be10`,
  and `59f1dfeb36fafc4af8b6fcf8b90b78f91a751be86ef4328ba0067c9514b84c7a`.

## Corrected search terminal contract

The shared contract has exactly two natural endings:

- `completed`, using the engine winner;
- `forced_pass_deadlock`, only after two consecutive turns whose complete
  legal action sets are each exactly `[PASS]`, using the established prestige
  then fewest-purchased-cards adjudication.

Inside AlphaZero tree search, after selecting a sole pass, inspect the next
player's complete legal set. If that set is also sole pass, apply the second
pass, adjudicate, and back up the exact outcome from the proper player
perspective. Do not expand or value the repeating position with the network.
`stalled`, repetition, no-progress, maximum-turn, and unknown endings remain
invalid.

Preflight must prove on a synthetic fixed-point deadlock that all eight search
visits receive the exact adjudicated root value `+1`, with no depth loop. It
must also seal the exact identities of reproduced states 48 and 294. Focused
tests cover shared adjudication, fixed-point identity, search backup, replay,
and repeated-return WDL recording.

## State and continuation schedule

Use the exact experiment-131 selection: source ordinals 112--171 provide 192
fitting and 48 development states; ordinals 172--186 provide 60 protected
states. The four sources and early/middle/late phases remain exactly balanced.

Regenerate every continuation in a fresh 132 output directory with the
corrected search operator. Do not copy any 131 state file. Use the same four
possible worlds, two training/development policy slots, four protected policy
slots, seed namespaces, 256 simulations, `c_puct=2.25`, real information-legal
search, zero root noise, deterministic root choice, and maximum turn 200.
This again totals 2,880 continuations.

Every continuation must end in one of the two natural endings above. Preserve
and report its actual reason. A deadlock receives the established adjudicated
WDL and score margin; it is never changed to a draw or replaced. Any other
ending is an integrity failure. Collection is atomic by state, resumable, and
hashed. All worker integrity failures must be aggregated rather than hiding
later worker errors.

## Matched value-only A/B and protected gate

Keep the exact 131 causal comparison. Both arms start from generation 11 and
train only the 4,705-parameter value head; 351,745 parameters remain exact.
The control target is world 0 / policy 0 WDL. The treatment target is the mean
of all eight state continuations. Both select independently on the same
48-state development repeated-return MSE, with epoch zero eligible, at most 50
epochs, batch 64, AdamW `1e-4`, weight decay `1e-4`, patience 8, minimum delta
`1e-4`, and seed `1131000`.

Seal both arms before reading protected targets. The 60-state / 960-
continuation protected gate, metrics, bootstrap, and thresholds are unchanged
from 131: nonzero treatment epoch; exact frozen policy; treatment MSE at most
90% of control and below generation 11; treatment MAE no worse than control;
both paired squared-error CI95 upper bounds below zero; and every phase within
control MSE plus 0.02.

## Conditional gameplay and decisions

Only the protected pass opens the same two sequential 200-pair panels: first
treatment versus single-outcome control, then treatment versus generation 11.
Search settings and strength gates remain unchanged. Gameplay accepts exactly
the two natural terminal reasons, and rejects every cutoff, stall, or unknown
ending. No 131 gameplay seed was used, so its frozen seed ranges remain clean.

The 131 decision ladder is retained. Integrity failure yields
`repeated_return_value_integrity_failed_no_strength_conclusion`; offline,
control-gameplay, parent-gameplay, and full-pass outcomes retain their exact
131 decision strings. No result installs or promotes a checkpoint, consults
`035`, changes the neural architecture, or starts a successor automatically.

The owner's 2026-08-30 approval authorizes the long 132 launch only after the
focused suite, full suite, frozen hash preflight, exact search-contract check,
and no-training launch guard all pass.
