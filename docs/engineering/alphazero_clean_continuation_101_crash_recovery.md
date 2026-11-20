# AlphaZero clean continuation 101 crash recovery

Status: frozen before post-crash execution on 2026-08-14.

The host crashed during generation-5 diagnostics. This is an operational recovery of the already frozen 101 schedule, not a new experiment or a change to any model, replay, seed, panel, or decision rule.

## Audited recovery point

- main 101 progress: SHA-256 `4e70ff2d2c83b4db90cbf454c4243228693493cce44ada3ec61a2e6150013417`, status `running`, stage `diagnostics_generation_5`;
- exact generation-4 checkpoint: `0cd7782714ca0b2c081847af92dc731711ecd15b14c72bb9fa25efb039d35669`;
- exact generation-5 checkpoint: `201e34128c9588722ce78edc85e8672675b1075dc0696651484718ddd7fa2444`;
- sanitized generation-3 replay: `e39440b4eeeba28b9c33e0c0fa222c5d89afb81d9285530e1829748f206b06ce`;
- sanitized generation-4 replay: `7e15847d2cdf51ecd467a2bb6f38f13e1901bd7337971bbe8f4817ace1dec244`;
- generation-4 training record: `0c77bf5e0c89d1d13ba08e98e008288a40928a1459c1cc1f40303da2bc9eb7f1`;
- generation-5 training record: `a61c4acaa92d7f8c0f15e8a2d60eebd7fdd3be3ac25176601ecbf2f45045d2a3`;
- complete generation-5 raw diagnostic: `10476b0dc5743bc09a12317769729e78b19626f3f10c6ddbc1b60bbb42cae069`, 16/16 games;
- partial generation-5 searched-child-versus-searched-parent diagnostic: `b0811df44ad6ce22e2c95292238309596b10d83979bdc8418cf008fee5d3b482`, exact schedule prefix 3/16 games;
- generation-5 search-versus-own-raw diagnostic has not started.

Generation-4 self-play completed 256/256 with only normal terminations. The sanitized generation-4 replay is byte-identical to the raw replay because no replacement was required. Generation 5 trained successfully from exact generation 4. No stderr was recorded.

## Recovery rule

The recovery runner must verify every hash and structural claim above, including checkpoint generations/parents, training lineage, sanitation identities, and exact arena prefixes. It must not regenerate self-play, sanitize replay, or retrain any checkpoint.

It then invokes only the original frozen generation-5 diagnostics using the original paths, names, factories, seeds, paired-game counts, search configuration, and `resume=True`. The already complete raw panel is read unchanged. The searched panel resumes after its first three committed games. Any game that was in memory during the crash was not committed and is rerun deterministically from the same seed/seat. The final search-versus-own-raw panel then runs normally.

After all three panels complete, the runner reconstructs the generation-5 result from the hash-locked training/sanitation artifacts, appends it to the existing generation-4 record, computes the original 101 trajectory, and seals the normal 101 result. Exact generation 2 remains installed.
