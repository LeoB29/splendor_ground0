# Live qualification of the final-token network optimization

31 October 2026. **Complete and independently accepted.**

The fixed faster256 implementation passed six fresh operational games against
unchanged256, followed by exact fresh-process reproduction of38 selected
decisions. No hard timeout, invalid move, operational failure or verification
failure occurred. Every game ended naturally. All recorded processes exited
successfully and all28 exact process identities are absent.

## What ran

Three new initial setups, each played with swapped seats. Both players retained
the same checkpoint, float32 precision,256 simulations, three worlds, cpuct0.4,
depth64, CPU placement and corrected rules. Only the candidate's previously
qualified final-token feed-forward implementation differed.

The internal9-second and complete-response10-second research guards were
unchanged. One second and700ms were descriptive thresholds, not forfeits or
admission gates. No decisions were retried, no games replaced and no budgets
extended. The original2,000-ply safety guard was never reached.

The run completed within approximately25minutes, including observed owner exit,
against its four-hour overall and three-hour model limits. Recorded stages were
199.21s qualification,880.66s live play,249.77s replay and71.65s first finalization.
The subsequent independently owned reader and external acceptance also passed.
The external saved-only audit took94.56s within its15-minute bound and the
original overall clock. Frozen evidence occupies approximately186.2MiB.

## Coverage and result

- Six natural games,356 completed live decisions and zero failures.
- Four fully qualified worker lifecycles: two live and two new replay workers.
  Across them:44 graph cases,152 repeated numerical fixtures and268 decisions
  in the actual corrected-rules qualification suites.
- All356 moves checked against the independent transition ledger; all six
  complete games independently checked through terminal state and winner.
- The outcome- and timing-blind rule selected38 replays:36 covering every
  game×variant×absolute-ply phase cell, plus two private-reserve representatives.
  No primary cell or observed private-reserve category was unrepresented.
- All38 replays matched the full original scientific output, action, trace and
  afterstate exactly within the same player variant.
-406 parent requests reconciled:356 live,38 replay and12 startup requests.
- All3,800 frozen pins,3,008 sealed evidence files and155 files of the earlier
  failed practical attempt were preserved and verified.

The descriptive score was3–3. Three paired setups cannot establish equal
strength or detect a modest strength change.

## Thinking time

| Completed live decisions | Optimized256 | Unchanged256 |
|---|---:|---:|
| Count |178|178|
| Mean |1.654s|1.779s|
| Median |1.703s|1.814s|
|95th percentile |2.384s|2.499s|
| Maximum |2.581s|2.776s|
| Over one second |163/178 (91.6%)|165/178 (92.7%)|
| Over700ms |166/178|166/178|

These live averages describe the games played. They do not isolate the
implementation's speed gain because the positions and execution order can
differ. The preceding controlled saved-position study remains the appropriate
estimate:5.73% median paired improvement, with all its declared speed gates
passed. This study does not replace that estimate.

## What this result establishes

The optimization now works as an integrated playing agent, not just on a fixed
panel. It completed lawful full games, retained the expected search settings,
and reproduced its selected decisions after restarting the player processes.
It is qualified for further research under the9/10-second guard.

This remains a non-bit-exact neural implementation admitted under the previously
fixed numerical tolerances. Exact same-variant replays do not prove universal
cross-variant decision equivalence. The earlier strict exact-output rejection
remains valid and preserved.

The implementation gain is useful but insufficient for one-second deployment
at256 simulations. Most moves exceeded that target even though no research
deadline failed. Passing this study does not promote a new strength reference.

## The next step

Recommend one preselected192-simulation version of this qualified implementation
against unchanged128, with fresh paired games after entry-point correctness
qualification. Design and freeze its sample size and resource limits separately.
Measure strength and the full thinking-time distribution in the same study;
retain one-second and700ms timing as descriptive.

This is a test of the combined192-simulation candidate against the reference,
not an isolated causal estimate of simulation count.192 may still exceed one
second. The useful question is how much of256's verified strength gain it
retains at lower computational cost. Another timing-only selection campaign
would not answer that question. No follow-up study has been launched.

## The broader picture

The project aims at exceptional Splendor play with lawful information and
practical response times. We adopted a strong pretrained policy/value search
player, established128 as the reference, and verified that256 is a stronger
research challenger. The recent work identified neural execution cost and
found a modest improvement without changing weights, precision or search.

This live qualification connects that implementation work back to complete
play. The next study should map the strength/time trade-off between128 and256,
instead of assuming either extra search or faster execution alone solves both
objectives.128 remains the reference;256 remains the verified stronger research
challenger. External testing, six worlds, training and larger-data work remain
deferred. Scheduled monitors remain cancelled.

## Evidence

- Protocol: `docs/engineering/splendorust_network_live_20261031.md`.
- Source: `tools/splendorust_network_live_20261031`.
- Output: `outputs/splendorust_network_live_20261031_01`.
- Freeze SHA256:
  `ba428d48f0e2ae70774770da5279607faffba0827e02d78d69fa0cb79187d5b1`.
- External acceptance:
  `outputs/splendorust_network_live_20261031_01_closeout/independent_acceptance.json`,
  SHA256 `a01a03648ba8ede8691aaba82e5c0640d4e0a690e2e66542c09479876e6b80a4`.
- Retained owner session14952: launch chunk976cf8, observed exit0 chunk0b0fe8.
- Prior controlled timing report:
  `docs/engineering/splendorust_network_practical_v2_20261031_results.md`.
