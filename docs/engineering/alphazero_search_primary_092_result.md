# AlphaZero Search-Primary Confirmation `092`: Result

Date: 2026-08-10  
Status: complete engineering result; searched candidate not installed

## Decision

`092` completed with the frozen decision:

`searched_candidate_not_confirmed_keep_generation_2`.

The result is close and encouraging rather than negative. The selected
`value_quarter` candidate beat generation 2 under equal-dose search by 21-11
(65.625%) on 32 fresh games, with positive margin and strength from both seats.
It also won the non-veto raw diagnostic 18-14. However, the 16-paired-seed
bootstrap CI95 lower bound was 46.875%, below the frozen requirement `>50%`.

The precision gate therefore stands. Generation 2 remains installed; no search
configuration discovery, self-play continuation, generation-4 training, or
`035` arena opened. No champion changed.

## Frozen identity

- Protocol: `docs/engineering/alphazero_search_primary_092.md`.
- Protocol SHA-256:
  `f44f01535d52988819f77347b71079e868b8349b4be54877459f8f5838bbfae4`.
- Result: `outputs/alphazero_search_primary_092/result.json`.
- Result SHA-256:
  `b3a5c2b85174bf73c119b4b10b5b821b6fb0391173c04b920fa523460061c3e1`.
- Runtime: 27.15 minutes.
- Runner stderr: zero bytes.
- All frozen `091` result, incumbent, candidate, and replay hashes reproduced.

## Candidate discovery

Both frozen `091` candidates played the incumbent on the same eight new paired
seeds under 64-simulation search.

| Candidate | W-L | Score share | Mean margin | Eligible |
|---|---:|---:|---:|---:|
| `balanced_half` | 9-7 | 56.25% | +0.7500 | yes |
| `value_quarter` | 11-5 | 68.75% | +2.6875 | yes |

Every game completed normally. The prospective searched-performance selector
chose `value_quarter`; no differently weighted objective levels were compared.
Discovery was used only for selection and was not pooled with confirmation.

## Independent searched confirmation

The selected candidate then played 16 fresh paired seeds / 32 games.

| Metric | Result | Frozen gate |
|---|---:|---:|
| W-L | 21-11 | diagnostic |
| Score share | 65.625% | `>=56.25%`: pass |
| Mean margin | +1.6250 | `>0`: pass |
| Candidate from seat 0 | 68.75% | `>=50%`: pass |
| Candidate from seat 1 | 62.50% | `>=50%`: pass |
| Normal terminations | 32/32 | pass |
| Paired-bootstrap CI95 | `[46.875%, 84.375%]` | lower `>50%`: **fail** |

The 20,000-resample bootstrap reproduced the arena mean exactly and used the
paired seed as its cluster. Pair outcomes were eight candidate sweeps, five
1-1 splits, and three incumbent sweeps. With only 16 independent clusters and
substantial between-seed variation, the point estimate can be 65.6% while the
clustered lower bound remains below 50%.

This is precisely the distinction `092` was intended to make: the candidate
looks likely better, but the fixed panel is not precise enough for the stated
confidence standard. The panel was not extended and discovery was not pooled.

## Raw diagnostic

On the same 16 paired seeds without search, the selected child scored:

- 18-14;
- 56.25% score share;
- mean margin +0.78125;
- 32/32 normal completions.

This reverses `091`'s 6-1-9 raw result and indicates that the earlier raw veto
was likely dominated by small-panel variation. Raw play was correctly
non-decisive in `092`; the deployed-style searched comparison remained primary.

## Dose and integrity

- Candidate discovery: 32 searched games total.
- Selected searched confirmation: 32 games.
- Selected raw diagnostic: 32 games.
- Total: 96/96 normal games.
- Four arena progress files are complete and contain the exact frozen schedule
  identities and prefixes.
- Installed checkpoint SHA-256 reproduces exact generation 2:
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`.
- Search discovery count: zero.
- Conditional self-play/training continuation: absent, as required.
- External `035` arena: absent, as required.

Pre-evidence verification passed 691/691 repository tests in 343.83 seconds;
focused `090--092` contracts passed 21/21. A final focused rerun exposed an
intermittent Windows permission race when atomically replacing the progress
file (20/21). The completed experiment itself had no such error and all files
were intact. A bounded retry was added after the result, then focused tests
passed 21/21 and the arena-specific suite passed 4/4 on an immediate repeat.
The final full suite after that hardening passed 691/691 in 311.80 seconds.

## Interpretation

The honest best estimate is now that `value_quarter` is probably a stronger
searched agent than generation 2. It has produced searched point estimates of
11-5 in `091`, 11-5 in `092` discovery, and 21-11 in independent `092`
confirmation. Only the last panel is confirmation evidence; the earlier panels
must not be pooled into its gate. The independent result is strong enough to
justify a larger exact replication, but not to waive the failed confidence
boundary.

The architecture diagnosis also improved:

- aggregate policy/value stabilization passed offline in `091`;
- search performance is consistently promising;
- fresh raw performance is no longer clearly worse;
- the immediate blocker is confirmation precision, not another architecture
  rewrite or a return to `035` optimization.

## Recommended next step: `093`

Freeze `value_quarter` in advance—no candidate discovery—and run a larger
searched confirmation against exact generation 2:

1. 50 fresh paired seeds / 100 games under the same 64-simulation search;
2. paired-seed bootstrap with fixed seed and at least 20,000 resamples;
3. require score share `>=55%`, CI95 lower `>50%`, positive margin, both seat
   shares `>=50%`, and all normal terminations;
4. keep raw play as a same-seed diagnostic, not a veto;
5. if and only if it passes, install generation 3 as the engineering incumbent
   and resume the already-designed search-improvement/self-play continuation.

At the observed runtime, 100 searched games should be a material but manageable
confirmation dose. It directly addresses the only failed `092` gate without
changing the candidate, architecture, comparator, or evaluation search.
