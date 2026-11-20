# AlphaZero Exact-Candidate Large Confirmation `093`: Result

Date: 2026-08-10  
Status: complete; candidate rejected, no continuation opened

## Decision

`093` completed with the frozen decision:

`large_confirmation_failed_keep_generation_2`.

On 50 fresh paired seeds / 100 searched games, exact `value_quarter` scored
48 wins, 2 draws, and 50 losses against exact generation 2. Its score share was
49.0%, mean margin was -0.33, and paired-bootstrap CI95 was `[39.5%,58.5%]`.
The candidate was non-losing from seat 0 (52%) but below half from seat 1 (46%).

This large fixed panel overturns the optimistic small-panel interpretation from
`091` and `092`. `value_quarter` is not a demonstrated searched improvement.
Generation 2 remains installed; no search-configuration screen, self-play,
generation-4 training, or `035` arena opened. No champion changed.

## Frozen identity

- Protocol: `docs/engineering/alphazero_large_confirmation_093.md`.
- Protocol SHA-256:
  `ba46cafde3128ab5585bc7cc5e74a8d2b15c0bd79e9a7ac2c492c230db103817`.
- Result: `outputs/alphazero_large_confirmation_093/result.json`.
- Result SHA-256:
  `3fcba198fbcdcf48b4909c84eae6e8402dfb8fee941231791efa98fe0423f54d`.
- Runtime: 59.48 minutes under concurrent `087` load.
- Runner stderr: zero bytes.
- All frozen `092` result, candidate, incumbent, and replay hashes reproduced.

## Searched confirmation

| Metric | Result | Frozen gate |
|---|---:|---:|
| W-D-L | 48-2-50 | diagnostic |
| Score share | 49.0% | `>=55%`: fail |
| Mean margin | -0.33 | `>0`: fail |
| Candidate from seat 0 | 52.0% | `>=50%`: pass |
| Candidate from seat 1 | 46.0% | `>=50%`: fail |
| Normal terminations | 100/100 | pass |
| Paired-bootstrap CI95 | `[39.5%,58.5%]` | lower `>50%`: fail |

The bootstrap used 50,000 deterministic paired-seed resamples and reproduced
the 49.0% mean exactly. Pair-level points comprised ten candidate sweeps, two
1.5-point pairs, 26 one-point pairs, and 12 incumbent sweeps. The larger dose
did what it was designed to do: substantially narrowed the uncertainty and
showed that the earlier positive point estimates were not stable.

Prior searched results must not be pooled into the formal gate. Informally,
their pattern illustrates the sampling problem:

- `091` development arena: 11-5;
- `092` selected discovery: 11-5;
- `092` independent confirmation: 21-11;
- `093` exact large confirmation: 48-2-50.

The first three small panels motivated replication; the predeclared 100-game
replication supplies the controlling conclusion.

## Raw diagnostic

On the same 50 paired seeds without search, the candidate scored:

- 52-48;
- 52.0% score share;
- mean margin +0.60;
- 100/100 normal terminations.

This is consistent with approximate raw-policy parity, not a material learned
improvement. It also confirms that the `091` raw 6-1-9 loss and `092` raw 18-14
win were both too small to characterize the candidate reliably.

## Integrity and verification

- Searched confirmation: exact 100/100 unique schedule-prefix games.
- Raw diagnostic: exact 100/100 unique schedule-prefix games.
- Total dose: 200/200 normal games.
- Both durable arena progress files complete.
- Installed checkpoint is unchanged exact generation 2, verified SHA-256:
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`.
- Search discovery count: zero.
- Conditional continuation: absent.
- External `035` arena: absent.
- Focused `090--093` contracts: 24/24 passed before and after evidence.
- Complete pre-evidence repository suite: 694/694 passed in 357.21 seconds.
- No code was changed after the evidence run.

## Programme interpretation

`091` successfully repaired aggregate held-out value/policy loss, but that did
not translate into reliable searched gameplay improvement. More evaluation of
this exact child is not warranted. The rollback/confirmation architecture has
done its job and prevented a noisy 11-5 or 21-11 panel from installing a model
that is approximately tied with, or slightly worse than, its parent under the
deployed search.

This does not invalidate the AlphaZero direction. It identifies the concrete
missing precondition: current PUCT visits are not yet a reliably stronger
policy-improvement operator for the accepted generation-2 network. Training a
child to imitate those visits cannot be expected to create stable strength if
search itself provides little or inconsistent gameplay advantage.

## Recommended next step: `094`

Stop evaluating `value_quarter`. Audit and calibrate the improvement operator
on accepted generation 2 before training another child:

1. Measure generation-2 value calibration/error by phase and outcome on locked
   replay, including late-game and older-generation slices already implicated
   by `091`.
2. Compare fixed PUCT doses such as 64, 128, and 256 simulations (with a small
   bounded `c_puct` grid) against generation-2 raw play on fresh paired seeds.
3. Select on discovery and independently confirm only one search configuration
   with a meaningful searched-over-raw margin and no seat collapse.
4. If no configuration passes, do not generate another visit corpus. Improve
   value learning/data/model calibration first while retaining terminal WDL as
   the primary AlphaZero target.
5. If a higher-dose search passes, use that confirmed improvement operator to
   generate the next self-play corpus, then reapply the acceptance/rollback
   machinery built in `091--093`.

This redirects compute toward the causal bottleneck—whether search actually
improves the current network—rather than another noisy child-vs-parent panel.
