# AlphaZero Head-Attribution Audit 096 — Result

## Outcome

Experiment 096 completed with the frozen attribution:

`no_clear_gameplay_head_attribution`.

Neither replacing the child's value with the parent value nor replacing its
policy with the parent policy produced a paired improvement over the full child
whose bootstrap CI95 lower bound exceeded zero. No hybrid independently
restored confirmed transfer over the full parent. Exact generation 2 remains
installed and no champion changed.

The point estimates are still directionally useful. The full child scored
58.33%, child-policy/parent-value 55.21%, and parent-policy/child-value 45.83%
against the full parent. Replacing the child policy reduced share by 12.5
points, while replacing the child value reduced it by only 3.125 points. Thus
the learned policy appears to carry most of the useful signal and the child
value is not an obvious primary blocker. This is an inference from noisy point
estimates, not the formal attribution result.

## Frozen provenance

- Protocol:
  `docs/engineering/alphazero_head_attribution_096.md`.
- Protocol SHA-256:
  `c3b78e215c254182c93117c76b4683ce4397f3743e0acbb4d68cad1eac46a638`.
- Result:
  `outputs/alphazero_head_attribution_096/result.json`.
- Result SHA-256:
  `11cc6d1fcc72a6744fa87646dd31ddac89a505c0d86a6e1396287a1dd48259e6`.
- Parent SHA-256:
  `60ca410b7d0d42c7beda66165df161cdc239a5e39a606d967e79adcf620534dc`.
- Child SHA-256:
  `36476fa8333a208c6d14559894cc491f9aa8f75a0a3c0c1b2972090d073090a4`.
- Replay SHA-256:
  `c58f9955362d1bb82b47ed34dd4bc0358cdf9a00116fca733b0171cb6d2cae47`.
- Controlling 095 result SHA-256:
  `6d2ec319982ced65855d526e1fe2c692c1f4bda49d8e3f60c473b693b4714655`.

All frozen inputs reproduced.

## Locked replay audit

Both exact networks were re-evaluated on all 4,106 complete positions from the
64-game 095 replay. Fresh parent probabilities reproduced the stored root
network priors to mean total variation `5.38e-8` and maximum total variation
`1.43e-7`, establishing exact replay/model alignment.

| metric | parent | child | child-parent |
| --- | ---: | ---: | ---: |
| policy cross-entropy | 2.4774 | 2.3630 | -0.1144 |
| visit-top agreement | 52.78% | 60.20% | +7.43 pp |
| value MAE | 1.0088 | 0.8823 | -0.1265 |
| value sign accuracy | 46.10% | 59.04% | +12.93 pp |

On 1,939 decisive positions (47.22% of replay) where the 256-simulation visit
argmax differed from the parent raw top action, the child selected the visit
argmax 23.62% of the time and assigned it mean probability 13.12%, compared
with 0% and 8.82% for the parent. The 0% parent top agreement is definitional
for this stratum; the probability movement is the more graded comparison.

The offline gains are therefore real and not an artifact of the earlier
training split. They are nevertheless incomplete: on more than three quarters
of decisive positions, the child's raw top still does not match search's visit
argmax.

## Matched hybrid gameplay

All variants played the full parent on the exact same 24 paired seeds
`1670000--1670023` under matched 256 / 2.25 search.

| variant | W-D-L | share | margin | seats | paired CI95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| full child | 28-0-20 | 58.33% | +0.19 | 66.67% / 50.00% | [43.75%, 70.83%] |
| child policy + parent value | 26-1-21 | 55.21% | +0.40 | 58.33% / 52.08% | [46.88%, 63.54%] |
| parent policy + child value | 22-0-26 | 45.83% | -0.17 | 58.33% / 33.33% | [31.25%, 60.42%] |

No panel's CI lower bound exceeded 50%, so no variant received the audit label
`restores_confirmed_transfer`.

Same-seed head-swap comparisons against the full child were:

| swap | mean difference | paired CI95 |
| --- | ---: | ---: |
| parent value replaces child value | -3.125 pp | [-18.75, +12.50] pp |
| parent policy replaces child policy | -12.50 pp | [-29.17, +4.17] pp |

Both intervals include zero, enforcing `no_clear_gameplay_head_attribution`.
The asymmetric point estimates mildly favor keeping the learned policy rather
than reverting it; they do not justify a head-specific causal claim.

## Integrity and operational record

- Every panel contains 48 unique exact seed-seat rows and 48/48 normal
  completions.
- All per-panel and paired-difference bootstraps independently reproduced.
- Exact parent remained installed and hash-verified.
- No self-play, training, checkpoint, `035` arena, or champion change.
- Complete pre-evidence suite passed 706/706 in 215.40 seconds.
- Focused 090--096 contracts passed 36/36 before evidence and 36/36 after the
  final evidence audit.

After the full-child panel completed, the first hybrid initially stopped
before playing a game because the hybrid inference adapter lacked the generic
`.checkpoint` identity used only for bot bookkeeping. The traceback was
preserved. A compatibility identity pointing to the operative policy
checkpoint was added and directly tested; the value identity remains separate.
The run resumed from the durable 48-game full-child artifact, which was not
replayed. The two hybrid panels then completed with zero resume stderr. Final
resumed runtime was 161.17 minutes; the initial phase before the adapter stop
was 64.82 minutes, for approximately 225.99 minutes total wall time.

## Interpretation

096 rules out the simplest high-confidence explanation that one obviously bad
learned head alone destroys an otherwise strong child. Both child heads improve
their replay targets offline. Reverting the value did not improve the child;
reverting the policy was directionally harmful. The remaining transfer failure
is more plausibly caused by incomplete imitation of decisive search changes,
distribution shift between self-play replay and evaluation play, shared
representation interactions, or noisy terminal value targets—not a plainly
reversed policy or value head.

This also reconciles the variable small panels. A modest child improvement
could produce 18-14 in 095 development and 28-20 here while still producing
50-1-49 in the controlling 095 confirmation. The available evidence does not
establish a durable searched gain.

## Recommended 097

Do not spend another large corpus or repeat the same training recipe. Use the
existing replay for a small offline objective/representation experiment aimed
at decisive search states:

1. freeze a validation split and identify states with high visit-to-parent
   policy divergence;
2. compare a small bounded set of policy objectives that emphasize those
   states (for example divergence weighting or a visit-argmax auxiliary loss)
   while preserving ordinary replay fit and value calibration;
3. require improvement specifically in decisive-state held-out visit-top
   agreement and probability mass, with no phase/value regression;
4. only one prospectively selected child may enter a modest searched
   development panel, followed by independent confirmation if warranted.

This targets the clearest remaining quantitative gap: despite genuine
learning, the child still misses search's preferred action on 76.38% of the
states where search overturns the parent raw policy. Any 097 implementation
must guard against merely overfitting a stratum defined from the same replay.
