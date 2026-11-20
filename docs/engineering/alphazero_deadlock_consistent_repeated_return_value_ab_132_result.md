# AlphaZero Deadlock-Consistent Repeated-Return Value A/B 132 Result

Date completed: 2026-08-30

Experiment 132 completed normally in 31,662.47 seconds (8h47m). It regenerated
all 300 selected states and 2,880 continuations from scratch under the repaired
shared terminal contract. Of the continuations, 2,876 ended through ordinary
`completed` and four through valid `forced_pass_deadlock`. Every collection,
possible-world, identity, policy-logit, checkpoint, and protected-continuation
integrity check passed.

The single-outcome control selected epoch zero. Its development repeated-mean
MSE was `0.25151644`, so the exact generation-11 value head remained the
control. The repeated-mean treatment selected epoch one with development MSE
`0.25068211`.

On the sealed 60-state protected panel:

| Model | MSE | MAE | Pearson |
|---|---:|---:|---:|
| generation 11 | 0.22924475 | 0.39636514 | 0.46308464 |
| single-outcome control | 0.22924475 | 0.39636514 | 0.46308464 |
| repeated-mean treatment | 0.22879765 | 0.39618136 | 0.46309347 |

The treatment improved protected MSE by only `0.00044710`, approximately
0.195%. It missed the required 10% causal improvement by a wide margin. The
treatment-minus-control squared-error bootstrap CI95 was
`[-0.00170056,+0.00081652]`; the treatment-minus-parent interval was
`[-0.00167044,+0.00079627]`. Both include zero. Phase safety, finite metrics,
nonzero treatment training, MAE safety, and the point estimate below parent
passed, but neither strict causal confidence gate passed.

Decision:
`repeated_return_value_target_not_causally_better_no_gameplay`.

No gameplay gate opened. Nothing was installed or promoted, and no automatic
continuation ran. The result confirms experiment 130's diagnosis that terminal
targets are noisy, but falsifies the proposed remedy that repeated-return means
alone materially improve the existing compact frozen-representation value
interface. Do not repeat the treatment with more continuations, another value
head schedule, or a post-hoc subgroup.

This closes the current compact AlphaZero target/representation repair recipe;
it does not establish that a materially larger fully trainable stochastic
AlphaZero programme cannot work.

Raw result SHA-256:
`347a8098494a882e72eff03b53d535734206efa0894cb31ccb2472ceae33ba45`.

