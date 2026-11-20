# Progressive Public-Belief Planner Feasibility 133 Result

Date completed: 2026-08-30

Experiment 133 completed normally on its frozen 36-state development panel,
balanced across both seats and early, middle, and late play. Both independent
searches completed for every state. The run evaluated 110,592 terminal
continuations, with no hard caps, loop adjudications, or unresolved
continuations. Every provenance, selection, simulation-accounting,
information-set stability, hidden-deck privacy, and exact-repeat integrity
gate passed.

The planner was not merely copying the learned `035` prior. Its root visit
distribution moved an average total-variation distance of `0.30367` from that
prior, changed the prior's top action on `80.56%` of states, and selected an
expanded (non-champion) root in at least one repeat on `52.78%` of states.
The prior-control margin was `0.05739`, above the frozen `0.02` gate.

However, the resulting decisions were not strong enough:

| Metric | Result | Frozen requirement |
|---|---:|---:|
| independent-search exact agreement | 41.67% | at least 60% |
| mean oracle regret | 0.09244 | at most 0.035 |
| left/right oracle-equivalent fraction | 36.11% / 27.78% | both at least 65% |
| mean Q-rank correlation | 0.49379 | at least 0.50 |
| mean gain over the four-root champion | -0.02087 | at least +0.025 |
| gain bootstrap CI95 | [-0.05515, +0.00887] | lower bound above zero |

Mean gain was `-0.04734` early, `-0.02668` in the middle, and `+0.01140`
late. Thus the planner was harmful where long-horizon credit assignment is
hardest, and only slightly positive late. Its two searches were reasonably
close in distribution (raw JSD `0.04632`, collapsed JSD `0.04077`) and had
low cross-regret (`0.01269` mean), so this is not best explained as a simple
reproducibility failure. The exact first-state repeats also matched.

Decision:
`progressive_public_belief_planner_133_failed_review_search_mechanism`.

No training, gameplay, installation, promotion, or automatic successor ran.
The experiment falsifies this particular shallow terminal-rollout PUCT
mechanism as the first deployable public-belief planner. It does not falsify
information-set search as the programme direction. It also does not establish
that the terminal continuation policy is the primary fault: experiment `086`
used the same continuation policy successfully when it evaluated complete
root-by-belief-by-opponent blocks and then applied the exact shared
information-set minimum.

A descriptive post-hoc decomposition strengthens that interpretation. The
average forced-coverage share was `64.32%`, but replacing visit argmax with the
final Q argmax would not have rescued the result: Q argmax had mean oracle
regret `0.09354` and mean champion-relative gain `-0.02197`, essentially the
same as the selected action's `0.09244` and `-0.02087`. The failure is therefore
inside the online allocation/value aggregation, not merely the final visit
tie-break. Do not tune PUCT constants, buy more simulations for this mechanism,
or jump directly to another neural value-head treatment.

The next scientific step is the programme-level review recorded in
`docs/strategy/information_set_search_mechanism_review_after_133.md`. Its first
recommendation is to resolve the still-open full-game status of the exact
expanded-root `086` candidate with a fresh prospective comparison against the
`084` research champion. If that succeeds, preserve its grouped resolver
semantics while making it cheaper through structured batch allocation and
confidence-aware elimination. Learned leaf values come only after that planner
frontier is established.

Raw result SHA-256:
`7f74736b4557a1b01b3e6febbf6a4f4cec2fa7643a78655c0f4adc529dde872c`.
