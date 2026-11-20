# AlphaZero Value-Uncertainty Audit 130 Result

Date completed: 2026-08-29

Experiment 130 completed normally in 1h13m. All 24 states, 96 possible-world
constructions, 192 public root searches, and 384 crossed continuations passed
their integrity checks. Every continuation ended normally. The audit trained
no model and played no strength arena; nothing was installed or promoted.

## Outcome uncertainty

The terminal-value target was materially under-observed by one trajectory:

- mean within-state WDL variance was 0.71468, with bootstrap CI95
  0.57210--0.83968;
- 20 of 24 states, or 83.33%, produced both wins and losses;
- 73.34% of individual terminal-target sum of squares lay within the same
  observable state, with bootstrap CI95 59.21%--87.07%;
- mean within-state WDL entropy was 0.7881 bits and mean score-margin standard
  deviation was 4.779 points.

Every frozen material-noise check passed. The phase breakdown was descriptive:
mean WDL variance was 0.80664 early, 0.81787 middle, and 0.51953 late. These
subgroups were not gates.

## Source of the uncertainty

The balanced two-way decomposition attributed 33.41% of within-state outcome
sum of squares to the possible-world main effect, 15.92% to the coupled
search-policy-seed main effect, and 50.67% to their residual interaction.

Possible-world variation exceeded policy-seed variation by more than the
frozen 0.10 margin, but it missed the required 0.40 absolute share. The audit
therefore does not authorize belief-aware features or a purely
possible-world-distributional head. Most variation arose from how the possible
world and the subsequent policy trajectory combined.

Repeated root estimates were numerically stable: mean search-value standard
deviation was 0.00210 at 256 simulations and 0.00195 at 1,024; modal semantic
action agreement was 83.33% and 88.54%. However, deeper search values were not
closer to the empirical 16-continuation means. MSE was 0.21632 for raw
generation 11, 0.21271 for the experiment-114 terminal control, 0.22591 for
mean 256-search value, and 0.22712 for mean 1,024-search value. This gives no
reason to revive direct search-value distillation.

Decision:
`trajectory_policy_value_uncertainty_material_review_joint_target`.

The next justified step is a paired causal test of target averaging: hold the
policy and shared representation exact, train otherwise matched value readouts
from either one fixed continuation outcome or the repeated-continuation mean,
and judge both against protected repeated-return means before any gameplay.

Raw result SHA-256:
`cc884522a98a6b45105d28a8d878bd31267ad6504f444b14cb9d1dfd0df4d417`.
