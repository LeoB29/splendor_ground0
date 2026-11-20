# Progressive Public-Belief Planner Feasibility 133

Date frozen: 2026-08-30

## Purpose

Experiment 133 opens the superhuman information-set search programme recorded
after experiment 132. It trains no model and plays no strength game. It asks
whether a progressive, expanded-root public-belief search can produce
value-supported decisions rather than merely repeat the sharp `035` prior.

The design responds directly to established evidence:

- experiment 086 replicated positive value from roots outside the champion
  top four;
- experiments 055--059 found that early decision instability is dominated by
  narrow belief sampling;
- experiment 060 found that a four-root visit search agreed mainly through its
  shared prior while independent Q estimates remained unsafe.

Experiment 133 is a development feasibility test on previously opened 086
states and value tables. It cannot be a fresh strength confirmation. A pass
authorizes only a separately frozen fresh-state/full-game design.

## Frozen inputs

- experiment-086 result SHA-256:
  `8ae985f6747b5f3bbbcf4997357daf299ef0f46cc23534d55b6de91367910eb5`;
- experiment-086 selection SHA-256:
  `7bebc07fede835be023c986701fd813e4134374f1a4559fdcaf9439910d7e46f`;
- raw legal `035` checkpoint SHA-256:
  `4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd`;
- experiment-132 result SHA-256:
  `347a8098494a882e72eff03b53d535734206efa0894cb31ccb2472ceae33ba45`;
- strategic-roadmap SHA-256:
  `414f1749fb146c5882a7774aa6ac5d5223efe57b74303a80c42e79612a34df79`.

Require the exact 086 pass decision, the exact 132 no-gameplay decision, and
unchanged champion/checkpoint identities. Do not read any outcome from the
partial owner-waived 087 progress artifact.

## Outcome-blind state panel

Use only `selection.json`, never 086 value rows, for selection. In each of the
six `seat x early/mid/late` cells, retain the first six rows in ascending
`sample_index`, yielding exactly 36 unique states. Selection is independent of
086 payoffs, selected actions, gains, regrets, and endings.

After the panel is sealed, join the corresponding already-open 086 pooled
expanded-root values only for development evaluation. Root ordering and state
snapshots must reproduce exactly.

## Candidate search

Run two independent searches, `left` and `right`, on every selected state.
Both use:

- 12 observer-legal determinizations;
- the exact 086 expanded semantic root list: the champion four plus every
  stable non-deck public root retained by 086;
- the top four distinct `035` opponent responses at each opponent information
  set;
- 1,536 terminal-rollout simulations;
- score-margin payoff scaled by 20;
- `035` stochastic continuation policy with epsilon `0.1`, temperature `0.5`,
  and top-k `3`;
- maximum 400 continuation turns and the shared loop/deadlock adjudication;
- root PUCT constant `1.5`, opponent minimizing PUCT constant `1.5`;
- root prior temperature `2.0` and uniform mix `0.25`;
- opponent prior temperature `1.0` and uniform mix `0.05`.

Every retained root must first receive one continuation for every
determinization/opponent-response combination. Cover the champion four first,
then add each lower-ranked root in order. This is the fixed progressive root
opening schedule. It guarantees value evidence for every opened root before
adaptive allocation.

After complete forced coverage, spend all remaining simulations with root and
opponent PUCT. Cycle determinizations deterministically. The selected action is
the highest-visit root, breaking ties by Q, prior, then frozen root order.

Left and right use disjoint belief and continuation namespaces. Each repeats
from zero on the first state and must match byte-for-byte after timing fields
are removed.

## Evaluation metrics

Report overall and by phase:

- structural completion and exact branch accounting;
- expanded root counts and forced/adaptive simulation counts;
- left/right raw and 086-value-equivalence-collapsed visit JSD;
- exact selected-action agreement;
- symmetric cross-regret under the frozen 086 pooled value table;
- each search's regret from the 086 pooled best root;
- each search's pooled-value gain over the 086 champion-prefix discovery
  selection;
- prior-to-search total variation and JSD;
- fraction changing the prior-top and champion-prefix selected actions;
- fraction selecting outside the champion four;
- root-Q Pearson and Spearman between independent searches;
- wall and aggregate process CPU time.

The prior-control margin is:

`mean(prior-to-search JSD) - mean(independent-search JSD)`.

This must be positive by a material amount; agreement caused only by a shared
prior is not a pass.

Bootstrap the 36 state clusters 20,000 times with seed `1330133` for mean
candidate pooled-value gain, using the per-state mean of left and right gain.

## Integrity gates

Require all:

- every frozen input hash and prerequisite decision reproduces;
- exactly 36 selected states, six per seat/phase cell, with selection depending
  only on the frozen selection fields;
- exact 086 root/value alignment for every state;
- at least 90% of states complete in both arms overall and in every phase;
- completed searches use exactly 1,536 simulations and exact forced coverage;
- all continuations resolve, hard caps are zero, and loop adjudications are at
  most 0.1%;
- opponent semantic action sets and priors are stable within every shared
  information set;
- hidden-deck-order reversal preserves the public root/prior construction on
  the first state;
- first-state repeats are byte-identical excluding timing;
- all metrics are finite;
- no optimizer, model mutation, human path, partial-087 outcome path, gameplay,
  installation, or promotion path exists.

## Feasibility gates

Require all:

- left/right exact selected-action agreement at least 60%;
- mean raw visit JSD at most 0.15 and early mean at most 0.18;
- mean equivalence-collapsed JSD at most 0.10 and early mean at most 0.12;
- mean symmetric cross-regret at most 0.025 and p90 at most 0.075;
- each arm has at least 65% 086-value-equivalent agreement within regret 0.025;
- each arm mean regret from the 086 pooled best is at most 0.035;
- mean independent root-Q Spearman at least 0.50;
- mean prior-to-search total variation at least 0.15;
- at least 15% of states change the prior-top action;
- prior-control JSD margin at least 0.02;
- mean pooled-value gain over the champion-prefix decision at least `+0.025`;
- bootstrap CI95 lower bound for mean gain strictly above zero;
- mean gain positive in early, middle, and late phases;
- at least 15% of states select outside the champion four and at least one in
  every phase.

## Decisions

- Integrity failure:
  `progressive_public_belief_planner_133_invalid_no_search_decision`.
- Valid run passing every feasibility gate:
  `progressive_public_belief_planner_133_pass_freeze_fresh_confirmation_design`.
- Valid run failing any feasibility gate:
  `progressive_public_belief_planner_133_failed_review_search_mechanism`.

A pass authorizes design of a separately frozen fresh-state confirmation and
then, only on its pass, a full-game comparison. It does not directly authorize
gameplay, training, corpus generation, installation, promotion, GPU rental, or
another search candidate. A failure requires a programme-level mechanism
review; do not tune these parameters on the opened 086 states.

## Launch guard

Default invocation verifies every input, seals the 36-state panel, reproduces
root/value alignment, and runs a temporary reduced first-state smoke that is
not persisted as evidence. It trains nothing and plays no strength game. The
full state panel requires explicit `--launch`. Completion never launches a
successor automatically.

