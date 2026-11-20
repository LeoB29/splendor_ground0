# Recommendation after160: can the stronger teacher improve the fast player?

2026-09-16. Recommendation and saved-data review only. The owner asked which
step should follow160. No successor training, inference, simulation, gameplay,
promotion or monitoring has been launched. Preserve all intentionally dirty work
and frozen predecessors. Confidence in the recommendation is moderate.

## One recommended experiment

Recommend a controlled incremental distillation experiment, provisionally161:

**Train matched versions of the current157 fast controller on128-world versus
32-world decisions from exactly the same160 positions. Test whether the128-trained
controller beats both its matched32-trained counterpart and the unchanged157
controller in fresh games.**

This is a single teacher-budget experiment with an attribution comparison and a
required practical-progress comparison. It uses the expensive decisions already
collected. It does not require another planner match or new teacher rollouts.

## Why this has the highest expected value now

160 completed400 games and every prescribed verification check. Exact155 with128
worlds scored65.5% against unchanged32-world155, paired95% interval60.875-70.125%,
261/2/137 wins/draws/losses. The improvement appeared from both seats. Only world
count changed; mean root wall time rose from19.482s to78.101s, about4.01x.
This establishes useful sampling-budget scaling in this matchup, not equal-time
superiority, an optimal128 budget, ground-truth labels or superhuman strength.

157 already established that some search knowledge can transfer into fast play:
57.5833% against035, versus49.8542% for the matched035-label control. Its three
fixed controllers took about11ms per decision. However, the large teacher/student
gap remains.158 did not establish that a stronger fast controller improves the
planner when substituted into its simulated continuations; its owner-stopped
audit also leaves that particular result provisional.159 identified conditional
policy-dependent differences without a practical quality oracle.

The new question is therefore specific and useful: **does the demonstrably
stronger teacher produce a better fast controller under the successful157
learning recipe?** A positive result advances fast playing strength and the
search-to-learning part of an improving system. It does not by itself close the
learning-to-better-search loop or license another continuation substitution.

This separation of planning and generalization is also the motivation of
[Expert Iteration](https://proceedings.neurips.cc/paper_files/paper/2017/hash/d8e1344e27a5b08cdfd5d027d9b8d6de-Abstract.html).
That paper demonstrates an iterative search/learning system in Hex; it does not
guarantee transfer for our Splendor targets, limited proposal support or model.
The recommendation follows our measured results and compute costs.

## What the existing data provide

All11,985 decisions made by the128-world player are already saved and audited,
from400 games/200 seed pairs. Each contains outcomes for every offered action in
all128 shared hidden worlds. We can recover exact32-prefix decisions from those
same positions and proposals. Using the separately visited32-player roots as the
control would confound teacher budget with state distribution and is unnecessary.

A read-only full scan verified all400 game hashes and all11,985128-root hashes.
The32-prefix and128 labels differ on4,081 roots (34.05%). This measures the size
of the training intervention, not128-label accuracy. All phases, actions, wins
and losses stay in the training pool; no confidence, margin or phase filter.
The full scan is recorded in `scripts/analysis/review_label_counts_after_160.py`
and `outputs/world_budget_160/review_label_counts_20260916.json`.

A separate inexpensive saved-data diagnostic selected the first candidate128
decision in each of three phases from one hash-assigned seat game per source
pair:600 positions/200 independent source pairs. Across64 disjoint64/64 world
partitions, choosing with8/16/32/64 worlds gave held-out conditional score gains
over proposal0 of0.933/1.437/2.052/2.660pp. The32-to64 increment was+0.608pp,
exploratory pair-bootstrap95% interval[0.509,0.710].64-world halves still chose
different moves51.73% of the time. The earliest-phase sample has39%32/128 label
changes; this is distinct from the34.05% all-root rate.

These diagnostics support useful sampling effects without making128 labels
noise-free. They evaluate conditional greedy035 futures, not full-game win
probabilities. The same128 outcomes cannot independently validate the128-selected
maximum. Repeated partitions do not create new independent games or labels.

Artifacts: `scripts/analysis/review_sampling_after_160.py` and
`outputs/world_budget_160/review_sampling_20260916/{summary,roots}.json`.
This diagnostic used no game/inference imports, new trajectories or training.

## Proposed controlled design

### Matched incremental training

- Use all11,985128-player roots. Treatment label is the recorded128-world first
  maximum; control label is the exact first32-world first maximum. Keep the
  same original representatives and canonical proposal groups, including
  supports smaller than four. Never treat unoffered legal actions as inferior.
- Split the200 source seed pairs80/20 by a frozen identity-only rule before
  fitting. Both seat games and every associated turn remain together. These
  are reused development data; only the later fresh games are confirmatory.
- Use all three frozen157 treatment checkpoints as paired initializers. For
  each initializer, produce one128-label and one32-label update: six fits.
  This retains the previous fast learning while testing a new fixed dose.
  Do not select157002 because it previously had the highest observed score.
- Within each matched pair, keep rows, shuffle stream, architecture, legal
  inputs, optimizer settings, four-pass dose and checkpoint-selection procedure
  identical. Start fresh AdamW state explicitly; the old optimizer state was
  not saved. Preserve the initial checkpoint as an eligible candidate.
- Retain157's offered-group classification loss and CPU implementation. Keep
  frozen035 as proposer at deployment. No value training, unrestricted learned
  argmax, new representation, replay-mixture sweep or GPU-backend change.

Approximately9,600 training roots remain after splitting, roughly half157's
original19,499 training rows. Better teacher play therefore does not guarantee
better fitting or transfer. Warm-starting157 retains earlier knowledge but does
not guarantee avoidance of forgetting. The anchor comparison measures that risk.

### Two prespecified gameplay comparisons, one experiment

For each of the three fixed initializer/model pairs:

1. Updated128-label controller versus its matched updated32-label controller.
2. Updated128-label controller versus its own unchanged157 initializer.

Propose400 fresh initial-seed pairs, shared across both comparisons and all three
fit pairs, with both seats:800 games per comparison/fit,4,800 total fast games.
Freeze every checkpoint and schedule before the arena. Use fresh lawful proposal
streams with coupling at matching observable states; never hidden-state seeds.
Keep these fast controllers separate from the expensive128-world teacher.

First average the two seats for each fit/comparison/seed, then the three fixed
fits. Bootstrap the400 complete initial-seed clusters, retaining all fits,
seats and comparisons jointly, for50,000 resamples. Intervals condition on these
three fixed training initializers/order streams, not arbitrary future training.

Proposed clear-success requirements:

- **Teacher-budget transfer:**128-trained versus32-trained paired95% lower
  score bound strictly above50%. No additional five-point point-estimate gate.
- **Useful fast-player progress:**128-trained versus unchanged157 score at
  least55%, with paired95% lower bound strictly above50%.
- Every prescribed source, fitting, legal-information and gameplay audit passes;
  the deployed controller retains the same fast inference structure and has no
  expensive search call hidden in its action selection.

Both comparisons are required. Beating a degraded32 control alone is not fast
progress; beating157 alone does not attribute the gain to128-world labels.
Report per-fit, seat, latency and margin results descriptively without selecting
a favorable subgroup. Validation loss or teacher agreement is not a strength gate.
Keep explicit inconclusive outcomes and no result-dependent sample extension.
Exact stream names, timing/integrity preflight and audit manifest remain to be
sealed during implementation if the owner authorizes this proposal.

### Verification and cost

Bind every source root to160's completed game/audit chain; independently recover
the32-prefix labels and both export tensors from lawful observations. Preserve
whole-pair splits and same-row matched controls. Independently reproduce the
fixed CPU fits/checkpoint selection and every fresh actual game; do not reopen
160's already completed sampled-inner audit as a new full rollout-replay task.

157's individual fits took486-488s on its larger corpus; its4,800 fast games took
358.5s and their full replay450.0s. Our two fast players per game and retained157
initialization differ from that run, so these are cost anchors, not guarantees.
Expect a few hours including extraction, checks, training and verification,
roughly1-3hours of run time subject to preparation/timing evidence. Preparation
engineering time is additional. Training and games stay on CPU for this test.

Use the tested windowless worker setup and a launcher that stays alive through
the startup acknowledgement; the160 short-launcher parent-diagnostic race must
not recur. Preserve its failed startup record and all frozen code. Monitoring
stays off. No automatic training launch, promotion, successor or restart follows
from this recommendation.

## Why the strongest alternatives wait

- **Another uniform increase:**128-to512 is plausible headroom, but no plateau
  or useful slope beyond128 has been established. A400-game512-versus128 match
  would cost roughly four times160's collection if game lengths stayed similar,
  approaching a week. It adds compute scale without testing retention of what
  we have already learned.
- **Adaptive allocation:**now well motivated, and likely the best subsequent
  search-engineering branch. Standard
  [best-arm identification](https://proceedings.mlr.press/v28/karnin13.html)
  provides algorithms for allocating a fixed sampling budget. Their assumptions
  do not by themselves establish stronger play for our correlated action
  estimates and policy-dependent futures. Prefer a future fixed-budget direct
  superiority test to claiming preserved128 strength from an inconclusive match.
  Using160's pair SD0.333 only as a rough planning proxy, a2.5pp noninferiority
  margin at80% power/one-sided5% would require about2,200 games; near-identical
  policies might have lower variance, but400 games cannot simply be assumed
  adequate. Even a cheaper adaptive-versus32 arena costs far more than fast games.
- **Broader proposals:**important unresolved headroom, but it changes support
  and estimation precision together unless carefully controlled. It needs a new
  support design;157's existing choices already lie in the old offered support.
- **More opponent validation:**useful later to assess generalization, without
  assuming transitivity from160. It is less informative now about whether
  stronger planning can improve the scalable fast player.

Independent reviewers agreed on matched stronger-teacher distillation as the
highest-value next paid direction. Common-anchor performance comparisons would
also be valid; the two direct matchups above make the requested transfer and
practical-progress claims particularly clear. A failure narrows this dataset,
warm-start recipe and training dose; it would not reject distillation generally.
