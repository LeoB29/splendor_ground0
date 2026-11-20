# Next after the verified second proposal-learning generation

2026-09-27. Recommendation only, following the owner's request for a considered
next step. AGENT_CURRENT.md was read first. Three independent read-only reviews
challenged the learning explanation, saved-root feasibility and competing research
directions. No new experiment, inference, simulation, fit, test, implementation,
promotion or monitor was started. This new unbound note supersedes the contingent
recommendation in next_during_proposal_iteration_128.md; frozen evidence is unchanged.

## Recommendation

Run **one bounded same-position diagnosis of fourth-proposal quality**. Include a
cheap descriptive analysis of existing matched roots within that study, then a
representative comparison using independent selection and assessment worlds.
Do not make the opening-only analysis a separate success gate or a reason to add
another approval/diagnostic cycle. The immediate question is where this update's
improved imitation fails to become useful extra choices for the planner.

Confidence is moderate that this is the highest-value next use of compute. It
addresses the demonstrated learning route and can discriminate concrete next
interventions for hours of simulation rather than another complete strength arena.
It is a mechanism study, not a claim of a new stronger player or a definitive
causal explanation for the completed arena.

## What the result changes

The completed study scored206W/2D/194L in402 games/201 pairs:51.4925%, paired95%
interval47.2637--55.7214%. All prescribed audits passed. Both Stage A criteria
failed and Stage B correctly did not run. The interval includes parity and55%:
there is no established improvement, equivalence, harm or exclusion of a55%
underlying score. Do not extend the fixed study to improve its classification.

The earlier learned128 result remains valid:65.7960% against the original128
planner, with interval60.9453--70.5224%. We have one demonstrated useful
search-to-learning-to-stronger-search cycle. This exact second update has not
established another. Successful repetition is now a weaker working assumption;
a fundamental limit on learned proposals or iterative learning is not established.

The three latest fit scores were44.7761%,58.5821%,51.1194%, each over134 games.
Each fit has a different initializer, batch stream and assigned67 opening pairs.
This is not a paired test of optimizer seeds. Selecting the58.58% fit from these
results would require fresh confirmation and would not validate the general
update recipe. Keep all three fits in the diagnostic.

## Leading hypotheses and evidence against simple explanations

1. **The measured learning objective inadequately constrains the deployed job.**
   Training chooses a winner among four offered moves. Deployment ranks all legal
   moves to supply one extra choice after the common first three are excluded.
   Better offered-choice cross-entropy does not establish better marginal proposals.
2. **Feedback coverage supplies too little useful new comparison information.**
   The new teacher compares the original top two, a random alternative and its
   learned fourth. This may reinforce familiar choices without discovering better
   omitted alternatives. It is not zero exploration: the random third and retained
   old data's two random alternatives do provide comparisons and may generalize.
3. **Winner-only labels discard useful ordering and confidence information.**
   If a top-two move wins, the target does not explicitly preserve the ordering of
   the other two. A tiny sampled lead and a decisive one receive the same class
   label. All recorded candidate outcomes could support a later outsider-ranking
   or margin-aware intervention, but the current result does not yet select it.
4. **Proposal gains are small, redundant, noisy, or fail to transfer to full play.**
   Better fourth moves might add little beyond the common first three; sampling
   noise might impede their selection; or greedy035 continuation outcomes might
   poorly represent repeated play against the stronger planner. These are distinct
   possibilities, not one proven search bottleneck.

Saved development reports show lower selected loss for every treatment versus
both anchor and replay, on both old and new validation partitions. Recognition
of teacher-preferred fourth moves improves; overall new-data choice agreement
improves only0.24--0.60 percentage points. Thus failure to optimize this loss,
gross forgetting on the measured old validation data, and blanket suppression of
fourth moves are weak leading explanations. All models changed and independent
fit reproduction passed. Audits reduce integrity concerns within their scope;
they do not eliminate shared-engine risk or unmeasured strategic forgetting.

Two treatment fits selected their final checkpoint, and the third was nearly tied
with its final loss. Do not repeat the older161/162 claim that every curve already
turned upward. More optimization may improve loss; that is not evidence for more
planning strength. Earlier useful hard-label learning also prevents a blanket
claim that hard labels cannot work.

## A sharper design than the earlier contingent proposal

Use **201 positions, one from each completed Stage A source pair**, rather than
216 positions from only36 pairs. All fits retain67 source pairs. This broadens
independent source coverage without increasing the continuation ceiling; a
guaranteed variance reduction is not claimed.

Within each fit assign pairs by fixed identity hashes to six source-role x phase
cells, with11 or12 pairs per cell. Source role is treatment/anchor; phases are
turns0--19,20--39,40+. Pool both seat games within a pair and hash-select one
eligible pre-decision root. Never use wins, scores, disagreement or logits for
selection. Saved final game metadata has minimum50 turns across all402 games,
so all phases exist; verify every assigned role/phase cell before freezing and
hold if unexpectedly absent. No adaptive replacements or new source games.

The primary estimand is the mean difference on this approximately balanced
201-position panel, equally weighting source pairs and therefore fits. It is
not an unbiased estimate of every arena move or full-game win rate. Retain
private-reserve and short-support states; report realized coverage. Analysis
respects source-pair and fit structure, with phase/role/fit subgroups descriptive.

On each same position:

- Keep the original first three candidates and all rule, observer-information,
  representative-action and tie contracts unchanged.
- Ask the corresponding frozen anchor, selected treatment and selected replay
  model for their actual highest-ranked remaining candidate. Add one fixed
  identity-sampled random fourth as an exploration control.
- Deduplicate the union: at most seven candidates. Evaluate all on128 fresh
  selection worlds and128 disjoint assessment worlds, with shared worlds across
  candidates and the existing greedy035 terminal-WDL continuation policy.
- Each four-candidate arm chooses using selection worlds only. Lock choices
  before assessment, and judge those fixed choices on the assessment worlds.
  Assessment outcomes must not select models, moves, subgroups or sample size.

Primary contrast: independently assessed treatment-minus-anchor selected-move
value. Secondary contrasts: treatment-minus-replay and random-minus-anchor.
Also report proposal/decision agreement, how often the fourth is selected, and
the independently assessed values of the fixed neural fourth proposals.

Two additional diagnostic selections cost no extra continuations: the common
first-three-only choice, and the choice from the larger candidate union. Both
must be selected on the selection block and judged on the assessment block.
They help distinguish redundant candidates from accessible conditional headroom.
The larger union has more candidates and is not an equal-budget player claim,
an optimal oracle, or proof of globally available headroom. A higher average
fourth-candidate value alone does not prove useful complementarity.

Fresh selection and assessment blocks avoid treating two copies of the same
saved worlds as independent evidence. Preserve128 selection worlds so the
diagnostic uses the deployed rule, rather than silently replacing it with64.

## Use the saved paired evidence as context, not the main gate

Each seat pair starts at an identical state with the same search stream; swapping
seats swaps treatment/anchor roles. All201 opening pairs, and their common
pre-decision prefixes, can therefore describe deployed proposal changes without
new inference or simulation. Include the first differing decision root; excluding
it would condition on current action agreement. Verify snapshots, legal actions,
world identities, common three actions and their outcome equality before matching.

Three representative pairs were inspected during this review (first assigned to
each fit:0,1,3). Exact opening identity was confirmed; their common prefixes had
5,2,2 roots including the first disagreement. These are feasibility examples,
not population frequencies. No corpus-wide diagnosis was performed in this review.

The two games share the SAME128 worlds. Cross-scoring one game's chosen move on
the other's outcomes is not independent assessment. Fixed neural-proposal outcome
contrasts avoid selecting a rollout maximum, but differences between the selected
maximum scores are optimistic. Prefixes emphasize early/agreed trajectories;
report opening and later-prefix results separately, with whole-pair accounting.

## Cost, verification and stopping boundary

Main ceiling:201 x7 x256 =360,192 continuations. A separate12-position timing
probe has ceiling21,504. A straight extrapolation from the completed arena's
recorded root times gives approximately1.73h main collection on eight workers:
1,645,810.70 worker-wall seconds /11,915,008 continuations =0.13813 seconds each.
This is not a forecast guarantee: phase mix, support, parallelism, audit and
setup differ. The older103,104-continuation screen took1h28 including full replay.

Aim for hours of execution. Propose an **eight-hour total execution ceiling**,
including probe and prescribed verification, subject to an outcome-blind timing
forecast before main collection. Preparation/implementation is additional. If
the forecast does not fit, hold before main work rather than quietly changing
sample size. No automatic extension for wide intervals or unfavorable scores.

Use the accepted sampled-verification policy for mature paths, all-state and
proposal/independence/budget checks, and fixed full stress/changed-path fixtures.
Freeze the exact audit contract before collection. Do not automatically replay
every continuation or add repeated broad test suites without a specific risk.
Use hidden independent workers if subsequently authorized; no recurring monitor.

## How this changes the next decision

- Little actual proposal movement favors investigating supervision and comparison
  coverage rather than assuming additional loss optimization improves this role.
- Changed but worse proposals favors a constrained update or better ranking target.
- Better fixed proposals without better independently assessed choices directs
  attention to complementarity and selection; it does not by itself isolate noise.
- Better independently assessed decisions alongside the inconclusive arena leaves
  effect size, full-game distribution and continuation-policy mismatch open. It
  is not permission to declare progress from a proxy or repeat training blindly.
- Useful replay or random/union headroom identifies a specific alternative worth
  one fresh test; none is promoted from the diagnostic.

Produce one bounded report even if uncertain. Do not sweep checkpoints, choose
the strongest fit, add states/worlds after seeing effects, or follow it with an
indefinite sequence of finer proxy studies. A subsequent intervention should be
one justified fixed change followed by fresh strength evidence. A contrastive
ordering of alternative moves using existing full outcome records is a plausible
learning candidate, not a preselected remedy or authorized successor.

## Why the other directions can wait

More Stage A games would chiefly improve precision on an unchanged package.
Stage B would measure new-data attribution without an established useful gain.
A new full-game test of the retrospective strongest fit could confirm a candidate
but would not diagnose a reproducible learning route. None currently earns its
roughly multi-day cost ahead of this bounded mechanism comparison.

Another training generation, arbitrary optimizer/loss sweep, larger model,
stronger continuations, or new MCTS architecture changes an unlocalized mechanism.
Prior failures constrain exact recipes, not these ideas universally. Batching
and broader independent opposition remain important for feasible superhuman
research, but neither answers this immediate learning-transfer question. The
old1.245x-versus1.25x batching gate remains failed; no old conditional authorization
is reactivated. Existing related opponent panels do not establish broad strength.

For eventual superhuman progress, the programme still needs repeatable improvement,
efficient search, independent strong opposition and declared-resource human tests.
This diagnostic serves the repeatable-improvement bottleneck. It cannot certify
the quality of greedy035 continuation values or any superhuman claim.

## Evidence

- outputs/proposal_iteration_128/{report.md,stage_a_result.json,stage_a_final_audit.json,final_status.json}
- docs/engineering/proposal_iteration_128.md; tools/proposal_iteration_128/{arena.py,learning.py}
- docs/strategy/next_during_proposal_iteration_128.md (saved validation details)
- docs/strategy/next_after_learned128.md; outputs/learned_proposal_128/report.md
- docs/engineering/learned_proposal_screen_result.md; learned_proposal_attribution_result.md
- docs/strategy/next_after_161.md; next_after_162.md; outputs/student_generalization_panel/report.md
- docs/engineering/saved_halving_screen_result.md
- docs/strategy/independent_superhuman_review_after_134_137.md (historical interpretations)

This note is a recommendation. No new executable, frozen protocol, output study,
training, inference, simulation, scheduled follow-up or experiment was created.
