# Reviewer Prompt

You are conducting an independent strategic and methodological review of a two-player Splendor AI project. Read `docs/project_review_packet_070.md` first, then inspect the cited local evidence in `AGENT_CURRENT.md` and the result artifacts.

Your job is not to endorse the project's current roadmap. Challenge its assumptions, gate design, branch closures, benchmark validity, and interpretation of negative results. In particular, distinguish formal preregistration failures from evidence that a method is genuinely ineffective.

Focus on five tensions:

1. Resolver `051` is a demonstrably stronger full-game decision procedure, yet its per-state targets and neural distillation were unstable.
2. Recurrent scale experiment `069` passed every condition except normalized entropy 0.294256 versus a frozen 0.30 floor and never reached its final broad gate.
3. Several strict gates produced threshold-sensitive rejections, although later evidence sometimes vindicated caution for different reasons.
4. `035` is legal at inference but inherits representation and teacher provenance from a checkpoint trained before the hidden-information repair.
5. The project has repeatedly used `035` as prior, opponent, continuation policy, and benchmark, potentially making the learning system self-referential.

Return:

- a concise blocker diagnosis;
- a table classifying the major closed directions as decisively closed, provisionally closed, or worth reopening;
- a critique of the statistical decision framework;
- a ranked top three next directions;
- one concrete preregisterable next experiment with sample sizes, metrics, gates, runtime/compute recommendation, and clear falsification conditions;
- a kill list;
- confidence from 0-100% and the key cruxes;
- any questions for the experienced Splendor player that could materially change your recommendation.

Do not propose retroactively passing a failed experiment. If you recommend a related follow-up, explain why it asks a genuinely new question. Do not recommend GPU rental unless the expected scientific benefit, rather than mere wall-clock acceleration, is explicit.

