# Roadmap update: pause external screens, diagnose saved decisions

27 October 2026. This additive update changes the next priority in the
[24 October roadmap](splendorust_revised_roadmap_2026-10-24.md). Historical
roadmaps, players, frozen attempts and evidence remain unchanged.

## Current decision

External-candidate testing is **paused at the user's request**. No further
adapter, qualification, opponent screen or external confirmation is authorized.
Both completed external policies scored0/60 against unchanged32 and are rejected
as competitive opposition for the proposed optimized64 confirmation. Preserve
the original competitiveness threshold and all previous attempts.

Prepare the [bounded saved-position decision-quality diagnostic](../engineering/splendorust_decision_quality_20261027.md).
It uses24 outcome/time-blind saved optimized64 decisions and separate research
searches at64/256/1024 simulations, with two held-out1024 seed families. Its
purpose is to identify a specific search-budget or sampling hypothesis, not to
prove a deeper same-network move is correct. Preparation is authorized; running
models, new games and implementation of the evaluator are not launched here.

Optimized64 remains the frozen candidate and unchanged32 its preserved control.
Larger-data collection, fitting and training remain deferred. There is no change
to the one-second practical target, corrected rules or lawful-information
requirement. The proposed30-second diagnostic allowance is extra research
computation and cannot qualify a deployment player.

## Why this priority

The project first moved from the earlier full-rollout planner to a verified
pretrained policy/value network with search. We then demonstrated an internal
strength gain from optimized64 over unchanged32 and addressed runtime failures,
including accepted recovered verification of the200-game reliability workload.
Those advances remain useful even though independent opposition is unresolved.

The two external screens were intended to test whether the improvement
generalizes beyond the parent. Both fixed exported policies were too weak for
that purpose: they lost naturally, without being cut off by the clock. Their
mean decisions were about65ms and62ms. Extra idle time alone cannot improve a
fixed network-only decision procedure; adding search would be a new player.
This does not identify whether training, model, absent search or adaptation to
corrected rules explains their weakness, and it is not a claim about all upstream
checkpoints or DDQN/PPO in general.

Further adapter work currently has uncertain value. A small saved-position
diagnostic can direct effort toward useful strength per second without another
large dataset or a new live tournament. More search may reveal promising
alternatives, amplify shared model errors, or remain unstable under sampled
worlds. The plan preserves those possibilities rather than assuming the answer.

## Decision after the diagnostic

Choose at most one narrowly supported follow-up: allocation of a fixed search
budget, sampling stability, or a clearly motivated later evaluator study.
Do not automatically increase simulations, world count, network size or training.
Any implemented successor needs fresh paired whole-game confirmation against
optimized64 at one second, with preserved timeouts and all verification checks.
Saved diagnostic roots cannot double as independent confirmation of that change.

Resume external work only through a later explicit decision, preferably with a
credible runnable search-based opponent and evidence that the adaptation effort
can produce useful competition. There are further public leads, but public code
is not the same as a strong, trained, independent, compatible one-second player.
For example, [AhinLendor](https://github.com/inhabae/AhinLendor) has author strength
claims but no verified released checkpoint in our bounded review;
[cestpasphoto AlphaZero](https://github.com/cestpasphoto/alpha-zero-general) has
pretrained models but related teacher-data ancestry to our inherited player;
[Rinascimento RAVE](https://github.com/EllingtonKirby/RinascimentoFramework/tree/ek/new_algorithms_and_params)
and [nbramia/play](https://github.com/nbramia/play/blob/main/docs/splendor.md)
are search-based code leads with no local qualification. This read-only
availability check does not resume testing or establish their local strength.

## Wider objective and unresolved claims

The objective remains demonstrably exceptional two-player base Splendor under
lawful information and practical response times. Improving a preserved parent
is evidence of local progress, not a human ranking or proof of generalization.
Strong independent opponents and eventually expert-human comparisons remain
part of that destination. Their timing in the roadmap is deferred, not their
scientific purpose abandoned.

The next immediate action, after this preparation, is separately authorized
additive implementation and correctness qualification of the small diagnostic.
No experiment, model, new game or training job runs as part of this update.
