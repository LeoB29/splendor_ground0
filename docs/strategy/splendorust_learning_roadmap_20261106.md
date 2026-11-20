# Return from collection engineering to learning

6 November 2026. This updates the research direction after the verified CPU8 versus
compiled GPU16 collector comparison. It prepares the next study; it launches no
games, fitting or larger-data collection.

The objective remains substantially stronger, ultimately superhuman two-player
Splendor using lawful information. The original EF8A checkpoint remains the
strength reference. Search improvements and training improvements are separate
claims; faster data generation alone is neither a stronger player nor evidence
of superhuman play.

## What we know

The small joint and policy-head-only training updates did not establish a useful
playing-strength gain. Their separate assessment schedules do not show that
policy-only training was superior. The postmortem found some policy-target
transfer and uncertain value changes; the failure mechanism is not diagnosed.
The corpus contains 192 independent training games, even though it supplies
10,660 correlated position records.

Collection is now qualified and measured. CPU8 produced 51.15 games/hour during
collection and 36.81 including per-batch checks; compiled GPU16 produced 55.37
and 38.93 respectively. The GPU gain was modest and did not meet the predeclared
consistency criterion in both blocks. Keep CPU8 as the practical default and
compiled GPU16 as a qualified alternative. Park further speculative GPU tuning.

## Recommended sequence

1. **Prepare, then qualify one saved-corpus learning diagnostic.** Use the
   [bounded design](../engineering/splendorust_learning_diagnostic_preparation_20261106.md):
   64 games × 48 roots versus 192 games × 16 roots, crossed with joint versus
   policy-only trainability. Eight fixed fits, equal records and optimizer steps,
   no new source games. Existing held-out games remain exploratory. This asks
   whether game breadth or update scope deserves attention before a larger spend.
2. **Use its result to choose one learning question.** Broader transfer could
   justify a prospective fresh-data comparison; an update-scope signal could
   justify a controlled objective/optimization study. Mixed evidence should be
   reported as such. Do not select the best of eight checkpoints for promotion.
3. **Only then propose fresh collection and independent strength confirmation.**
   Price complete games, storage, final verification and assessment separately.
   Keep the existing useful-gain criteria and fresh paired-game requirement.
   A future recipe needs independent evidence beyond already opened held-out data.

This sequence avoids assuming that either data scarcity or the network's size
caused the previous null results. The next diagnostic can study diversity within
the existing 192-game support; it cannot establish the benefit of thousands of
new games. It also compares trainable scopes within the same architecture, not
small versus large networks.

## Fixed boundaries

Preserve every frozen player, failure and accepted result, including the collector's
failed original final checker and separately recovered verification. New work
uses separate versioned files and the mandatory corrected-rules source/loader
qualification. Do not relabel engineering benchmark trajectories as training data.
No training chain starts from an unpromoted successor.

The one-second playing target remains distinct from concurrent teacher throughput.
Resource estimates are planning information; they are not RAM-reserve or forecast
stops. Actual bounds, real clocks, exact ownership and evidence preservation remain
required. Larger-data work, external testing and six-world work stay deferred.

Current authorization completes the next study's **preparation**. The execution
freeze, implementation qualification and scientific fits are the next concrete
action to propose after this design is reviewed; no execution job or automation
has been created.
