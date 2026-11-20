# Saved-position decision-quality calibration — 13 November 2026

Status: prospective separate implementation; freeze and actual qualification are
required before the diagnostic. User authorized this next step. No training,
new games, checkpoint promotion or subsequent live study is authorized here.

## Question

Can a cheaper saved-position score detect the already verified strength gains
from increasing EF8A search from64 to128 to256 to512 simulations? The recent
training-duration study found declining generalization after four passes, but
offline fitting errors have not been calibrated to playing strength. This study
tests the sensitivity of a possible screening tool before another long live test.

## Panel and unchanged players

Use512 saved Main positions, four from each of all128 held-out corrected-rules
games in the accepted November7 corpus (64 validation,64 offline). These games
have been reused in development; this is not a new independent assessment.
For each game sort eligible roots chronologically, divide into four equal-count
bands, and choose the minimum prospective identity hash in each band. Selection
uses no outcomes, moves, values, timing or current model output. Eligibility is
the accepted source eligibility, at least two supported Main actions, turn<124.
Insufficient eligibility fails the panel; no substitutions or threshold changes.

Only the lawful acting-player observation enters a model request. Source game,
split and ply metadata remain in the controller. The actual corrected corpus
decoder authenticates and validates all512 consumed records. No old snapshot
or historical training loader is an alternate route.

All players use the original EF8A checkpoint, exact float32 graph, three worlds,
depth64,cpuct0.4,dynamicFPU0.02965 and the original search implementation. No
weights, precision, legal rules or auxiliary choice logic changes. Test64/128/
256/512 on every root using seedsA/B. A is the canonical lawful-observer hash;
B is a new domain-separated lawful-observer hash. Reference searches use2048
simulations with two separate seedsR/S. All ten conditions are prospectively
rotated within each root:5120 ordinary completed decisions.

The reference shares the same network and algorithm. It is a fallible deeper
search, not ground truth. Its edgeQ is an internal win-credit estimate, not
calibrated probability or factual regret. Only the Main action is scored;
complete compound actions are nevertheless executed and checked for legality.
The saved positions also come from512-search teacher games, so sensitivity is
measured on the states that teacher visits, not every possible Splendor state.

## Qualification and evidence

Before diagnostic work: eleven exact graph fixtures; all52 actual-entrypoint
counterexample decisions,26 malformed-input rejections, corrected catalogue,
nobles, returns, payments, final-round/private-reserve and native-support checks;
18 fixed parity decisions. Compare original-native versus telemetry-native at
64/128/256/512 with seedA, telemetry on/off at those quotas,2048R/S and64B.
Require exact complete decisions, scientific traces and callback float32 bits.
The native67 Main-action index is distinct from the81 policy logits; missing
unvisited edgeQ stays missing. The2048 dynamic-FPU numerical bound explicitly
accounts for float32 export rounding and binary64 accumulated arithmetic.

Every new worker, including model-free evaluation/audit workers, calls
verify_runtime_sources once at startup. Corrected ruleset/catalogue and turn
contract are bound in the source/profile/panel/freeze receipts. The historical
checkpoint is explicitly an unchanged reference, not newly corrected training.

Retain every result before checking it, and all failure evidence. Fixed32 exact
reproductions use16 hash-selected roots, one rotated control and one reference
per root. Reproduce complete actions, callbacks and telemetry exactly; do not
retry a failed request. The saved-evidence audit checks every ordinary and
reproduction result, action, work count, profile, source hash and request clock.
Independent arithmetic recomputes the report from retained normalized records.
Every model service must exit0 gracefully, with observed exact descendant
absence. Completion also requires independent acceptance and actual owner exit.

## Prospective measurement and reading rules

For each candidate action, take its edgeQ in each independent2048 reference.
The primary comparison averages both candidate seeds and both references, using
one common mask: a root is covered only if all eight candidate chosen actions
were visited in both reference searches. Report missing support and excluded
roots/games; never impute zero or rerun a missing edge. A support conclusion
requires at least90% of512 roots and at least100 source games. The estimand is
conditional on this shared subset; selection-related bias remains possible.

Average covered positions within game, then weight games equally. Use20000
paired whole-game bootstrap draws, stratified by the64/64 source partitions,
fixed seed202611135120. Record reference-specific sensitivity, minimum visit
support, all-root agreement with reference choices and reference disagreement.
Timing is descriptive, with counts over one second. No scientific action is
discarded merely for exceeding one second.

- Strong positive-control support: every adjacent gain has a positive lower
 98.333% interval (Bonferroni family0.05), with positive broad reference signs.
- Provisional broad support: nondecreasing means, positive512-minus64 lower95%
 interval, and positive broad difference independently inR andS, but adjacent
 ranks remain unresolved.
- Contradicted: an adjacent increase has a negative upper98.333% interval.
- Otherwise unresolved; insufficient shared coverage is separately unresolved.

The accepted separate live controls are128vs64:56.125% (95%CI51.5–60.75),
256vs128:61% (56.5–65.375),512vs256:56.5% (51.875–61.125). Preserve their
disclosed recovery histories, including the256 sensitivity60.75% if its affected
game is a loss. Do not pool those different games or fit an exchange rate from
proxy units to win rate. Correctly ordering these controls does not demonstrate
validity for changed network weights. No model is promoted by this diagnostic.

## Fixed operational bounds

One model service and its native players at a time, CPU0; controllerCPU2 where
available. Windows console-helper affinities are recorded, not changed; only
the pinned System32 console executable with the exact owned parent is allowed.
Maximum three
services (qualification,diagnostic,reproduction), each with eleven graph fixtures.
All services share one immutable model among their native adapter instances.
Complete ordinary response deadline120s, internal119s. Qualification's nested
complete decisions have120s watchdogs; its finite batch has a6000s external cap.
No reset of an active clock. Startup240s, graph qualification300s; these startup
allowances are clipped by stage clocks. A new ordinary decision requires its
full120s to fit within the original stage deadline, otherwise the stage ends
incomplete. No automatic retries or game/model substitutions.

Original-clock global cap24h; model cutoff18h. Stage caps: qualification2h,
diagnostic12h,reproduction2h,analysis4h,acceptance4h, all clipped to global/model
deadlines. Actual model-tree RSS4GiB, audit/root8GiB, new evidence16GiB, actual
disk reserve5GiB. No available-RAM admission rule or forecast refusal. Estimates
of roughly4–5h search work plus checks are planning information, not stop gates.
Externally owned jobs record actual exits and clean exact descendants.

Before launch require source-bound model-free tests, independent reviews of
scientific/evidence/lifecycle paths and a detached-launch fixture. Failed
prelaunch checks remain preserved. Scientific negative or inconclusive results
end this study; they are not technical failures to retry. No scheduled monitor.
After final acceptance produce at most two charts and a recommendation; leave
EF8A and deferred collection, training, external testing and six-world work intact.
