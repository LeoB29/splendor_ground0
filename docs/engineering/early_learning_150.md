# Experiment 150: dense early checkpoint diagnostic

2026-09-09. Authorized by the owner's "ok do it" following
[the proposal](C:/Users/leoba/Desktop/splendor_ground0/docs/strategy/next_after_149.md).
Freeze this protocol, new runner/verifier/tests, runtime and explicit input
hashes before any 150 fit on real development data. Preserve the intentionally
dirty worktree and all closed prior experiments. No automatic successor.

## Intervention and exposure

Reuse 149's 768 training and 192 validation states, features, masks and targets
exactly; one state per source game, disjoint splits. Read no protected test
panels, features, predictions or outcomes. Verify the existing audited input
hash chain, then reuse the 960 target files without any continuation replay.
Reading the old completion/audit metadata is permitted to validate provenance.

Both original 1085-input, 320001-parameter representations and all three paired
seeds 149001/149002/149003 are retained. Same AdvantageModel architecture,
initialization, float32 CPU arithmetic, single Torch thread and deterministic
algorithms as 149. AdamW lr=.0003, decay=.0001, gradient cap 1; exactly 4096
updates, 64 states per update with all fixed candidates. Batch generator seed
is model seed+149000. No model, feature, target, learning-rate, batch or seed
selection change. The exact zero baseline and padded-candidate masks remain.

Validate at every update 0–128 inclusive, then multiples of 128 through 4096:
160 checks per fit, 127 more than 149. Both original coarse-grid and expanded
grid selections retain the earliest strictly lowest validation state-mean MSE.
Selection uses original float32 computation; no minimum improvement tolerance.
Keep update 0 eligible, run all six full schedules regardless of selection,
and retain both selected tensors, final tensors and every history row.

Require added evaluation to leave Torch global and minibatch RNG states
unchanged. Each rerun must reproduce all 33 original coarse history rows,
original selected update/tensors and final tensors exactly before acceptance.
Six independent full refits then reproduce all 160 history rows and tensors
with a separately implemented training loop. No tuning or restart on a
scientific failure; preserve engineering failures and diagnose them explicitly.

## Development outcomes

For each representation/seed/selection rule, predict all four fixed candidates
on the 192 validation states. Choose maximum predicted advantage among valid
candidates; exact ties use first index, always baseline. The zero predictor
therefore keeps 035. Validation outcomes are the existing 64-world averages
under exact greedy-035 continuation, not an optimal-play oracle.

Report per-seed and equal-seed mean state-balanced MSE, selected WDL minus035,
baseline-choice fractions, coarse-to-dense move changes and selected updates.
The MSE denominator is real nonbaseline candidates per state. Statistics use
float64 after float32 predictions/targets; checkpoint selection remains the
original float32 computation. Baseline WDL is +1/0/-1, so +.02 advantage
corresponds to one percentage point of expected terminal score under these
continuations. No ensemble prediction, best-seed selection or subgroup rescue.

A representation passes the development screen only if ALL conditions hold:

1. At least two of three expanded-grid selections are updates 1–127.
2. Equal-seed validation MSE is at least 3% lower than its coarse reference.
3. That MSE is also at least 3% lower than the zero predictor.
4. Equal-seed selected WDL gain over 035 is at least +.02.
5. Selected WDL gain is strictly greater than its coarse reference.

Report all criteria and both representations. Any pass is only a development
screen; a control-only pass supplies no afterstate-specific support. No
confidence intervals, significance claims, equivalence claims or protected
strength conclusions are appropriate from this reused selection set.

## Integrity, stopping and cost

Freeze the explicit 149 and 148 dependency/input inventories instead of invoking
their dynamic file-discovery identities after adding new source. Bind149's
manifest, successful audit, selection/development locks, six models, both
feature/panel/target locks and all 960 training/validation targets. Exclude
149's protected test artifacts from the 150 input and evaluation paths.

Preflight validates provenance and dataset structure without fitting real
data. Synthetic unit tests establish unchanged optimization trajectories,
independent reproduction, baseline tie/mask behavior and both positive and
negative screening cases before freezing. Run appropriate repository checks.

Main execution has exactly six 4096-update fits, followed by six independent
4096-update audit fits; six concurrent workers per stage. No new source games,
continuations, hyperparameter searches or fresh evaluation stage. Record all
timings; historical fit times are not a promised runtime. A separate-process
verifier independently aggregates targets, recomputes every prediction and
statistic/gate and checks frozen hashes again. Final status binds result/audit.

If the screen fails, close the missed-early-window remedy for this recipe at
the declared development threshold, without rejecting other learners. A pass
permits only a later fresh-confirmation proposal. Neither result changes 149,
champions 084/035, or authorizes new data, training, arena or promotion.
