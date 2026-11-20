# Separate repair of the live-game final verifier

19 October 2026, before any execution of the repaired full analysis.

The six prescribed games and all 153 continuation audits completed. The original
final verifier then failed its first initial-world comparison. Its reconstruction
passed `ruleset` to the existing initial-snapshot helper, while the corrected
snapshot envelope requires `ruleset_id`. A model-free diagnostic confirmed that
all six saved initial worlds equal the deterministic reconstruction when the
correct envelope key is supplied. This diagnoses the verifier defect; it does
not substitute for the remaining independent checks or accept the study.

Keep the original verifier, its tests, the original freeze, the failed execution
01 log and receipt, the sealed failure diagnostic, and all experimental evidence
unchanged. Do not rerun players, games or continuation inference. Do not change
the opponent, local planner, model selection, search work, rules, clocks, sample,
stress selection, success criteria or interpretation of failures. Matchup totals
remain closed and these six games remain excluded from strength analysis.

## Narrow repair and authentication

Create a separately named v2 verifier. Correct only the initial-world envelope
key in its scientific checks. Keep the exact full initial-state, schedule and
hash comparison, and retain every other original acceptance check. Keep the
original analyzer's required membership and hash verification in the original
experiment freeze. Authenticate the new verifier separately; do not pretend
that it existed in that older freeze.

The mechanical versioning changes must give v2 a distinct success output and
sealed failure filename, and bind it to a new repair freeze. That repair freeze
must identify the original freeze and verifier, failed execution and diagnostic,
this protocol, v2 source, separate regression tests, the passing test receipt and
the completed-run/source-preservation receipts. Each bound artifact has its
path, byte count and SHA256. The repaired analyzer must check this repair
authentication before accepting evidence. Preserve the repair freeze itself
exclusively before launching the full v2 analysis.

Add a model-free regression that reconstructs the authoritative literal
`ruleset_id` envelope and requires equality with a valid initial world. Include
a negative case proving that corruption of the starting world is still
rejected. Run the original analyzer tests plus the new regression tests; retain
their logs in a separate output directory. Have another reviewer inspect the
complete original-to-v2 diff, test relevance and authentication before launch.

## One separately recorded v2 execution

After tests, review and repair freezing pass, execute the full v2 analysis once.
Save its output as `independent_analysis_v2.json`, its execution receipt/log as
execution 02, and any diagnostic as `sealed/independent_analysis_failure_02.json`.
Use exclusive writes. Never replace the failed original execution or silently
reuse its output names. A further unexplained failure remains preserved and
blocks acceptance; it does not authorize repeated attempts or weaker checks.

Retain the original absolute deadline, 19 October 2026 at 08:53:58.639351 UTC,
and the final 600-second reserve, evidence ceiling and process-closure rules.
Repair work and finalization consume that existing allowance. No new deadline
starts, and no additional games, data collection or larger-data study are
authorized. Final acceptance requires the repaired independent analysis to
pass all unchanged scientific checks and both original and repair inputs to
remain authenticated. The result report must disclose the preserved original
verifier failure and the separately accepted repair.
