param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$outputDir = Join-Path $root "data\corpus_reserve_choice_policy035_vs016_500"
New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

Add-Type @"
using System;
using System.Runtime.InteropServices;

public static class OvernightPowerRequest {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint executionState);
}
"@

$esContinuous = [uint32]::Parse(
    "80000000",
    [System.Globalization.NumberStyles]::HexNumber
)
$esSystemRequired = [uint32]0x00000001
$previous = [OvernightPowerRequest]::SetThreadExecutionState(
    $esContinuous -bor $esSystemRequired
)
if ($previous -eq 0) {
    throw "Could not create a temporary Windows system-awake request."
}

try {
    Set-Location $root
    & $Python "run_reserve_generalization_audit.py" `
        "--replay-path" "data/corpus_generation1_policy035_vs016_038_retry/replays.jsonl" `
        "--policy-checkpoint" "outputs/legal_action_scorer_035_legal_observation/legal_action_attention_scorer_best.pt" `
        "--output-path" "$outputDir/reserve_matched_returns.json" `
        "--report-path" "$outputDir/reserve_matched_returns.md" `
        "--progress-path" "$outputDir/reserve_matched_returns.progress.json" `
        "--samples-per-band" "125" `
        "--selection-pool-multiplier" "10" `
        "--rollout-determinizations" "3" `
        "--rollouts-per-determinization" "8" `
        "--log-every-states" "5" `
        "--device" "cpu"
    if ($LASTEXITCODE -ne 0) {
        throw "Reserve corpus process exited with code $LASTEXITCODE."
    }
}
finally {
    [void][OvernightPowerRequest]::SetThreadExecutionState($esContinuous)
}
