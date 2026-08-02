# Recurrent Scale-or-Stop `069` Result

Experiment `069` stopped validly at scale generation 2. It did not reach the
third generation or final broad gameplay gate.

## Result

Scale generation 1 passed every contract after 2,880 eligible games and 90,290
actor decisions. Its validation KL was `0.00393466`, greedy movement
`3.4831%`, normalized entropy `0.309344`, overall/midlate surrogate
`+0.00068065/+0.00070557`, and history-rotation TV `0.0257367`. It scored
`0.495` over 100 immediate-parent games with `+0.28` paired margin.

Scale generation 2 collected 3,360/3,360 eligible games and 105,087 decisions.
All integrity conditions passed. It had validation KL `0.00513131`, movement
`3.6341%`, overall/midlate surrogate `+0.00039317/+0.00024386`, history TV
`0.0360446`, exact empty-history fallback/reload, and a `0.520` parent score
share with `-0.10` paired margin.

Its only failed gate was normalized validation entropy: `0.294256`, below the
frozen `0.30` minimum. The fixed conjunction therefore rejected generation 2
and stopped the experiment before another collection or any final broad gate.

Formal decision:
`recurrent_scale_learning_contract_failed_stop_outcome_only_branch`.

Primary evidence:
`outputs/recurrent_scale_or_stop_069/result.json`, SHA-256
`0970851f468ffae65ac68846a8aba439f27b7815985915c0f69542c86e7c8f3f`.
