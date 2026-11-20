"""Prospective latency-permissive panel and fresh finite-game workload."""
from copy import deepcopy
from . import common as c
from tools.splendorust_gpu_concurrency_20261105.design import cells

PRIOR = c.ROOT/'outputs/splendorust_gpu_concurrency_20261105_01'
REPORT = c.ROOT/'outputs/splendorust_gpu_concurrency_report_20261105_01'
PLAN_SHA = '43126553774d6f07a9451bc33eb051c510b61cac77d26a82835a0c9a67fddf91'
AUDIT_SHA = '0058022406908c0a57fd1bb0aaffb060be170b69b74a89954a410c109154887d'
CLOSEOUT_SHA = '0b342c7c97d2e9180f70216ced4dbab63fec8307f29deadeb400d02b1570bd52'
REVIEW_SHA = '08a47ab9b9db422f42830b4311ab5f2daaf23e9453f6d224ed3c70b04c9004fd'


def build_plan():
    from tools.future_training_rules_20261021 import verify_runtime_sources
    from tools.splendorust_gpu_probe_20261105.control import require_absent
    from . import qualification, games
    for path, expected in ((PRIOR/'plan.json',PLAN_SHA),(REPORT/'audit_saved.json',AUDIT_SHA),
                           (REPORT/'closeout.json',CLOSEOUT_SHA),(REPORT/'closeout_review.json',REVIEW_SHA)):
        c.require(c.sha(path)==expected,'Authenticated accepted predecessor: '+str(path))
    previous=c.read(PRIOR/'plan.json'); accepted=c.read(REPORT/'audit_saved.json')
    c.require(accepted['status']=='saved_concurrency_study_verified' and accepted['plan_sha256']==PLAN_SHA,
              'Accepted complete prior study, with preserved negative cells')
    c.verify(previous['inputs'])
    execution=c.read(PRIOR/'execution.json'); claim=c.read(PRIOR/'claim.json')
    for identity in execution['observed_owned_identities']:
        require_absent(identity)
    require_absent(dict(pid=claim['pid'],created=claim['created']))
    sources=verify_runtime_sources()
    c.require(sources==previous['source_guard'],'Same corrected source identities')
    runtime=deepcopy(previous['runtime'])
    runtime.update(internal_seconds=c.INTERNAL_SECONDS,outer_seconds=c.OUTER_SECONDS,
                   execution_policy_id=c.EXECUTION_POLICY_ID,qualification=qualification.build_plan())
    game_plan=games.build_plan(); runtime['games']=deepcopy(game_plan)
    runtime['gameplay_seconds_per_route']=7200
    prior_c1={route:{case['case_id']:c.artifact(PRIOR/'cells'/f'{route}_c1'/'worker/cell'/case['case_id']/'result.json')
                    for case in previous['cases']} for route in ('cpu','gpu')}
    return dict(schema='splendorust_latency_permissive_throughput_v1',time=c.now(),
        authorization='User authorized the next throughput test accepting late moves and separating throughput from latency',
        execution_policy_id=c.EXECUTION_POLICY_ID,
        prior=dict(plan=c.artifact(PRIOR/'plan.json'),closeout=c.artifact(REPORT/'closeout.json'),
            audit=c.artifact(REPORT/'audit_saved.json'),closeout_review=c.artifact(REPORT/'closeout_review.json'),
            execution=c.artifact(PRIOR/'execution.json'),claim=c.artifact(PRIOR/'claim.json'),
            frozen_inputs=previous['inputs'],
            preserved_evidence=[c.artifact(p) for p in sorted(PRIOR.rglob('*')) if p.is_file()],
            preserved_reports=[c.artifact(p) for p in sorted(REPORT.rglob('*')) if p.is_file()]),
        prior_c1=prior_c1,runtime=runtime,cases=deepcopy(previous['cases']),cells=cells(),games=game_plan,
        worker_affinity=runtime['worker_affinity'],supervisor_affinity=runtime['supervisor_affinity'],
        worker_limits=dict(qualification_seconds=1800,cell_seconds=900,games_seconds=10800),
        gameplay_seconds_per_route=7200,total_seconds=c.TOTAL_SECONDS,model_seconds=c.MODEL_SECONDS,
        maximum_worker_rss_bytes=c.MAX_RSS,maximum_output_bytes=c.MAX_OUTPUT,minimum_free_disk_bytes=c.MIN_DISK,
        available_ram_reserve_bytes=0,forecast_admission_gate=False,group_gap_seconds=0.,
        maximum_diagnostic_decisions=64,maximum_games=16,maximum_selected_model_replays=48,
        warmup=dict(calls_per_cell=5,case_index=0,client_id='warmup',slot=0,
            actual_service=True,inside_measured_panel=False,exact_within_route=True,
            internal_seconds=c.INTERNAL_SECONDS,outer_seconds=c.OUTER_SECONDS,failure_stops_study=True),
        qualification_decisions_per_route=67,qualification_not_counted_as_diagnostic_panel=True,
        internal_seconds=c.INTERNAL_SECONDS,outer_seconds=c.OUTER_SECONDS,deployment_target_seconds=1.,
        descriptive_latency_threshold_seconds=[1.,9.,10.],late_completed_responses_accepted=True,
        fixed_slot='Permanent case or game index0..7 for every concurrency; unused GPU rows remain zero',
        cell_failure_isolation='Only authenticated decision60/75s safeguards permit later independent cells/routes after exact cleanup',
        integrity_failure_policy='Stop study on source, status, startup, cap, numerical, result, replay or cleanup failure',
        exact_same_route_baseline='Validated wrapper-only normalization to accepted prior C1; new C2/4/8 exact scientific results and root bytes equal new C1',
        cross_route_root_numeric_gate=True,cross_route_actions_descriptive=True,
        primary_timing='Panel: first barrier to last durable group writes; games: first release to eighth durable natural finish; include gaps/tails',
        eligibility='All8 complete and correct, C1 exact scientific reproduction, numerical gates and clean closure; partial cells have no rate',
        selection='Highest completed8-request panel throughput per route among eligible cells; lower concurrency breaks exact ties',
        game_admission='Both routes selected after all8 cells saved audits; no retry or fallback concurrency',
        retries=0,training_updates=0,larger_data_deferred=True,automatic_followup_games=True,
        games_are_training_data=False,automatic_followup_training=False,
        protocol=c.artifact(c.PROTOCOL),source_guard=sources)
