"""Prospective small logging diagnostic; no player or placement selection."""
from . import common as c
from tools.splendorust_search_scale_20261024 import qualification as original
from tools.future_training_rules_20261021 import decode_training_snapshot
from tools.external_strength_screen_20261018 import rules

PRESERVED_ATTEMPT=c.ROOT/'outputs/splendorust_pool6_core_compare_20261027_02'
PRESERVED_FREEZE_SHA='92e35152b24276baaa676572b8891b723ae64eec4c4320be8389147eb5915745'
PRESERVED_REVIEW_SHA='4a5cdd7aa91e92f10bebf70621b990bead9ad2afb31d524ff268c933e5131cb0'
BASELINE=c.ROOT/'outputs/splendorust_pool6_core_compare_20261027_01'
PANEL=c.OUT/'saved_panel.json'
MATRIX=c.OUT/'request_matrix.json'
FIXTURES=('private_reserves','gold_returns')
ARMS=('frozen_cpu0','frozen_cpu2','research_cpu0','research_cpu2')

def fixed(path,value):
    if path.exists():c.require(c.read(path)==value,'Prepared artifact changed: '+str(path))
    else:c.save(path,value)
    return c.artifact(path)

def build_panel():
    suite=original.build_plan(64);positions=[]
    for index,case in enumerate(FIXTURES):
        fixture=next(x for x in suite['counterexamples'] if x['id']==case)
        observer=rules.observation_payload(decode_training_snapshot(fixture['snapshot']));targets={}
        for route in ('frozen','research'):
            source=BASELINE/'workers'/('qualification_'+route+'_cpu0')/'entrypoint_forensics.jsonl'
            events=[__import__('json').loads(line)['event'] for line in source.read_text(encoding='utf-8').splitlines()]
            if route=='frozen':
                selected=[x['result'] for x in events if x.get('event')=='result' and x.get('operation')=='ordinary_worker_request' and x.get('case_id')==case and x.get('arm')=='search']
            else:
                indices=[i for i,x in enumerate(events) if x.get('event')=='result' and x.get('operation')=='ordinary_worker_request' and x.get('case_id')==case and x.get('arm')=='search']
                c.require(len(indices)==1,'Unique named research fixture')
                i=indices[0];raw,projection=events[i-2],events[i-1]
                c.require(raw.get('event')=='raw_research_result_before_checker_projection' and raw.get('request')==dict(player='search',observer=observer) and projection.get('event')=='explicit_frozen_checker_projection' and projection['raw_result_sha256']==c.digest(raw['result']) and projection['projected_result']==events[i]['result'],'Exact named raw research/projection binding')
                selected=[raw['result']]
            c.require(len(selected)==1,'Unique saved original fixture result')
            target=c.OUT/'prelaunch/expected_results'/f'{route}_{case}.json'
            targets[route]=fixed(target,dict(route=route,case_id=case,observer=observer,result=selected[0],source=c.artifact(source)))
        positions.append(dict(position_index=index,case_id=case,role='saved_failure_target' if index==0 else 'fixed_rule_edge_control',
            observer=observer,observer_sha256=c.digest(observer),before_snapshot=fixture['snapshot'],expected=targets))
    return dict(schema='startup_logging_panel_v1',positions=positions,selection='Exact saved failure and prospectively fixed gold-return rule-edge control; no latency matching or fresh roots')

def build_matrix(panel=None):
    panel=build_panel() if panel is None else panel;rows=[]
    for stage,mode_order in (('overhead',(False,True,True,False,True,False,False,True)),('diagnostic',(True,True,True,True))):
        for round_index,enabled in enumerate(mode_order):
            positions=panel['positions'] if round_index%2==0 else list(reversed(panel['positions']))
            for pos in positions:
                arms=ARMS if (round_index+pos['position_index'])%2==0 else tuple(reversed(ARMS))
                for arm in arms:
                    route,cpu=arm.split('_cpu');sequence=len(rows)
                    rows.append(dict(index=sequence,sequence=sequence,stage=stage,round=round_index,
                        position_index=pos['position_index'],case_id=pos['case_id'],observer=pos['observer'],
                        observer_sha256=pos['observer_sha256'],before_snapshot=pos['before_snapshot'],
                        route=route,logical_cpu=int(cpu),arm_id=arm,quota=64,world_pool=3 if route=='frozen' else 6,
                        seed_label='A',telemetry=False,observer_enabled=False,logging_timing_enabled=enabled,
                        expected_result_file=pos['expected'][route],core_pair_index=sequence//2))
    c.require(len(rows)==96,'Fixed 64 overhead and32 detailed decisions')
    return dict(schema='startup_logging_matrix_v1',requests=rows,no_retries=True,no_replacements=True,
        order='ABBA then BAAB timer modes; balanced fixture/arm order; four final ON rounds',no_warmup_dropping=True)

def engineering_contract():
    return dict(margin_seconds=.7,ordinary_deadline_seconds=1.,internal_seconds=.9,
        placement_selection=False,admission_decision=False,margin_descriptive_only=True)

def host_observer_contract():
    from tools.splendorust_pool6_core_compare_20261027_02.design import host_observer_contract as old
    value=old();value.update(requests=96,disabled_requests=96);return value

def build_plan():
    return dict(schema='startup_logging_plan_v1',phase=c.PHASE,fixtures=list(FIXTURES),arms=list(ARMS),
        saved_write_replays=64,saved_replay_gate=dict(added_mean_seconds=.005,added_p95_seconds=.025,per_fixture_per_mode=16),
        overhead_requests=64,diagnostic_requests=32,maximum_ordinary_requests=96,
        actual_overhead_gate=dict(maximum_mean_ratio=1.05,added_mean_seconds=.010,added_p95_seconds=.025,per_arm_per_mode=8),
        startup_complete_decisions=208,graph_cases=44,parent_requests=104,scientific_decisions=304,
        extra_host_sampler_enabled=False,players_unchanged=True,logging_bytes_and_sync_unchanged=True,
        timer_metadata_only_buffered=True,seconds_per_move=1.,internal_seconds=.9,margin_seconds=.7,
        startup_timers_enabled_after_saved_replay_gate=True,stop_first_failure=True,retries=0,
        new_games=0,training=False,larger_data_deferred=True,external_testing='paused',
        job_seconds=c.JOB_SECONDS,collection_cutoff_seconds=c.COLLECTION_CUTOFF_SECONDS,stage_seconds=c.STAGES)

def prepare():
    from tools.future_training_rules_20261021 import verify_runtime_sources
    fixed(c.OUT/'runtime_sources.json',verify_runtime_sources())
    panel=build_panel();fixed(PANEL,panel);fixed(MATRIX,build_matrix(panel));fixed(c.OUT/'study_plan.json',build_plan())
    return dict(status='prepared_model_free',ordinary_requests=96,new_games=0)
