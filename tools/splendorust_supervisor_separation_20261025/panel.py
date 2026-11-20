"""Prospective, fixed paired qualification for supervisor placement only."""
from copy import deepcopy
import math
from pathlib import Path
import statistics

from . import common as c
from tools import future_training_rules_20261021 as corrected
from tools.external_strength_screen_20261018 import rules, audit as ledger
from tools.splendorust_search_scale_20261024 import audit as science

SOURCE = c.ROOT / 'outputs/splendorust_live_boundary_20261025_01'
SOURCE_FREEZE = '50ca8363a364b06dbf452778b90aa27c5ea763d8f2d76d2c4efab8bca582ec6b'
QUOTAS = (64, 32)
PLACEMENTS = ('original', 'isolated')
FOCUS = [('operational',4,16), ('main',3,28), ('main',15,34),
    ('main',36,26), ('main',44,6), ('main',46,21),
    ('main',72,16), ('main',72,18), ('main',72,20), ('main',72,22), ('main',72,28)]
CONTROLS = [(stage, game, ply-1) for stage, game, ply in FOCUS] + [('main',72,0), ('main',71,52)]


def row_path(stage, game, ply):
    return SOURCE / f'games/{stage}/p{game//2:03d}_g{game}/moves/{ply:04d}.json'


def make_schedule(fixtures):
    c.require(len(fixtures)==24 and len({r['fixture_id'] for r in fixtures})==24,
              'Exactly24 unique saved positions required')
    result=[]
    block_index=0
    for quota in QUOTAS:
        for pass_index in range(2):
            indices=list(range(24))
            modes=list(PLACEMENTS)
            if pass_index:
                indices.reverse();modes.reverse()
            for placement in modes:
                for ordinal,index in enumerate(indices):
                    result.append(dict(index=len(result),block_index=block_index,
                        ordinal=ordinal,fixture_id=fixtures[index]['fixture_id'],fixture_index=index,
                        quota=quota,player_role='challenger' if quota==64 else 'opponent',
                        placement_mode=placement,pass_index=pass_index))
                block_index+=1
    return result


def validate_plan(plan):
    c.require(plan['schema']=='splendorust_supervisor_separation_panel_v1', 'Panel schema')
    c.require(plan['source_freeze_sha256']==SOURCE_FREEZE, 'Source freeze identity')
    fixtures=plan['fixtures']
    c.require(len(fixtures)==24 and [tuple(x['source_identity']) for x in fixtures]==FOCUS+CONTROLS,
              'Fixed focus/control selection differs')
    c.require(sum(x['kind']=='failed_saved_position' for x in fixtures)==1 and
              sum(x['kind']=='slow_completed' for x in fixtures)==10 and
              sum(x['kind']=='control' for x in fixtures)==13, 'Fixed selection categories')
    c.require(plan['schedule']==make_schedule(fixtures), 'Balanced eight-block schedule differs')
    c.require(len(plan['schedule'])==192, 'Exactly192 timed requests')
    return plan


def build_plan():
    runtime=corrected.verify_runtime_sources()
    c.require(c.file_hash(SOURCE/'freeze.json')==SOURCE_FREEZE, 'Original live freeze changed')
    completion=c.read(SOURCE/'completion.json');execution=c.read(SOURCE/'execution.json')
    verification=c.read(SOURCE/'verification_summary.json')
    c.require(completion['status']=='captured_fault_verified' and
        execution['status']=='completed' and execution['exit_code']==0 and not execution['cleanup_errors'] and
        verification['status']=='passed', 'Verified original failure prefix required')
    seal=c.read(SOURCE/'completed_evidence_seal.json')
    index={item['path']:item for item in seal['artifacts']}
    c.require(len(index)==len(seal['artifacts']), 'Duplicate original seal entries')
    sources={}
    for name in ('freeze.json','completion.json','execution.json','verification_summary.json',
                 'completed_evidence_seal.json','collection_stop.json'):
        item=c.artifact(SOURCE/name);sources[item['path']]=item
    fixtures=[]
    for ordinal,(stage,game,ply) in enumerate(FOCUS+CONTROLS):
        path=row_path(stage,game,ply);item=c.artifact(path)
        c.require(item==index[item['path']], 'Selected saved move differs from original seal')
        sources[item['path']]=item;row=c.read(path)
        c.require((row['stage'],row['game_index'],row['ply'])==(stage,game,ply), 'Saved identity')
        snapshot=row['before'];state=corrected.decode_training_snapshot(snapshot)
        observer=rules.observation_payload(state)
        c.require(observer==row['observer'], 'Strict corrected snapshot/observer mismatch')
        ledger.check_observer(snapshot,observer)
        for name in ('worker_clock_file','child_profile_file','parent_profile_file'):
            source=c.artifact(c.ROOT/row[name]);c.require(source==index[source['path']], 'Selected raw source differs')
            sources[source['path']]=source
        c.require(c.read(c.ROOT/row['worker_clock_file'])==row['clock'], 'Saved clock copy differs')
        history=None
        if row['completed']:
            c.require(row['clock']['status']=='ok' and row['clock']['receipt_elapsed_seconds']<=1.,
                'Historical accepted completion required')
            science.validate_result(row['clock']['result'],observer,'search',expected_quota=row['quota'])
            science.audit_decision(observer,row['action'],row['trace'],'search',
                before=snapshot,after=row['after'],expected_quota=row['quota'])
            history=dict(quota=row['quota'],result=deepcopy(row['clock']['result']),
                action=deepcopy(row['action']),trace=deepcopy(row['trace']),after=deepcopy(row['after']))
        else:
            c.require((stage,game,ply)==('main',72,28) and row['quota']==64 and
                row['clock']['status']=='worker_error' and row['clock']['result'] is None,
                'Only the disclosed failed saved position has no target')
        kind='control' if ordinal>=11 else 'failed_saved_position' if ordinal==10 else 'slow_completed'
        if kind=='slow_completed':
            c.require(row['completed'] and row['clock']['receipt_elapsed_seconds']>.7 and row['quota']==64,
                'Ten original slow completed moves')
        fixtures.append(dict(fixture_id=f'{stage}_g{game}_p{ply}',kind=kind,
            source_identity=[stage,game,ply],source=item,snapshot=deepcopy(snapshot),observer=observer,
            original_quota=row['quota'],historical=history))
    plan=dict(schema='splendorust_supervisor_separation_panel_v1',
        source_freeze_sha256=SOURCE_FREEZE,fixtures=fixtures,schedule=make_schedule(fixtures),
        sources=sorted(sources.values(),key=lambda x:x['path']),source_guard=runtime,
        selection='All10slow completions plus failure; preceding11opposite-role controls; earlyg72p0 and latest64moveg71p52.',
        original_failed_attempt_unchanged=True,failed_saved_root_is_new_diagnostic_not_replacement=True,
        historical_comparisons_expected=92,agreement_cells_expected=48,
        independent_population_sample=False,paired_order='Eight closed-worker blocks; reverse placement and root order on pass2')
    return validate_plan(plan)


def distribution(values):
    c.require(all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in values),
              'Finite nonnegative timing values required')
    if not values:return dict(count=0,mean=None,p95=None,max=None)
    values=sorted(values);at=(len(values)-1)*.95;low=math.floor(at);high=math.ceil(at)
    return dict(count=len(values),mean=statistics.mean(values),p95=values[low]+(values[high]-values[low])*(at-low),max=max(values))


def timing_gate(rows):
    """An engineering margin failure never changes a completed move into a loss."""
    fixtures=[dict(fixture_id=f'{s}_g{g}_p{p}') for s,g,p in FOCUS+CONTROLS]
    expected={(r['fixture_id'],r['quota'],r['placement_mode'],r['pass_index'])
              for r in make_schedule(fixtures)}
    identities=[(r['fixture_id'],r['quota'],r['placement_mode'],r['pass_index']) for r in rows]
    complete=(len(rows)==192 and len(set(identities))==192 and set(identities)==expected and
              all(r['completed'] and r['clock']['status']=='ok' and r['clock']['deadline_seconds']==1.
                  for r in rows))
    per_quota={}
    inventory=[]
    for quota in QUOTAS:
        groups={mode:[r for r in rows if r['quota']==quota and r['placement_mode']==mode]
                for mode in PLACEMENTS}
        timings={mode:distribution([r['clock']['receipt_elapsed_seconds'] for r in group
                   if r['completed'] and r['clock']['status']=='ok']) for mode,group in groups.items()}
        full=all(len(groups[m])==48 and timings[m]['count']==48 for m in PLACEMENTS)
        conditions=dict(complete_inventory=full,
            original_under_one_second=full and timings['original']['max']<=1.,
            isolated_under_700ms=full and timings['isolated']['max']<.7,
            isolated_mean_nonregression=full and timings['isolated']['mean']<=1.05*timings['original']['mean'],
            isolated_p95_nonregression=full and timings['isolated']['p95']<=timings['original']['p95']+.025)
        passed=all(conditions.values());inventory.append(full)
        per_quota[str(quota)]=dict(qualified=passed,conditions=conditions,distributions=timings)
    qualified=complete and all(inventory) and all(x['qualified'] for x in per_quota.values())
    return dict(status='passed' if qualified else 'not_qualified',qualified=qualified,
        complete_inventory=complete and all(inventory),per_quota=per_quota,
        margin_seconds=.7,seconds_per_move=1.,mean_ratio_ceiling=1.05,p95_added_seconds_ceiling=.025,
        engineering_margin_is_not_timeout=True,no_causal_fix_claim=True)
