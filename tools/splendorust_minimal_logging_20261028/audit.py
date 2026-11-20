"""Independent saved-position repeat science; no model or game execution."""
from collections import Counter
from . import common as c
from tools.splendorust_pool6_live_20261027 import audit as accepted
from tools.external_strength_screen_20261018 import audit as ledger
from tools.future_training_rules_20261021 import decode_training_snapshot

check_clock = accepted.check_clock
validate_result = accepted.validate_result
validate_research_result = accepted.validate_research_result
profile_identity = accepted.profile_identity
classify_failure = accepted.classify_failure


def validate_record(row,spec,freeze,*,boundary=lambda:None):
    c.require(row['spec']==spec and row['route']==spec['route'] and row['observer']==spec['observer'],
              'Exact prospective repeat identity')
    c.require(spec['stage'] in ('overhead','diagnostic') and spec['observer_enabled'] is False,'Lean OFF-only scientific record')
    c.require(type(spec['logical_cpu']) is int and spec['logical_cpu'] in (0,2) and
        row['logical_cpu']==spec['logical_cpu'] and row['arm_id']==spec['arm_id']==f"{spec['route']}_cpu{spec['logical_cpu']}",
        'Exact fixed player/core arm')
    c.require(spec['quota']==64 and spec['seed_label']=='A' and spec['telemetry'] is False and
        spec['world_pool']==(6 if spec['route']=='research' else 3),'Unchanged players/configuration')
    before=spec['before_snapshot'];decode_training_snapshot(before);ledger.check_observer(before,row['observer'])
    c.require(row['before']==before,'Original saved root binding')
    check_clock(row['clock']);c.require(row['completed']==(row['clock']['status']=='ok'),'Original completion status')
    c.require(not row.get('profile_capture_error') and not row.get('placement_error'),'Integrity failure is not scientific acceptance')
    if row['completed']:
        boundary();identity=profile_identity(freeze,64,6,boundary=boundary) if spec['route']=='research' else None
        checked=validate_result(row['clock']['result'],row['observer'],spec['route'],identity=identity)
        c.require(row['action']==row['clock']['result']['action'] and row['trace']==row['clock']['result']['diagnostics'],
                  'Response/action/trace binding')
        after=ledger.transition(before,row['action']);decode_training_snapshot(after)
        c.require(after==row['after'],'Independent saved-root dictionary successor')
        return dict(status='passed',completed=True,scientific=checked,transition_checked=True)
    c.require(row['action'] is None and row['trace'] is None and row['after'] is None,'Rejected result remains unplayed evidence')
    return dict(status='preserved_failed_request',completed=False,original_status=row['clock']['status'],
                transition_checked=False,retained_rejected_result=row['clock']['result'] is not None)


def audit_prefix(rows,matrix,freeze,*,boundary=lambda:None):
    specs=matrix['requests'];c.require(len(specs)==96 and len(rows)<=96,'Frozen96-request inventory')
    c.require([r['spec'] for r in rows]==specs[:len(rows)],'No repeat omission, substitution or retry')
    c.require(all(r['completed'] for r in rows[:-1]),'No request after first failure')
    checks=[]
    for row,spec in zip(rows,specs):
        boundary();checks.append(validate_record(row,spec,freeze,boundary=boundary))
    complete=len(rows)==96 and all(r['completed'] for r in rows)
    counts=Counter((r['spec']['position_index'],r['arm_id']) for r in rows if r['completed'])
    if complete:c.require(counts==Counter((s['position_index'],s['arm_id']) for s in specs),'All prospective root/player/core repetitions')
    return dict(status='complete_verified' if complete else 'verified_prefix',complete=complete,
        planned_requests=96,attempted_requests=len(rows),completed_requests=sum(r['completed'] for r in rows),
        failed_requests=sum(not r['completed'] for r in rows),transitions=sum(x['transition_checked'] for x in checks),
        completed_cells=[dict(position_index=k[0],arm_id=k[1],count=v) for k,v in sorted(counts.items())],
        no_games=True,no_admission_or_strength_claim=True,seven_hundred_ms_descriptive_only=True)
