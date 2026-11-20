"""Independent ordered assessment with explicit retained control deadlines.

This module validates saved decisions; it never creates a model or changes an
outcome. The runner additionally binds every row to raw transport, placement,
clock/input artifacts and the fresh-worker continuation inventory.
"""
from copy import deepcopy

from . import common as c, panel, game_audit as audit
from tools.splendorust_search_scale_20261024 import audit as science


def _integrity(row):
    c.require('profile_capture_error' not in row and 'placement_error' not in row,
        'Observer/placement evidence failure cannot authorize control continuation')


def _empty_failure(row):
    c.require(row.get('completed') is False and all(key in row and row[key] is None for key in ('action','trace','after')) and
        'result' in row['clock'] and row['clock']['result'] is None, 'Failed request cannot invent completed science')


def control_timeout(row):
    """Authorize only an original-control deadline after independent raw checks.

    The caller must first authenticate raw/observer/placement evidence. This
    helper shares the unchanged adjudicator and adds the prospective narrower
    continuation rule; malformed/illegal/unclassified/resource failures stop.
    """
    _integrity(row)
    c.require(row.get('placement_mode')=='original' and row.get('arm')=='search' and
        type(row.get('quota')) is int and row['quota'] in (32,64) and
        row.get('player_role')==('challenger' if row['quota']==64 else 'opponent') and
        row.get('mode')==('candidate' if row['quota']==64 else 'baseline') and
        row.get('observation_mode')=='profiled', 'Only original controls may continue after a timeout')
    _empty_failure(row)
    receipt=row['clock'];classification=science.classify_failure(receipt)
    c.require(receipt['deadline_seconds']==1., 'Control timeout requires unchanged full one-second clock')
    c.require(classification=='timeout' or classification=='worker_error' and
        'one_second_internal_deadline' in str(receipt.get('detail','')).lower(),
        'Non-timeout control failure cannot continue qualification')
    if classification=='timeout':
        received=receipt['receipt_elapsed_seconds']
        c.require(receipt['elapsed_seconds']>=1. and receipt['received_before_deadline'] is False and
            (received is None or received>1.), 'Original outer-timeout clock evidence is inconsistent')
    return classification


def _failure_record(row, index, classification):
    return dict(index=index,fixture_id=row['fixture_id'],quota=row['quota'],placement_mode=row['placement_mode'],
        block_index=row['block_index'],pass_index=row['pass_index'],classification=classification,
        clock=deepcopy(row['clock']),worker_clock_file=row['worker_clock_file'],
        no_retry=True,no_fabricated_completed_output=True)


def validate_timing_rows(rows, plan, *, boundary=lambda:None, allow_prefix=False):
    """Require the exact schedule or a prefix ending at an isolated failure.

    Real original-control deadlines may occur anywhere in that schedule. Every
    successful output still agrees across placements/passes and with its
    matching historical target. An isolated error always ends the prefix.
    """
    panel.validate_plan(plan)
    c.require(type(allow_prefix) is bool, 'Explicit timing-prefix policy required')
    c.require(type(rows) is list and 0<len(rows)<=192 and (allow_prefix or len(rows)==192), 'Timing panel inventory')
    groups={};historical={'isolated':0,'original':0};control_failures=[];failure=None
    completed={'isolated':0,'original':0}
    for index,row in enumerate(rows):
        boundary();spec=plan['schedule'][index];fixture=plan['fixtures'][spec['fixture_index']]
        c.require(type(row) is dict and all(type(row.get(key)) is type(value) and row[key]==value
            for key,value in spec.items()), 'Timing ordered schedule differs')
        c.require(row['before']==fixture['snapshot'] and row['observer']==fixture['observer'], 'Saved timing fixture differs')
        c.require(failure is None, 'Timing requests continued after isolated failure')
        _integrity(row)
        audit.validate_boundary_row(row)
        science.check_clock(row['clock'])
        c.require(row['clock']['deadline_seconds']==1., 'Timing full one-second clock required')
        c.require(type(row.get('completed')) is bool and row['completed']==(row['clock']['status']=='ok'),
            'Completed flag differs from original clock')
        if not row['completed']:
            _empty_failure(row)
            if row['placement_mode']=='original':
                classification=control_timeout(row)
                control_failures.append(_failure_record(row,index,classification))
            else:
                classification=science.classify_failure(row['clock'])
                failure=_failure_record(row,index,classification)
                c.require(allow_prefix, 'Full isolated qualification cannot include a failure')
            continue
        science.validate_result(row['clock']['result'],row['observer'],'search',expected_quota=row['quota'])
        c.require(row['action']==row['clock']['result']['action'] and row['trace']==row['clock']['result']['diagnostics'],
            'Timing action/trace differs from original response')
        science.audit_decision(row['observer'],row['action'],row['trace'],'search',
            before=row['before'],after=row['after'],expected_quota=row['quota'])
        value={key:row[key] for key in ('action','trace','after')};value['result']=row['clock']['result']
        key=(row['fixture_id'],row['quota'])
        if key in groups:
            c.require(groups[key]['science']==value, 'Placement/repeat full scientific output differs')
        else:
            groups[key]=dict(science=value,isolated=0,original=0)
        groups[key][row['placement_mode']]+=1
        completed[row['placement_mode']]+=1
        target=fixture['historical']
        if target is not None and target['quota']==row['quota']:
            c.require(value=={key:target[key] for key in ('result','action','trace','after')},
                'Original-quota historical result differs')
            historical[row['placement_mode']]+=1
    complete=len(rows)==192 and failure is None
    if complete:
        c.require(len(groups)==48 and all(group['isolated']==2 and 0<=group['original']<=2 for group in groups.values()) and
            completed['isolated']==96 and completed['original']+len(control_failures)==96,
            'Full48cells/two isolated successes/all96control attempts required')
        c.require(historical['isolated']==46 and 0<=historical['original']<=46,
            'Mandatory46isolated historical comparisons required')
        expected_controls=sum(plan['fixtures'][row['fixture_index']]['historical'] is not None and
            plan['fixtures'][row['fixture_index']]['historical']['quota']==row['quota']
            for row in rows if row['placement_mode']=='original' and row['completed'])
        c.require(historical['original']==expected_controls, 'Actual successful-control historical count differs')
    elif allow_prefix:
        c.require(failure is not None and failure['index']==len(rows)-1 and failure['placement_mode']=='isolated',
            'Valid stopped prefix must end at the first isolated failure')
    return dict(status='passed',requests=len(rows),completed=sum(completed.values()),
        isolated_completed=completed['isolated'],control_completed=completed['original'],
        completed_scientific_groups=len(groups),
        fully_replicated_isolated_groups=sum(group['isolated']==2 for group in groups.values()),
        fully_replicated_groups=sum(group['isolated']==2 and group['original']==2 for group in groups.values()),
        isolated_historical_comparisons=historical['isolated'],
        control_historical_comparisons=historical['original'],historical_comparisons=sum(historical.values()),
        historical_comparisons_are_actual_successful_checks=True,
        control_timeout_count=len(control_failures),control_timeouts=control_failures,
        complete=complete,failure=failure,original_control_timeouts_do_not_stop_isolated_qualification=True,
        relative_comparison_is_admission_gate=False)
