"""Unchanged actual-entrypoint checks and exact saved repeat comparisons."""
from copy import deepcopy
from collections import Counter

from tools.splendorust_pool6_live_20261027 import qualification as immutable
from tools.splendorust_search_scale_20261024 import qualification as original
from . import common as c, adapters

# These functions only use supplied context/evidence paths. They neither rebind
# the frozen package nor write to its old output directory.
run_entrypoint_checks = immutable.run_entrypoint_checks
verify_entrypoint = immutable.verify_entrypoint
validate_raw_qualification_annotations = immutable.validate_raw_qualification_annotations
exact_scientific_outputs = immutable.exact_scientific_outputs
ordinary_payload = adapters.ordinary_payload


def build_plan():
    panel_path,matrix_path=c.OUT/'saved_panel.json',c.OUT/'request_matrix.json'
    panel,matrix=c.read(panel_path),c.read(matrix_path);requests=matrix['requests']
    c.require(len(panel['positions'])==18 and len(requests)==216,'Fixed18-root overhead/diagnostic inventory')
    c.require([x['index'] for x in requests]==list(range(216)) and
              [x['sequence'] for x in requests]==list(range(216)),'Fixed request order')
    c.require(Counter(x['stage'] for x in requests)==Counter(overhead=72,diagnostic=144),'Fixed stage counts')
    c.require(all(x['quota']==64 and type(x['quota']) is int and x['world_pool']==(3 if x['route']=='frozen' else 6)
        and x['seed_label']=='A' and x['telemetry'] is False for x in requests),'Unchanged ordinary players')
    c.require(all(type(x['observer_enabled']) is bool for x in requests),'Explicit observer state')
    return dict(schema='pool6_hostdiag_qualification_v1',original_suite=original.build_plan(64),
        complete_decisions=104,startup_complete_decisions=104,total_scientific_decisions=320,
        graph_cases=22,ordinary_requests=216,maximum_parent_requests=220,maximum_resident_workers=2,
        ordinary_seconds=1.,internal_seconds=.9,graph_seconds=60.,actual_entrypoint_seconds=300.,
        panel=c.artifact(panel_path),matrix=c.artifact(matrix_path),repeat_requests=requests,
        historical_comparison='Exact saved same-player result where available; no invented counterpart target',
        repeated_comparison='Exact full result against first successful observation in every root/player cell across observer modes',
        margin_700ms='descriptive_only',admission_decision=False,new_games=0)


def historical_result(spec,*,boundary=lambda:None):
    """Authenticate an available same-player target; counterpart has none."""
    boundary();item=spec.get('expected_record')
    c.require((item is not None) != (spec.get('expected_first_observation') is True),
              'Exactly one historical target or explicitly new counterpart baseline')
    if item is None:return dict(result=None,projection='no_historical_counterpart',source=None,
                               observer_sha256=spec['observer_sha256'])
    c.verify([item]);saved=c.read(c.ROOT/item['path'])
    c.require(saved['completed'] is True and saved['clock']['status']=='ok' and saved['clock']['accepted'] is True
        and saved['route']==spec['route'] and saved['observer']==spec['observer'],'Historical ordinary result binding')
    result=saved['clock']['result']
    c.require(result['observer_sha256']==spec['observer_sha256']==c.digest(spec['observer']) and
        result['simulations_quota']==64 and result['player']=='search' and
        result['diagnostics']['observer_seed']==adapters.observer_seed(spec['observer']),
        'Historical target observer/quota/seed')
    return dict(result=result,projection='none',source=item,observer_sha256=spec['observer_sha256'])


def validate_repeat_row(row, spec, first_results, *, boundary=lambda: None):
    """Fail immediately on a successful response whose exact science differs.

    The caller saves the original row first. This job-local cache avoids reading
    the same immutable baseline repeatedly; final readback uses a fresh cache.
    """
    boundary()
    c.require(row['spec'] == spec, 'Exact ordered saved repeat specification')
    clock = row['clock']
    c.require(clock['status'] == 'ok' and clock['accepted'] is True and row.get('completed', True) is True,
              'Single-row scientific comparison requires original completed success')
    cell = (spec['position_index'], spec['route'])
    if cell not in first_results:
        target = historical_result(spec, boundary=boundary)
        first_results[cell] = dict(target=target, first=None)
    saved = first_results[cell]
    # Even cached cells retain exactly the original descriptor/observer binding.
    c.require(saved['target']['source'] == spec.get('expected_record') and
              saved['target']['observer_sha256'] == spec['observer_sha256'] == c.digest(spec['observer']),
              'Repeat cell target/observer changed')
    value = c.canonical(clock['result'])
    historical=saved['target']['result'] is not None
    if historical:c.require(value==c.canonical(saved['target']['result']),'Exact saved historical result/callback output mismatch')
    repeated = saved['first'] is not None
    if repeated:
        c.require(value == saved['first'], 'Exact full within-cell repeated result mismatch')
    else:
        saved['first'] = value
    return dict(status='passed', historical_comparison=historical, first_repeat_comparison=repeated,
                historical_projection=saved['target']['projection'])


def validate_repeat_rows(rows, plan, *, allow_prefix=False, boundary=lambda: None):
    """Validate successes exactly; a rejected final prefix row remains a failure.

    The parent separately authenticates original clocks, raw streams, sidecars,
    placement and every full scientific result. No failure is dropped or retried.
    """
    expected = plan['repeat_requests']
    c.require(len(expected) == 216 and len(rows) <= len(expected), 'Fixed repeat row bound')
    if not allow_prefix:
        c.require(len(rows) == len(expected), 'Complete repeat inventory required')
    cache, counts, comparisons = {}, Counter(), Counter();historical_count=0
    failures = []
    for index, row in enumerate(rows):
        boundary()
        spec, clock = row['spec'], row['clock']
        c.require(spec == expected[index], 'Exact ordered saved repeat specification')
        successful = clock['status'] == 'ok' and clock['accepted'] is True
        if not successful:
            c.require(allow_prefix and index == len(rows)-1 and clock['accepted'] is False,
                      'Only final rejected prefix row; no continuation or fabricated success')
            c.require(row.get('completed', False) is False, 'Rejected repeat cannot be complete')
            failures.append(dict(index=index, route=spec['route'], case_id=spec['case_id'],
                                 status=clock['status'], detail=clock.get('detail')))
            continue
        cell = (spec['position_index'], spec['route'])
        proof = validate_repeat_row(row, spec, cache, boundary=boundary)
        historical_count+=int(proof['historical_comparison'])
        if proof['first_repeat_comparison']:
            comparisons[cell] += 1
        counts[cell] += 1
    complete = len(rows) == 216 and not failures
    if complete:
        c.require(counts == Counter((x['position_index'],x['route']) for x in expected),
                  'Every successful root/player cell has its full prospective inventory')
    return dict(status='passed' if complete else 'verified_prefix', complete=complete,
        attempted=len(rows), successful=sum(counts.values()), failures=failures,
        exact_historical_comparisons=historical_count,
        exact_first_repeat_comparisons=sum(comparisons.values()),
        cells=[dict(position_index=p, route=r, successful=counts[p, r],
                    historical_projection=cache[p, r]['target']['projection'],
                    first_repeat_comparisons=comparisons[p, r]) for p, r in sorted(counts)],
        new_games=0, admission_decision=False, margin_700ms='descriptive_only', model_calls=0)
