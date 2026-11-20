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
    panel_path, matrix_path = c.OUT/'saved_panel.json', c.OUT/'request_matrix.json'
    panel, matrix = c.read(panel_path), c.read(matrix_path)
    requests = matrix['requests']
    c.require(len(panel['positions']) == 6 and len(requests) == 144, 'Fixed six-root repeat inventory')
    c.require([x['index'] for x in requests] == list(range(144)) and
              [x['sequence'] for x in requests] == list(range(144)), 'Fixed repeat request order')
    cells = Counter((x['position_index'], x['route']) for x in requests)
    c.require(cells == Counter({(p, r): 12 for p in range(6) for r in ('frozen', 'research')}),
              'Twelve repeats of every root/player cell')
    c.require(all(x['quota'] == 64 and type(x['quota']) is int and
                  x['world_pool'] == (3 if x['route'] == 'frozen' else 6) and
                  x['seed_label'] == 'A' and x['telemetry'] is False for x in requests),
              'Fixed unchanged players in ordinary repeat inventory')
    return dict(schema='pool6_repeat_qualification_v1', original_suite=original.build_plan(64),
        complete_decisions=104, startup_complete_decisions=104, total_scientific_decisions=248,
        graph_cases=22, ordinary_requests=144, maximum_parent_requests=148,
        maximum_resident_workers=2, ordinary_seconds=1., internal_seconds=.9,
        graph_seconds=60., actual_entrypoint_seconds=300.,
        panel=c.artifact(panel_path), matrix=c.artifact(matrix_path), repeat_requests=requests,
        historical_comparison='Exact full result for eleven cells; frozen gold_returns removes only phase.inference from captured startup target',
        repeated_comparison='Exact full unprojected result against first successful observation in each cell',
        margin_700ms='descriptive_only', admission_decision=False, new_games=0)


def _without_startup_capture(result):
    """Only the old frozen gold startup record has additional capture arrays."""
    value = deepcopy(result)
    for phase in value['diagnostics']['phases']:
        phase.pop('inference', None)
    return value


def historical_result(spec, *, boundary=lambda: None):
    """Authenticate the old raw target without executing a model or transition."""
    boundary()
    c.require(('expected_record' in spec) != ('expected_entrypoint' in spec),
              'Exactly one historical target kind')
    item = spec.get('expected_record', spec.get('expected_entrypoint'))
    c.verify([item])
    saved = c.read(c.ROOT/item['path'])
    if 'expected_record' in spec:
        c.require(saved['completed'] is True and saved['clock']['status'] == 'ok' and
                  saved['clock']['accepted'] is True and saved['route'] == spec['route'] and
                  saved['observer'] == spec['observer'], 'Historical ordinary result binding')
        result = saved['clock']['result']
        projection = 'none'
    else:
        c.require(spec['route'] == 'frozen' and spec['case_id'] == spec['expected_case'] == 'gold_returns',
                  'Only frozen gold_returns may use startup capture projection')
        c.require(saved['status'] == 'ok' and saved['accepted'] is True and
                  saved['result']['status'] == 'entrypoint_counterexamples_passed',
                  'Historical frozen actual-entrypoint receipt')
        cases = [x for x in saved['result']['fixtures'] if x['id'] == 'gold_returns']
        c.require(len(cases) == 1 and cases[0]['observer_sha256'] == spec['observer_sha256'],
                  'Historical frozen gold observer binding')
        result = _without_startup_capture(cases[0]['results']['search'])
        projection = 'remove_phase_inference_from_frozen_startup_target_only'
    c.require(result['observer_sha256'] == spec['observer_sha256'] == c.digest(spec['observer']) and
              result['simulations_quota'] == 64 and result['player'] == 'search' and
              result['diagnostics']['observer_seed'] == adapters.observer_seed(spec['observer']),
              'Historical scientific target observer/quota/seed')
    return dict(result=result, projection=projection, source=item)


def validate_repeat_row(row, spec, first_results, *, boundary=lambda: None):
    """Fail immediately on a successful response whose exact science differs.

    The caller saves the original row first. This job-local cache avoids reading
    the same immutable baseline twelve times; final readback uses a fresh cache.
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
    c.require(saved['target']['source'] == spec.get('expected_record', spec.get('expected_entrypoint')) and
              saved['target']['result']['observer_sha256'] == spec['observer_sha256'] == c.digest(spec['observer']),
              'Repeat cell target/observer changed')
    value = c.canonical(clock['result'])
    c.require(value == c.canonical(saved['target']['result']),
              'Exact saved historical result/callback output mismatch')
    repeated = saved['first'] is not None
    if repeated:
        c.require(value == saved['first'], 'Exact full within-cell repeated result mismatch')
    else:
        saved['first'] = value
    return dict(status='passed', historical_comparison=True, first_repeat_comparison=repeated,
                historical_projection=saved['target']['projection'])


def validate_repeat_rows(rows, plan, *, allow_prefix=False, boundary=lambda: None):
    """Validate successes exactly; a rejected final prefix row remains a failure.

    The parent separately authenticates original clocks, raw streams, sidecars,
    placement and every full scientific result. No failure is dropped or retried.
    """
    expected = plan['repeat_requests']
    c.require(len(expected) == 144 and len(rows) <= len(expected), 'Fixed repeat row bound')
    if not allow_prefix:
        c.require(len(rows) == len(expected), 'Complete repeat inventory required')
    cache, counts, comparisons = {}, Counter(), Counter()
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
        if proof['first_repeat_comparison']:
            comparisons[cell] += 1
        counts[cell] += 1
    complete = len(rows) == 144 and not failures
    if complete:
        c.require(counts == Counter({(p, r): 12 for p in range(6) for r in ('frozen', 'research')}),
                  'Every successful repeat cell has all twelve observations')
    return dict(status='passed' if complete else 'verified_prefix', complete=complete,
        attempted=len(rows), successful=sum(counts.values()), failures=failures,
        exact_historical_comparisons=sum(counts.values()),
        exact_first_repeat_comparisons=sum(comparisons.values()),
        cells=[dict(position_index=p, route=r, successful=counts[p, r],
                    historical_projection=cache[p, r]['target']['projection'],
                    first_repeat_comparisons=comparisons[p, r]) for p, r in sorted(counts)],
        new_games=0, admission_decision=False, margin_700ms='descriptive_only', model_calls=0)
