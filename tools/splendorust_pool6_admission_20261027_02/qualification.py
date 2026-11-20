"""Fresh admission inventory with unchanged actual52 and scientific checks."""
from collections import Counter
from copy import deepcopy

from tools.splendorust_pool6_live_20261027 import qualification as immutable
from tools.external_strength_screen_20261018 import audit as ledger
from . import common as c, adapters

run_entrypoint_checks = immutable.run_entrypoint_checks
verify_entrypoint = immutable.verify_entrypoint
validate_raw_qualification_annotations = immutable.validate_raw_qualification_annotations
exact_scientific_outputs = immutable.exact_scientific_outputs
ordinary_payload = adapters.ordinary_payload


def _old_plan():
    c.require(c.file_hash(adapters.PRIOR/'freeze.json') == adapters.PRIOR_FREEZE_SHA,
              'Immutable original admission freeze')
    item = c.read(adapters.PRIOR/'freeze.json')['qualification_plan']
    c.verify([item])
    return c.read(c.ROOT/item['path']), item


def build_plan():
    """Only fresh reserved inputs and prospective order differ from old plan."""
    previous, previous_artifact = _old_plan()
    reserved_path = c.OUT/'reserved_panel.json'
    reserved = c.read(reserved_path)
    positions = {p['id']: p for p in reserved['positions']}
    c.require(len(positions) == len(reserved['positions']) == 18,
              'Exactly eighteen distinct reserved positions')
    replacements = []
    for old in previous['parity_requests']:
        row = deepcopy(old)
        row['historical_plan_index'] = old['index']
        if old['kind'] == 'fresh_reserved':
            pos = positions[old['reserved_id']]
            c.require(pos['target_ply'] == old['target_ply'] and pos['seat'] == old['seat'],
                      'Fresh panel preserves original ply/seat strata')
            c.require(c.digest(pos['observer']) != old['observer_sha256'] and
                      pos['seed_decimal'] != old['reserved_seed_decimal'],
                      'Fresh reserved position/seed must differ from old admission')
            row.update(observer=deepcopy(pos['observer']), observer_sha256=c.digest(pos['observer']),
                before_snapshot=deepcopy(pos['before_snapshot']), reserved_seed_decimal=pos['seed_decimal'])
        replacements.append(row)
    frozen = [r for r in replacements if r['route'] == 'frozen']
    paired_candidate = {(r['kind'], r['case_id']): r for r in replacements
                        if r['route'] == 'research' and r['kind'] != 'live_edge'}
    edges = [r for r in replacements if r['kind'] == 'live_edge']
    c.require(len(frozen) == len(paired_candidate) == 42 and len(edges) == 6,
              'Forty-two paired roots plus six candidate edges')
    ordered = []
    for index, baseline in enumerate(frozen):
        candidate = paired_candidate[baseline['kind'], baseline['case_id']]
        c.require(baseline['observer'] == candidate['observer'] and
                  baseline['before_snapshot'] == candidate['before_snapshot'],
                  'Both routes receive the same paired lawful root')
        ordered.extend((baseline, candidate) if index % 2 == 0 else (candidate, baseline))
        if (index + 1) % 7 == 0:
            ordered.append(edges[(index + 1)//7 - 1])
    for index, row in enumerate(ordered):
        row['index'] = index
    c.require(len(ordered) == 90 and {r['historical_plan_index'] for r in ordered} == set(range(90)),
              'Every original qualification cell represented once')
    result = deepcopy(previous)
    result.update(schema='pool6_fresh_admission_qualification_v1', parity_requests=ordered,
        maximum_simultaneous_qualification_trees=2,
        reserved_panel=c.artifact(reserved_path), original_qualification_plan=previous_artifact,
        ordering='24 saved-root pairs then18 fresh-root pairs; alternating first arm; one fixed candidate edge after each seventh pair',
        historical_plan_index='Original cell identity only; fresh reserved observers/seeds explicitly replaced',
        prior_negative_preserved=True, no_retry_until_pass=True)
    return result


def _saved(item, cache, boundary):
    boundary()
    key = (item['path'], item['sha256'], item['bytes'])
    saved = cache.setdefault('artifacts', {})
    if key not in saved:
        c.verify([item])
        saved[key] = c.read(c.ROOT/item['path'])
    return saved[key]


def validate_admission_row(row, spec, cache, *, boundary=lambda: None, frozen_entrypoint=None):
    """Exact original scientific gates immediately after saving each response.

    Timings remain the parent's original receipts. The700ms gate is applied to
    the full predeclared inventory, never by changing these scientific outputs.
    """
    boundary()
    c.require(row['spec'] == spec, 'Qualification exact ordered specification')
    clock = row['clock']
    c.require(clock['status'] == 'ok' and clock['accepted'] is True and row.get('completed', True) is True,
              'Qualification request failed; no retry')
    result = clock['result']
    c.require(result['observer_sha256'] == spec['observer_sha256'] == c.digest(spec['observer']),
              'Parity observer binding')
    c.require(result['simulations_quota'] == spec['quota'] == 64, 'Parity simulation quota')
    c.require(result['diagnostics']['observer_seed'] == adapters.observer_seed(spec['observer'], spec['seed_label']),
              'Parity observer-only seed')
    ledger.check_observer(spec['before_snapshot'], spec['observer'])
    if spec['route'] == 'research':
        c.require(result['research']['world_pool'] == 6 and result['research']['seed_label'] == 'A' and
                  result['research']['telemetry_enabled'] is False and
                  all(x['telemetry'] is None for x in result['research']['phases']),
                  'Fixed live pool6-A/telemetryOFF')
    late = 0
    if spec['kind'] == 'saved_reproduction':
        prior = _saved(spec['expected_record'], cache, boundary)
        c.require(prior['completed'] and prior['clock']['status'] == 'ok' and prior['observer'] == spec['observer'],
                  'Accepted saved target input/receipt')
        c.require(exact_scientific_outputs(result) == exact_scientific_outputs(prior['clock']['result']),
                  'Live saved action/output/callback reproduction differs')
        if spec['route'] == 'research':
            c.require([p['callback_bits'] for p in result['research']['phases']] ==
                      [p['callback_bits'] for p in prior['clock']['result']['research']['phases']],
                      'Live callback float32 bits differ from accepted sampling')
            category = 'candidate_saved'
        else:
            category = 'frozen_saved'
    elif spec['kind'] == 'live_edge':
        previous = _saved(spec['expected_entrypoint'], cache, boundary)
        c.require(previous['status'] == 'ok', 'Accepted edge source receipt')
        expected = next(x for x in previous['result']['normalized_suite']['fixtures']
                        if x['id'] == spec['expected_case'])['results']['search']
        c.require(exact_scientific_outputs(result) == exact_scientific_outputs(expected),
                  'Live edge differs from accepted exact science')
        if spec['exact_frozen_shortcut']:
            if frozen_entrypoint is None:
                frozen_entrypoint = c.read(c.OUT/'entrypoint_checks/frozen.json')
            c.require(frozen_entrypoint['status'] == 'ok', 'Current unchanged64 suite receipt')
            baseline = next(x for x in frozen_entrypoint['result']['fixtures']
                            if x['id'] == spec['expected_case'])['results']['search']
            c.require(exact_scientific_outputs(result) == exact_scientific_outputs(baseline),
                      'Late heuristic changed from frozen64')
            c.require(result['diagnostics']['inference_calls'] == 0 and
                      all(p['heuristic_phase'] and p['simulations'] == p['leaf_calls'] == 0
                          for p in result['diagnostics']['phases']),
                      'Late heuristic unexpectedly searched')
            late = 1
        category = 'live_edges'
    elif spec['kind'] == 'fresh_reserved':
        category = 'fresh_reserved'
    else:
        raise ValueError('Unknown live qualification kind')
    return dict(status='passed', category=category, exact_frozen_late_shortcuts=late,
                model_calls=0, no_retries=True)


def validate_parity_rows(rows, plan, *, boundary=lambda: None, frozen_entrypoint=None, allow_prefix=False):
    expected = plan['parity_requests']
    c.require(len(expected) == 90 and len(rows) <= 90, 'Finite90 live qualification request bound')
    if not allow_prefix:
        c.require(len(rows) == 90, 'Finite90 live qualification request coverage')
    counts = dict(frozen_saved=0, candidate_saved=0, live_edges=0, fresh_reserved=0,
                  exact_frozen_late_shortcuts=0)
    cache, failures = {}, []
    successful = 0
    for index, row in enumerate(rows):
        spec = expected[index]
        c.require(row['spec'] == spec, 'Qualification exact ordered specification')
        clock = row['clock']
        if clock['status'] != 'ok' or clock['accepted'] is not True:
            c.require(allow_prefix and index == len(rows)-1 and clock['accepted'] is False and
                      row.get('completed', False) is False,
                      'Only final failed prefix request; never continue or invent successful science')
            failures.append(dict(index=index, status=clock['status'], detail=clock.get('detail')))
            continue
        proof = validate_admission_row(row, spec, cache, boundary=boundary, frozen_entrypoint=frozen_entrypoint)
        counts[proof['category']] += 1
        counts['exact_frozen_late_shortcuts'] += proof['exact_frozen_late_shortcuts']
        successful += 1
    complete = len(rows) == 90 and not failures
    if complete:
        c.require(counts == dict(frozen_saved=24, candidate_saved=24, live_edges=6, fresh_reserved=36,
                                exact_frozen_late_shortcuts=1), 'All mandatory fixed live qualification cells')
    return dict(status='passed' if complete else 'verified_prefix', complete=complete,
        complete_decisions=successful, attempted=len(rows), failures=failures, counts=counts,
        exact_recorded_numeric_outputs=True, exact_candidate_callback_bits=True,
        model_calls=0, no_retries=True)
