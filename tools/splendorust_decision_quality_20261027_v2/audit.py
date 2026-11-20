"""Finite saved-observation and diagnostic evidence checks; never runs a model.

The corrected atomic projection and safe-choice implementation are inherited
read-only. The larger-quota accounting and root-statistic checks are additive.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import struct

from tools import future_training_rules_20261021 as guard
from tools.external_strength_screen_20261018 import audit as ledger, rules
from tools.external_strength_screen_20261018.common import digest
from tools.external_strength_screen_20261018_02 import bridge
from tools.splendorust_search_scale_20261024 import audit as original

ROOT = Path(__file__).resolve().parents[2]
DOMAIN = 'decision_quality_saved_panel_20261027_v1'
SHA = re.compile(r'[0-9a-f]{64}\Z')
QUOTAS = (64, 256, 1024)
RESEARCH_SCHEMA = 'splendorust_saved_decision_research_v1'
BASELINE_PROFILE_SHA = '2b60b56419c938ecd5e96d5b9c9a5153f36b2796d4e4b35219b746dc7dfed6d9'
BASE_PHASE = {'action', 'simulations', 'leaf_calls', 'native_legal_count',
              'heuristic_phase', 'neural_callbacks', 'neural_callback_count',
              'neural_callback_rows'}
TRACE_FIELDS = {'opponent_id', 'profile_id', 'observer_seed', 'phases',
                'inference_calls', 'inference_rows', 'forced_pass'}
TELEMETRY_FIELDS = {'visits', 'q', 'root_value', 'network_value', 'selected_edge_value'}


def require(value, message):
    if not value:
        raise ValueError('DECISION_QUALITY_AUDIT: ' + message)


def same(a, b):
    return json.dumps(a, sort_keys=True, separators=(',', ':'), allow_nan=False) == \
        json.dumps(b, sort_keys=True, separators=(',', ':'), allow_nan=False)


def finite(value, name, minimum=None, maximum=None):
    require(type(value) in (int, float) and math.isfinite(value), name + ' finite number')
    require(minimum is None or value >= minimum, name + ' lower bound')
    require(maximum is None or value <= maximum, name + ' upper bound')
    return value


def integer(value, name, minimum=0, maximum=2**63-1):
    require(type(value) is int and minimum <= value <= maximum, name + ' integer bound')
    return value


def sha(value, name):
    require(type(value) is str and SHA.fullmatch(value), name + ' SHA256')
    return value


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def float32_sha(values):
    return hashlib.sha256(b''.join(struct.pack('<f', x) for x in values)).hexdigest()


def seed_for(observer, label):
    require(label in ('A', 'B', 'C'), 'seed family')
    if label == 'A':
        return int(digest(observer)[:16], 16)
    encoded = json.dumps(observer, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return int(hashlib.sha256((DOMAIN+'|'+label+'|'+encoded).encode()).hexdigest()[:16], 16)


def _read_bound(item, freeze, *, root=ROOT, boundary=lambda: None, maximum=4*1024*1024):
    """Read one explicitly frozen regular source file, without tree enumeration."""
    boundary()
    require(type(item) is dict and set(item) == {'path', 'bytes', 'sha256'}, 'artifact descriptor')
    relative = Path(item['path'])
    require(not relative.is_absolute() and bool(relative.parts) and '..' not in relative.parts,
            'source relative path')
    path = Path(root)/relative
    require(path.resolve().is_relative_to(Path(root).resolve()) and not path.is_symlink(),
            'source path boundary')
    pins = {x['path']: x for x in freeze['inputs']}
    require(len(pins) == len(freeze['inputs']) and same(pins.get(item['path']), item),
            'source artifact absent from execution freeze')
    integer(item['bytes'], 'source byte count', 0, maximum)
    sha(item['sha256'], 'source hash')
    raw = path.read_bytes()
    require(len(raw) == item['bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256'],
            'source bytes changed')
    value = json.loads(raw)
    boundary()
    return value


def load_saved_root(descriptor, freeze, *, root=ROOT, boundary=lambda: None):
    """Actual new loader: authenticate the old success, expose lawful input only.

    Returned historical evidence is for the parent auditor, never a worker
    payload. The worker receives returned observer alone (plus frozen research
    condition). No fresh transition, game, model or search is performed.
    """
    require(type(descriptor) is dict and descriptor.get('stage') == 'main' and
            descriptor.get('mode') == 'candidate' and descriptor.get('quota') == 64,
            'saved root player/stage')
    row = _read_bound(descriptor['source'], freeze, root=root, boundary=boundary)
    require(row.get('completed') is True and row.get('arm') == 'search' and
            row.get('player_role') == 'challenger' and row.get('placement_mode') == 'isolated' and
            row.get('observation_mode') == 'profiled', 'saved accepted isolated64 request')
    for field in ('stage', 'mode', 'quota', 'seat', 'game_index', 'pair_index', 'ply'):
        require(type(row.get(field)) is type(descriptor.get(field)) and row[field] == descriptor[field],
                'saved root metadata: '+field)
    before, after, observer = row['before'], row['after'], row['observer']
    state = guard.decode_training_snapshot(before)
    guard.decode_training_snapshot(after)
    ledger.validate_ledger(before)
    ledger.validate_ledger(after)
    ledger.check_observer(before, observer)
    require(same(rules.observation_payload(state), observer) and
            same(observer, descriptor['observer']), 'saved lawful observer binding')
    require(digest(before) == descriptor['before_sha256'] and
            digest(observer) == descriptor['observer_sha256'], 'saved snapshot/observation hashes')
    require(state.current_player == descriptor['seat'] and state.turn_index == descriptor['ply'],
            'saved actor/ply binding')
    require(state.turn_index < 124, 'saved Main heuristic cutoff is not searched')
    receipt = row['clock']
    original.check_clock(receipt)
    require(receipt['status'] == 'ok' and receipt['accepted'] is True and
            receipt['deadline_seconds'] == 1., 'original accepted one-second clock')
    result = receipt['result']
    original.validate_result(result, observer, 'search', expected_quota=64)
    require(result['profile_sha256'] == BASELINE_PROFILE_SHA and
            same(result['action'], row['action']) and same(result['diagnostics'], row['trace']),
            'saved scientific envelope coupling')
    original.audit_decision(observer, row['action'], row['trace'], 'search',
                            before, after, expected_quota=64)
    mains = bridge.main_actions(observer)
    require(len(mains) >= 2 and len(mains) == descriptor['main_legal_count'] and
            row['trace']['phases'][0]['heuristic_phase'] is False and
            row['trace']['phases'][0]['simulations'] == 64,
            'saved actual searched Main support/quota')
    clock_path = descriptor['source_worker_clock_file']
    require(row['worker_clock_file'] == clock_path, 'saved raw clock path')
    pins = {x['path']: x for x in freeze['inputs']}
    require(clock_path in pins, 'saved raw clock not frozen')
    raw_clock = _read_bound(pins[clock_path], freeze, root=root, boundary=boundary)
    require(same(raw_clock, receipt), 'saved clock-file/row identity')
    return dict(status='passed', root_id=descriptor['root_id'], observer=deepcopy(observer),
                original_result=deepcopy(result), source=descriptor['source'],
                source_worker_clock_file=clock_path,
                before_sha256=digest(before), observer_sha256=digest(observer),
                independent_saved_transition_checked=True, model_input_private_state=False)


def scientific_core(result):
    """Only declared identities/extra capture are excluded from exact parity."""
    value = deepcopy(result)
    value.pop('research', None)
    for key in ('profile_id', 'profile_sha256', 'wrapper_id'):
        value.pop(key, None)
    trace = value.get('diagnostics', {})
    trace.pop('profile_id', None)
    for phase in trace.get('phases', []):
        phase.pop('inference', None)
    return value


def validate_baseline_result(result, observer, expected_original=None):
    original.validate_result(result, observer, 'search', expected_quota=64)
    require(result['profile_sha256'] == BASELINE_PROFILE_SHA, 'frozen64 profile')
    if expected_original is not None:
        require(same(result, expected_original), 'frozen64 exact historical scientific reproduction')
    return dict(status='passed', exact_historical_reproduction=expected_original is not None)


def validate_callback_capture(phase, bit_records, *, require_capture=True):
    count = integer(phase.get('neural_callback_count'), 'callback count')
    require(type(phase.get('neural_callbacks')) is list and len(phase['neural_callbacks']) == count,
            'callback hash stream count')
    require(type(phase.get('neural_callback_rows')) is int and phase['neural_callback_rows'] == count,
            'batch-one row count')
    for value in phase['neural_callbacks']:
        sha(value, 'callback hash')
    require(type(bit_records) is list and len(bit_records) == count, 'callback bits inventory')
    for bits in bit_records:
        require(type(bits) is dict and set(bits) == {'input_float32_sha256', 'output_float32_sha256'},
                'callback bit fields')
        for value in bits.values():
            sha(value, 'float32 bit hash')
    capture = phase.get('inference')
    require(capture is not None or not require_capture, 'required callback capture missing')
    if capture is None:
        return None
    require(type(capture) is list and len(capture) == count, 'callback capture count')
    for index, (item, callback, bits) in enumerate(zip(capture, phase['neural_callbacks'], bit_records)):
        base = {'input_sha256', 'output_sha256', 'batch'}
        full = base | {'tokens', 'outputs', 'output83'}
        require(type(item) is dict and set(item) == (full if index == 0 else base),
                'bounded callback capture fields')
        require(type(item['batch']) is int and item['batch'] == 1, 'callback batch')
        sha(item['input_sha256'], 'callback input')
        sha(item['output_sha256'], 'callback output')
        require(callback == digest([item['input_sha256'], item['output_sha256'], 1]),
                'callback canonical digest coupling')
        if index == 0:
            tokens, outputs = item['tokens'], item['outputs']
            require(type(tokens) is list and len(tokens) == 31*48, 'root token shape')
            require(type(outputs) is list and len(outputs) == 1 and
                    type(outputs[0]) is list and len(outputs[0]) == 83, 'root output shape')
            for v in tokens+outputs[0]:
                finite(v, 'root float32 scalar')
                require(f32(v) == v, 'root scalar is not exactly float32')
            require(digest(tokens) == item['input_sha256'] and digest(outputs) == item['output_sha256'] and
                    same(item['output83'], outputs[0]), 'root arrays/hash coupling')
            require(float32_sha(tokens) == bits['input_float32_sha256'] and
                    float32_sha(outputs[0]) == bits['output_float32_sha256'], 'root bit hash coupling')
    return capture[0]['output83'] if capture else None


def initial_network_win_credit(raw83):
    """The acting-player head converted exactly as the native float64 code.

    This is the first callback's prediction, not SearchTargets.network_value:
    that upstream field reads the root node after dynamic-FPU backups.
    """
    require(type(raw83) is list and len(raw83) == 83, 'root callback value required')
    value = finite(raw83[81], 'initial signed network value', -1., 1.)
    require(f32(value) == value, 'initial network output float32')
    return (value + 1.) / 2.


def dynamic_fpu_rounding_bound(visits):
    """Absolute bound for two float32 reports and positive float64 arithmetic.

    The exact running mean is (initial + sum(backups))/(N+1). Both the
    reported all-edge mean and reported node mean are separately rounded to
    float32; each has at most 2**-24 absolute error on [0,1] (conservative).
    At most 6*N+16 binary64 operations cover the positive running-mean path,
    edge sums and their regrouping, and this reconstruction. gamma_k bounds
    their error; the values and convex means are all in [0,1]. This includes
    source arithmetic error rather than treating rounded Q as exact backups.
    """
    integer(visits, 'dynamic-FPU visits', 1, 1024)
    k_u = (6 * visits + 16) * 2.**-53
    return 2.**-23 + k_u / (1. - k_u)


def validate_root_statistics(telemetry, native_observer, choices, action, quota, raw83):
    require(type(telemetry) is dict and set(telemetry) == TELEMETRY_FIELDS, 'root statistic fields')
    visits, qs = telemetry['visits'], telemetry['q']
    require(type(visits) is list and type(qs) is list and len(visits) == len(qs) == 67,
            'canonical67 root arrays')
    safe = original.safe_choices(native_observer, choices)
    supported = {original.canonical_index(a) for a in safe}
    for i, (n, q) in enumerate(zip(visits, qs)):
        integer(n, 'edge visit count', 0, quota)
        require((q is None) == (n == 0), 'unvisited Q must be missing, visited Q present')
        require(i in supported or n == 0, 'visits outside safe legal support')
        if q is not None:
            finite(q, 'edge win credit', 0., 1.)
            require(f32(q) == q, 'edge Q float32')
    require(sum(visits) == quota, 'root visits equal completed quota')
    selected = original.canonical_index(action)
    require(selected in supported and visits[selected] > 0 and visits[selected] == max(visits),
            'selected root edge must attain maximum visits')
    for field in ('root_value', 'network_value', 'selected_edge_value'):
        finite(telemetry[field], field, 0., 1.)
        require(f32(telemetry[field]) == telemetry[field], field+' float32')
    require(telemetry['selected_edge_value'] == qs[selected], 'selected-edge value/Q')
    # Each reported Q and aggregate was separately rounded to float32. The
    # weighted reconstruction therefore has at most two float32 roundings.
    weighted = sum(n*q for n, q in zip(visits, qs) if q is not None)/quota
    require(abs(weighted-telemetry['root_value']) <= 2**-22, 'root visit-weighted value')
    require(native_observer['viewer'] == native_observer['current'], 'root acting-player perspective')
    initial = initial_network_win_credit(raw83)
    # The unchanged native source updates node.value on each dynamic-FPU
    # backup, then exports it under the historical name "network_value".
    expected_node = (initial + quota * telemetry['root_value']) / (quota + 1)
    bound = dynamic_fpu_rounding_bound(quota)
    residual = abs(telemetry['network_value'] - expected_node)
    require(residual <= bound, 'post-search dynamic-FPU node mean')
    return dict(selected_index=selected, visited_edges=sum(n>0 for n in visits),
                safe_edges=len(supported), missing_safe_edges=sum(visits[i] == 0 for i in supported),
                initial_network_win_credit=initial,
                post_search_root_node_value=telemetry['network_value'],
                dynamic_fpu_expected_node_mean=expected_node,
                dynamic_fpu_absolute_residual=residual,
                dynamic_fpu_absolute_rounding_bound=bound)


def validate_research_result(result, observer, *, quota, seed_label, profile_id,
                             profile_sha256, wrapper_id, native_binary_sha256,
                             telemetry_required=True, schema=RESEARCH_SCHEMA):
    """Full completed action/phase/root work audit, without model inference."""
    require(type(quota) is int and quota in QUOTAS, 'research quota')
    require(type(result) is dict and set(result) == original.RESULT_FIELDS | {'research'}, 'research result fields')
    state = rules.state_from_observation(observer)
    require(not state.terminal and same(rules.observation_payload(state), observer), 'strict lawful observer')
    legal = [rules.serialize_action(a) for a in rules.legal_actions(state)]
    require(result['player'] == 'search' and result['observer_sha256'] == digest(observer) and
            result['legal_actions_sha256'] == digest(legal) and type(result['legal_count']) is int and
            result['legal_count'] == len(legal), 'observer/legal envelope binding')
    require(result['ruleset'] == guard.RULESET_ID and result['catalogue_sha256'] == guard.CATALOGUE_SHA256 and
            result['turn_contract'] == guard.TURN_CONTRACT, 'corrected identity')
    require(type(result['simulations_quota']) is int and result['simulations_quota'] == quota and
            result['profile_id'] == profile_id and result['profile_sha256'] == profile_sha256 and
            result['wrapper_id'] == wrapper_id, 'externally bound profile/quota identity')
    sha(profile_sha256, 'profile'); sha(native_binary_sha256, 'native binary')
    require(any(same(result['action'], a) for a in legal), 'complete action legality')
    trace, research = result['diagnostics'], result['research']
    require(type(trace) is dict and set(trace) == TRACE_FIELDS, 'original scientific trace fields')
    fields = {'schema', 'quota', 'seed_label', 'seed', 'telemetry_enabled', 'internal_seconds',
              'native_binary_sha256', 'phases'}
    require(type(research) is dict and set(research) == fields, 'research annotation fields')
    seed = seed_for(observer, seed_label)
    require(research['schema'] == schema and type(research['quota']) is int and research['quota'] == quota and
            research['seed_label'] == seed_label and type(research['seed']) is int and research['seed'] == seed and
            type(research['telemetry_enabled']) is bool and research['telemetry_enabled'] is telemetry_required and
            research['internal_seconds'] == 29. and research['native_binary_sha256'] == native_binary_sha256,
            'research seed/clock/native identity')
    require(trace['opponent_id'] == 'splendorust' and trace['profile_id'] == profile_id and
            type(trace['observer_seed']) is int and trace['observer_seed'] == seed, 'scientific seed/profile')
    phases, extra = trace['phases'], research['phases']
    require(type(phases) is list and type(extra) is list and len(phases) == len(extra) <= 4,
            'finite phase inventory')
    mains = bridge.main_actions(observer)
    forced = mains == [{'kind': 'Pass'}]
    require(type(trace['forced_pass']) is bool and trace['forced_pass'] == forced, 'forced pass')
    cursor = 0; calls = leaves = simulations = 0; checks = []
    def select(native, choices):
        nonlocal cursor, calls, leaves, simulations
        require(cursor < len(phases), 'missing atomic phase')
        phase, annotation = phases[cursor], extra[cursor]
        require(type(phase) is dict and set(phase) == BASE_PHASE | {'inference'}, 'phase fields/capture')
        require(type(annotation) is dict and set(annotation) == {'phase_index', 'phase_kind', 'telemetry', 'callback_bits'} and
                type(annotation['phase_index']) is int and annotation['phase_index'] == cursor and
                annotation['phase_kind'] == native['phase']['kind'], 'research phase coupling')
        cursor += 1
        retained = original.safe_choices(native, choices)
        require(any(same(phase['action'], a) for a in retained), 'phase action outside safe support')
        require(type(phase['native_legal_count']) is int and phase['native_legal_count'] == len(choices),
                'native full legal count')
        heuristic = native['phase']['kind'] != 'Main' or len(choices) == 1 or native['turns'] >= 124
        require(type(phase['heuristic_phase']) is bool and phase['heuristic_phase'] == heuristic,
                'native shortcut classification')
        n = integer(phase['simulations'], 'phase simulations', 0, quota)
        leaf = integer(phase['leaf_calls'], 'leaf calls', 0, quota+1)
        count = integer(phase['neural_callback_count'], 'callbacks', 0, quota+1)
        require(n == (0 if heuristic else quota), 'actual fixed simulation quota')
        raw83 = validate_callback_capture(phase, annotation['callback_bits'])
        if heuristic:
            require(leaf == count == 0 and annotation['telemetry'] is None,
                    'heuristic phase has no search/telemetry')
        else:
            require(1 <= count <= leaf <= quota+1, 'searched leaf/callback bounds')
            if native['turns']+min(quota, 64) < 124:
                require(count == leaf, 'heuristic leaf unreachable')
            elif native['turns'] == 123:
                require(count == 1, 'turn123 root-only neural call')
            if telemetry_required:
                checks.append(validate_root_statistics(annotation['telemetry'], native, choices,
                                                       phase['action'], quota, raw83))
            else:
                require(annotation['telemetry'] is None, 'telemetry-off route must remain off')
        calls += count; leaves += leaf; simulations += n
        return phase['action']
    main = mains[0] if forced else select(bridge.encode_observation(observer), mains)
    resolved = rules.serialize_action(bridge.resolve_compound(observer, main, select))
    require(cursor == len(phases) and same(resolved, result['action']), 'complete atomic phase/action reconstruction')
    require(type(trace['inference_calls']) is int and trace['inference_calls'] == calls and
            type(trace['inference_rows']) is int and trace['inference_rows'] == calls, 'aggregate neural work')
    return dict(status='passed', main_action=deepcopy(main), phases=cursor, simulations=simulations,
                neural_callbacks=calls, leaf_calls=leaves, root_statistic_checks=checks)


def check_clock(receipt, *, limit):
    """Authenticate the original receipt without discarding rejected evidence.

    The trusted parent assigns ``result`` before invoking its validator. An
    ``illegal`` receipt therefore retains the rejected JSON value; a final
    supervision sample can likewise reject a decoded response. Neither route
    grants scientific acceptance. Raw-byte and child-hash binding is separate.
    """
    received = original.check_clock(receipt, limit=limit)
    require(receipt['deadline_seconds'] == limit, 'exact unshortened planned request deadline')
    post_validation_failures = {'resource_guard', 'process_tree_unobservable', 'process_tree_limit',
        'rss_limit', 'affinity_changed', 'log_limit', 'log_compression_error', 'message_flood'}
    statuses = {'ok', 'illegal', 'timeout', 'worker_error', 'worker_exit', 'malformed', 'wrong_request',
        'unsolicited', 'invalid_request', 'reader_error', 'write_error'} | post_validation_failures
    status = receipt['status']
    require(status in statuses and 'result' in receipt, 'original status/result receipt contract')
    if status == 'ok':
        require(receipt['result'] is not None, 'successful result absent')
    elif status == 'illegal':
        require(receipt['accepted'] is False and receipt['received_before_deadline'] is True and
                received is not None and 0 <= received <= limit and
                type(receipt.get('detail')) is str and 0 < len(receipt['detail']) <= 4096,
                'rejected response requires timely original validator receipt')
        # JSON null is also a legitimate rejected payload. Retain it exactly.
    elif status in post_validation_failures and receipt['result'] is not None:
        require(receipt['accepted'] is False and receipt['received_before_deadline'] is True and
                received is not None and 0 <= received <= limit,
                'retained supervision-rejected response requires timely original receipt')
    else:
        require(receipt['result'] is None, 'status cannot retain an undecoded or absent response')
    if status == 'timeout':
        require(receipt['received_before_deadline'] is False, 'timeout cannot have a timely accepted arrival')
    return received


def failure_kind(receipt):
    status = receipt['status']; detail = str(receipt.get('detail', '')).lower()
    if status == 'timeout' or 'internal_deadline' in detail or 'deadline' in detail:
        return 'operational_request_deadline_failure'
    return 'request_failure_requires_diagnosis'


def profile_identity(freeze, quota, *, root=ROOT, boundary=lambda: None):
    require(type(quota) is int and quota in QUOTAS, 'profile quota')
    item = freeze['research_profiles'][str(quota)]
    profile = _read_bound(item, freeze, root=root, boundary=boundary) if set(item) == {'path','bytes','sha256'} else item
    profile_id = profile['profile_id']
    require(profile['iterations'] == quota and type(profile['iterations']) is int and
            profile_id == f'research_puct{quota}_saved_diagnostic_20261027_v1', 'profile quota/id contract')
    return dict(profile_id=profile_id,
                profile_sha256=item['sha256'] if set(item) == {'path','bytes','sha256'} else profile['profile_sha256'],
                wrapper_id=profile.get('wrapper_id', profile_id+'_30s'),
                native_binary_sha256=profile.get('native_binary_sha256', freeze.get('research_native_sha256')))


def audit_record(row, spec, freeze, *, saved_root, root=ROOT, boundary=lambda: None):
    """Validate one prospective ordinary row; low-level transport proofs are separate."""
    boundary()
    require(type(row) is dict and same(row.get('spec'), spec), 'exact prospective request identity')
    require(same(row.get('observer'), saved_root['observer']) and
            digest(row['observer']) == spec['observer_sha256'], 'request lawful observer')
    clock = row['clock']; completed = row['completed']
    require(type(completed) is bool and completed == (clock['status'] == 'ok'), 'completion/clock')
    check_clock(clock, limit=spec['complete_response_seconds'])
    require(spec['seed_decimal'] == str(seed_for(row['observer'], spec['seed_family'])), 'prospective seed binding')
    require(not row.get('profile_capture_error') and not row.get('placement_error'),
            'infrastructure evidence capture failure')
    if not completed:
        return dict(status='preserved_request_failure', sequence=spec['sequence'],
                    failure_kind=failure_kind(clock), complete=False)
    require(clock['deadline_seconds'] == spec['complete_response_seconds'], 'complete request full planned clock')
    result = clock['result']
    for key, expected in (('result', result), ('action', result['action']), ('trace', result['diagnostics'])):
        if key in row:
            require(same(row[key], expected), 'saved row/result coupling: '+key)
    if spec['stage'] == 'frozen_reference_reproduction':
        checks = validate_baseline_result(result, row['observer'], saved_root['original_result'])
    else:
        identity = profile_identity(freeze, spec['quota'], root=root, boundary=boundary)
        checks = validate_research_result(result, row['observer'], quota=spec['quota'],
                                          seed_label=spec['seed_family'], **identity)
        require(not result['diagnostics']['forced_pass'] and
                result['diagnostics']['phases'][0]['heuristic_phase'] is False,
                'selected saved Main must actually search')
        if spec['quota'] == 64 and spec['seed_family'] == 'A':
            require(same(scientific_core(result), scientific_core(saved_root['original_result'])),
                    'Research64-A exact frozen scientific trajectory')
    return dict(status='passed', sequence=spec['sequence'], complete=True, science=checks)


def audit_prefix(records, freeze, matrix, panel, *, saved_roots=None, root=ROOT,
                 boundary=lambda: None, allow_partial=False):
    """Require an exact ordered prefix, with no failed-request replay/replacement."""
    specs = matrix['requests']; descriptors = {r['root_id']:r for r in panel['roots']}
    require(len(specs) == 156 and [s['sequence'] for s in specs] == list(range(156)), 'fixed156 request inventory')
    require(type(records) is list and len(records) <= len(specs), 'bounded observed prefix')
    loaded = {} if saved_roots is None else saved_roots
    seen_requests = set(); seen_paths = set(); originals = {}; checks = []; failed = None
    for i, row in enumerate(records):
        boundary(); spec = specs[i]
        require(failed is None, 'request after first failure')
        rid = spec['root_id']
        if rid not in loaded:
            loaded[rid] = load_saved_root(descriptors[rid], freeze, root=root, boundary=boundary)
        # The worker path disambiguates persistent-worker local request IDs.
        identity = (row.get('worker_clock_file'), row['clock'].get('request_id'))
        require(identity[0] is not None and identity not in seen_requests and identity[0] not in seen_paths,
                'request inventory duplicated/unbound')
        seen_requests.add(identity)
        seen_paths.add(identity[0])
        check = audit_record(row, spec, freeze, saved_root=loaded[rid], root=root, boundary=boundary)
        checks.append(check)
        if not row['completed']:
            failed = dict(sequence=i, root_id=rid, status=row['clock']['status'],
                          category=check['failure_kind'], no_retry=True)
        elif spec['stage'] == 'research_reproduction':
            key = (rid, spec['quota'], spec['seed_family'])
            require(key in originals and same(row['clock']['result'], originals[key]),
                    'exact full research reproduction')
        elif spec['stage'] in ('research_pilot', 'research_main'):
            originals[(rid, spec['quota'], spec['seed_family'])] = deepcopy(row['clock']['result'])
    full = len(records) == len(specs) and failed is None
    require(full or allow_partial, 'complete diagnostic required')
    counts = Counter(row['spec']['stage'] for row in records if row['completed'])
    if full:
        require(dict(counts) == {'frozen_reference_reproduction':24, 'research_pilot':30,
                                'research_main':90, 'research_reproduction':12}, 'complete coverage counts')
    return dict(status='passed_complete_science' if full else 'passed_science_prefix',
                complete=full, planned_requests=156, observed_requests=len(records),
                completed_requests=sum(r['completed'] for r in records), counts=dict(counts),
                preserved_failure=failed, all_seen_requests_unique=True,
                strength_or_true_quality_claim=False, checks=checks)


def verify_saved_record(row, freeze, *, matrix=None, panel=None, saved_root=None,
                        root=ROOT, boundary=lambda: None):
    """Runner entry point; the saved source is checked on the actual audit route."""
    spec = row['spec']
    if matrix is not None:
        require(same(matrix['requests'][spec['sequence']], spec), 'row differs from frozen matrix')
    if saved_root is None:
        require(panel is not None, 'saved root or prepared panel required')
        matches = [r for r in panel['roots'] if r['root_id'] == spec['root_id']]
        require(len(matches) == 1, 'exactly one frozen saved root')
        saved_root = load_saved_root(matches[0], freeze, root=root, boundary=boundary)
    return audit_record(row, spec, freeze, saved_root=saved_root, root=root, boundary=boundary)


def finalize(freeze, records, matrix, panel, *, boundary=lambda: None, allow_partial=False,
             saved_roots=None, root=ROOT):
    """Science component only; owning runner must also bind transport/source proofs."""
    from . import analysis
    coverage = audit_prefix(records, freeze, matrix, panel, saved_roots=saved_roots,
                            root=root, boundary=boundary, allow_partial=allow_partial)
    summary = analysis.analyze(records, panel, matrix, complete=coverage['complete'])
    return dict(status='passed_saved_science_component', coverage=coverage, analysis=summary,
                external_transport_runtime_placement_seal_and_cleanup_acceptance_required=True)
