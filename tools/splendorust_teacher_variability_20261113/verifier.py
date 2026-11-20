"""Independent saved-result audit: exact inventory, clocks, profiles and search.

This module loads no neural model and issues no decisions. Native canonical67
coordinates remain unchanged, including unused channels and missing edge Q.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from . import adapters, panel as panel_module, qualification
from tools.splendorust_decision_quality_20261027_v2 import audit as clock_audit
from tools.splendorust_search_scale_20261024 import audit as native_audit

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'teacher_variability_saved_result_verification_v1'
MAX_JSON_BYTES = 512 * 1024**2
require, digest, canonical = panel_module.require, panel_module.digest, panel_module.canonical


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate saved JSON key'); result[key] = value
        return result
    def constant(value):
        raise ValueError('PROXY_PANEL: nonfinite saved JSON constant: ' + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=constant)


def _read_path(path):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), 'saved audit path within workspace')
    before = path.stat()
    require(0 < before.st_size <= MAX_JSON_BYTES, 'bounded saved JSON artifact')
    raw = path.read_bytes(); after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), 'stable saved artifact read')
    item = dict(path=path.relative_to(ROOT).as_posix(), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    return _json(raw), item


def _read_bound(item):
    require(type(item) is dict and set(item) == {'path', 'bytes', 'sha256'}, 'exact artifact descriptor')
    require(type(item['path']) is str and not Path(item['path']).is_absolute(), 'workspace-relative artifact path')
    value, actual = _read_path(ROOT / item['path'])
    require(actual == item, 'saved artifact physical hash, size and canonical path')
    return value


def _freeze(folder, freeze):
    if freeze is not None:
        require(type(freeze) is dict, 'explicit frozen plan object')
        return freeze
    for parent in (Path(folder).resolve(), *Path(folder).resolve().parents):
        candidate = parent / 'freeze.json'
        if parent.is_relative_to(ROOT) and candidate.is_file():
            return _read_path(candidate)[0]
    raise ValueError('PROXY_PANEL: explicit freeze or ancestor freeze.json required')


def _profiles(freeze, *, boundary):
    artifacts = freeze['research_profiles']
    expected = {adapters.profile_key(q, pool=p, mode=m)
                for q in adapters.ALL_QUOTAS for p in (3, 6) for m in ('legacy', 'split')}
    require(type(artifacts) is dict and set(artifacts) == expected, 'complete frozen mode/pool/quota profiles')
    result = {}
    for quota in adapters.QUOTAS:
        for pool in (3, 6):
            for mode in ('legacy', 'split'):
                boundary(); key = adapters.profile_key(quota, pool=pool, mode=mode)
                item = artifacts[key]; document = _read_bound(item)
                require(document == adapters.profile(quota, pool=pool, mode=mode),
                        'frozen profile equals authenticated diagnostic construction')
                result[key] = (document, deepcopy(item))
    return result


def _index(folder, request_specs, *, boundary):
    folder = Path(folder).resolve()
    require(folder.is_relative_to(ROOT / 'outputs'), 'worker evidence folder within outputs')
    boundary(); index, item = _read_path(folder / 'index.json')
    require(type(index) is dict and type(index.get('requests')) is list and
            type(request_specs) is list and bool(request_specs), 'explicit nonempty saved request inventory')
    rows = index['requests']
    require(len(rows) == len(request_specs), 'exact request count; no skips or extras')
    planned_ids, receipt_paths = set(), set()
    for i, (row, spec) in enumerate(zip(rows, request_specs, strict=True)):
        boundary()
        require(type(row) is dict and set(row) == {'spec', 'receipt'} and row['spec'] == spec, 'exact ordered prospective request specification')
        require(spec['request_id'] not in planned_ids, 'unique prospective request identity')
        planned_ids.add(spec['request_id'])
        expected_path = folder / 'transport' / f'request-{i+2:08d}.json'
        require(type(row['receipt']) is dict and row['receipt'].get('path') == expected_path.relative_to(ROOT).as_posix(),
                'original contiguous transport sequence after graph qualification')
        require(row['receipt']['path'] not in receipt_paths, 'unique physical request receipt')
        receipt_paths.add(row['receipt']['path'])
    expected_names = {f'request-{i+1:08d}.json' for i in range(len(rows)+1)}
    actual_names = {p.name for p in (folder / 'transport').glob('request-*.json')}
    require(actual_names == expected_names, 'transport contains only graph request and planned ordinary requests')
    return rows, item


def _root_lookup(panel):
    roots = panel['roots']
    require(type(roots) is list and len({r['root_id'] for r in roots}) == len(roots), 'unique saved root inventory')
    result = {r['root_id']: r for r in roots}
    require(set(result) == set(panel['host_metadata']), 'complete host metadata join')
    for root in roots:
        require(set(root) == {'root_id', 'observer', 'observer_sha256'} and
                digest(root['observer']) == root['observer_sha256'], 'lawful payload hash binding')
    return result


def _verify_one(row, root, metadata, profiles, *, boundary):
    boundary(); spec = row['spec']; quota = spec['quota']; pool = spec['pool']; mode = spec['mode']
    world = spec['world_seed_label']; encoding = spec['encoding_seed_label']
    require(type(quota) is int and quota in (512, 2048) and type(pool) is int and pool in (3, 6),
            'ordinary fixed quota and world pool')
    require(mode in ('legacy', 'split') and world in ('A', 'B', 'C', 'D') and encoding in ('A', 'B', 'C', 'D'),
            'ordinary fixed mode and lawful seed labels')
    require((mode == 'legacy' and world == encoding and spec['replicate'] == world+'_'+encoding) or
            (mode == 'split' and world in ('A', 'B') and encoding in ('A', 'B') and spec['replicate'] == world+'_'+encoding),
            'exact prospective coupled or crossed seed condition')
    require(spec['root_id'] == root['root_id'], 'requested lawful root identity')
    receipt = _read_bound(row['receipt'])
    require(re.fullmatch(r'request-[0-9]{8}', receipt.get('request_id', '')) is not None and
            Path(row['receipt']['path']).name == receipt['request_id'] + '.json', 'clock request and physical receipt identity')
    elapsed = clock_audit.check_clock(receipt, limit=adapters.COMPLETE_SECONDS)
    require(receipt['status'] == 'ok' and receipt['accepted'] is True and receipt['received_before_deadline'] is True,
            'every planned request completed successfully within original allowance')
    key = adapters.profile_key(quota, pool=pool, mode=mode)
    result = receipt['result']; profile, item = profiles[key]
    checks = qualification.validate_result(result, root['observer'], quota=quota, world_seed_label=world,
        pool=pool, mode=mode, encoding_seed_label=encoding,
        profile_id=profile['profile_id'], profile_sha256=item['sha256'], wrapper_id=profile['wrapper_id'],
        native_binary_sha256=profile['native_binary_sha256'], telemetry_required=True, player='search')
    require(result['player'] == 'search', 'qualification-only original native control cannot enter ordinary diagnostic')
    phases = result['research']['phases']; scientific = result['diagnostics']['phases']
    require(len(phases) >= 1 and phases[0]['phase_index'] == 0 and phases[0]['phase_kind'] == 'Main' and
            scientific[0]['heuristic_phase'] is False and result['diagnostics']['forced_pass'] is False and
            len(checks['root_statistic_checks']) == 1, 'exactly one genuinely searched Main root')
    telemetry = phases[0]['telemetry']; action_index = native_audit.canonical_index(checks['main_action'])
    root_check = checks['root_statistic_checks'][0]
    require(root_check['selected_index'] == action_index, 'independently reconstructed Main choice')
    record = dict(request_id=spec['request_id'], root_id=spec['root_id'], game_index=metadata['game_index'],
        split=metadata['split'], quota=quota, pool=pool, mode=mode, replicate=spec['replicate'],
        world_seed_label=world, encoding_seed_label=encoding, main_action_index67=action_index,
        visits67=deepcopy(telemetry['visits']), q67=deepcopy(telemetry['q']),
        root_value=telemetry['root_value'], root_network_value=root_check['initial_network_win_credit'],
        elapsed_seconds=elapsed, result_sha256=digest(result),
        observer_sha256=root['observer_sha256'], receipt=deepcopy(row['receipt']))
    proof = dict(request_id=spec['request_id'], receipt=deepcopy(row['receipt']), result_sha256=record['result_sha256'],
        original_transport_request_id=receipt['request_id'], profile=deepcopy(item), scientific_checks_sha256=digest(checks))
    return record, proof, result


def verify_forensics(folder, requests, *, boundary=lambda: None):
    """Stream every pre-validation completed result and bind its original receipt.

    A failure event, missing event, extra event or changed numerical field fails
    acceptance. The raw stream remains preserved even when a receipt is rejected.
    """
    folder = Path(folder).resolve(); path = folder / 'service' / 'ordinary_forensics.jsonl'
    require(folder.is_relative_to(ROOT / 'outputs'), 'forensic worker folder within outputs')
    require(type(requests) is list and bool(requests), 'nonempty exact forensic request inventory')
    before = path.stat(); require(0 < before.st_size <= 8 * 1024**3, 'finite forensic file bound')
    previous = None; count = 0; physical = hashlib.sha256(); checks = []
    with path.open('rb') as stream:
        while True:
            boundary(); raw = stream.readline(16 * 1024**2 + 1)
            if not raw: break
            require(len(raw) <= 16 * 1024**2 and raw.endswith(b'\n'), 'complete bounded forensic line')
            physical.update(raw); chain = _json(raw)
            require(count < len(requests) and type(chain) is dict and set(chain) == {'sequence', 'previous_sha256', 'event'} and
                    type(chain['sequence']) is int and chain['sequence'] == count and chain['previous_sha256'] == previous,
                    'exact forensic count, order and hash chain')
            wrapper = chain['event']
            require(type(wrapper) is dict and set(wrapper) == {'transport_request_id', 'event'}, 'exact forensic transport wrapper')
            event = wrapper['event']; row = requests[count]; receipt = _read_bound(row['receipt'])
            require(receipt['status'] == 'ok' and receipt['accepted'] is True and receipt['received_before_deadline'] is True,
                    'forensic acceptance cannot promote a rejected original receipt')
            require(wrapper['transport_request_id'] == receipt['request_id'], 'forensic event binds exact original transport request')
            require(type(event) is dict and set(event) == {'event', 'arm', 'observer', 'action', 'diagnostics', 'research'} and
                    event['event'] == 'complete_result_before_worker_validation', 'only successful raw completed results; no failed events')
            result = receipt['result']
            require(event['arm'] == result['player'] and digest(event['observer']) == result['observer_sha256'] and
                    all(event[k] == result[k] for k in ('action', 'diagnostics', 'research')),
                    'raw pre-validation lawful observer and complete scientific response equal original receipt')
            previous = digest(chain); count += 1
            checks.append(dict(request_id=row['spec']['request_id'], original_transport_request_id=receipt['request_id'],
                               event_sha256=digest(event), result_sha256=digest(result)))
    after = path.stat()
    require(count == len(requests), 'no missing or silently discarded forensic decision')
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), 'stable complete forensic evidence')
    item = dict(path=path.relative_to(ROOT).as_posix(), bytes=after.st_size, sha256=physical.hexdigest())
    return dict(status='passed', artifact=item, events=count, terminal_chain_sha256=previous,
                exact_prevalidation_response_binding=True, requests=checks, model_calls=0, search_calls=0)


def verify_records(folder, panel, request_specs, *, boundary=lambda: None, freeze=None):
    """Return (normalized native67 records, independent saved-evidence proof)."""
    frozen = _freeze(folder, freeze); profiles = _profiles(frozen, boundary=boundary)
    rows, index = _index(folder, request_specs, boundary=boundary); roots = _root_lookup(panel)
    records, checks = [], []; sampling_groups = {}
    for row in rows:
        rid = row['spec']['root_id']; require(rid in roots, 'only prospectively selected saved roots')
        normalized, check, raw_result = _verify_one(row, roots[rid], panel['host_metadata'][rid], profiles, boundary=boundary)
        records.append(normalized); checks.append(check)
        spec=row['spec'];identity=('search',spec['quota'],spec['pool'],spec['mode'],spec['world_seed_label'],spec['encoding_seed_label'],True)
        group=sampling_groups.setdefault(rid,{})
        require(identity not in group,'unique per-root crossed sampling cell')
        first=raw_result['research']['phases'][0]
        group[identity]={'research':{'phases':[{'sampling':deepcopy(first['sampling']),
            'callback_bits':[deepcopy(first['callback_bits'][0])]}]}}
    sampling_checks=[]
    for rid,group in sampling_groups.items():
        boundary()
        expected_quotas=(512,2048) if rid in panel['deep_root_ids'] else (512,)
        expected={('search',q,p,m,w,e,True) for q in expected_quotas for p,m,w,e in panel_module.CONDITIONS}
        require(set(group)==expected,'every root has all prospective crossed and legacy cells')
        sampling_checks.append(dict(root_id=rid,checks=qualification.validate_crossed_sampling(group)))
    forensic = verify_forensics(folder, rows, boundary=boundary)
    proof = dict(schema=SCHEMA, status='passed', index=index, requests=len(records), ordered_specs_sha256=digest(request_specs),
        panel_sha256=digest(panel), normalized_records_sha256=digest(records), request_checks=checks,
        prevalidation_forensics=forensic, all_root_sampling_checks=sampling_checks,
        profiles={str(q): item for q, (_, item) in profiles.items()}, complete_response_seconds=adapters.COMPLETE_SECONDS,
        raw_search_and_callback_evidence_checked=True, native_action_coordinates=67, no_skipped_or_replaced_requests=True,
        model_calls=0, new_games=0, search_calls=0)
    return records, proof


def verify_reproductions(original_folder, repeat_folder, panel, *, boundary=lambda: None, freeze=None):
    """Verify all32 repeats and the corresponding originals, comparing full results.

    Outer arrival clocks naturally differ. Every returned scientific field,
    float32 callback hash and complete atomic action must remain identical.
    """
    frozen = _freeze(original_folder, freeze); profiles = _profiles(frozen, boundary=boundary)
    main_specs = panel_module.request_matrix(panel); repeated_specs = panel_module.reproduction_matrix(panel)
    original_rows, original_index = _index(original_folder, main_specs, boundary=boundary)
    repeated_rows, repeated_index = _index(repeat_folder, repeated_specs, boundary=boundary)
    require(len(repeated_rows) == 32 and len({x['spec']['root_id'] for x in repeated_rows}) == 16, 'fixed32 exact repetitions on16 roots')
    originals = {x['spec']['request_id']: x for x in original_rows}; roots = _root_lookup(panel); checks = []
    for repeat in repeated_rows:
        spec = repeat['spec']; rid = spec['root_id']; original = originals[spec['original_request_id']]
        require({k: v for k, v in spec.items() if k not in ('request_id', 'original_request_id')} ==
                {k: v for k, v in original['spec'].items() if k != 'request_id'}, 'identical reproduced experiment condition')
        _, left_proof, left = _verify_one(original, roots[rid], panel['host_metadata'][rid], profiles, boundary=boundary)
        _, right_proof, right = _verify_one(repeat, roots[rid], panel['host_metadata'][rid], profiles, boundary=boundary)
        require(canonical(left) == canonical(right), 'exact full scientific result reproduction; no tolerances')
        checks.append(dict(original=left_proof, repetition=right_proof, full_result_exact=True))
    forensic = verify_forensics(repeat_folder, repeated_rows, boundary=boundary)
    return dict(schema='teacher_variability_exact_reproductions_v1', status='passed', repetitions=32, roots=16,
        original_index=original_index, repeated_index=repeated_index, checks=checks,
        repeated_prevalidation_forensics=forensic, original_full_forensics_checked_by_verify_records=True,
        normalized_result_fields_omitted=False, outer_clocks_excluded=True, model_calls=0, search_calls=0, new_games=0)
