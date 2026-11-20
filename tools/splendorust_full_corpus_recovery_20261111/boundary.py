"""Saved-only authentication of the sole disclosed, potentially interrupted move.

The historical attempt remains failed.  Its missing request-70 receipt is never
interpreted as proof that dispatch did not happen.  Only the explicit amendment
permits one new allowance, durably consumed before entering the live transport.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zlib

import psutil

from . import common as c

FAILED_SEGMENT = 'main_block_terminal_parent_r1_g030'
PARTIAL = 'strength/terminal_parent/r1/games/main/g032'
WORKERS = 'strength/terminal_parent/r1/workers'
AUTH_SCHEMA = 'full_corpus_single_disclosed_retry_authorization_v1'
CONSUMPTION_SCHEMA = 'full_corpus_single_disclosed_retry_consumed_v1'
HELPER_SECONDS = 20 * 60
HELPER_RSS = 8 * 1024**3
PINNED = {
    'freeze.json': '30b715c249b34e6dcae108c5a4633ada93b566b6d9af73f30bd40cc82059ff74',
    'claim.json': '1b97f790dcac66b64f26e1c8ccd41f111d39cf883236d4b96da2a116dd6b6e73',
    'execution.json': '76f4ad74ff5bef4a2bae452b351c61ac21d1afcf05bc1c0215e23d4824af0c73',
    'failure.json': '562873502f861baae3f24da59ec7bbee91821743a92dcd9ff641efb1a4c8c168',
    'stages/strength_start.json': 'd64c505b2f9b6b9eef156d142201a7f0b6523f6c4ccc4560ad4a1ad62d4b59a2',
    'segments/' + FAILED_SEGMENT + '/execution.json': '68c9e655195b893f094a0777530b1597d1fbccb49594a471bd9f93965e8881f6',
    PARTIAL + '/moves/000021.json': '454028a1b84af00acb61ed716bd373b3cca66a461b391f7e6659b89e822dd65b',
    WORKERS + '/main_g030_control/inputs/request-00000070.json': 'd339cea5df3725749d38e5fc8aa5a2da8ca139f0599092e695d8fa726e20b75d',
    WORKERS + '/main_g030_control/stdout.log.gz': 'fd38724fa2bea8292057526c101ec8fd9056f1b9e4b330fea9304b37184a6b07',
    WORKERS + '/main_g030_control/events.jsonl.gz': 'c91ae84059a10a93886375772788b29cdebb68e28b860051faee260f5e52ddbb',
    WORKERS + '/main_g030_candidate/stdout.log.gz': '63fe0b219a003b4ce8ef15edfc2b2a6bac147f923ee0927f101fa76b4cf009cd',
    WORKERS + '/main_g030_candidate/events.jsonl.gz': '3935e6e2c281900c565219bec11dd9ba2dc983d9de3441733a626a8527b4981e',
}


def original(path):
    return c.ORIGINAL_OUT / path


def _pinned(boundary=lambda: None):
    result = []
    for relative, expected in PINNED.items():
        boundary()
        item = c.artifact(original(relative))
        c.require(item['sha256'] == expected, 'Exact preserved failure boundary: ' + relative)
        result.append(item)
    return result


def validate_retry_authorization(value=None, boundary=lambda: None):
    value = c.read(c.AUTHORIZATION) if value is None else value
    c.require(type(value) is dict and value.get('schema') == AUTH_SCHEMA,
              'Explicit single disclosed retry authorization schema')
    c.require(c.canonical(value.get('retry_spec')) == c.canonical(c.RETRY_SPEC),
              'Only the exact one-move retry is authorized')
    c.require(value.get('original_freeze') == c.artifact(original('freeze.json')) and
              value['original_freeze']['sha256'] == c.ORIGINAL_PLAN_SHA,
              'Authorization binds the original failed study freeze')
    c.require(type(value.get('user_reply')) is str and bool(value['user_reply'].strip()),
              'User reply is retained verbatim in the amendment')
    c.require(value.get('original_attempt_remains_incomplete') is True and
              all(type(value.get(key)) is int and value[key] == 0
                  for key in ('deadline_extension_seconds','new_training_calls','replacement_games')),
              'The amendment authorizes no clock extension, training, or replacement game')
    c.require(type(value.get('inspection')) is dict, 'Prior ambiguity inspection is retained')
    c.verify([value['original_freeze'], value['inspection']], boundary=boundary)
    inspection = c.read(c.resolve(value['inspection']))
    c.require(type(inspection.get('root_actual_exit_code')) is int and inspection['root_actual_exit_code'] == 1 and
              inspection.get('interrupted_game') == PARTIAL and
              inspection.get('completed_plies') == 22 and inspection.get('next_ply') == 22 and
              inspection.get('ambiguous_worker') == 'main_g030_control' and
              inspection.get('ambiguous_request') == 70,
              'Authorization preserves the actual unresolved dispatch boundary')
    return value


def retry_beforestate():
    """Authenticated state; decoding checks all corrected catalogue faces."""
    from tools.future_training_rules_20261021 import decode_training_snapshot
    from tools.external_strength_screen_20261018 import rules
    last_path = original(PARTIAL + '/moves/000021.json')
    input_path = original(WORKERS + '/main_g030_control/inputs/request-00000070.json')
    for path in (last_path, input_path):
        relative = path.relative_to(c.ORIGINAL_OUT).as_posix()
        c.require(c.sha(path) == PINNED[relative], 'Original last move and ambiguous input bytes')
    last = c.read(last_path)
    c.require(last.get('ply') == 21 and last.get('completed') is True and last['clock']['status'] == 'ok',
              'The retry follows the last completed saved move')
    state = decode_training_snapshot(last['after'])
    c.require(not state.terminal, 'No retry after an official terminal state')
    payload = dict(player='search', observer=rules.observation_payload(state))
    c.require(c.read(input_path) == payload, 'Retry observer is exactly the saved ambiguous input')
    spec = c.read(original(PARTIAL + '/spec.json'))
    c.require(state.current_player != spec['challenger_seat'], 'Only original control turn may be retried')
    return last['after']


def _prefix_inventory():
    total = 0
    for matchup, repetitions in c.COMPLETED_PREFIXES.items():
        for rep, count in repetitions.items():
            base = original(f'strength/{matchup}/r{rep}/games')
            main = base / 'main'
            expected = {f'g{i:03d}' for i in range(count)}
            partial = matchup == 'terminal_parent' and rep == 1
            c.require({p.name for p in main.iterdir()} == expected | ({'g032'} if partial else set()),
                      'Exact original main-game directories, no hidden replacement')
            c.require({p.parent.name for p in main.glob('g*/result.json')} == expected,
                      'Exact original completed main-game prefix')
            c.require({p.name for p in (base/'operational').iterdir()} == {f'g{i:03d}' for i in range(6)},
                      'All six original operational games retained')
            for stage, limit in (('main', count), ('operational', 6)):
                for index in range(limit):
                    folder = base/stage/f'g{index:03d}'
                    result = c.read(folder/'result.json')
                    c.require(result.get('complete') is True and result.get('reason') == 'natural' and
                              type(result.get('game_index')) is int and result['game_index'] == index and
                              result.get('stage') == stage and result.get('matchup') == matchup and
                              type(result.get('repetition')) is int and result['repetition'] == rep and
                              (folder/'online_replay.json').is_file() and (folder/'cost.json').is_file(),
                              'Original complete game identity and verification receipts')
            total += count
    partial = original(PARTIAL)
    c.require({p.name for p in partial.iterdir()} == {'spec.json','initial.json','workers.json','moves'},
              'Partial game has no invented result, extra file, or replay')
    c.require({p.name for p in (partial/'moves').iterdir()} == {f'{i:06d}.json' for i in range(22)},
              'Exactly the immutable 22-move prefix')
    c.require(c.read(partial/'workers.json') == {'control':'main_g030_control','candidate':'main_g030_candidate'},
              'Original failed worker pair')
    c.require(total == 212, 'Exactly 212 original complete main games')
    return total


def check_source_boundary(boundary=lambda: None, deep=False):
    from tools.splendorust_gpu_probe_20261105.control import require_absent
    pins = _pinned(boundary)
    validate_retry_authorization(boundary=boundary)
    old = c.read(original('claim.json'))
    execution = c.read(original('execution.json'))
    failed = c.read(original('segments/'+FAILED_SEGMENT+'/execution.json'))
    c.require(execution['status'] == 'failed' and execution['original_claim'] == c.artifact(original('claim.json')) and
              execution['freeze_sha256'] == c.ORIGINAL_PLAN_SHA and old['freeze'] == c.artifact(original('freeze.json')),
              'Original failure remains failed with its exact owner and freeze')
    c.require(failed['status'] == 'failed' and failed['worker_exit_code'] == 0 and
              failed['cleanup_errors'] == ["ValueError('Exact previous process still alive')"] and
              'Actual free disk floor' in failed['error'] and not original('segments/'+FAILED_SEGMENT+'/result.json').exists(),
              'Preserve the disk stop, missing segment result, and original cleanup error')
    identities = {(old['pid'], old['created']): dict(pid=old['pid'], created=old['created'])}
    c.require(len(execution['segments']) == 42, 'Exact 42 completed stages before the failed block')
    c.verify(execution['segments'], boundary=boundary)
    for receipt in [*[c.read(c.resolve(item)) for item in execution['segments']], failed]:
        boundary()
        if receipt is not failed:
            c.require(receipt['status'] == 'passed' and receipt['worker_exit_code'] == 0 and
                      receipt['error'] is None and receipt['cleanup_errors'] == [],
                      'Every completed original stage has an actual clean exit')
        for identity in receipt['observed_owned_identities'] + [receipt['identity']]:
            identities[(identity['pid'],identity['created'])] = identity
    for identity in identities.values():
        boundary(); require_absent(identity)
    count = _prefix_inventory()
    before = retry_beforestate()
    result = dict(schema='full_corpus_saved_retry_boundary_v1', status='preserved_ambiguous_dispatch',
        original_claim=c.artifact(original('claim.json')), original_freeze=c.artifact(original('freeze.json')),
        original_execution=c.artifact(original('execution.json')), original_failure=c.artifact(original('failure.json')),
        failed_execution=c.artifact(original('segments/'+FAILED_SEGMENT+'/execution.json')),
        original_strength_clock=c.artifact(original('stages/strength_start.json')),
        pinned_evidence=pins, original_owned_identities_checked=len(identities), completed_main_games=count,
        completed_operational_games=36, completed_prefixes=c.COMPLETED_PREFIXES,
        completed_prefix_plies=22, retry_spec=deepcopy(c.RETRY_SPEC), before_sha256=c.digest(before),
        dispatch_never_started_proven=False, original_attempt_stays_incomplete=True,
        last_saved_move=c.artifact(original(PARTIAL+'/moves/000021.json')),
        ambiguous_input=c.artifact(original(WORKERS+'/main_g030_control/inputs/request-00000070.json')))
    if deep:
        result['saved_prefix_audit'] = run_saved_audit(boundary)
        c.require(_pinned(boundary) == pins, 'Original failure bytes unchanged by saved-only audit')
    return result


def _read_unclosed_stream(path, expected_sha, boundary=lambda: None):
    """Only a pinned failed stream: no footer fabrication or source rewriting."""
    path = Path(path)
    c.require(path.stat().st_size <= 16*c.MiB and c.sha(path) == expected_sha, 'Pinned bounded failed gzip bytes')
    decoder = zlib.decompressobj(16+zlib.MAX_WBITS)
    raw = bytearray()
    with path.open('rb') as stream:
        while chunk := stream.read(65536):
            boundary()
            raw.extend(decoder.decompress(chunk, 32*c.MiB-len(raw)+1))
            c.require(len(raw) <= 32*c.MiB and not decoder.unconsumed_tail,
                      'Bounded readable failed stream prefix')
    c.require(not decoder.eof and not decoder.unused_data and raw.endswith(b'\n'),
              'Exactly the known unclosed gzip with whole JSONL records')
    lines = bytes(raw).splitlines(keepends=True)
    for line in lines:
        c.require(len(line) <= 8*c.MiB and line.endswith(b'\n'), 'Complete bounded saved raw frame')
        json.loads(line, object_pairs_hook=c._unique,
                   parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite raw JSON')))
    return lines, dict(artifact=c.artifact(path), gzip_eof=False, unclosed_stream_preserved=True,
                      raw_bytes=len(raw), raw_sha256=hashlib.sha256(raw).hexdigest(), records=len(lines))


def audit_saved_prefix(boundary=lambda: None):
    """Call only in a separate, model-free original terminal-parent/r1 namespace."""
    from tools.splendorust_full_corpus_20261110.strength import common as old, control, transport
    from tools.splendorust_live_boundary_20261025.raw_transport import _packet_evidence, _bind_parent_wire, _bind_child_wire
    from tools.splendorust_gpu_probe_20261105.control import require_absent
    c.require(old.MATCHUP == 'terminal_parent' and old.REPETITION == 1 and
              old.STUDY_OUT == c.ORIGINAL_OUT, 'Dedicated original saved-audit namespace')
    freeze = c.read(old.OUT/'freeze.json')
    class SavedEnvelope:
        def guard(self): boundary()
        def effective_deadline(self): return time.monotonic()+HELPER_SECONDS
    envelope = SavedEnvelope()
    worker_results = []
    for route in ('control','candidate'):
        folder = old.OUT/'workers'/('main_g030_'+route)
        ids = [f'request-{i:08d}' for i in range(1,70)]
        c.require({p.name for p in folder.glob('request-????????.json')} == {rid+'.json' for rid in ids},
                  'Exactly the 69 completed receipts per failed worker')
        c.require({p.name for p in (folder/'startup_inputs').iterdir()} == {rid+'.json' for rid in ids[:2]},
                  'Exactly two retained startup inputs')
        ordinary_ids = ids[2:]+(['request-00000070'] if route == 'control' else [])
        c.require({p.name for p in (folder/'inputs').iterdir()} == {rid+'.json' for rid in ordinary_ids},
                  'Only the one disclosed prepared request may lack a receipt')
        for directory in ('parent_profiles','placement_requests'):
            c.require({p.name for p in (folder/directory).iterdir()} == {rid+'.json' for rid in ids},
                      'Complete exact saved profile and placement inventory')
        c.require({p.name for p in folder.glob('request-*_child_profile*')} == {rid+'_child_profile.json' for rid in ids},
                  'Exact completed child sidecars, no hidden pending request')
        c.require(not (folder/'cleanup.json').exists() and (folder/'stderr.log').read_bytes() == b'',
                  'The abrupt old worker closure remains unmodified')
        stream_rows, streams = {}, {}
        for name in ('stdout.log.gz','events.jsonl.gz'):
            relative = folder.relative_to(c.ORIGINAL_OUT).as_posix()+'/'+name
            stream_rows[name], streams[name] = _read_unclosed_stream(folder/name, PINNED[relative], boundary)
        packets = [_packet_evidence(line) for line in stream_rows['stdout.log.gz']]
        init = c.read(folder/'initialization.json')
        require_absent(init['identity'])
        c.require(len(packets) == 70 and packets[0]['packet'] == {'type':'ready','metadata':init['result']},
                  'Exact ready packet plus 69 original completed responses')
        events = [json.loads(line) for line in stream_rows['events.jsonl.gz']]
        c.require(len(events) == 69, 'Exactly the completed request event prefix')
        for index, rid in enumerate(ids):
            boundary()
            receipt = c.read(folder/(rid+'.json'))
            packet = packets[index+1]
            c.require(receipt['status'] == 'ok' and receipt['accepted'] is True and receipt['request_id'] == rid and
                      packet['packet'] == dict(type='response',request_id=rid,result=receipt['result']) and
                      events[index] == {k:v for k,v in receipt.items() if k != 'result'},
                      'Every original raw response/event equals its successful trusted receipt')
            payload = c.read(folder/('startup_inputs' if index < 2 else 'inputs')/(rid+'.json'))
            parent = c.read(folder/'parent_profiles'/(rid+'.json'))
            child = c.read(folder/(rid+'_child_profile.json'))
            _bind_parent_wire(parent,[packet],completed=True)
            _bind_child_wire(child,[packet],receipt)
            c.require(child['payload_sha256'] == c.digest(payload) and
                      parent['request_raw_sha256'] == child['request_raw_sha256'] and
                      parent['request_raw_complete'] is True and parent['request_raw_overflow'] is False,
                      'Original request payload and actual written bytes authenticate each other')
            transport.validate_profiles(parent,child,receipt,payload,route,worker_pid=init['identity']['pid'],
                                        graph_identity_sha256=init['result']['graph_identity_sha256'])
            transport.verify_placement_request(folder,rid,freeze,parent)
        plan = c.read(old.OUT/'qualification_plan.json')
        control.qualification_evidence(folder.name,envelope,freeze,plan,writing=False)
        worker_results.append(dict(worker=folder.name, route=route, requests=69, startup_requests=2,
            ordinary_requests=67, failed_requests=0, prepared_extra_request='request-00000070' if route=='control' else None,
            exact_raw_responses_checked=True, request_profiles_checked=True, actual_startup_checked=True, streams=streams))
    seen = set()
    for index, count in ((30,56),(31,56),(32,22)):
        folder = old.OUT/'games/main'/f'g{index:03d}'
        spec = c.read(folder/'spec.json'); initial = c.read(folder/'initial.json')['initial']
        control.design.validate_spec(spec)
        c.require(initial == control.design.initial_snapshot(spec), 'Original game setup and seed binding')
        previous = initial; names = c.read(folder/'workers.json')
        c.require({p.name for p in (folder/'moves').iterdir()} == {f'{ply:06d}.json' for ply in range(count)},
                  'No missing, extra, or overwritten original game-prefix move')
        for ply in range(count):
            boundary(); row = c.read(folder/'moves'/f'{ply:06d}.json')
            c.require(type(row['ply']) is int and row['ply'] == ply and row['before'] == previous and row['completed'] is True,
                      'Contiguous, successful original corrected transitions')
            c.require(control.validate_row(row,spec,freeze,envelope,worker_name=names[row['route']]) is None,
                      'No failed saved move may be relabelled completed')
            key = (row['route'],row['clock']['request_id'])
            c.require(key not in seen, 'No duplicated original ordinary request')
            seen.add(key); previous = row['after']
        if index == 32:
            c.require(previous == retry_beforestate(), 'Independently reconstructed retry beforestate')
    c.require(seen == {(role,f'request-{i:08d}') for role in ('control','candidate') for i in range(3,70)},
              'All original ordinary requests map exactly once to the two games and partial prefix')
    c.require('torch' not in sys.modules, 'Saved boundary audit must not load tensor runtime/models')
    return dict(status='saved_prefix_verified_with_disclosed_dispatch_ambiguity', checked_partial_moves=22,
                checked_completed_block_moves=112, raw_requests=138, failed_saved_requests=0,
                unknown_dispatch_request=70, unknown_dispatch_ply=22, dispatch_never_started_proven=False,
                before_sha256=c.digest(retry_beforestate()), workers=worker_results, models=0, games=0)


def run_saved_audit(boundary=lambda: None):
    """Bounded read-only helper; the polling parent retains both original clocks."""
    c.require(c.sha(original('freeze.json')) == c.ORIGINAL_PLAN_SHA, 'Pinned source freeze before importing its audit code')
    source_files = [item for item in c.read(original('freeze.json'))['inputs'] if Path(item['path']).suffix == '.py']
    c.require(bool(source_files), 'Nonempty original Python source closure')
    c.verify(source_files,boundary=boundary)
    env = os.environ.copy()
    env.update(INDEPENDENT_DATA_REPETITION='1',FULL_CORPUS_MATCHUP='terminal_parent',PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(c.ROOT/'src')+os.pathsep+str(c.ROOT))
    proc = subprocess.Popen([sys.executable,'-B','-m',c.PACKAGE+'.boundary','--saved-audit'],
        cwd=c.ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    started = time.monotonic(); process = psutil.Process(proc.pid)
    identity = dict(pid=proc.pid,created=process.create_time())
    try:
        while proc.poll() is None:
            boundary()
            c.require(time.monotonic()-started < HELPER_SECONDS, 'Saved helper 20-minute ceiling')
            try:
                c.require(process.memory_info().rss < HELPER_RSS, 'Saved helper 8GiB RSS ceiling')
            except psutil.NoSuchProcess:
                pass
            time.sleep(.1)
        stdout, stderr = proc.communicate(timeout=5)
        c.require(proc.returncode == 0, 'Saved boundary helper failed: '+stderr.decode(errors='replace')[-6000:])
        c.require(len(stdout) < 256*1024 and not stderr, 'Bounded clean saved helper output')
        from tools.splendorust_gpu_probe_20261105.control import require_absent
        require_absent(identity)
        c.verify(source_files,boundary=boundary)
        value=json.loads(stdout)
        value['helper_execution']=dict(identity=identity,worker_exit_code=0,identity_absent=True,
            elapsed_seconds=time.monotonic()-started,maximum_seconds=HELPER_SECONDS,maximum_rss_bytes=HELPER_RSS,
            original_python_sources_checked=len(source_files),models=0,games=0)
        return value
    finally:
        if proc.poll() is None:
            proc.kill(); proc.wait(timeout=5)


def _consumer_paths():
    return (c.OUT/'strength/terminal_parent/r1/segments/resume_partial/start.json',
            c.OUT/'segments/resume_partial_terminal_parent_r1/claim.json')


def _consumption_contract(value, before):
    start_path, segment_path = _consumer_paths()
    expected = dict(schema=CONSUMPTION_SCHEMA, retry_spec=c.RETRY_SPEC,
        authorization=c.artifact(c.AUTHORIZATION), original_freeze=c.artifact(original('freeze.json')),
        last_saved_move=c.artifact(original(PARTIAL+'/moves/000021.json')),
        ambiguous_input=c.artifact(original(WORKERS+'/main_g030_control/inputs/request-00000070.json')),
        before_sha256=c.digest(before), new_worker='main_g032_control', new_request_id='request-00000003',
        amendment_retry_number=1, fresh_allowance_seconds=10., internal_seconds=9.,
        original_dispatch_ambiguous=True, original_attempt_stays_incomplete=True,
        consumed_before_transport=True, reusable_after_failure=False,
        segment_start=c.artifact(start_path), segment_claim=c.artifact(segment_path))
    c.require(all(c.canonical(value.get(key)) == c.canonical(item) for key,item in expected.items()),
              'Exact nonrenewable one-move retry consumption')
    return expected


def verify_retry_consumption(value=None, before=None, boundary=lambda: None):
    validate_retry_authorization(boundary=boundary)
    value = c.read(c.OUT/'authorized_retry_consumed.json') if value is None else value
    original_before = retry_beforestate()
    if before is not None:
        c.require(before == original_before, 'No changed state at the one authorized retry')
    _consumption_contract(value, original_before)
    c.verify(list(c.nested_artifacts(value)), boundary=boundary)
    c.require(type(value.get('monotonic')) is float and type(value.get('unix_time')) is float and
              type(value.get('owner')) is dict and type(value['owner'].get('pid')) is int and
              type(value['owner'].get('created')) is float, 'Actual retry consumer identity and dual time')
    claim = c.read(c.OUT/'claim.json'); phase = c.read(c.OUT/'stages/strength_start.json')
    old_claim = c.read(original('claim.json')); old_phase = c.read(original('stages/strength_start.json'))
    for key in ('monotonic_origin','unix_origin','total_seconds','model_seconds'):
        c.require(type(claim[key]) is type(old_claim[key]) and claim[key] == old_claim[key],
                  'Retry cannot restart the original study/model clock')
    for key in ('monotonic','unix_time','monotonic_deadline','unix_deadline','stage','cap_seconds'):
        c.require(type(phase[key]) is type(old_phase[key]) and phase[key] == old_phase[key],
                  'Retry cannot restart or extend the original strength phase')
    c.require(value.get('recovery_claim') == c.artifact(c.OUT/'claim.json') and
              value.get('recovery_freeze') == c.artifact(c.PLAN), 'Retry belongs to this frozen recovery owner')
    c.require(c.artifact(c.AUTHORIZATION) in c.read(c.PLAN)['inputs'], 'Frozen exact retry authorization')
    start_path, segment_path = _consumer_paths()
    start = c.read(start_path); segment = c.read(segment_path)
    c.require(start['owner'] == value['owner'] and start['phase'] == 'resume_partial' and
              start['matchup'] == segment['matchup'] == 'terminal_parent' and
              type(start['repetition']) is int and start['repetition'] == segment['repetition'] == 1 and
              start['root_claim'] == segment['claim'] == c.artifact(c.OUT/'claim.json') and
              start['root_phase_clock'] == segment['phase_clock'] == c.artifact(c.OUT/'stages/strength_start.json') and
              start['segment_claim'] == c.artifact(segment_path) and segment['mode'] == 'resume_partial' and
              segment['freeze_sha256'] == c.sha(c.PLAN) and segment['block_start'] is None and
              start['monotonic'] <= value['monotonic'] and start['unix_time'] <= value['unix_time'],
              'The exact admitted resume controller consumes the token after admission')
    c.require(claim['monotonic_origin'] <= value['monotonic'] < phase['monotonic_deadline'] and
              claim['unix_origin'] <= value['unix_time'] < phase['unix_deadline'], 'Original retry phase clocks')
    return value


def consume_authorized_retry(path, before, *, matchup, repetition, stage, game_index, ply,
                             route, worker_name, request_id):
    c.require(Path(path).resolve() == (c.OUT/'authorized_retry_consumed.json').resolve(),
              'Single global durable retry token path')
    passed = dict(matchup=matchup,repetition=repetition,stage=stage,game_index=game_index,ply=ply,route=route)
    c.require(c.canonical(passed) == c.canonical({k:c.RETRY_SPEC[k] for k in passed}),
              'No other game, player, or turn receives the retry exception')
    c.require(worker_name == 'main_g032_control' and request_id == 'request-00000003',
              'Fresh qualified pair first ordinary request only')
    validate_retry_authorization()
    c.require(before == retry_beforestate(), 'Exact preserved beforestate for the disclosed retry')
    plan = c.read(c.PLAN)
    c.require(c.artifact(c.AUTHORIZATION) in plan['inputs'], 'Authorization is bound into the recovery freeze')
    start_path, segment_path = _consumer_paths()
    value = dict(schema=CONSUMPTION_SCHEMA,retry_spec=deepcopy(c.RETRY_SPEC),authorization=c.artifact(c.AUTHORIZATION),
        original_freeze=c.artifact(original('freeze.json')),last_saved_move=c.artifact(original(PARTIAL+'/moves/000021.json')),
        ambiguous_input=c.artifact(original(WORKERS+'/main_g030_control/inputs/request-00000070.json')),
        before_sha256=c.digest(before),new_worker=worker_name,new_request_id=request_id,
        amendment_retry_number=1,fresh_allowance_seconds=10.,internal_seconds=9.,original_dispatch_ambiguous=True,
        original_attempt_stays_incomplete=True,consumed_before_transport=True,reusable_after_failure=False,
        owner=dict(pid=os.getpid(),created=psutil.Process().create_time()),monotonic=time.monotonic(),unix_time=time.time(),
        recovery_claim=c.artifact(c.OUT/'claim.json'),recovery_freeze=c.artifact(c.PLAN),
        segment_start=c.artifact(start_path),segment_claim=c.artifact(segment_path))
    verify_retry_consumption(value,before)
    # Exclusive creation + fsync: even an interrupted/failed dispatch spends this token.
    return c.save(path,value)


if __name__ == '__main__':
    c.require(sys.argv[1:] == ['--saved-audit'], 'Only saved boundary auditing is exposed')
    from tools.future_training_rules_20261021 import verify_runtime_sources
    runtime_sources = verify_runtime_sources()
    end = time.monotonic()+HELPER_SECONDS
    def guard():
        c.require(time.monotonic() < end and psutil.Process().memory_info().rss < HELPER_RSS,
                  'Saved-only worker time/RSS bounds')
    proof = audit_saved_prefix(guard)
    proof['runtime_sources'] = runtime_sources
    print(c.canonical(proof),flush=True)
