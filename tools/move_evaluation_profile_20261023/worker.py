"""Fresh worker around the unchanged evaluator, with optional wall profiling."""
import cProfile
import gzip
import os
from pathlib import Path
import time

from tools.local_player_qualification_128.clock import serve
from tools.move_evaluation_20261023.worker import progress
from . import common as c


def initialize():
    from tools.future_training_rules_20261021 import verify_runtime_sources, RULESET_ID, CATALOGUE_SHA256
    folder = Path(os.environ['STUDY_WORKER_DIR']).resolve()
    c.require(folder.is_relative_to(c.OUT), 'Worker output boundary')
    c.save(folder/'runtime_sources.json', {'sources': verify_runtime_sources(),
           'ruleset_id': RULESET_ID, 'catalogue_sha256': CATALOGUE_SHA256})
    mode = os.environ['STUDY_MODE']
    c.require(mode in ('plain', 'profile', 'qualify', 'audit'), 'Diagnostic mode')
    from tools.corrected_planner_qualification_20261018 import local as q
    from tools.move_evaluation_20261023.greedy_adapter import GreedyOpponent
    c.save(folder/'models.json', q.initialize())
    return {'mode': mode, 'folder': folder, 'used': False, 'opponent': GreedyOpponent(folder)}


def scientific_body(ctx, payload):
    from tools.move_evaluation_20261023 import evaluator
    from tools.external_strength_screen_20261018 import rules
    if ctx['mode'] == 'qualify':
        c.require(payload == {'operation': 'entrypoint_checks'}, 'Qualification payload')
        from tools.move_evaluation_20261023.qualification import run_entrypoint_checks
        return run_entrypoint_checks(ctx['opponent'], evaluator.select_raw)
    if ctx['mode'] == 'audit':
        c.require(type(payload) is dict and set(payload) == {'selector', 'result', 'trace'}, 'Audit payload')
        path = (c.ROOT/payload['trace']['path']).resolve()
        c.require(path.is_relative_to(c.OUT) and path.name == 'continuations.jsonl.gz', 'Audit path boundary')
        c.verify([payload['trace']])
        from tools.move_evaluation_20261023.audit import audit_selection
        return audit_selection(payload['selector'], payload['result'], path, ctx['opponent'], boundary=progress)
    c.require(ctx['mode'] in ('plain', 'profile'), 'Selection mode')
    path = ctx['folder']/'continuations.jsonl.gz'
    count = 0
    with path.open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=3, mtime=0) as compressed:
            def sink(row):
                nonlocal count
                compressed.write((c.canonical(row)+'\n').encode())
                count += 1
            decision = evaluator.choose(payload, ctx['opponent'], boundary=progress, trace_sink=sink)
        raw.flush()
        os.fsync(raw.fileno())
    return {'decision': decision, 'trace': {**c.artifact(path), 'records': count},
            'profile_id': evaluator.PROFILE, 'observer_sha256': c.digest(payload['observer']),
            'ruleset_id': rules.RULESET_ID, 'catalogue_sha256': rules.CATALOGUE_SHA256}


def handle(ctx, payload):
    c.require(not ctx['used'], 'Exactly one request per fresh worker')
    ctx['used'] = True
    profiler = cProfile.Profile() if ctx['mode'] in ('profile', 'qualify') else None
    start = time.perf_counter()
    if profiler:
        profiler.enable()
    try:
        scientific = scientific_body(ctx, payload)
    finally:
        if profiler:
            profiler.disable()
        body_seconds = time.perf_counter()-start
    # Nothing from profiling is added to the scientific decision or gzip trace.
    # Export is outside the measured body but inside the trusted response clock.
    export_start = time.perf_counter()
    profile_artifact = None
    if profiler:
        profile_path = ctx['folder']/'scientific.pstats'
        c.require(not profile_path.exists(), 'No profile overwrite')
        profiler.dump_stats(str(profile_path))
        with profile_path.open('r+b') as stream:
            os.fsync(stream.fileno())
        profile_artifact = c.artifact(profile_path)
    export_seconds = time.perf_counter()-export_start
    return {'scientific': scientific, 'measurement': {
        'mode': ctx['mode'], 'scientific_body_seconds': body_seconds,
        'profile_export_seconds': export_seconds, 'profile': profile_artifact,
        'timer': 'cProfile default wall clock; native child internals not attributed',
        'body_includes': 'decode, fixed selection, trace JSON/gzip close/fsync, hash and scientific wrapper',
        'body_excludes': 'initialization, profile export, response serialization and delivery'}}


if __name__ == '__main__':
    raise SystemExit(serve(initialize, handle, max_request_bytes=64*1024**2))
