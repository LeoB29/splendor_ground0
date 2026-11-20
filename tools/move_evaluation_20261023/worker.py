"""One supervised model worker; no parent referee secrets in policy requests."""
import gzip
import json
import os
from pathlib import Path
import sys

from tools.local_player_qualification_128.clock import serve
from . import common as c


def initialize():
    from tools.future_training_rules_20261021 import verify_runtime_sources, RULESET_ID, CATALOGUE_SHA256
    folder = Path(os.environ['STUDY_WORKER_DIR']).resolve()
    c.require(folder.is_relative_to(c.OUT), 'Worker output boundary')
    c.save(folder/'runtime_sources.json', {'sources': verify_runtime_sources(),
           'ruleset_id': RULESET_ID, 'catalogue_sha256': CATALOGUE_SHA256})
    mode = os.environ['STUDY_MODE']
    c.require(mode in ('qualify', 'select', 'audit_selection', 'assess', 'qualify_assessor', 'audit_branch', 'statistics'),
              'Unsupported worker mode')
    ctx = {'mode': mode, 'folder': folder, 'used': False}
    if mode != 'statistics':
        from tools.corrected_planner_qualification_20261018 import local as q
        c.save(folder/'models.json', q.initialize())
    if mode in ('qualify', 'select', 'audit_selection'):
        from .greedy_adapter import GreedyOpponent
        ctx['opponent'] = GreedyOpponent(folder)
    elif mode in ('assess', 'qualify_assessor'):
        from tools.external_strength_screen_20261018_02.splendorust_adapter import Opponent
        c.require(c.file_hash(c.PROFILE) == c.PROFILE_SHA, 'Qualified PUCT16 profile changed')
        ctx['opponent'] = Opponent(c.read(c.PROFILE))
    return ctx


def progress():
    # Scientific trace contains no timing. Progress is only a stderr heartbeat.
    progress.calls += 1
    if progress.calls <= 2 or progress.calls % 512 == 0:
        sys.stderr.write(json.dumps({'event': 'progress', 'calls': progress.calls})+'\n')
        sys.stderr.flush()
progress.calls = 0


def handle(ctx, payload):
    from . import evaluator
    from tools.external_strength_screen_20261018 import rules
    mode = ctx['mode']
    if mode not in ('assess', 'audit_branch'):
        c.require(not ctx['used'], 'One request per fresh qualification/audit worker')
        ctx['used'] = True
    if mode in ('qualify', 'qualify_assessor'):
        c.require(payload == {'operation': 'entrypoint_checks'}, 'Unexpected qualification request')
        from .qualification import run_entrypoint_checks
        return run_entrypoint_checks(ctx['opponent'], evaluator.select_raw)
    if mode == 'select':
        path = ctx['folder']/'continuations.jsonl.gz'
        count = 0
        with path.open('xb') as raw:
            with gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=3, mtime=0) as compressed:
                def sink(row):
                    nonlocal count
                    compressed.write((c.canonical(row)+'\n').encode())
                    count += 1
                decision = evaluator.choose(payload, ctx['opponent'], boundary=progress, trace_sink=sink)
            raw.flush(); os.fsync(raw.fileno())
        return {'decision': decision, 'trace': {**c.artifact(path), 'records': count},
                'profile_id': evaluator.PROFILE, 'observer_sha256': c.digest(payload['observer']),
                'ruleset_id': rules.RULESET_ID, 'catalogue_sha256': rules.CATALOGUE_SHA256}
    if mode == 'audit_selection':
        c.require(set(payload) == {'selector', 'result', 'trace'}, 'Unexpected audit payload')
        path = (c.ROOT/payload['trace']['path']).resolve()
        c.require(path.is_relative_to(c.OUT) and path.name == 'continuations.jsonl.gz', 'Audit path boundary')
        c.verify([payload['trace']])
        from .audit import audit_selection
        return audit_selection(payload['selector'], payload['result'], path,
                               ctx['opponent'], boundary=progress)
    if mode == 'assess':
        c.require(type(payload) is dict and set(payload) == {'observer', 'player'} and
                  payload['player'] in ('raw035', 'opponent'), 'Assessment observer-only request')
        observer = payload['observer']
        state = evaluator.decode_observer(observer)
        legal = rules.CorrectedEnv().legal_actions(state)
        if payload['player'] == 'raw035':
            action, diag = evaluator.select_raw(observer)
        else:
            action, diag = ctx['opponent'].choose(observer)
        c.require(action in legal, 'Illegal assessment response')
        return {'action': rules.serialize_action(action), 'diagnostics': diag,
                'profile_sha256': c.PROFILE_SHA if payload['player'] == 'opponent' else None,
                'player': payload['player'], 'observer_sha256': c.digest(observer),
                'legal_actions_sha256': c.digest([rules.serialize_action(a) for a in legal]),
                'legal_count': len(legal), 'ruleset': rules.RULESET_ID, 'catalogue_sha256': rules.CATALOGUE_SHA256}
    if mode == 'audit_branch':
        from .audit import audit_branch, audit_assessment_initial
        c.require(set(payload) == {'branch', 'observer', 'world_seed'}, 'Assessment audit payload')
        initial = audit_assessment_initial(payload['observer'], payload['world_seed'], payload['branch']['initial'])
        return {'initial_world': initial, 'branch': audit_branch(payload['branch'], boundary=progress)}
    if mode == 'statistics':
        c.require(set(payload) == {'rows', 'candidate_lock'}, 'Unexpected analysis request')
        from .audit import statistics
        from .primary_statistics import calculate
        from tools.matched_strength_20261020.statistics import runtime_contract
        expected_runtime = c.read(c.OUT/'statistics_runtime.json')
        c.require(runtime_contract() == expected_runtime, 'Statistics runtime changed before calculation')
        primary = calculate(payload['rows'], boundary=progress)
        independent = statistics(payload['rows'], payload['candidate_lock'], checks_complete=True, boundary=progress)
        c.require(primary['mean_gain'] == independent['mean_gain'] and
                  primary['by_fit_mean_gain'] == independent['by_fit_mean_gain'] and
                  primary['conditional_ci95'] == independent['conditional_ci95'] and
                  primary['bootstrap_integer_totals_sha256'] == independent['bootstrap_integer_totals_sha256'],
                  'Independent statistical computation differs')
        c.require(runtime_contract() == expected_runtime, 'Statistics runtime changed during calculation')
        return {'primary': primary, 'independent': independent, 'comparison': 'exact_match',
                'runtime': c.artifact(c.OUT/'statistics_runtime.json')}
    raise ValueError('Unhandled mode')


if __name__ == '__main__':
    raise SystemExit(serve(initialize, handle, max_request_bytes=64*1024**2))
