"""Guarded fresh64 selectors/replays and observer-only assessment workers."""
import gzip
import os
from pathlib import Path

from tools.local_player_qualification_128.clock import serve
from tools.move_evaluation_20261023.worker import progress
from . import common as c


class Trace:
    def __init__(self, path):
        self.path, self.count = path, 0
        self.raw = path.open('xb')
        self.compressed = gzip.GzipFile(filename='', fileobj=self.raw, mode='wb', compresslevel=3, mtime=0)

    def sink(self, row):
        self.compressed.write((c.canonical(row)+'\n').encode())
        self.count += 1

    def finalize(self):
        try:
            self.compressed.close()
            self.raw.flush()
            os.fsync(self.raw.fileno())
        finally:
            self.raw.close()
        return {**c.artifact(self.path), 'records': self.count}


def initialize():
    from tools.future_training_rules_20261021 import verify_runtime_sources, RULESET_ID, CATALOGUE_SHA256, TURN_CONTRACT
    folder = Path(os.environ['STUDY_WORKER_DIR']).resolve()
    c.require(folder.is_relative_to(c.OUT), 'Worker output boundary')
    c.save(folder/'runtime_sources.json', {'sources': verify_runtime_sources(),
           'ruleset_id': RULESET_ID, 'catalogue_sha256': CATALOGUE_SHA256, 'turn_contract': TURN_CONTRACT})
    mode = os.environ['STUDY_MODE']
    c.require(mode in ('qualify', 'select', 'audit_selection', 'assess', 'audit_assessment',
                      'audit_branch', 'statistics'), 'Unsupported64 worker mode')
    ctx = {'mode': mode, 'folder': folder, 'used': False, 'opponent': None}
    if mode in ('qualify', 'select', 'audit_selection'):
        from .evaluator import arm_profile
        ctx['arm'] = os.environ['STUDY_ARM']
        profile = arm_profile(ctx['arm'])
        c.save(folder/'implementation.json', {'arm': ctx['arm'], 'profile_id': profile,
            'world_count': 64, 'proposal_world_count': 128, 'full_prefix_derivation': True,
            'inference': 'original_torch_module', 'cross_decision_cache': False,
            'opponent': 'none_raw_both' if ctx['arm'] == 'A' else
                        'original_reference' if mode == 'audit_selection' else 'accepted_per_decision_reuse'})
    if mode != 'statistics':
        from tools.corrected_planner_qualification_20261018 import local as q
        c.save(folder/'models.json', q.initialize())
    if mode in ('qualify', 'select', 'audit_selection') and ctx['arm'] == 'B':
        if mode == 'audit_selection':
            from tools.move_evaluation_20261023.greedy_adapter import GreedyOpponent
        else:
            from tools.move_evaluation_reuse_20261023.greedy_adapter import GreedyOpponent
        ctx['opponent'] = GreedyOpponent(folder)
    elif mode == 'assess':
        from tools.external_strength_screen_20261018_02.splendorust_adapter import Opponent
        c.require(c.file_hash(c.PROFILE) == c.PROFILE_SHA, 'Qualified PUCT16 profile changed')
        ctx['opponent'] = Opponent(c.read(c.PROFILE))
    return ctx


def handle(ctx, payload):
    from . import evaluator
    from tools.external_strength_screen_20261018 import rules
    mode = ctx['mode']
    if mode not in ('assess', 'audit_assessment', 'audit_branch'):
        c.require(not ctx['used'], 'Exactly one request per fresh worker')
        ctx['used'] = True
    if mode == 'qualify':
        c.require(payload == {'operation': 'entrypoint_checks'}, 'Qualification payload')
        from .qualification import run_entrypoint_checks
        return run_entrypoint_checks(ctx['opponent'], evaluator.select_raw, arm=ctx['arm'], boundary=progress)
    if mode == 'select':
        trace = Trace(ctx['folder']/'continuations.jsonl.gz')
        try:
            decision = evaluator.choose(payload, ctx['opponent'], arm=ctx['arm'],
                                         boundary=progress, trace_sink=trace.sink)
        finally:
            trace_artifact = trace.finalize()
        return {'decision': decision, 'trace': trace_artifact,
                'profile_id': evaluator.arm_profile(ctx['arm']), 'observer_sha256': c.digest(payload['observer']),
                'ruleset_id': rules.RULESET_ID, 'catalogue_sha256': rules.CATALOGUE_SHA256}
    if mode == 'audit_selection':
        c.require(type(payload) is dict and set(payload) == {'selector', 'result', 'trace'}, 'Audit payload')
        path = (c.ROOT/payload['trace']['path']).resolve()
        c.require(path.is_relative_to(c.OUT) and path.name == 'continuations.jsonl.gz', 'Audit path boundary')
        c.require(payload['trace'] == payload['result']['trace'], 'Audit trace copy mismatch')
        # Common verify expects three-field artifacts; records are separately audited.
        c.verify([{k: payload['trace'][k] for k in ('path', 'sha256', 'bytes')}])
        from .audit import audit_selection
        return audit_selection(payload['selector'], payload['result'], path, ctx['opponent'],
                               arm=ctx['arm'], boundary=progress)
    if mode == 'assess':
        c.require(type(payload) is dict and set(payload) == {'observer', 'player'} and
                  payload['player'] in ('raw035', 'opponent'), 'Assessment observer-only request')
        observer = payload['observer']; state = evaluator.decode_observer(observer)
        legal = rules.CorrectedEnv().legal_actions(state)
        action, diag = evaluator.select_raw(observer) if payload['player'] == 'raw035' else ctx['opponent'].choose(observer)
        c.require(action in legal, 'Illegal assessment response')
        return {'action': rules.serialize_action(action), 'diagnostics': diag,
                'profile_sha256': c.PROFILE_SHA if payload['player'] == 'opponent' else None,
                'player': payload['player'], 'observer_sha256': c.digest(observer),
                'legal_actions_sha256': c.digest([rules.serialize_action(a) for a in legal]),
                'legal_count': len(legal), 'ruleset': rules.RULESET_ID, 'catalogue_sha256': rules.CATALOGUE_SHA256}
    if mode in ('audit_assessment', 'audit_branch'):
        from .assessment import audit_assessment_initial
        from tools.move_evaluation_20261023.audit import audit_branch
        c.require(type(payload) is dict and set(payload) == {'branch', 'observer', 'world_seed'}, 'Assessment audit payload')
        initial = audit_assessment_initial(payload['observer'], payload['world_seed'], payload['branch']['initial'])
        return {'initial_world': initial, 'branch': audit_branch(payload['branch'], boundary=progress)}
    if mode == 'statistics':
        c.require(type(payload) is dict and set(payload) == {'rows', 'candidate_lock'}, 'Statistics payload')
        from .analysis import calculate_and_verify
        return calculate_and_verify(payload['rows'], payload['candidate_lock'], boundary=progress)
    raise ValueError('Unhandled64 worker mode')


if __name__ == '__main__':
    raise SystemExit(serve(initialize, handle, max_request_bytes=64*1024**2))
