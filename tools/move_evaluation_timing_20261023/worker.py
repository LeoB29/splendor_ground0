"""Fresh corrected-rules worker with selected Python-region instrumentation."""
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
    c.require(mode in ('plain', 'timed', 'qualify_plain', 'qualify_timed', 'audit'), 'Timing mode')
    from tools.corrected_planner_qualification_20261018 import local as q
    from tools.move_evaluation_20261023.greedy_adapter import GreedyOpponent
    c.save(folder/'models.json', q.initialize())
    return {'mode': mode, 'folder': folder, 'used': False, 'opponent': GreedyOpponent(folder)}


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


def scientific_body(ctx, payload):
    from tools.move_evaluation_20261023 import evaluator
    from tools.external_strength_screen_20261018 import rules
    if ctx['mode'].startswith('qualify_'):
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
    c.require(ctx['mode'] in ('plain', 'timed'), 'Selection mode')
    trace = Trace(ctx['folder']/'continuations.jsonl.gz')
    try:
        decision = evaluator.choose(payload, ctx['opponent'], boundary=progress, trace_sink=trace.sink)
    finally:
        trace_artifact = trace.finalize()
    return {'decision': decision, 'trace': trace_artifact,
            'profile_id': evaluator.PROFILE, 'observer_sha256': c.digest(payload['observer']),
            'ruleset_id': rules.RULESET_ID, 'catalogue_sha256': rules.CATALOGUE_SHA256}


def targets(ctx):
    from tools.move_evaluation_20261023 import evaluator, qualification
    from tools.move_evaluation_20261023.greedy_adapter import GreedyOpponent, NativeOnly
    from tools.corrected_planner_qualification_20261018 import local as q
    from tools.external_strength_screen_20261018 import rules
    from tools.external_strength_screen_20261018_02 import bridge
    # Only function boundaries are observed. No function, model or global is replaced.
    result = {
        'scientific_body': scientific_body,
        'entrypoint_checks': qualification.run_entrypoint_checks,
        'actual_trajectory_check': qualification.verify_actual_trajectory,
        'choose': evaluator.choose, 'trajectory': evaluator.trajectory,
        'sample_world': evaluator.sample_world, 'decode_observer': evaluator.decode_observer,
        'raw_select': evaluator.select_raw, 'raw_score': q.score_actions,
        'proposal': q.proposal_for_root, 'greedy_choose': GreedyOpponent.choose,
        'entity_infer': GreedyOpponent._infer, 'native_exchange': NativeOnly.exchange,
        'bridge_prepare': bridge._state_and_legal, 'bridge_encode': bridge.encode_observation,
        'bridge_actions': bridge.main_actions, 'bridge_resolve': bridge.resolve_compound,
        'observer_decode': rules.state_from_observation, 'observer_encode': rules.observation_payload,
        'legal': rules.CorrectedEnv.legal_actions, 'step': rules.CorrectedEnv.step,
        'shared_attention_forward': q._RAW._model.forward,
        'trace_sink': Trace.sink, 'trace_finalize': Trace.finalize}
    models = [q.registered_model('anchor_'+str(fit))[0] for fit in (161001, 161002, 161003)]
    c.require(all(m.forward.__code__ is models[0].forward.__code__ for m in models),
              'Expected common proposal forward implementation')
    c.require(models[0].forward.__code__ is q._RAW._model.forward.__code__,
              'Expected shared raw/proposal attention forward implementation')
    c.require(len({id(fn.__code__) for fn in result.values()}) == len(result), 'Ambiguous selected code identities')
    return result


def handle(ctx, payload):
    c.require(not ctx['used'], 'Exactly one request per fresh worker')
    ctx['used'] = True
    if ctx['mode'] in ('timed', 'qualify_timed'):
        from .monitor import SelectedTimer
        timer = SelectedTimer(targets(ctx))
        try:
            scientific, measurement = timer.run(scientific_body, ctx, payload)
        except BaseException:
            if timer.last_failure is not None:
                c.save(ctx['folder']/'timing_failure.json', timer.last_failure)
            raise
    else:
        start = time.perf_counter_ns()
        scientific = scientific_body(ctx, payload)
        elapsed = time.perf_counter_ns()-start
        measurement = {'schema': 'ordinary_body_wall_time_v1', 'body_ns': elapsed, 'body_seconds': elapsed/1e9}
    measurement.update({'mode': ctx['mode'], 'scientific_body_seconds': measurement['body_seconds'],
        'body_includes': 'scientific algorithm, JSON/gzip trace close/fsync/hash and scientific wrapper',
        'request_clock_includes': 'all request work including timing installation/teardown and response delivery'})
    return {'scientific': scientific, 'measurement': measurement}


if __name__ == '__main__':
    raise SystemExit(serve(initialize, handle, max_request_bytes=64*1024**2))
