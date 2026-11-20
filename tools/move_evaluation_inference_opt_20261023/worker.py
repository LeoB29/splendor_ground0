"""Fresh guarded Stage B selections; independent audits retain original inference."""
import os
from pathlib import Path
import time

from tools.local_player_qualification_128.clock import serve
from tools.move_evaluation_20261023.worker import progress
from tools.move_evaluation_timing_20261023.worker import Trace
from tools.move_evaluation_timing_20261023.monitor import SelectedTimer
from . import common as c

MODES = ('qualify_plain', 'qualify_timed', 'probe_reference', 'probe_plain', 'probe_timed', 'plain', 'timed', 'audit')
SCIENTIFIC_POLICY_ID = 'entity_onehot_greedy_canonical_atomic_prereveal_cpu_20261023_v1'


def implementation_metadata(mode):
    c.require(mode in MODES, 'Inference diagnostic mode')
    return {'schema': 'entity_prebound_fused_implementation_v1',
        'implementation_id': c.IMPLEMENTATION_ID,
        'active_opponent': 'original_reference' if mode in ('audit', 'probe_reference') else 'prebound_fused_transformer',
        'scientific_policy_id': SCIENTIFIC_POLICY_ID,
        'arithmetic_unchanged': True, 'optimization_enabled': mode not in ('audit', 'probe_reference'),
        'prediction_cache': False}


def initialize():
    from tools.future_training_rules_20261021 import verify_runtime_sources, RULESET_ID, CATALOGUE_SHA256
    folder = Path(os.environ['STUDY_WORKER_DIR']).resolve()
    c.require(folder.is_relative_to(c.OUT), 'Worker output boundary')
    c.save(folder/'runtime_sources.json', {'sources': verify_runtime_sources(),
        'ruleset_id': RULESET_ID, 'catalogue_sha256': CATALOGUE_SHA256})
    mode = os.environ['STUDY_MODE']
    implementation = implementation_metadata(mode)
    from tools.corrected_planner_qualification_20261018 import local as q
    from tools.move_evaluation_20261023.greedy_adapter import GreedyOpponent as OriginalOpponent
    from .inference import PreboundOpponent, IMPLEMENTATION_ID
    c.require(IMPLEMENTATION_ID == c.IMPLEMENTATION_ID, 'Prebound implementation identity')
    c.save(folder/'implementation.json', implementation)
    c.save(folder/'models.json', q.initialize())
    opponent = (OriginalOpponent if mode in ('audit', 'probe_reference') else PreboundOpponent)(folder)
    eligibility = ({'schema': 'entity_prebound_reference_dispatch_v1',
        'active_opponent': 'original_reference', 'optimization_enabled': False}
        if mode in ('audit', 'probe_reference') else opponent.eligibility_receipt)
    eligibility_artifact = c.save(folder/'eligibility.json', eligibility)
    return {'folder': folder, 'mode': mode, 'used': False, 'opponent': opponent,
            'implementation': implementation, 'eligibility': eligibility_artifact}


def probe_panel(ctx, payload):
    import json
    from tools.external_strength_screen_20261018.common import serialize_action
    from tools.external_strength_screen_20261018 import rules
    from tools.move_evaluation_20261023.worker import progress
    c.require(payload == {'operation': 'public_inference_panel', 'repetitions': c.PANEL_REPETITIONS},
              'Fixed public inference panel request')
    lock = c.read(c.PREP/'candidate_lock.json')
    roots = lock['roots']
    c.require(len(roots) == 18 and [r['fit'] for r in roots] ==
              [fit for fit in (161001, 161002, 161003) for _ in range(6)], 'Fixed panel identities')
    rows = []
    for repetition in range(c.PANEL_REPETITIONS):
        for index, root in enumerate(roots):
            progress()
            live = c.selector_payload(root)['observer']
            c.require(set(live) == {'schema', 'context', 'ruleset_id', 'catalogue_sha256', 'observation'}
                      and live['schema'] == 'corrected_live_planner_observer_v1'
                      and live['context']['fit'] == root['fit'], 'Qualified live public observer')
            # Same public projection as the frozen live decoder, without game
            # context or a hidden-state round trip. The actual adapter validates it.
            observer = json.loads(c.canonical({k: v for k, v in live.items() if k != 'context'}))
            observer['schema'] = rules.OBSERVATION_SCHEMA
            before = c.canonical(observer)
            action, trace = ctx['opponent'].choose(observer, diagnostic_capture=True)
            c.require(before == c.canonical(observer), 'Probe changed its public input')
            rows.append({'repetition': repetition, 'index': index, 'id': root['id'],
                         'fit': root['fit'], 'observer_sha256': c.digest(observer),
                         'action': serialize_action(action), 'trace': trace})
    return {'schema': 'entity_public_inference_panel_v1', 'repetitions': c.PANEL_REPETITIONS,
            'panel_size': 18, 'rows': rows}


def scientific_body(ctx, payload):
    from tools.move_evaluation_20261023 import evaluator
    from tools.external_strength_screen_20261018 import rules
    if ctx['mode'].startswith('qualify_'):
        c.require(payload == {'operation': 'entrypoint_checks'}, 'Qualification payload')
        from tools.move_evaluation_20261023.qualification import run_entrypoint_checks
        return run_entrypoint_checks(ctx['opponent'], evaluator.select_raw)
    if ctx['mode'].startswith('probe_'):
        return probe_panel(ctx, payload)
    if ctx['mode'] == 'audit':
        c.require(ctx['implementation']['active_opponent'] == 'original_reference', 'Original reference audit required')
        c.require(type(payload) is dict and set(payload) == {'selector', 'result', 'trace'}, 'Audit payload')
        path = (c.ROOT/payload['trace']['path']).resolve()
        c.require(path.is_relative_to(c.OUT) and path.name == 'continuations.jsonl.gz', 'Audit path boundary')
        c.verify([payload['trace']])
        from tools.move_evaluation_20261023.audit import audit_selection
        return audit_selection(payload['selector'], payload['result'], path, ctx['opponent'], boundary=progress)
    c.require(ctx['mode'] in ('plain', 'timed') and
              ctx['implementation']['active_opponent'] == 'prebound_fused_transformer', 'Optimized selection mode')
    trace = Trace(ctx['folder']/'continuations.jsonl.gz')
    try:
        decision = evaluator.choose(payload, ctx['opponent'], boundary=progress, trace_sink=trace.sink)
    finally:
        trace_artifact = trace.finalize()
    # Policy and scientific result identity remain identical; implementation
    # provenance is recorded separately, never hidden by a changed policy label.
    return {'decision': decision, 'trace': trace_artifact,
            'profile_id': evaluator.PROFILE, 'observer_sha256': c.digest(payload['observer']),
            'ruleset_id': rules.RULESET_ID, 'catalogue_sha256': rules.CATALOGUE_SHA256}


def targets(ctx):
    from tools.move_evaluation_reuse_20261023.worker import targets as reuse_targets
    from .inference import PreboundOpponent, fine_targets
    values = reuse_targets(ctx)
    values.update(scientific_body=scientific_body, entity_infer=PreboundOpponent._infer,
                  probe_panel=probe_panel)
    values.update(fine_targets())
    c.require(len({id(fn.__code__) for fn in values.values()}) == len(values),
              'Ambiguous inference selected-code identities')
    return values


def handle(ctx, payload):
    c.require(not ctx['used'], 'Exactly one request per fresh worker')
    ctx['used'] = True
    if ctx['mode'] == 'timed' or ctx['mode'].endswith('_timed'):
        timer = SelectedTimer(targets(ctx))
        try:
            science, measurement = timer.run(scientific_body, ctx, payload)
        except BaseException:
            if timer.last_failure is not None:
                c.save(ctx['folder']/'timing_failure.json', timer.last_failure)
            raise
    else:
        start = time.perf_counter_ns()
        science = scientific_body(ctx, payload)
        elapsed = time.perf_counter_ns()-start
        measurement = {'schema': 'ordinary_body_wall_time_v1', 'body_ns': elapsed,
                       'body_seconds': elapsed/1e9}
    measurement.update(mode=ctx['mode'], scientific_body_seconds=measurement['body_seconds'],
        body_includes='Actual qualification, fixed public panel, complete selection including trace finalization, or original-policy replay.',
        request_clock_includes='All request work and response serialization/delivery.')
    return {'scientific': science, 'measurement': measurement, 'implementation': ctx['implementation'],
            'eligibility': ctx['eligibility']}


if __name__ == '__main__':
    raise SystemExit(serve(initialize, handle, max_request_bytes=64*1024**2))
