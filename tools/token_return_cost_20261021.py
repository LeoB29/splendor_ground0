"""Bounded, isolated corrected-rule continuation cost diagnosis; no deployment.

Run with python -m tools.token_return_cost_20261021. All new outputs are exclusive.
Only this diagnostic process is pinned/instrumented. No current-game data is read.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import time
from dataclasses import replace
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/token_return_cost_20261021_01'
SOURCE = ROOT / 'outputs/corrected_planner_qualification_20261018_01'
MAIN = ROOT / 'outputs/matched_strength_20261020_01/freeze.json'
MAIN_SHA = '41244628326e23ea6a8685752f367bd4ca5e13809cfb47d983d0cd6f139b248e'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, value):
    with (OUT / name).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def main():
    import psutil
    from tools.simulation_backend_diagnosis_128.topology import cpu_sets
    process = psutil.Process()
    original_affinity, original_priority = process.cpu_affinity(), process.nice()
    topology = cpu_sets()
    chosen = next(r for r in topology if r['logical_processor_index'] == 4)
    protected = [r for r in topology if r['logical_processor_index'] in (0, 2)]
    assert chosen['efficiency_class'] == 1 and all(chosen['core_index'] != r['core_index'] for r in protected)
    runner = psutil.Process(30052)
    assert runner.create_time() == 1792537125.6896 and sha(MAIN) == MAIN_SHA
    assert psutil.virtual_memory().available >= 3 * 1024**3
    process.cpu_affinity([4])
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    OUT.mkdir(exist_ok=False)
    started = time.perf_counter()
    cpu_started = time.process_time()
    peak_rss = 0

    def guard():
        nonlocal peak_rss
        peak_rss = max(peak_rss, process.memory_info().rss)
        assert time.perf_counter() - started < 300, 'Diagnostic 300-second cap'
        assert peak_rss < 1024**3, 'Diagnostic 1-GiB cap'
        assert psutil.virtual_memory().available > 1536 * 1024**2, 'Protect main RAM headroom'

    try:
        import torch
        from tools.future_training_rules_20261021 import verify_runtime_sources, decode_training_snapshot
        from tools.external_strength_screen_20261018 import rules
        from tools.corrected_planner_qualification_20261018 import local
        from tools.simulation_speed_probe_128_v2 import encoder as fast
        from splendor_ai.encoding import canonical_successor_action_key, actions_by_canonical_successor_key
        from splendor_ai.bots import model_bot as mb
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        sources = verify_runtime_sources()
        selection = json.loads((SOURCE / 'selection.json').read_text())
        assert len(selection['core']) == 18
        fixture_paths = [SOURCE / 'fixtures' / f'{i:02d}.json' for i in range(18)]
        frozen = {x['path']: x['sha256'] for x in json.loads(MAIN.read_text())['inputs']}
        assert str(Path(__file__).relative_to(ROOT)).replace('\\', '/') not in frozen
        for path in fixture_paths:
            key = path.relative_to(ROOT).as_posix()
            assert key in frozen and sha(path) == frozen[key], key
        local.initialize()
        raw = local._RAW
        assert raw._model.config.observation_size == 295 and raw._model.config.action_feature_size == 79
        inputs = {p.relative_to(ROOT).as_posix(): sha(p) for p in fixture_paths}
        for p in [Path(__file__), SOURCE/'selection.json', ROOT/'tools/simulation_speed_probe_128_v2/encoder.py']:
            inputs[p.relative_to(ROOT).as_posix()] = sha(p)
        inputs.update(sources)
        save('plan.json', {
            'scope': 'Engineering cost diagnosis, not strength/training/deployment',
            'sample': '18 previously accepted stratified core fixtures; first scheduled hidden world; all four actual learned proposals; every continuation state retained',
            'fixture_caveat': 'Corrected counterfactual historical breadth-player positions, not natural-frequency sample of current matched games',
            'variants': ['reference_full', 'fast_full', 'canonical_first_representative', 'one_return_per_family'],
            'repeats': 2, 'counterfactuals_change_policy': True,
            'timing': 'Latin rotation over variants per state; second repeat reverses state and variant order; same CPU4 throughout; exact full scoring and fast tensors/scores checked outside timed portions',
            'caps': {'seconds': 300, 'rss_bytes': 1024**3, 'physical_headroom_bytes': 1536*1024**2},
            'cpu': chosen, 'protected_cores': protected, 'main_freeze_sha256': MAIN_SHA,
            'inputs': inputs, 'ruleset': rules.RULESET_ID, 'model': json.loads((ROOT/'outputs/proposal_breadth_arena_128/models_lock.json').read_text())['continuation_model'],
        })
        original_env, original_score = rules.CorrectedEnv, local.score_actions
        meters = defaultdict(float)

        class TimedEnv(original_env):
            def _token_gain_actions(self, *args, **kwargs):
                player = kwargs['player']
                overflow = sum(player.tokens.values()) + len(kwargs['gained_tokens']) > 10
                start = time.perf_counter()
                value = super()._token_gain_actions(*args, **kwargs)
                if overflow:
                    meters['overflow_generation_seconds'] += time.perf_counter() - start
                    meters['overflow_generation_calls'] += 1
                return value

            def legal_actions(self, state):
                start = time.perf_counter()
                value = super().legal_actions(state)
                meters['legal_seconds'] += time.perf_counter() - start
                meters['legal_calls'] += 1
                return value

            def step(self, state, action):
                before = meters['legal_seconds']
                start = time.perf_counter()
                value = super().step(state, action)
                elapsed = time.perf_counter() - start
                regenerated = meters['legal_seconds'] - before
                meters['step_excluding_legal_seconds'] += elapsed - regenerated
                meters['regenerated_legal_seconds'] += regenerated
                meters['steps'] += 1
                return value

        # Compile/warm both encoders and model without including warmup in timings.
        first = decode_training_snapshot(json.loads(fixture_paths[0].read_text())['snapshot'])
        first_legal = original_env().legal_actions(first)
        for _ in range(3):
            raw.score_actions(first, first_legal)
            with torch.no_grad():
                raw._model(*fast.tensors(first, first_legal))
        states, info, baseline_scores, trajectories = [], [], [], []
        active = {}

        def measured_score(state, legal):
            start = time.perf_counter()
            scores = original_score(state, legal)
            elapsed = time.perf_counter() - start
            meters['score_seconds'] += elapsed
            families = {replace(a, return_tokens=()) for a in legal}
            canonical = actions_by_canonical_successor_key(state, legal)
            states.append(state)
            baseline_scores.append(tuple(scores))
            info.append({**active, 'turn': state.turn_index,
                         'tokens': sum(state.players[state.current_player].tokens.values()),
                         'actions': len(legal), 'return_actions': sum(bool(a.return_tokens) for a in legal),
                         'return_families': len(families), 'canonical': len(canonical),
                         'extra_return_rows': len(legal)-len(families),
                         'score_seconds': elapsed})
            return scores

        try:
            for index, path in enumerate(fixture_paths):
                guard()
                fixture = json.loads(path.read_text())
                state = decode_training_snapshot(fixture['snapshot'])
                entry = selection['core'][index]
                legal = original_env().legal_actions(state)
                assert [rules.serialize_action(a) for a in legal] == fixture['legal_actions']
                scores = original_score(state, legal)
                proposal = local.proposal_for_root(state, entry['context'], legal, scores)
                seed = proposal['schedule']['world_seeds'][0]
                world = local.determinize_for_observer(state, state.current_player, random.Random(seed))
                local.assert_corrected_cards(world)
                actions = [local.old.decode_action(a) for a in proposal['candidates']['actions']]
                rules.CorrectedEnv, local.score_actions = TimedEnv, measured_score
                for ci, action in enumerate(actions):
                    active.clear(); active.update(root=index, candidate=ci)
                    before = len(states)
                    result, _ = local.trajectory(world, action, boundary=guard, capture=False)
                    trajectories.append({'root': index, 'candidate': ci, 'world_seed': seed,
                                         'scored_positions': len(states)-before,
                                         'turns': result['turns'], 'reason': result['reason'],
                                         'trace_sha256': result['trace_sha256']})
                rules.CorrectedEnv, local.score_actions = original_env, original_score
            print(json.dumps({'stage': 'continuations_done', 'trajectories': len(trajectories), 'positions': len(states)}), flush=True)
        finally:
            rules.CorrectedEnv, local.score_actions = original_env, original_score
        save('collection.json', {'metrics': dict(meters), 'positions': info, 'trajectories': trajectories})
        with gzip.open(OUT/'states.jsonl.gz', 'xt', encoding='utf-8') as f:
            for state in states:
                f.write(json.dumps(rules.serialize_snapshot(state), separators=(',', ':'))+'\n')

        def score_timed(state, actions, use_fast=False):
            t0 = time.perf_counter()
            if use_fast:
                tensors = fast.tensors(state, actions)
                t1 = t2 = time.perf_counter()
            else:
                obs = mb.encode_public_observation_tensor(state, state.current_player)
                obsvec = mb.fit_feature_vector(obs.vector, 295)
                features = mb.encode_legal_action_features(state, actions, state.current_player)
                vectors = [mb.fit_feature_vector(e.vector, 79) for e in features]
                t1 = time.perf_counter()
                tensors = (torch.tensor(obsvec, dtype=torch.float32).unsqueeze(0),
                           torch.tensor(vectors, dtype=torch.float32).unsqueeze(0),
                           torch.ones((1, len(actions)), dtype=torch.bool))
                t2 = time.perf_counter()
            with torch.no_grad():
                logits, _ = raw._model(*tensors)
                masked = mb.masked_legal_action_scores(logits, tensors[2])[0]
            t3 = time.perf_counter()
            scores = tuple(float(v) for v in masked.detach().cpu().tolist())
            t4 = time.perf_counter()
            return scores, {'prepare': t1-t0, 'tensor': t2-t1, 'forward': t3-t2,
                            'output': t4-t3, 'total': t4-t0}

        variant_names = ['reference_full', 'fast_full', 'canonical_first_representative', 'one_return_per_family']
        timing_rows, tensor_checks = [], 0
        # Full-state paired comparisons; no subset selected after seeing the timings.
        for repeat in range(2):
            indices = range(len(states)) if repeat == 0 else reversed(range(len(states)))
            for i in indices:
                guard()
                state = states[i]
                legal = original_env().legal_actions(state)
                if repeat == 0:
                    a, b = fast.reference_tensors(state, legal), fast.tensors(state, legal)
                    assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True)), f'Tensor mismatch at {i}'
                    tensor_checks += 1
                groups = actions_by_canonical_successor_key(state, legal)
                families = {}
                for action in legal:
                    families.setdefault(replace(action, return_tokens=()), action)
                variants = [legal, legal, list(groups.values()), list(families.values())]
                order = [(i+j) % 4 for j in range(4)]
                if repeat:
                    order.reverse()
                reference_key = canonical_successor_action_key(state, legal[max(range(len(legal)), key=lambda k: baseline_scores[i][k])])
                for variant in order:
                    actions = variants[variant]
                    scores, measured = score_timed(state, actions, variant == 1)
                    if variant in (0, 1):
                        assert scores == baseline_scores[i], f'Full-score mismatch at {i}/{variant}'
                    choice = actions[max(range(len(actions)), key=lambda k: scores[k])]
                    timing_rows.append({'state': i, 'repeat': repeat, 'variant': variant_names[variant],
                                        'actions': len(actions), **measured,
                                        'changed_successor': canonical_successor_action_key(state, choice) != reference_key})
            print(json.dumps({'stage': 'paired_repeat_done', 'repeat': repeat}), flush=True)
        save('timings.json', timing_rows)
        totals = {name: {field: sum(r[field] for r in timing_rows if r['variant'] == name)
                         for field in ('prepare', 'tensor', 'forward', 'output', 'total')}
                  for name in variant_names}
        strata = {}
        for label, predicate in [('no_returns', lambda x: x['return_actions'] == 0),
                                 ('has_returns', lambda x: x['return_actions'] > 0)]:
            selected = {i for i, r in enumerate(info) if predicate(r)}
            strata[label] = {'positions': len(selected),
                             'actions': sum(info[i]['actions'] for i in selected),
                             'extra_return_rows': sum(info[i]['extra_return_rows'] for i in selected),
                             'timing_totals': {name: sum(r['total'] for r in timing_rows if r['state'] in selected and r['variant'] == name) for name in variant_names}}
        imported_sources = {}
        import sys
        for module in list(sys.modules.values()):
            value = getattr(module, '__file__', None)
            if value:
                path = Path(value).resolve()
                if path.is_relative_to(ROOT) and path.suffix == '.py':
                    relative = path.relative_to(ROOT).as_posix()
                    imported_sources[relative] = sha(path)
                    if relative in frozen:
                        assert imported_sources[relative] == frozen[relative], relative
        for path, digest in inputs.items():
            assert sha(ROOT/path) == digest, path
        assert sha(MAIN) == MAIN_SHA and runner.create_time() == 1792537125.6896 and runner.is_running()
        result = {'complete': True, 'positions': len(states), 'trajectories': len(trajectories),
                  'exact_tensor_checks': tensor_checks, 'exact_full_score_comparisons': len(states)*4,
                  'collection': dict(meters), 'timing_totals': totals, 'strata': strata,
                  'action_rows': sum(r['actions'] for r in info),
                  'extra_return_rows': sum(r['extra_return_rows'] for r in info),
                  'canonical_rows': sum(r['canonical'] for r in info),
                  'changed_successors': {name: sum(r['changed_successor'] for r in timing_rows if r['variant'] == name and r['repeat'] == 0) for name in variant_names},
                  'repeats': {str(rep): {name: sum(r['total'] for r in timing_rows if r['variant'] == name and r['repeat'] == rep) for name in variant_names} for rep in range(2)},
                  'wall_seconds': time.perf_counter()-started, 'cpu_seconds': time.process_time()-cpu_started,
                  'peak_sampled_rss': peak_rss, 'main_freeze_unchanged': True, 'main_identity_alive': True,
                  'imported_sources': imported_sources, 'production_enabled': False}
        save('result.json', result)
        print(json.dumps({k: v for k, v in result.items() if k != 'imported_sources'}), flush=True)
    except BaseException:
        import traceback
        save('failure.json', {'traceback': traceback.format_exc(), 'wall_seconds': time.perf_counter()-started})
        raise
    finally:
        process.cpu_affinity(original_affinity)
        process.nice(original_priority)


if __name__ == '__main__':
    main()
