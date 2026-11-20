"""Prospective source parity and outcome-free fixed-quota timing selection."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

from tools.external_strength_screen_20261018 import rules, audit
from tools.external_strength_screen_20261018_02 import common as c
from tools.external_strength_screen_20261018_02.runner import worker, decision
from tools.local_player_qualification_128.clock import processor_topology

BASE = c.ROOT / 'outputs/external_strength_screen_20261018_02'
SEARCH = c.ROOT / 'outputs/external_opponent_search_20261018_02'
SOURCE = SEARCH / 'splendorust'
GRID = [128, 64, 32, 16, 8, 4, 1]


def fixtures():
    return c.read(c.ROOT / 'outputs/external_strength_screen_20261018_01/fixtures.json')


def make_profile(iterations):
    binary = SEARCH / 'native_build_v2/target/x86_64-pc-windows-gnu/release/splendor-observer-bridge.exe'
    model = SEARCH / 'splendorust_model.pt'
    definition = SOURCE / 'research/architecture_pivots/models.py'
    paths = {p for p in SOURCE.rglob('*') if p.is_file()}
    paths.update((binary, model, SEARCH / 'native_build_v2/build_receipt_v2.json',
                  SEARCH / 'native_build_v2/Cargo.lock', SEARCH / 'model_receipt.json',
                  c.ROOT / 'tools/external_strength_screen_20261018_02/native_bridge/src/main.rs',
                  c.ROOT / 'tools/external_strength_screen_20261018_02/bridge.py',
                  c.ROOT / 'tools/external_strength_screen_20261018_02/splendorust_adapter.py'))
    for p in SEARCH.glob('*.json'):
        paths.add(p)
    return {'schema': c.PROFILE_SCHEMA, 'opponent_id': c.OPPONENT_ID,
            'profile_id': f'entity_onehot_puct{iterations}_canonical_atomic_prereveal_cpu_v1',
            'source_pin': '1828507b8d2f0e02bc03d102b9c6d29a54fa8d57',
            'deterministic_fixed_work': True, 'internal_budget_seconds': 0.9,
            'iterations': iterations, 'depth': 64, 'cpuct': 0.4, 'world_pool': 3,
            'chance_universes': 3, 'dynamic_fpu': True, 'fpu_reduction': 0.02965,
            'uniform_prior': 0, 'inference_batch_size': 1, 'fresh_agent_per_turn': True,
            'native_binary': binary.relative_to(c.ROOT).as_posix(),
            'checkpoint': model.relative_to(c.ROOT).as_posix(),
            'model_definition': definition.relative_to(c.ROOT).as_posix(),
            'artifacts': [c.artifact(p) for p in sorted(paths)]}


def start():
    target = BASE / 'qualification/execution_start.json'
    if not target.exists():
        c.save_new(target, {'time': c.now(), 'monotonic_start': time.monotonic(),
                            'execution_cap_seconds': 14400})
    return c.read(target)['monotonic_start'] + 14400


def source_check(attempt):
    deadline = start()
    c.boundary(BASE, deadline)
    import psutil
    psutil.Process().cpu_affinity([processor_topology()['selected_logical_processor']])
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    from tools.external_strength_screen_20261018_02.splendorust_adapter import Opponent
    from tools.external_strength_screen_20261018_02 import bridge
    profile = make_profile(1)
    target = BASE / 'qualification' / f'source_attempt_{attempt:02d}.json'
    if target.exists(): raise ValueError('Never overwrite source qualification')
    rows = []; failure = None; external = None
    try:
        external = Opponent(profile)
        native = external.exchange({'op': 'catalogue'})
        expected_cards = rules.corrected_cards_by_id()
        for index, local_id in enumerate(bridge.RUST_CARD_IDS):
            card, other = expected_cards[local_id], native['cards'][index]
            assert other == {'tier': card.tier-1, 'bonus': bridge.COLORS.index(card.bonus_color),
                             'points': card.points, 'cost': [card.cost.get(k, 0) for k in bridge.COLORS]}
        for index, local_id in enumerate(bridge.RUST_NOBLE_IDS):
            noble = rules.corrected_nobles_by_id()[local_id]
            assert native['nobles'][index] == [noble.requirement.get(k, 0) for k in bridge.COLORS]
        for index, item in enumerate(fixtures()):
            c.boundary(BASE, deadline, memory_headroom=False)
            state = rules.deserialize_snapshot(item['snapshot'])
            observer = rules.observation_payload(state)
            obs = bridge.encode_observation(observer)
            seed = int(c.digest(observer)[:16], 16)
            expected = bridge.main_actions(observer)
            actual = external.exchange({'op': 'inspect', 'observation': obs, 'seed': seed})['legal']
            assert actual == ([] if expected == [{'kind': 'Pass'}] else expected), (index, actual, expected)
            row = {'fixture': index, 'observer_sha256': c.digest(observer), 'main_choices': len(actual),
                   'native_legal_order_equal': True, 'forced_pass': not actual}
            if obs['turns'] < 124 and actual:
                parity = external.exchange({'op': 'parity', 'observation': obs, 'seed': seed,
                    'model_bin': str(SOURCE / 'web/public/models/57f6e227f8ac0382b7fa67dba6b58ec6663d1f635c7b9f583937867cd6692deb.bin')})
                assert parity['max_abs_delta'] <= 0.0005, (index, parity['max_abs_delta'])
                row['model_parity'] = parity
            rows.append(row)
            print(c.canonical({'source_fixture': index+1, 'max_abs_delta': row.get('model_parity', {}).get('max_abs_delta')}), flush=True)
    except BaseException:
        failure = traceback.format_exc()
    finally:
        if external is not None: external.close()
    receipt = {'status': 'passed' if failure is None else 'failed', 'failure': failure,
               'cards_matched': 90 if rows else None, 'nobles_matched': 10 if rows else None,
               'policy_value_abs_tolerance': 0.0005, 'rows': rows,
               'profile_artifacts': profile['artifacts'], 'time': c.now()}
    c.save_new(target, receipt)
    if failure: raise RuntimeError(failure)


def timing(attempt, source_attempt):
    deadline = start()
    source = BASE / 'qualification' / f'source_attempt_{source_attempt:02d}.json'
    source_receipt = c.read(source)
    if source_receipt['status'] != 'passed': raise ValueError('Source check failed')
    c.verify(source_receipt['profile_artifacts'])
    cpu = processor_topology()['selected_logical_processor']
    receipt_path = BASE / 'qualification' / f'timing_attempt_{attempt:02d}.json'
    if receipt_path.exists(): raise ValueError('Never overwrite timing qualification')
    attempts = []; chosen = None; failure = None
    try:
        with c.execution_lease():
            for quota in GRID:
                pilot = BASE / 'qualification' / f'timing_{attempt:02d}_quota_{quota:03d}'
                profile = make_profile(quota)
                c.save_new(pilot / 'opponent_profile.json', profile)
                rows = []; ok = True; prior = {}
                with worker(pilot, 'pilot', deadline, cpu) as child:
                    for repeat in range(2):
                        for index, item in enumerate(fixtures()):
                            c.boundary(BASE, deadline, memory_headroom=False)
                            state = rules.deserialize_snapshot(item['snapshot'])
                            row = decision(child, state, 'opponent')
                            row.update({'repeat': repeat, 'fixture': index})
                            c.save_new(pilot / 'requests' / f'{len(rows):03d}.json', row)
                            rows.append(row)
                            clock = row['clock']
                            elapsed = clock.get('elapsed_seconds', float('inf'))
                            # The frozen supervisor reports elapsed_seconds for its real request clock.
                            if clock['status'] != 'ok' and not (clock['status'] == 'timeout' or
                                    (clock['status'] == 'worker_error' and 'TimeoutError' in json.dumps(clock))):
                                raise ValueError('Non-timing prerequisite failed: ' + json.dumps(clock))
                            if clock['status'] != 'ok' or elapsed > .70:
                                ok = False; break
                            if repeat and row['clock']['result'] != prior[index]:
                                raise ValueError('Deterministic fixed-work result changed')
                            prior[index] = row['clock']['result']
                        if not ok: break
                outcome = {'quota': quota, 'passed_margin': ok, 'requests': len(rows),
                           'maximum_elapsed_seconds': max(r['clock'].get('elapsed_seconds', float('inf')) for r in rows),
                           'profile': c.artifact(pilot / 'opponent_profile.json')}
                attempts.append(outcome); print(c.canonical(outcome), flush=True)
                if ok:
                    chosen = profile; break
            if chosen is None: raise ValueError('No fixed quota passed the one-second deployment margin')
    except BaseException:
        failure = traceback.format_exc()
    receipt = {'status': 'passed' if failure is None else 'failed', 'failure': failure,
               'grid': GRID, 'maximum_accepted_seconds': .70, 'attempts': attempts,
               'selected_quota': chosen['iterations'] if chosen else None,
               'source_check': c.artifact(source), 'no_match_outcomes_used': True, 'time': c.now()}
    c.save_new(receipt_path, receipt)
    if failure: raise RuntimeError(failure)
    c.save_new(BASE / 'opponent_profile.json', chosen)
    c.save_new(BASE / 'qualification/opponent_qualification.json', {
        'status': 'passed_fixture_qualification', 'opponent_id': c.OPPONENT_ID,
        'profile_id': chosen['profile_id'], 'profile_sha256': c.file_hash(BASE / 'opponent_profile.json'),
        'deterministic_fixed_work': True, 'ruleset_id': rules.RULESET_ID,
        'catalogue_sha256': rules.CATALOGUE_SHA256, 'source_check': c.artifact(source),
        'timing_check': c.artifact(receipt_path), 'fixtures': 23, 'requests': 46,
        'selected_quota': chosen['iterations'], 'time': c.now()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['source', 'timing'])
    parser.add_argument('--attempt', type=int, default=1); parser.add_argument('--source-attempt', type=int, default=1)
    args = parser.parse_args()
    source_check(args.attempt) if args.mode == 'source' else timing(args.attempt, args.source_attempt)
