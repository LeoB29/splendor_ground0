"""Prospective, outcome-blind saved-observer panel; no models or fresh games.

Only 512 consumed moves are decoded. The accepted corrected corpus/cache
receipts certify the complete eligible-root inventories used for selection.
Source game outcomes, source actions and source values are never selection keys.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOMAIN = 'splendorust_proxy_calibration_20261113_panel_v1'
SCHEMA = 'corrected_saved_observer_proxy_panel_v1'
BUDGETS = (64, 128, 256, 512)
CONDITIONS = tuple((q, seed, 'control') for seed in ('A', 'B') for q in BUDGETS) + ((2048, 'R', 'reference'), (2048, 'S', 'reference'))
SELECTION = dict(path='outputs/splendorust_full_corpus_20261110_01/selection.json', bytes=19335311,
                 sha256='07f2b18cc99c0e51393d9e1cc5f670c2f0eae9e4389f51ab133bc40939593527')
CACHE_ACCEPTANCE = dict(path='outputs/splendorust_full_corpus_20261110_01/corpus/cache_acceptance.json', bytes=805,
                        sha256='447163bcb1ff900bf4a77d53a433faf9e2ecdd7c9105123858f7371818142e03')
CORPUS_ACCEPTANCE = dict(path='outputs/splendorust_independent_data_20261107_01/collection/corpus_acceptance.json', bytes=14819,
                         sha256='d65e806ed856bef16a0710c284bffe07ab44261a253e96bd44fa0597fff7524e')


def require(condition, message):
    if not condition:
        raise ValueError('PROXY_PANEL: ' + message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def root_id(game_index, ply):
    return f'g{game_index:04d}_p{ply:06d}'


def choose_four(roots, game_index):
    """Four chronological equal-count bands, deterministic hash-min per band.

    This function receives only accepted root identities, not decoded moves.
    Extra dictionary fields cannot affect selection.
    """
    require(type(game_index) is int and 1024 <= game_index < 1152, 'held-out game identity')
    identities = []
    for row in roots:
        require(row['game_index'] == game_index and row['split'] in ('validation', 'offline'), 'held-out root identity')
        require(type(row['ply']) is int and 0 <= row['ply'] < 124, 'eligible chronological root')
        identities.append((row['game_index'], row['ply']))
    require(len(identities) >= 4 and len(set(identities)) == len(identities), 'at least four unique roots; no relaxation')
    ordered = sorted(identities)
    result = []
    for band in range(4):
        start, stop = band * len(ordered) // 4, (band + 1) * len(ordered) // 4
        entries = ordered[start:stop]
        chosen = min(entries, key=lambda x: (digest([DOMAIN, 'within_chronological_band', band, list(x)]), x))
        result.append(dict(game_index=chosen[0], ply=chosen[1], quantile_band=band,
                           band_start=start, band_stop=stop, eligible_roots_in_game=len(ordered)))
    return result


def _identity():
    from tools import future_training_rules_20261021 as guard
    return dict(ruleset_id=guard.RULESET_ID, catalogue_sha256=guard.CATALOGUE_SHA256,
                turn_contract=guard.TURN_CONTRACT, origin='generated_under_declared_rules')


def build_panel(*, boundary=lambda: None, runtime_sources=None):
    from tools import future_training_rules_20261021 as guard
    from tools.splendorust_independent_data_20261107.collection import data
    from tools.external_strength_screen_20261018_02 import bridge

    sources = guard.verify_runtime_sources() if runtime_sources is None else runtime_sources
    require(sources == guard.SOURCE_HASHES, 'startup corrected source verification receipt')
    bindings = {}

    def read(item):
        boundary()
        value = data.read_artifact(item)
        previous = bindings.setdefault(item['path'], deepcopy(item))
        require(previous == item, 'one immutable binding per source path')
        return value

    selection = read(SELECTION)
    corpus_acceptance = read(CORPUS_ACCEPTANCE)
    cache_acceptance = read(CACHE_ACCEPTANCE)
    cache = read(cache_acceptance['manifest'])
    identity = _identity()
    for item in (selection, corpus_acceptance, cache_acceptance, cache):
        require(all(item.get(k) == v for k, v in identity.items()), 'corrected source identity')
    require(selection['source_receipt']['acceptance'] == CORPUS_ACCEPTANCE, 'selection binds accepted collection')
    require(corpus_acceptance['passed'] is True and corpus_acceptance['status'] == 'accepted' and
            corpus_acceptance['game_count'] == 1152 and corpus_acceptance['raw_semantics_and_all_transitions_verified'] is True,
            'independently accepted corrected collection')
    require(cache_acceptance['status'] == 'independently_verified' and cache_acceptance['rows'] == 64108 and
            all(cache_acceptance[k] is True for k in ('all_feature_bytes_checked', 'all_q_actor_and_terminal_targets_checked', 'all_source_rows_checked')),
            'complete accepted cache checks')
    require(cache['selection'] == SELECTION and cache['rows'] == 64108 and cache['source_moves_authenticated'] == 64108,
            'accepted cache authenticates exact source selection and all source moves')
    games = [g for g in selection['games'] if g['spec']['split'] in ('validation', 'offline')]
    require(len(games) == 128 and {g['spec']['game_index'] for g in games} == set(range(1024, 1152)), 'all 128 held-out games')
    require(Counter(g['spec']['split'] for g in games) == {'validation': 64, 'offline': 64}, 'both complete held-out splits')
    ranges = {g['game_index']: g for g in cache['game_ranges'] if g['split'] != 'train'}
    accepted_audits = {x['path']: x for x in corpus_acceptance['independent_cell_audits']}
    audits = {}
    roots, metadata = [], {}
    for game in sorted(games, key=lambda g: g['spec']['game_index']):
        boundary()
        spec = game['spec']; gid = spec['game_index']; split = spec['split']
        require(spec['purpose'] == 'independent_corrected_training_source' and spec['quota'] == 512 and
                spec['world_pool'] == 3 and spec['cpuct'] == .4, 'accepted fixed original teacher source')
        inventory = game['roots']
        require(all(x['split'] == split for x in inventory), 'one declared split per source game')
        cached = ranges[gid]
        require(cached['source_game'] == game['source_game'] and cached['target'] == game['artifact'] and
                cached['stop'] - cached['start'] == len(inventory), 'accepted cache binds complete game inventory')
        audit_item = game['independent_cell_audit']
        require(accepted_audits.get(audit_item['path']) == audit_item, 'source game proof accepted by corpus receipt')
        if audit_item['path'] not in audits:
            audits[audit_item['path']] = read(audit_item)
        audit = audits[audit_item['path']]
        require(audit['full_cohort_verified'] is True and type(audit['owned_processes_absent']) is list and
                bool(audit['owned_processes_absent']), 'accepted source cell complete with recorded absent identities')
        proof = next(x for x in audit['game_proofs'] if x['game_id'] == spec['game_id'])
        require(proof['document'] == game['source_game'] and proof['target'] == game['artifact'], 'exact game proof bindings')
        target = read(game['artifact'])
        require(target['spec'] == spec and target['teacher_simulations'] == 512 and
                all(target.get(k) == v for k, v in identity.items()), 'corrected target game identity')
        expected_inventory = [dict(game_index=gid, ply=x['ply'], split=split,
            root_input_sha256=x['root_input_sha256'], row_sha256=x['row_sha256']) for x in target['examples']]
        require(expected_inventory == inventory and all(x['eligible'] is True for x in target['examples']), 'complete accepted eligible-root inventory')
        chosen = choose_four(inventory, gid)
        source_game = read(game['source_game'])
        require(source_game['spec'] == spec and all(source_game.get(k) == v for k, v in identity.items()), 'corrected physical source game')
        by_ply = {x['ply']: x for x in inventory}
        for selected in chosen:
            ply = selected['ply']; item = source_game['rows'][ply]
            row = read(item)
            decoded = data.decode_row(row, expected_spec=spec, expected_ply=ply, expected_quota=512)
            original = by_ply[ply]
            require(decoded['eligible'] is True and decoded['root_input_sha256'] == original['root_input_sha256'] and
                    decoded['row_sha256'] == original['row_sha256'], 'actual corrected decoder and root identity')
            observer = row['observer']
            actions = bridge.main_actions(observer)
            require(observer['observation']['turn_index'] < 124 and len(actions) >= 2 and actions != [{'kind': 'Pass'}],
                    'supported multi-choice Main root before late fallback')
            key = root_id(gid, ply)
            roots.append(dict(root_id=key, observer=deepcopy(observer), observer_sha256=digest(observer)))
            metadata[key] = dict(**selected, split=split, source_move=deepcopy(item), source_game=deepcopy(game['source_game']),
                source_target=deepcopy(game['artifact']), independent_cell_audit=deepcopy(audit_item),
                source_row_sha256=decoded['row_sha256'], root_input_sha256=decoded['root_input_sha256'],
                source_game_proof_sha256=digest(proof), source_clock_sha256=digest(row['clock']),
                clock_source='saved physical move.clock; not used for selection or fresh timing',
                source_quota=512, source_main_support=len(actions), actor=observer['observation']['current_player'])
    game_order = sorted(range(1024, 1152), key=lambda g: (digest([DOMAIN, 'game_order', g]), g))
    game_rank = {g: i for i, g in enumerate(game_order)}
    roots.sort(key=lambda row: (metadata[row['root_id']]['quantile_band'], game_rank[metadata[row['root_id']]['game_index']]))
    panel = dict(schema=SCHEMA, domain=DOMAIN, **identity, runtime_sources=deepcopy(sources), roots=roots,
        host_metadata=metadata, source_bindings=sorted(bindings.values(), key=lambda x: x['path']),
        design=dict(games=128, roots=512, roots_per_game=4, heldout_splits={'validation': 64, 'offline': 64},
            selection='four chronological equal-count bands; independent hash-min within each; hash-interleaved game order',
            selection_uses_outcomes=False, selection_uses_actions=False, selection_uses_values=False, selection_uses_timing=False,
            source_population='previously examined development held-out games; not a fresh confirmation sample',
            eligibility='accepted corrected multi-choice Main roots with turn_index below124',
            complete_root_inventories_reused=True, physical_moves_decoded=512, historical_corpus_rescanned=False,
            new_games=0, new_models=0, control_budgets=list(BUDGETS), control_seed_labels=['A', 'B'],
            reference_budget=2048, reference_seed_labels=['R', 'S'], main_requests=5120, exact_reproduction_requests=32,
            same_network_reference_bias=True, reference_is_independent_oracle=False))
    validate_panel(panel)
    return panel


def validate_panel(panel, *, authenticate=False, boundary=lambda: None):
    require(panel['schema'] == SCHEMA and panel['domain'] == DOMAIN, 'panel schema and domain')
    require(all(panel.get(k) == v for k, v in _identity().items()), 'panel corrected identity')
    require(len(panel['roots']) == 512 and len(panel['host_metadata']) == 512, 'complete512root panel')
    keys = [r['root_id'] for r in panel['roots']]
    require(len(set(keys)) == 512 and set(keys) == set(panel['host_metadata']), 'unique complete metadata join')
    counts, splits = Counter(), {}
    for row in panel['roots']:
        require(set(row) == {'root_id', 'observer', 'observer_sha256'} and digest(row['observer']) == row['observer_sha256'], 'lawful-observer-only payload and hash')
        m = panel['host_metadata'][row['root_id']]
        require(row['root_id'] == root_id(m['game_index'], m['ply']) and m['split'] in ('validation', 'offline'), 'host identity')
        require(0 <= m['quantile_band'] < 4 and m['source_main_support'] >= 2 and row['observer']['observation']['turn_index'] < 124, 'eligible root and band')
        counts[(m['game_index'], m['quantile_band'])] += 1
        require(m['game_index'] not in splits or splits[m['game_index']] == m['split'], 'split consistency')
        splits[m['game_index']] = m['split']
    require(set(splits) == set(range(1024, 1152)) and Counter(splits.values()) == {'validation': 64, 'offline': 64}, 'all held-out source games')
    require(set(counts) == {(g, b) for g in range(1024, 1152) for b in range(4)} and set(counts.values()) == {1}, 'exactly one root per band and game')
    if authenticate:
        from tools.splendorust_independent_data_20261107.collection import data
        for item in panel['source_bindings']:
            boundary(); data.read_artifact(item)
    return panel


def request_matrix(panel):
    """Ten conditions per root, deterministic cyclic interleave; no observations."""
    result = []
    for ordinal, root in enumerate(panel['roots']):
        rotated = CONDITIONS[ordinal % 10:] + CONDITIONS[:ordinal % 10]
        for quota, seed, role in rotated:
            key = root['root_id']
            result.append(dict(request_id=f'{key}_q{quota:04d}_{seed}', root_id=key,
                               quota=quota, seed_label=seed, role=role))
    return result


def reproduction_matrix(panel):
    """Sixteen prospective hash-selected roots; one control and reference each."""
    selected = sorted((r['root_id'] for r in panel['roots']), key=lambda x: (digest([DOMAIN, 'exact_reproduction_root', x]), x))[:16]
    by_key = {(x['root_id'], x['quota'], x['seed_label']): x for x in request_matrix(panel)}
    result = []
    for i, key in enumerate(selected):
        for quota, seed in ((BUDGETS[i % 4], ('A', 'B')[(i // 4) % 2]), (2048, ('R', 'S')[i % 2])):
            original = by_key[key, quota, seed]
            result.append(dict(original, request_id='repeat_' + original['request_id'], original_request_id=original['request_id']))
    return result


def known_live_controls(*, boundary=lambda: None):
    """Bind the accepted adjacent live comparisons; never pool their games."""
    from tools.splendorust_independent_data_20261107.collection import data
    definitions = [
        (128, 64, .56125, [.515, .6075],
         ('outputs/splendorust_search128_resume_20261029_01/analysis.json', 3144, '6332d255871c91ef99561c97b26eed87d45698fad609f837f74a998e91c9acde'),
         ('outputs/splendorust_search128_saved_audit_20261029_continuity/recovered_verification_accepted.json', 3081, '8571ce7f3ee6adc52a3aee4b7e18b9dba3587dae1878ab629613d30e9b098680')),
        (256, 128, .61, [.565, .65375],
         ('outputs/splendorust_search256_recovery_20261030_02/main_analysis.json', 3958, '3ac252ca8ec996b67f4a5a150c5bdc8944cf9bbeaa59797c25ed1813459f57b1'),
         ('outputs/splendorust_search256_recovery_20261030_02_closeout/independent_acceptance_01.json', 6813, '40eb825c191ba57fd7e17738f4dd38426404592c7a29997f60c4fcc9ccc7c202')),
        (512, 256, .565, [.51875, .61125],
         ('outputs/splendorust_search512_20261101_01_closeout/report_data.json', 2103, '75e002d108d302c5a1566e7f676feaa8bfe6b2516a2fc04e826fa00fbcddb2fb'),
         ('outputs/splendorust_search512_20261101_01_closeout/final_closeout.json', 3455, '9ec2bef3534335fd816097c78e439a5bc5cfe2e83ea39cac5b6577aa42ab10c8')),
    ]
    result = []
    for candidate, reference, score, interval, summary, acceptance in definitions:
        descriptors = [dict(zip(('path', 'bytes', 'sha256'), x)) for x in (summary, acceptance)]
        boundary(); report = data.read_artifact(descriptors[0]); accepted = data.read_artifact(descriptors[1])
        if candidate == 128:
            require(report['search_mean_score'] == score and report['paired_ci95'] == interval and
                    report['games'] == 400 and report['status'] == 'complete_verified_fixed_sample' and
                    accepted['status'] == 'separately_recovered_verification_accepted' and accepted['actual_owner_exit_code'] == 0,
                    'accepted128 versus64 fixed-sample comparison')
            recovery = 'Amended recovered benchmark; one already-emitted response adopted; original attempts preserved incomplete.'
        elif candidate == 256:
            require(report['official']['mean_score'] == score and report['official']['paired_ci95'] == interval and
                    report['games'] == 400 and report['complete_verification'] is True and accepted['approved'] is True,
                    'accepted256 versus128 fixed-sample comparison')
            recovery = 'Amended recovery with one disclosed fresh-clock move retry; whole affected game as loss still gives60.75%, CI56.25-65.125%.'
        else:
            require(report['strength']['score'] == score and report['strength']['paired_ci95'] == interval and
                    accepted['status'] == 'accepted_completed_verified', 'accepted512 versus256 fixed-sample comparison')
            recovery = 'Completed verified fixed sample; no move retries or replacements.'
        result.append(dict(candidate_simulations=candidate, reference_simulations=reference, games=400, pairs=200,
                           score=score, paired_ci95=interval, accepted_sources=descriptors, recovery_disclosure=recovery))
    return dict(comparisons=result, interpretation='Three separate adjacent positive controls, not pooled games or a validated conversion from proxy units to win rate.',
                proxy_uses_new_disjoint_seed_families=True, known_controls_validate_training_generalization=False)
