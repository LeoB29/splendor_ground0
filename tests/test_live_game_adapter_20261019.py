"""Live boundary/transfer fixtures with fake neural models only."""
from copy import deepcopy
import ast
import gzip
import json
from types import SimpleNamespace

import pytest
import torch

from tools.live_game_operational_20261019 import common as c, local, worker
from tools.live_game_operational_20261018 import common as pilot_c, local as pilot_local
from tools.corrected_planner_qualification_20261018 import local as old
from tools.external_strength_screen_20261018 import rules, audit as ledger


def test_unchanged_player_and_audit_implementation_matches_accepted_pilot_ast():
    def functions(path):
        return {node.name: ast.dump(node, include_attributes=False)
                for node in ast.parse(path.read_text(encoding='utf-8')).body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    names = {
        'local.py': ('initialize','make_observer_payload','choose'),
        'common.py': ('opponent_observer','sample_worlds','stress_priority','stress_cells',
                      'expected_stress_cells','check_wrapper','check_audit_proof'),
        'verify.py': ('sample_worlds','_hash','check_outcome','check_trace_structure','audit_decision'),
        'worker.py': ('trace_path',),
    }
    for filename, functions_to_compare in names.items():
        old_functions=functions(c.ROOT/'tools/live_game_operational_20261018'/filename)
        new_functions=functions(c.ROOT/'tools/live_game_operational_20261019'/filename)
        assert all(new_functions[name]==old_functions[name] for name in functions_to_compare)
    assert local.qualified is pilot_local.qualified
    for name in ('registered_model','score_actions','trajectory','proposal_for_root'):
        assert getattr(local,name) is getattr(pilot_local,name)


class FakeRaw:
    def score_actions(self, state, legal):
        return tuple(0.0 for _ in legal)


class FakeProposal:
    config = SimpleNamespace(observation_size=500, action_feature_size=200)
    def __call__(self, observation, features, mask):
        return torch.zeros((1, features.shape[1])), torch.zeros(1)


@pytest.fixture
def fake_models(monkeypatch):
    monkeypatch.setattr(old, '_INITIALIZED', True)
    monkeypatch.setattr(old, '_RAW', FakeRaw())
    monkeypatch.setattr(old, '_MODELS', {f'anchor_{fit}': FakeProposal() for fit in c.FITS})
    monkeypatch.setattr(old, '_ARTIFACTS', {
        f'anchor_{fit}': {'sha256': str(fit), 'source_fit': fit} for fit in c.FITS})


def near_terminal(actor=1):
    state = rules.CorrectedEnv(seed=20261018).initial_state()
    selected = [card for card in rules.corrected_cards_by_id().values() if card.points == 5][:3]
    ids = {card.card_id for card in selected}
    for tier in (1, 2, 3):
        for groups in (state.visible_tier_cards, state.hidden_tier_decks):
            groups[tier] = [card for card in groups[tier] if card.card_id not in ids]
        while len(state.visible_tier_cards[tier]) < 4 and state.hidden_tier_decks[tier]:
            state.visible_tier_cards[tier].append(state.hidden_tier_decks[tier].pop())
        state.deck_counts[tier] = len(state.hidden_tier_decks[tier])
    for card in selected:
        state.players[1-actor].purchased_cards.append(card)
        state.players[1-actor].score += card.points
        state.players[1-actor].bonuses[card.bonus_color] += 1
    state.start_player = 1-actor
    state.current_player, state.turn_index, state.pending_round_end = actor, 1, True
    ledger.validate_ledger(rules.serialize_snapshot(state))
    return state


def observe(state=None, game=7):
    state = near_terminal() if state is None else state
    return local.make_observer_payload(rules.serialize_snapshot(state), c.operational_context(game, state.turn_index))


def write_trace(path, rows):
    with path.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=3) as stream:
            for row in rows: stream.write((c.canonical(row) + '\n').encode())


def wrapper(observer, decision, path, rows):
    write_trace(path, rows)
    return {'decision': decision, 'trace': {'sha256': c.file_hash(path), 'bytes': path.stat().st_size,
            'records': len(rows)}, 'ruleset_id': c.RULESET_ID, 'catalogue_sha256': c.CATALOGUE_SHA256,
            'profile_id': c.LOCAL_PROFILE_ID, 'observer_sha256': c.digest(observer)}


def test_all_nine_pairs_keep_prescribed_fit_seat_world_and_separate_streams():
    schedule = c.make_schedule()
    assert tuple(row['game'] for row in schedule) == (6,8,10,7,9,11,12,14,16,13,15,17,18,20,22,19,21,23)
    assert sorted(row['game'] for row in schedule) == list(range(6,24))
    assert len({row['seed'] for row in schedule}) == 9
    assert len({row['seed'] for row in schedule} & {row['seed'] for row in pilot_c.make_schedule()}) == 0
    for pair in range(3,12):
        fit = c.FITS[(pair-3)%3]
        first, second = [c.operational_context(2*pair+leg, 0) for leg in (0,1)]
        assert first['fit'] == second['fit'] == fit
        assert first['stream'] != second['stream'] and first['pair'] == second['pair'] == pair
        rows = [row for row in schedule if row['pair'] == pair]
        assert len(rows)==2 and rows[0]['seed']==rows[1]['seed']
        assert {row['local_seat'] for row in rows}=={0,1}
        assert all(row['local_seat']==1-((pair+row['game']%2)%2) for row in rows)
    assert [c.operational_context(g,0)['fit'] for g in c.VALIDATION_GAMES] == list(c.FITS)
    assert all(c.local_seat(g)==0 for g in c.VALIDATION_GAMES)
    assert c.MAX_SECONDS==24*3600 and c.POST_GAME_RESERVE==6*3600 and c.MAX_BYTES==4*1024**3


@pytest.mark.parametrize('game', [False,True,0,5,24,-1,6.0,'6'])
def test_operational_boundary_excludes_pilot_and_outside_games(game):
    with pytest.raises(ValueError): c.operational_context(game,0)


@pytest.mark.parametrize('mode', ['legacy_transfer','unknown'])
def test_unregistered_context_mode_rejected(mode):
    context=c.operational_context(7,1);context['mode']=mode
    with pytest.raises(ValueError): c.check_context(context)


@pytest.mark.parametrize('damage', ['fit', 'stream', 'pair', 'identity', 'turn', 'namespace', 'extra', 'actor'])
def test_live_context_and_observer_boundary_fail_closed(damage):
    payload = observe()
    if damage == 'actor': payload['observation']['current_player'] = 0
    elif damage == 'extra': payload['context']['referee_seed'] = 42
    elif damage == 'fit': payload['context']['fit'] = c.FITS[1]
    elif damage == 'stream': payload['context']['stream'] = '0'*64
    elif damage == 'pair': payload['context']['pair'] = True
    elif damage == 'identity': payload['context']['identity'][0] = 'main'
    elif damage == 'turn': payload['observation']['turn_index'] += 1
    else: payload['context']['namespace'] = 'different'
    with pytest.raises(ValueError): local.state_from_observer_payload(payload)


def test_pilot_transfer_admits_only_bound_observations_and_exact_prior_context():
    for index in c.TRANSFER_INDICES:
        original = deepcopy(c.pilot_observer(index))
        payload = {**original, 'schema': c.SCHEMA, 'context': c.pilot_context(index)}
        local.state_from_observer_payload(payload)
        bad = deepcopy(payload); bad['context']['stream'] = '0'*64
        with pytest.raises(ValueError): local.state_from_observer_payload(bad)
        bad = deepcopy(payload); bad['observation']['turn_index'] += 2
        with pytest.raises(ValueError): local.state_from_observer_payload(bad)
    with pytest.raises(ValueError): c.pilot_context(1)


@pytest.mark.parametrize('game', [7, 8, 11])
def test_full128_orchestration_preserves_policy_ties_and_hidden_order(fake_models, game):
    state = near_terminal(); observer = observe(state, game)
    traces = []
    decision = local.choose(observer, trace_sink=traces.append)
    assert decision['record']['proposal']['model_id'] == f"anchor_{c.FITS[(game//2-3)%3]}"
    assert len(decision['record']['worlds']) == 128 and len(traces) == 512
    assert decision['record']['choice'] == 0
    assert all(row['turns'] == 1 and len(row['steps']) == 1 for row in traces)
    changed = deepcopy(state)
    for values in changed.hidden_tier_decks.values(): values.reverse()
    assert observe(changed, game) == observer


@pytest.mark.parametrize('fixture,pilot_game', [(0,1),(6,2),(12,5)])
def test_identical_pilot_primitives_transfer_except_profile_label(fake_models, monkeypatch, tmp_path, fixture, pilot_game):
    # Synthetic one-ply full4x128 transfer; no checkpoint or saved-root rollout.
    state = near_terminal(actor=0)
    snapshot = rules.serialize_snapshot(state)
    before_observer = pilot_local.make_observer_payload(snapshot, pilot_c.operational_context(pilot_game,state.turn_index))
    monkeypatch.setattr(c,'pilot_observer',lambda index: deepcopy(before_observer))
    after_observer = local.make_observer_payload(snapshot,c.pilot_context(fixture))
    assert after_observer['context'] == {**before_observer['context'],'mode':'pilot_transfer','fixture_index':fixture}
    before_trace,after_trace=[],[]
    before=pilot_local.choose(before_observer,trace_sink=before_trace.append)
    after=local.choose(after_observer,trace_sink=after_trace.append)
    assert after['record']['profile']==c.LOCAL_PROFILE_ID
    after['record']['profile']=before['record']['profile']
    assert before==after and before_trace==after_trace
    one,two=tmp_path/'one.gz',tmp_path/'two.gz'
    write_trace(one,before_trace);write_trace(two,after_trace)
    assert one.read_bytes()==two.read_bytes()


def test_pilot_source_reader_never_opens_hidden_fixture_files(monkeypatch):
    c.pilot_observer.cache_clear()
    source=pilot_local.make_observer_payload(rules.serialize_snapshot(near_terminal(actor=0)),pilot_c.operational_context(1,1))
    seen=[]
    def read(path):
        seen.append(path)
        assert path==c.PRIOR_PILOT/'decision_attempts/validation/v002.json'
        return {'id':'validation/v002','status':'completed','correct':True,'observer':deepcopy(source)}
    monkeypatch.setattr(c,'read',read)
    try:
        assert c.pilot_observer(0)==source
        assert len(seen)==1
    finally:
        c.pilot_observer.cache_clear()


def test_worker_writes_exact_trace_inside_result_and_rejects_reuse(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'OUTPUT', tmp_path)
    path = tmp_path/'continuations.jsonl.gz'
    monkeypatch.setenv('LIVE_TRACE_PATH', str(path))
    decision = {'action': {'synthetic': True}, 'record': {}}
    def choose(payload, *, boundary, trace_sink):
        boundary(); trace_sink({'test': 'trace'}); return decision
    context = {'mode': 'local', 'local': SimpleNamespace(choose=choose), 'used': False}
    result = worker.handle(context, {'observer': {'public': 'fixture'}})
    assert result['decision'] == decision and result['trace']['records'] == 1
    assert result['trace']['sha256'] == c.file_hash(path)
    assert gzip.decompress(path.read_bytes()) == b'{"test":"trace"}\n'
    with pytest.raises(ValueError, match='Exactly one'): worker.handle(context, {'observer': {}})


def test_worker_trace_boundary_never_overwrites_or_leaves_output(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'OUTPUT', tmp_path/'allowed')
    for path in (tmp_path/'continuations.jsonl.gz', tmp_path/'allowed'/'other.gz'):
        with pytest.raises(ValueError): worker.trace_path(path)
    path = tmp_path/'allowed'/'continuations.jsonl.gz'; path.parent.mkdir(); path.write_bytes(b'preserve')
    monkeypatch.setenv('LIVE_TRACE_PATH', str(path))
    context = {'mode': 'local', 'local': SimpleNamespace(choose=lambda *_: pytest.fail('Overwritten')), 'used': False}
    with pytest.raises(FileExistsError): worker.handle(context, {'observer': {}})
    assert path.read_bytes() == b'preserve'


def test_worker_initialization_rejects_unknown_mode_before_models(monkeypatch):
    monkeypatch.setenv('LIVE_WORKER_MODE', 'opponent')
    monkeypatch.setattr(local, 'initialize', lambda: pytest.fail('Models initialized'))
    with pytest.raises(ValueError): worker.initialize()
