"""Model-free codec, public-information and complete-action counterexamples."""
from copy import deepcopy
from collections import Counter
import ast

import numpy as np
import pytest

from tools.dom_screen_20261027 import adapter as a, reference as r
from tools.external_strength_screen_20261018 import rules
from tools import future_training_rules_20261021 as training


def initial(seed=41, seat=0):
    state = rules.CorrectedEnv(seed=seed).initial_state()
    state.current_player = seat
    state.start_player = seat
    return state


def tokens(state, seat, counts):
    player = state.players[seat]
    for color in a.TOKENS:
        amount = counts.get(color, 0)
        state.bank_tokens[color] += player.tokens[color] - amount
        player.tokens[color] = amount


def reserve(state, seat, tier, source='deck'):
    if source == 'deck':
        card = state.hidden_tier_decks[tier].pop()
        state.deck_counts[tier] -= 1
    else:
        card = state.visible_tier_cards[tier].pop()
    state.players[seat].reserve_card(card, source)
    return card


def grant_bonus(state, seat, color, count):
    player = state.players[seat]
    for _ in range(count):
        located = next((tier, i, card) for tier, deck in state.hidden_tier_decks.items()
                       for i, card in enumerate(deck) if card.bonus_color == color and card.points == 0)
        tier, index, card = located
        state.hidden_tier_decks[tier].pop(index)
        state.deck_counts[tier] -= 1
        player.purchased_cards.append(card)
        player.bonuses[color] += 1


def observer(state):
    payload = rules.serialize_snapshot(state)
    training.decode_training_snapshot(payload)
    return rules.observation_payload(state)


def assert_parity(obs):
    p=a.prepare(obs)
    assert np.array_equal(p['features'],r.encode(obs))
    assert np.array_equal(p['source_mask'],r.source_mask(obs))
    assert p['features'].shape==(267,) and p['features'].dtype==np.float32
    assert p['legal_mask'].shape==(77,) and p['legal_mask'].dtype==bool
    legal=rules.legal_actions(rules.state_from_observation(obs))
    assert all(rules.deserialize_action(x) in legal for x in p['actions'] if x is not None)
    return p

def predictor(indices):
    index=0
    def run(features):
        nonlocal index
        out=np.zeros(77,dtype=np.float64);out[indices[min(index,len(indices)-1)]]=10;index+=1;return out
    return run

def check_trace(obs,result):
    assert a.validate_result(result,obs)['status']=='passed'
    for phase in result['diagnostics']['phases']:
        assert np.array_equal(np.asarray(phase['features'],dtype=np.float32),r.encode_phase(phase['context']))
        assert np.array_equal(phase['source_mask'],r.source_mask_phase(phase['context']))
        assert all(not enabled or phase['source_mask'][i] for i,enabled in enumerate(phase['mask']))
    assert rules.deserialize_action(result['action']) in rules.legal_actions(rules.state_from_observation(obs))
    return result

def test_source77_contract_corrected90cards10nobles():
    assert r.validate_mapping_contract()['static_indices_checked']==77
    cards,nobles=r.catalogue_mapping()
    assert len(cards)==90 and len(nobles)==10 and set(rules.CORRECTED_COSTS)<=set(cards)
    assert set(x['id'] for x in cards.values())==set(range(90))
    assert set(x['id'] for x in nobles.values())==set(range(10))

@pytest.mark.parametrize('seat',[0,1])
@pytest.mark.parametrize('seed',[1,41,99])
def test_initial_source_features_masks_and_corrected_short_take_restriction(seat,seed):
    p=assert_parity(observer(initial(seed,seat)))
    assert not p['legal_mask'][:15].any() and p['source_mask'][:15].all()
    assert p['legal_mask'].sum()==30
    check_trace(observer(initial(seed,seat)),a.decide(observer(initial(seed,seat)),predictor([15])))

@pytest.mark.parametrize('main',[30,42])
@pytest.mark.parametrize('return_index',[60,65])
def test_reserve_gold_overflow_complete_turn_and_prereveal_context(main,return_index):
    s=initial();tokens(s,0,{'white':2,'blue':2,'green':2,'red':2,'black':1,'gold':1});obs=observer(s);p=assert_parity(obs)
    result=check_trace(obs,a.decide(obs,predictor([main,return_index])))
    phases=result['diagnostics']['phases'];assert [x['selected_index'] for x in phases]==[main,return_index]
    assert result['action']['take_tokens']==['gold'] and result['action']['return_tokens']==[a.TOKENS[return_index-60]]
    ctx=phases[1]['context'];assert ctx['players'][0]['tokens'][5]==2 and ctx['bank'][5]==3 and ctx['decks'][0]==p['context']['decks'][0]-1
    if main==30:assert ctx['display'][0][0] is None
    else:assert ctx['display']==p['context']['display']
    assert len(ctx['players'][0]['reserved'])==1

@pytest.mark.parametrize('count',[7,8,9,10])
def test_repeated_policy_returns_use_updated_context_and_whole_turn_bound(count):
    s=initial();tokens(s,0,{'white':3,'blue':3,'green':count-6});obs=observer(s)
    result=check_trace(obs,a.decide(obs,predictor([15,60,61,62])))
    assert result['diagnostics']['policy_calls']==1+max(0,count+3-10)
    assert len(result['action']['return_tokens'])==max(0,count+3-10)
    phases=result['diagnostics']['phases']
    for first,second in zip(phases[1:],phases[2:]):
        assert sum(second['context']['players'][0]['tokens'])==sum(first['context']['players'][0]['tokens'])-1

def noble_state():
    s=initial()
    for c in ('white','blue','green'):grant_bonus(s,0,c,4)
    nm=r.catalogue_mapping()[1];desired=[k for k,v in nm.items() if v['id'] in (0,1,5)]
    s.nobles=[rules.corrected_nobles_by_id()[x] for x in desired]
    return s

def test_main_three_returns_then_noble_is_five_forwards():
    s=noble_state();tokens(s,0,{'white':3,'blue':3,'green':4});obs=observer(s)
    result=check_trace(obs,a.decide(obs,predictor([15,60,61,62,67])))
    assert [x['phase'] for x in result['diagnostics']['phases']]==['play','discard','discard','discard','choose_noble']
    assert result['diagnostics']['policy_calls']==a.MAX_POLICY_CALLS==5
    assert r.catalogue_mapping()[1][result['action']['noble_id']]['id']==1

def test_single_noble_is_compulsory_without_extra_forward():
    s=noble_state();nm=r.catalogue_mapping()[1];s.nobles=[rules.corrected_nobles_by_id()[k] for k,v in nm.items() if v['id'] in (0,3,4)];obs=observer(s)
    result=check_trace(obs,a.decide(obs,predictor([15])))
    assert result['diagnostics']['policy_calls']==1 and result['action']['noble_id']==s.nobles[0].noble_id

def test_unknown_reserves_and_deck_order_never_enter_policy():
    s=initial();reserve(s,1,1);reserve(s,1,2);reserve(s,0,3);obs=observer(s)
    alt=s.copy_shallow()
    for tier in (1,2):
        i=tier-1;alt.players[1].reserved_cards[i],alt.hidden_tier_decks[tier][0]=alt.hidden_tier_decks[tier][0],alt.players[1].reserved_cards[i]
    for deck in alt.hidden_tier_decks.values():deck.reverse()
    assert obs==observer(alt)
    assert a.decide(obs,predictor([42]))==a.decide(observer(alt),predictor([42]))
    p=assert_parity(obs);assert len(p['completion_assignments'])==2
    assert rules.observation_payload(rules.deserialize_snapshot(p['completion']))==obs

def test_own_public_opponent_reserves_preserve_identity_and_slot():
    s=initial();own=reserve(s,0,1);public=reserve(s,1,2,'visible');reserve(s,1,3);obs=observer(s);p=assert_parity(obs)
    cm,_=r.catalogue_mapping()
    assert p['context']['players'][0]['reserved'][0]==cm[own.card_id]['id']
    assert p['context']['players'][1]['reserved'][0]==cm[public.card_id]['id']
    assert p['features'][179]==1 and p['features'][229]==1

@pytest.mark.parametrize('mutation',['private_id','seed','legacy','extra'])
def test_private_or_unapproved_input_rejected(mutation):
    s=initial();reserve(s,1,1);obs=observer(s)
    if mutation=='private_id':obs['observation']['players'][1]['reserves'][0]['card_id']=s.players[1].reserved_cards[0].card_id
    elif mutation=='seed':obs['observation']['seed']=41
    elif mutation=='legacy':obs['ruleset_id']='legacy'
    else:obs['extra']=1
    with pytest.raises(ValueError):a.prepare(obs)

@pytest.mark.parametrize('mutation',['features','mask','logits','context','selected','action','extra_phase'])
def test_full_phase_trace_rejects_tampering(mutation):
    s=initial();tokens(s,0,{'white':3,'blue':3,'green':4});obs=observer(s);out=a.decide(obs,predictor([15,60,61,62]));phase=out['diagnostics']['phases'][1]
    if mutation=='features':phase['features'][0]+=.1
    elif mutation=='mask':phase['mask'][76]=True
    elif mutation=='logits':phase['logits'][0]+=.1
    elif mutation=='context':phase['context']['bank'][0]+=1
    elif mutation=='selected':phase['selected_index']=61
    elif mutation=='action':out['action']['return_tokens']=[]
    else:out['diagnostics']['phases'].append(deepcopy(phase))
    with pytest.raises(ValueError):a.validate_result(out,obs)

def test_output_dtype_shape_nan_rejected():
    obs=observer(initial())
    for x in (np.zeros(77,dtype=np.float32),np.zeros(76),np.full(77,np.nan)):
        with pytest.raises(ValueError):a.decide(obs,lambda f:x)

def test_source_browser_arithmetic_on_synthetic_weights_only(tmp_path,monkeypatch):
    from tools.dom_screen_20261027 import network
    import hashlib
    flat=np.zeros(1757492//4,dtype='<f4');flat[0]=.25;flat[267*512]=.5
    second=267*512+512;flat[second]=.75;flat[second+512*512]=.125
    third=second+512*512+512;flat[third]=.3;flat[third+77*512]=-.1
    raw=flat.tobytes();path=tmp_path/'synthetic.bin';path.write_bytes(raw);sha=hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(network,'WEIGHT',path);monkeypatch.setattr(network,'WEIGHT_SHA256',sha)
    monkeypatch.setattr(r,'WEIGHTS',path);monkeypatch.setattr(r,'WEIGHTS_SHA256',sha)
    policy=network.FrozenPolicy();ref=r.load_policy();features=np.zeros(267,dtype=np.float32);features[0]=.7
    assert policy.weights[0].shape==(512,267) and policy.weights[1].shape==(512,512) and policy.weights[2].shape==(77,512)
    actual=policy.predict(features);expected=r.predict(features,ref)
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-12)
    assert actual.dtype==np.float64 and np.argmax(actual)==np.argmax(expected)

def test_float32_feature_rounding_matches_source_exactly_all_slots():
    s=noble_state();tokens(s,0,{'white':1,'blue':2,'green':3,'gold':1});reserve(s,0,1);reserve(s,1,2)
    s.turn_index=181;s.current_player=1;s.start_player=0;p=assert_parity(observer(s))
    assert p['features'][166]==np.float32(1.5)
    assert np.array_equal(a.encode_phase(p['context']),r.encode_phase(p['context']))
