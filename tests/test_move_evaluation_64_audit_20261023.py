"""Synthetic audit counterexamples only: no checkpoints, native process or study."""
from copy import deepcopy
import gzip
import hashlib
import json

import numpy as np
import pytest

from tools.move_evaluation_64_20261023 import audit as a
from tools.external_strength_screen_20261018 import rules


def final_pass():
    state = rules.CorrectedEnv(seed=301).initial_state()
    state.current_player, state.turn_index, state.pending_round_end = 1, 1, True
    for tier in (1,2,3):
        for card in state.visible_tier_cards[tier]+state.hidden_tier_decks[tier]:
            p = state.players[0]; p.purchased_cards.append(card); p.bonuses[card.bonus_color] += 1; p.score += card.points
        state.visible_tier_cards[tier], state.hidden_tier_decks[tier], state.deck_counts[tier] = [],[],0
    for seat,counts in enumerate(({'white':4,'blue':4,'green':2},{'green':2,'red':4,'black':4})):
        for color,count in counts.items():
            state.players[seat].tokens[color] = count; state.bank_tokens[color] -= count
    return state


def two_step_terminal():
    state = rules.CorrectedEnv(seed=304).initial_state(); cards = rules.corrected_cards_by_id()
    fives = [c for c in cards.values() if c.points == 5][:2]
    four = next(c for c in cards.values() if c.points == 4)
    owned = {c.card_id for c in fives+[four]}|{c.card_id for c in cards.values() if c.points == 0}
    for tier in (1,2,3):
        for group in (state.visible_tier_cards,state.hidden_tier_decks):
            keep = []
            for card in group[tier]:
                if card.card_id in owned:
                    p = state.players[0]; p.purchased_cards.append(card); p.bonuses[card.bonus_color] += 1; p.score += card.points
                else: keep.append(card)
            group[tier] = keep
        state.deck_counts[tier] = len(state.hidden_tier_decks[tier])
    forced = next(x for x in rules.legal_actions(state) if x.action_type.name == 'BUY_VISIBLE')
    return rules.serialize_snapshot(state),rules.serialize_action(forced)


def first_entity(observer):
    state = rules.state_from_observation(observer)
    return rules.legal_actions(state)[0], {'fake_policy':'first','observer_sha256':a.digest(observer)}


def trace(world, forced, length=1):
    current = deepcopy(world); steps = []; hasher = hashlib.sha256()
    for i in range(length):
        state = rules.deserialize_snapshot(current); objects = rules.legal_actions(state)
        legal = [rules.serialize_action(x) for x in objects]
        if i == 0: action,policy,diag = forced,'forced',None
        else:
            selected,diag = first_entity(rules.observation_payload(state))
            action,policy = rules.serialize_action(selected),'greedy_entity'
        steps.append({'state_hash':a.digest(current['state']),'action':action,'turn':state.turn_index,'actor':state.current_player,
                      'legal_count':len(legal),'selected_index':legal.index(action),'forced':i == 0,'scores_sha256':None,
                      'policy':policy,'policy_trace':diag})
        hasher.update(a.digest(action).encode()); current = a.ledger.transition(current,action)
    return {'initial_state_sha256':a.digest(world['state']),'steps':steps,'reason':'completed',
            'winner':current['state']['winner'],'turns':len(steps),'trace_sha256':hasher.hexdigest(),
            'final_state_sha256':a.digest(current['state'])}


def observation(state,game=0):
    turn = state.turn_index
    context = {'mode':'operational','namespace':a.NAMESPACE,'pair':game//2,'game':game,'turn':turn,
               'fit':a.FITS[(game//2)%3],'stream':a.digest(['splendor_live_operational_search_v1',a.NAMESPACE,game]),
               'identity':[a.NAMESPACE,game,turn]}
    return {**rules.observation_payload(state),'schema':'corrected_live_planner_observer_v1','context':context}


def full_fixture(tmp_path,monkeypatch,arm='B'):
    state = final_pass(); observer = observation(state); world = a.reference.world_from_observer(observer)
    legal = [rules.serialize_action(x) for x in rules.legal_actions(state)]
    proposal = a.reference.derive_proposal(observer,world,legal,[0.],[0.],{'sha256':'a'*64})
    monkeypatch.setattr(a,'_root_scores',lambda *_:([0.],[0.],{'sha256':'a'*64}))
    payload = {'observer':observer,'fit':observer['context']['fit'],'candidates':proposal['candidates'],
               'selection_schedule':{**deepcopy(proposal['schedule']), 'world_seeds':proposal['schedule']['world_seeds'][:64]}}
    frozen = deepcopy({'roots':[{'payload':payload, 'original_full_schedule_sha256':a.digest(proposal['schedule'])}]})
    monkeypatch.setattr(a.c, 'read', lambda _: deepcopy(frozen))
    value = trace(world,legal[0]); rows=[]; worlds=[]
    if arm == 'A':
        for step in value['steps']:
            del step['policy']; del step['policy_trace']
    for wi,seed in enumerate(payload['selection_schedule']['world_seeds']):
        rows.append({**deepcopy(value),'world_index':wi,'candidate_index':0,'seed':seed})
        worlds.append({'seed':seed,'state_sha256':a.digest(world['state']),
                       'outcomes':[{**{k:value[k] for k in a.OUTCOME_FIELDS},'wdl':-1.}]})
    record = {'root_identity':observer['context']['identity'],'planner':'reference' if arm == 'A' else 'opponent_policy_substitution',
              'privacy':{'own':0,'opponent':0},'schedule':payload['selection_schedule'],'candidates':proposal['candidates'],
              'proposal':proposal,'scores':[0.],'worlds':worlds,'profile':a.PROFILES[arm],'ruleset_id':rules.RULESET_ID,
              'catalogue_sha256':rules.CATALOGUE_SHA256,'choice':0,'means':[-1.]}
    result = {'decision':{'action':legal[0],'record':record},'trace':{},'profile_id':a.PROFILES[arm],
              'observer_sha256':a.digest(observer),'ruleset_id':rules.RULESET_ID,'catalogue_sha256':rules.CATALOGUE_SHA256}
    path=tmp_path/'trace.gz'
    def write():
        with gzip.open(path,'wt',encoding='utf-8') as f:
            for row in rows: f.write(json.dumps(row)+'\n')
        result['trace']={'path':str(path),'bytes':path.stat().st_size,'sha256':a.file_hash(path),'records':64}
    write()
    return payload,result,path,rows,write


@pytest.mark.parametrize('arm', ['A','B'])
def test_full64_independent_replay_and_first_tie(tmp_path,monkeypatch,arm):
    payload,result,path,_,_=full_fixture(tmp_path,monkeypatch,arm)
    proof=a.audit_selection(payload,result,path,first_entity if arm == "B" else None,arm=arm)
    assert proof['audited_worlds']==proof['audited_continuations']==proof['audited_transitions']==64
    assert proof['entity_choices_reexecuted']==proof['raw035_choices_reexecuted']==0
    assert proof['proposal_model_reexecuted'] and proof['all_continuations_replayed']


@pytest.mark.parametrize('damage',['extra_payload','fit_bool','context','actor','seed','candidate','score','proposal','choice_bool',
                                  'mean','wdl','missing','extra','duplicate','trace_path','wrapper','privacy'])
@pytest.mark.parametrize('arm', ['A','B'])
def test_full_replay_rejects_tampering(tmp_path,monkeypatch,damage,arm):
    p,r,path,rows,write=full_fixture(tmp_path,monkeypatch,arm); rec=r['decision']['record']
    if damage=='extra_payload': p['assessment_seeds']=['secret']
    elif damage=='fit_bool': p['fit']=True
    elif damage=='context': p['observer']['context']['stream']='bad'
    elif damage=='actor': p['observer']['observation']['current_player']=False
    elif damage=='seed': rec['worlds'][0]['seed']='bad'
    elif damage=='candidate': p['candidates']['actions'][0]['noble_id']='N01'
    elif damage=='score': rec['scores']=[1.]
    elif damage=='proposal': rec['proposal']['model_sha256']='b'*64
    elif damage=='choice_bool': rec['choice']=False
    elif damage=='mean': rec['means']=[1.]
    elif damage=='wdl': rec['worlds'][0]['outcomes'][0]['wdl']=1.
    elif damage=='missing': rows.pop()
    elif damage=='extra': rows.append(deepcopy(rows[-1]))
    elif damage=='duplicate': rows[1]=deepcopy(rows[0])
    elif damage=='wrapper': r['ruleset_id']='legacy'
    elif damage=='privacy': rec['privacy']['own']=1
    write()
    if damage=='trace_path': r['trace']['path']=str(path)+'bad'
    with pytest.raises(ValueError): a.audit_selection(p,r,path,first_entity if arm == "B" else None,arm=arm)


def test_entity_policy_is_reexecuted_on_lawful_observer():
    world,forced=two_step_terminal(); row=trace(world,forced,2); seen=[]
    def policy(obs):
        assert 'hidden_tier_decks' not in obs['observation']
        seen.append(obs); return first_entity(obs)
    outcome,counts=a.audit_trajectory(world,forced,row,policy,arm="B")
    assert outcome['reason']=='completed' and outcome['winner']==0
    assert counts['entity_choices_reexecuted']==len(seen)==1
    assert counts['raw035_choices_reexecuted']==0


@pytest.mark.parametrize('damage',['policy','diag','score','state','selected_bool','forced','action','winner','hash','extra','missing'])
def test_entity_trace_mutations_rejected(damage):
    world,forced=two_step_terminal(); row=trace(world,forced,2)
    if damage=='policy': row['steps'][1]['policy']='raw035'
    elif damage=='diag': row['steps'][1]['policy_trace']['observer_sha256']='bad'
    elif damage=='score': row['steps'][1]['scores_sha256']='bad'
    elif damage=='state': row['steps'][1]['state_hash']='bad'
    elif damage=='selected_bool': row['steps'][1]['selected_index']=False
    elif damage=='forced': row['steps'][1]['forced']=True
    elif damage=='action': row['steps'][1]['action']['noble_id']='N01'
    elif damage=='winner': row['winner']=False
    elif damage=='hash': row['trace_sha256']='bad'
    elif damage=='extra': row['steps'].append(deepcopy(row['steps'][-1]))
    else: row['steps'].pop()
    with pytest.raises(ValueError): a.audit_trajectory(world,forced,row,first_entity,arm="B")


def test_a_replay_reexecutes_raw_on_both_seats(monkeypatch):
    from tools.move_evaluation_64_20261023 import qualification as q
    from tools.move_evaluation_64_20261023 import evaluator as e
    fixtures = q.old.build_fixtures()
    state = q.guard.decode_training_snapshot(next(f['snapshot'] for f in fixtures if f['id'] == 'noble_final_round'))
    state.current_player, state.turn_index = 1, 1
    forced = next(x for x in rules.legal_actions(state) if x.action_type.name == 'TAKE_TOKENS')
    scorer = lambda state, legal: [-float(i) for i in range(len(legal))]
    monkeypatch.setattr(e.q, 'score_actions', scorer)
    result, steps = e.trajectory(state, forced, None, arm='A', capture=True)
    world = rules.serialize_snapshot(state)
    row = {**result, 'initial_state_sha256': a.digest(world['state']), 'steps': steps}
    visited = []
    def replay_score(current, legal):
        visited.append(current.current_player)
        return scorer(current, legal)
    outcome, counts = a.audit_trajectory(world, rules.serialize_action(forced), row, arm='A', scorer=replay_score)
    assert outcome == result and visited == [0, 1]
    assert counts['raw035_choices_reexecuted'] == 2 and counts['entity_choices_reexecuted'] == 0

