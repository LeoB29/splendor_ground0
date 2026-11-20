"""Synthetic audit counterexamples only: no checkpoints, native process or study."""
from copy import deepcopy
import gzip
import hashlib
import json

import numpy as np
import pytest

from tools.move_evaluation_20261023 import audit as a
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


def full_fixture(tmp_path,monkeypatch):
    state = final_pass(); observer = observation(state); world = a.reference.world_from_observer(observer)
    legal = [rules.serialize_action(x) for x in rules.legal_actions(state)]
    proposal = a.reference.derive_proposal(observer,world,legal,[0.],[0.],{'sha256':'a'*64})
    monkeypatch.setattr(a,'_root_scores',lambda *_:([0.],[0.],{'sha256':'a'*64}))
    payload = {'observer':observer,'fit':observer['context']['fit'],'candidates':proposal['candidates'],
               'selection_schedule':proposal['schedule']}
    value = trace(world,legal[0]); rows=[]; worlds=[]
    for wi,seed in enumerate(proposal['schedule']['world_seeds']):
        rows.append({**deepcopy(value),'world_index':wi,'candidate_index':0,'seed':seed})
        worlds.append({'seed':seed,'state_sha256':a.digest(world['state']),
                       'outcomes':[{**{k:value[k] for k in a.OUTCOME_FIELDS},'wdl':-1.}]})
    record = {'root_identity':observer['context']['identity'],'planner':'opponent_policy_substitution',
              'privacy':{'own':0,'opponent':0},'schedule':proposal['schedule'],'candidates':proposal['candidates'],
              'proposal':proposal,'scores':[0.],'worlds':worlds,'profile':a.PROFILE,'ruleset_id':rules.RULESET_ID,
              'catalogue_sha256':rules.CATALOGUE_SHA256,'choice':0,'means':[-1.]}
    result = {'decision':{'action':legal[0],'record':record},'trace':{},'profile_id':a.PROFILE,
              'observer_sha256':a.digest(observer),'ruleset_id':rules.RULESET_ID,'catalogue_sha256':rules.CATALOGUE_SHA256}
    path=tmp_path/'trace.gz'
    def write():
        with gzip.open(path,'wt',encoding='utf-8') as f:
            for row in rows: f.write(json.dumps(row)+'\n')
        result['trace']={'path':str(path),'bytes':path.stat().st_size,'sha256':a.file_hash(path),'records':128}
    write()
    return payload,result,path,rows,write


def test_full128_independent_replay_and_first_tie(tmp_path,monkeypatch):
    payload,result,path,_,_=full_fixture(tmp_path,monkeypatch)
    proof=a.audit_selection(payload,result,path,first_entity)
    assert proof['audited_worlds']==proof['audited_continuations']==proof['audited_transitions']==128
    assert proof['entity_choices_reexecuted']==proof['raw035_choices_reexecuted']==0
    assert proof['proposal_model_reexecuted'] and proof['all_continuations_replayed']


@pytest.mark.parametrize('damage',['extra_payload','fit_bool','context','actor','seed','candidate','score','proposal','choice_bool',
                                  'mean','wdl','missing','extra','duplicate','trace_path','wrapper','privacy'])
def test_full_replay_rejects_tampering(tmp_path,monkeypatch,damage):
    p,r,path,rows,write=full_fixture(tmp_path,monkeypatch); rec=r['decision']['record']
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
    with pytest.raises(ValueError): a.audit_selection(p,r,path,first_entity)


def test_entity_policy_is_reexecuted_on_lawful_observer():
    world,forced=two_step_terminal(); row=trace(world,forced,2); seen=[]
    def policy(obs):
        assert 'hidden_tier_decks' not in obs['observation']
        seen.append(obs); return first_entity(obs)
    outcome,counts=a.audit_trajectory(world,forced,row,policy)
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
    with pytest.raises(ValueError): a.audit_trajectory(world,forced,row,first_entity)


def branch_final_pass():
    state=final_pass(); initial=rules.serialize_snapshot(state); action=rules.serialize_action(rules.legal_actions(state)[0])
    after=a.ledger.transition(initial,action)
    return {'initial':initial,'forced_action':action,'root_actor':1,'steps':[{
        'before':initial,'observer':rules.observation_payload(state),'action':action,'after':after,
        'player':'forced','clock':None,'receipt':None}], 'final':after,'reason':'completed','score':0.}


def test_official_terminal_target_and_no_forced_move_clock():
    b=branch_final_pass(); proof=a.audit_branch(b)
    assert proof['official_terminal'] and proof['score_halfpoints']==0 and proof['audited_transitions']==1
    assert not proof['puct16_internal_edges_reexecuted']


@pytest.mark.parametrize('damage',['deadlock','score','actor_bool','before','after','observer','forced','clock','empty','final'])
def test_assessment_rejects_nonofficial_or_corrupted_branch(damage):
    b=branch_final_pass()
    if damage=='deadlock': b['reason']='forced_pass_deadlock'
    elif damage=='score': b['score']=1.
    elif damage=='actor_bool': b['root_actor']=True
    elif damage=='before': b['steps'][0]['before']=deepcopy(b['initial']); b['steps'][0]['before']['state']['turn_index']=3
    elif damage=='after': b['steps'][0]['after']=deepcopy(b['final']); b['steps'][0]['after']['state']['winner']=1
    elif damage=='observer': b['steps'][0]['observer']['observation']['bank_tokens']['gold']=4
    elif damage=='forced': b['forced_action']=deepcopy(b['forced_action']); b['forced_action']['noble_id']='N01'
    elif damage=='clock': b['steps'][0]['clock']={}
    elif damage=='empty': b['steps']=[]
    else: b['final']=deepcopy(b['initial'])
    with pytest.raises(ValueError): a.audit_branch(b)


def stats_fixture():
    roots=[]; rows=[]
    for i in range(18):
        fit=a.FITS[i//6]
        roots.append({'id':f'r{i}','fit':fit,'control_choice':0,'candidates':{'actions':[{},{}]}})
        aa=[0]*32; bb=[1 if j%3==0 else 0 for j in range(32)]
        rows.append({'id':f'r{i}','fit':fit,'control_choice':0,'alternative_choice':1,
                     'a_scores_halfpoints':aa,'b_scores_halfpoints':bb,'differences_halfpoints':bb[:]})
    return rows,{'roots':roots}


def test_prescribed_bootstrap_exact_scalar_draw_order_including_zero_roots():
    differences=np.array([[(i+2*j)%5-2 for j in range(32)] for i in range(18)],dtype=np.int64)
    differences[1]=0; differences[11]=0
    got=a.bootstrap_totals(differences.tolist())
    # Independent scalar replicate loop, preserving all zero-root RNG draws.
    rng=np.random.Generator(np.random.PCG64(2026102301)); expected=[]
    for _ in range(50000):
        indices=rng.integers(0,32,size=(18,32),dtype=np.int64)
        expected.append(sum(int(differences[r,indices[r]].sum()) for r in range(18)))
    assert got.tolist()==expected


def test_statistics_conditional_denominators_and_progression():
    rows,lock=stats_fixture(); got=a.statistics(rows,lock,checks_complete=True)
    assert got['roots']==18 and got['worlds_per_root']==32 and got['terminal_branches']==1152
    assert got['integer_difference_numerator']==198 and got['mean_gain']==198/1152
    assert all(x==66/384 for x in got['by_fit_mean_gain'].values())
    assert got['conditional_ci95'][0]>0 and got['promising_on_development_panel']
    assert not got['population_strength_claim'] and not got['fresh_game_launch_authorized']


def test_structural_zero_is_kept_in_denominator_and_breadth(monkeypatch):
    rows,lock=stats_fixture()
    for i,row in enumerate(rows):
        if i%3:
            row.update(alternative_choice=0,a_scores_halfpoints=None,b_scores_halfpoints=None,differences_halfpoints=[0]*32)
    monkeypatch.setattr(a,'bootstrap_totals',lambda d,boundary:np.full(50000,sum(map(sum,d)),dtype=np.int64))
    got=a.statistics(rows,lock,checks_complete=True)
    assert got['structural_zero_roots']==12 and got['mean_gain']==66/1152 and got['terminal_branches']==384
    assert got['intervention_breadth']['changed_roots']==6


@pytest.mark.parametrize('damage',['missing','order','duplicate','fit','choice_bool','zero_imputed','short','bool_diff','arithmetic','outcome','incomplete','breadth'])
def test_statistics_fail_closed_before_bootstrap(monkeypatch,damage):
    rows,lock=stats_fixture(); complete=True
    monkeypatch.setattr(a,'bootstrap_totals',lambda *_:pytest.fail('Must reject before bootstrap'))
    if damage=='missing': rows.pop()
    elif damage=='order': rows[0],rows[1]=rows[1],rows[0]
    elif damage=='duplicate': rows[0]=deepcopy(rows[1])
    elif damage=='fit': rows[0]['fit']=161002
    elif damage=='choice_bool': rows[0]['alternative_choice']=True
    elif damage=='zero_imputed': rows[0]['alternative_choice']=0
    elif damage=='short': rows[0]['differences_halfpoints'].pop()
    elif damage=='bool_diff': rows[0]['differences_halfpoints'][0]=True
    elif damage=='arithmetic': rows[0]['differences_halfpoints'][0]=-1
    elif damage=='outcome': rows[0]['a_scores_halfpoints'][0]=-1
    elif damage=='incomplete': complete=False
    else:
        for row in rows: row['alternative_choice']=0
    with pytest.raises(ValueError): a.statistics(rows,lock,checks_complete=complete)


def test_saturated_losses_are_not_certain_equality(monkeypatch):
    rows,lock=stats_fixture()
    for row in rows:
        row.update(a_scores_halfpoints=[0]*32,b_scores_halfpoints=[0]*32,differences_halfpoints=[0]*32)
    monkeypatch.setattr(a,'bootstrap_totals',lambda *_:np.zeros(50000,dtype=np.int64))
    got=a.statistics(rows,lock,checks_complete=True)
    assert got['classification']=='saturated_uninformative' and got['conditional_ci95']==[0.,0.]
    assert not got['promising_on_development_panel']


def test_one_negative_fit_blocks_even_positive_overall(monkeypatch):
    rows,lock=stats_fixture()
    for i,row in enumerate(rows):
        aa,bb=([1]*32,[0]*32) if i<6 else ([0]*32,[2]*32)
        row.update(a_scores_halfpoints=aa,b_scores_halfpoints=bb,differences_halfpoints=[y-x for x,y in zip(aa,bb)])
    monkeypatch.setattr(a,'bootstrap_totals',lambda d,boundary:np.full(50000,sum(map(sum,d)),dtype=np.int64))
    got=a.statistics(rows,lock,checks_complete=True)
    assert got['mean_gain']>.02 and got['conditional_ci95'][0]>0
    assert not got['promising_on_development_panel']


def test_runtime_guard_called_by_explicit_startup_only(monkeypatch):
    calls=[]; monkeypatch.setattr(a,'verify_runtime_sources',lambda:calls.append(1) or {'source':'hash'})
    receipt=a.initialize_audit()
    assert calls==[1] and receipt['runtime_sources']=={'source':'hash'}


def test_fresh_assessment_world_is_independently_reconstructed_and_bound():
    state=rules.CorrectedEnv(seed=411).initial_state(); observer=observation(state,game=1)
    root_id='games/g001/t0000'; seed=int(a.digest(['move_evaluation_20261023_assessment_v1',root_id,7]),16)
    world=a.reference.world_from_observer(observer,seed)
    got=a.audit_assessment_initial(observer,seed,world)
    assert got['world_index']==7 and got['root_id']==root_id
    assert a.reference.world_from_observer(observer,str(seed)) != world
    with pytest.raises(ValueError,match='integer'): a.audit_assessment_initial(observer,str(seed),world)
    with pytest.raises(ValueError,match='integer'): a.audit_assessment_initial(observer,True,world)
    broken=deepcopy(world); broken['state']['hidden_tier_decks']['1'].reverse()
    with pytest.raises(ValueError,match='initial world'): a.audit_assessment_initial(observer,seed,broken)
    with pytest.raises(ValueError,match='prescribed'): a.audit_assessment_initial(observer,'unknown',world)


def three_ply_raw_trace():
    world,_=two_step_terminal(); world['state']['current_player']=1; world['state']['turn_index']=1
    current=deepcopy(world); steps=[]; hashes=hashlib.sha256()
    for i in range(3):
        state=rules.deserialize_snapshot(current); objects=rules.legal_actions(state)
        legal=[rules.serialize_action(x) for x in objects]
        if i==1:
            action=next(x for x in legal if x['action_type']=='BUY_VISIBLE')
            policy,diag,score_hash='greedy_entity',{'fake_policy':'buy'},None
        else:
            action=legal[0]; policy='forced' if i==0 else 'raw035'
            diag=None; score_hash=None if i==0 else a.digest([0.]*len(legal))
        steps.append({'state_hash':a.digest(current['state']),'action':action,'turn':state.turn_index,'actor':state.current_player,
                      'legal_count':len(legal),'selected_index':legal.index(action),'forced':i==0,'scores_sha256':score_hash,
                      'policy':policy,'policy_trace':diag})
        hashes.update(a.digest(action).encode()); current=a.ledger.transition(current,action)
    assert current['state']['terminal']
    row={'initial_state_sha256':a.digest(world['state']),'steps':steps,'reason':'completed','winner':current['state']['winner'],
         'turns':3,'trace_sha256':hashes.hexdigest(),'final_state_sha256':a.digest(current['state'])}
    return world,steps[0]['action'],row


def test_raw_replay_after_entity_uses_full_score_digest_and_first_tie():
    world,forced,row=three_ply_raw_trace()
    def policy(observer):
        objects=rules.legal_actions(rules.state_from_observation(observer))
        return next(x for x in objects if x.action_type.name=='BUY_VISIBLE'),{'fake_policy':'buy'}
    outcome,counts=a.audit_trajectory(world,forced,row,policy,scorer=lambda s,l:[0.]*len(l))
    assert outcome['turns']==3 and counts=={'raw035_choices_reexecuted':1,'entity_choices_reexecuted':1}
    row['steps'][2]['scores_sha256']='bad'
    with pytest.raises(ValueError,match='score'): a.audit_trajectory(world,forced,row,policy,scorer=lambda s,l:[0.]*len(l))


def clock_for(result):
    return {'request_id':'request-00000001','clock':'perf_counter_ns','elapsed_seconds':.21,'deadline_seconds':1.,
            'receipt_elapsed_seconds':.2,'received_before_deadline':True,'status':'ok','accepted':True,'result':result,
            'telemetry':{'affinity':[0],'cpu_seconds':.1,'request_cpu_seconds':.1,'rss_bytes':100,'peak_sampled_rss_bytes':100,
                         'observed_threads':1,'resource_scope':'supervised_worker_and_all_descendants','processes_observed':1,
                         'process_tree':[{'role':'python','pid':55,'affinity':[0],'cpu_seconds':.1,'rss_bytes':100,
                                          'executable_sha256':'a'*64}]}}


def branch_with_receipt(tmp_path,monkeypatch):
    from tools.analyze_external_strength_screen_20261018_02 import main_action
    world,forced=two_step_terminal(); after=a.ledger.transition(world,forced)
    state=rules.deserialize_snapshot(after); legal=[rules.serialize_action(x) for x in rules.legal_actions(state)]
    action=legal[0]; assert action['action_type']=='TAKE_TOKENS' and not action['return_tokens'] and action['noble_id'] is None
    observer=rules.observation_payload(state); count=len({a.digest(main_action(x)) for x in legal})
    diag={'opponent_id':'splendorust','profile_id':'entity_onehot_puct16_canonical_atomic_prereveal_cpu_v1',
          'observer_seed':int(a.digest(observer)[:16],16),'forced_pass':False,
          'phases':[{'action':main_action(action),'simulations':16,'leaf_calls':17,'native_legal_count':count,'heuristic_phase':False}],
          'inference_calls':17,'inference_rows':17}
    result={'action':action,'diagnostics':diag,'profile_sha256':a.PUCT_SHA256,'player':'opponent',
            'observer_sha256':a.digest(observer),'legal_actions_sha256':a.digest(legal),'legal_count':len(legal),
            'ruleset':rules.RULESET_ID,'catalogue_sha256':rules.CATALOGUE_SHA256}
    receipt=clock_for(result); path=tmp_path/'r.json'
    monkeypatch.setattr(a,'ROOT',tmp_path)
    def write():
        path.write_text(json.dumps(receipt),encoding='utf-8')
        return {'path':'r.json','bytes':path.stat().st_size,'sha256':a.file_hash(path)}
    final=a.ledger.transition(after,action)
    b={'initial':world,'forced_action':forced,'root_actor':0,'steps':[
        {'before':world,'observer':rules.observation_payload(rules.deserialize_snapshot(world)),'action':forced,'after':after,
         'player':'forced','clock':None,'receipt':None},
        {'before':after,'observer':observer,'action':action,'after':final,'player':'opponent','clock':receipt,'receipt':write()}],
       'final':final,'reason':'completed','score':1.}
    return b,receipt,write


def test_puct16_actual_edge_receipt_and_quota_verified(tmp_path,monkeypatch):
    b,_,_=branch_with_receipt(tmp_path,monkeypatch); got=a.audit_branch(b)
    assert got['puct16_actual_edges_checked']==1 and got['work']['simulations']==16
    assert got['maximum_move_receipt_seconds']==.2 and not got['puct16_internal_edges_reexecuted']


@pytest.mark.parametrize('damage',['late','clock_flag','quota','profile','observer','legal_digest','action','source','affinity','rss'])
def test_assessment_receipt_tampering_blocks_target(tmp_path,monkeypatch,damage):
    b,r,write=branch_with_receipt(tmp_path,monkeypatch)
    if damage=='late': r['receipt_elapsed_seconds']=1.001; r['elapsed_seconds']=1.01; r['received_before_deadline']=False
    elif damage=='clock_flag': r['accepted']=False
    elif damage=='quota': r['result']['diagnostics']['phases'][0]['simulations']=15
    elif damage=='profile': r['result']['profile_sha256']='bad'
    elif damage=='observer': r['result']['observer_sha256']='bad'
    elif damage=='legal_digest': r['result']['legal_actions_sha256']='bad'
    elif damage=='action': r['result']['action']=deepcopy(r['result']['action']); r['result']['action']['noble_id']='N01'
    elif damage=='affinity': r['telemetry']['affinity']=[2]
    elif damage=='rss': r['telemetry']['peak_sampled_rss_bytes']=3*1024**3
    b['steps'][1]['receipt']=write()
    if damage=='source': b['steps'][1]['receipt']['sha256']='bad'
    with pytest.raises(ValueError): a.audit_branch(b)
