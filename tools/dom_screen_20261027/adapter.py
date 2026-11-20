"""Lawful corrected PPO512 policy, with prereveal sequential auxiliary choices.

One frozen lawful hidden completion is reused throughout a complete atomic
turn. No referee afterstate, actual hidden draw or market replacement is read.
"""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
import hashlib
import itertools
import json
import numpy as np
from tools.external_strength_screen_20261018 import rules
from tools import future_training_rules_20261021 as training_rules
from .catalogue import catalogue_mapping
PROFILE_ID = 'dom_ppo512_policy_corrected_public_v1'
COMPLETION_DOMAIN = 'dom_public_completion_20261027_v1'
RETURN_DOMAIN = 'dom_prereveal_policy_auxiliaries_20261027_v1'
COLORS = ('white', 'blue', 'green', 'red', 'black')
TOKENS = COLORS + ('gold',)
MAX_POLICY_CALLS = 5
AUXILIARY_CONVENTION = ('One whole-turn clock; main choice followed by up to three policy token returns and, '
 'only if multiple nobles qualify, one policy noble choice. Public purchases/payment/token effects are applied '
 'to a local prereveal context. Removed display slots stay empty until the turn completes; public deck '
 'counts decrease for a withheld replacement. A blind reserve is represented only by the last card of '
 'the single observation-hash-ordered lawful surrogate fixed before the main choice, never by the real draw. '
 'No new market or real blind-card information enters any auxiliary decision. Single eligible nobles '
 'are compulsory without a policy call; payment uses source colored-first then gold.')

def require(condition, message):
    if not condition: raise ValueError(message)

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

def _table():
    out=[]
    for n in (1,2,3):
        for combo in itertools.combinations(COLORS,n):
            out.append(dict(index=len(out),family='take_distinct',take_tokens=list(combo)))
    for c in COLORS: out.append(dict(index=len(out),family='take_double',take_tokens=[c,c]))
    for tier in (1,2,3):
        for slot in range(4): out.append(dict(index=len(out),family='reserve_visible',tier=tier,market_index=slot))
    for tier in (1,2,3): out.append(dict(index=len(out),family='reserve_deck',tier=tier))
    for tier in (1,2,3):
        for slot in range(4): out.append(dict(index=len(out),family='buy_visible',tier=tier,market_index=slot))
    for slot in range(3): out.append(dict(index=len(out),family='buy_reserved',reserved_index=slot))
    for c in TOKENS: out.append(dict(index=len(out),family='discard',token=c))
    for i in range(10): out.append(dict(index=len(out),family='choose_noble',source_noble_id=i))
    out.append(dict(index=76,family='pass'))
    assert len(out)==77
    return tuple(out)
ACTION_TABLE=_table()
ACTION_TABLE_SHA256=digest(ACTION_TABLE)

def lawful_completion(observer):
    """Exactly one observation-seeded allocation, never actual private data.

    Hash-sort each tier's unseen IDs, assign opponent blind slots in slot order,
    and retain the remaining IDs as a dummy deck. No hidden draw is inspected.
    """
    state = rules.state_from_observation(observer)  # Strict privacy/schema gate.
    require(not state.terminal, 'Dom cannot decide a terminal position')
    observation_hash = digest(observer)
    view = observer['observation']
    cards = rules.corrected_cards_by_id()
    assigned = []
    for tier in (1, 2, 3):
        slots = [(seat, slot) for seat, player in enumerate(view['players'])
                 for slot, reserve in enumerate(player['reserves'])
                 if reserve['card_id'] is None and reserve['tier'] == tier]
        unseen = [c.card_id for c in state.hidden_tier_decks[tier]]
        unseen += [state.players[s].reserved_cards[i].card_id for s, i in slots]
        unseen.sort(key=lambda card_id: (digest([COMPLETION_DOMAIN, observation_hash, tier, card_id]), card_id))
        for (seat, slot), card_id in zip(slots, unseen):
            state.players[seat].reserved_cards[slot] = cards[card_id]
            assigned.append(dict(seat=seat, slot=slot, tier=tier, card_id=card_id))
        state.hidden_tier_decks[tier] = [cards[key] for key in unseen[len(slots):]]
    snapshot = rules.serialize_snapshot(state)
    training_rules.decode_training_snapshot(snapshot)
    require(rules.observation_payload(state) == observer, 'Completion changes public information')
    return state, snapshot, assigned


def phase_context(state, phase='play'):
    cards,nobles=catalogue_mapping()
    return dict(schema='dom_prereveal_phase_context_v1',phase=phase,current=state.current_player,
        round=state.turn_index//2,
        display=[[cards[c.card_id]['id'] for c in state.visible_tier_cards[t]]+[None]*(4-len(state.visible_tier_cards[t])) for t in (1,2,3)],
        decks=[state.deck_counts[t] for t in (1,2,3)],bank=[state.bank_tokens[c] for c in TOKENS],
        nobles=[nobles[n.noble_id]['id'] for n in state.nobles],
        players=[dict(tokens=[p.tokens[c] for c in TOKENS],bonuses=[p.bonuses[c] for c in COLORS],
            points=p.score,reserved=[cards[c.card_id]['id'] for c in p.reserved_cards],
            owned_count=len(p.purchased_cards),noble_count=len(p.nobles)) for p in state.players])

def encode_phase(context):
    cards,_=catalogue_mapping(); by_id={c['id']:c for c in cards.values()}
    out=np.zeros(267,dtype=np.float32); cursor=0
    def card_slot(identity):
        nonlocal cursor
        if identity is not None:
            card=by_id[identity];out[cursor]=1;out[cursor+1]=np.float32(card['points'])/np.float32(5)
            out[cursor+2+card['bonus']]=1
            out[cursor+7:cursor+12]=np.asarray(card['cost'],dtype=np.float32)/np.float32(7)
        cursor+=12
    for row in context['display']:
        for identity in row:card_slot(identity)
    out[144:147]=np.asarray(context['decks'],dtype=np.float32)/np.float32(40)
    out[147:153]=np.asarray(context['bank'],dtype=np.float32)/np.float32(7)
    for identity in context['nobles']:out[153+identity]=1
    out[163+('play','discard','choose_noble').index(context['phase'])]=1
    out[166]=min(np.float32(context['round'])/np.float32(60),np.float32(1.5));cursor=167
    for offset in range(2):
        player=context['players'][(context['current']+offset)%2]
        out[cursor:cursor+6]=np.asarray(player['tokens'],dtype=np.float32)/np.float32(10)
        out[cursor+6:cursor+11]=np.asarray(player['bonuses'],dtype=np.float32)/np.float32(7)
        out[cursor+11]=np.float32(player['points'])/np.float32(15);cursor+=12
        for slot in range(3):card_slot(player['reserved'][slot] if slot<len(player['reserved']) else None)
        out[cursor]=np.float32(player['owned_count'])/np.float32(20)
        out[cursor+1]=np.float32(player['noble_count'])/np.float32(5);cursor+=2
    require(cursor==267 and np.isfinite(out).all(),'Invalid Dom phase encoding')
    return out

def encode_state(state):return encode_phase(phase_context(state))

def _spend(actor, card):
    cost=[max(0,card.cost.get(c,0)-actor.bonuses[c]) for c in COLORS]
    payment={c:min(actor.tokens[c],n) for c,n in zip(COLORS,cost)}
    payment['gold']=sum(cost)-sum(payment.values())
    return tuple(c for c in TOKENS for _ in range(payment[c]))

def source_mask_phase(context):
    cards,nobles=catalogue_mapping(); by_id={c['id']:c for c in cards.values()};ns={n['id']:n for n in nobles.values()}
    player=context['players'][context['current']];out=np.zeros(77,dtype=bool)
    if context['phase']=='discard':
        out[60:66]=np.asarray(player['tokens'])>0;return out
    if context['phase']=='choose_noble':
        for n in context['nobles']:out[66+n]=all(b>=r for b,r in zip(player['bonuses'],ns[n]['req']))
        return out
    for d in ACTION_TABLE[:60]:
        i,f=d['index'],d['family']
        if f.startswith('take_'):
            out[i]=(context['bank'][COLORS.index(d['take_tokens'][0])]>=4 if f=='take_double' else all(context['bank'][COLORS.index(c)]>0 for c in d['take_tokens']))
        elif f=='reserve_deck':out[i]=len(player['reserved'])<3 and context['decks'][d['tier']-1]>0
        elif f=='reserve_visible':out[i]=len(player['reserved'])<3 and context['display'][d['tier']-1][d['market_index']] is not None
        else:
            identity=(context['display'][d['tier']-1][d['market_index']] if f=='buy_visible' else player['reserved'][d['reserved_index']] if d['reserved_index']<len(player['reserved']) else None)
            if identity is not None:out[i]=sum(max(0,c-b-t) for c,b,t in zip(by_id[identity]['cost'],player['bonuses'],player['tokens'][:5]))<=player['tokens'][5]
    out[76]=not out.any();return out

def source_mask(state):return source_mask_phase(phase_context(state))

def prepare(observer):
    state,completion,assigned=lawful_completion(observer);actor=state.players[state.current_player]
    legal=rules.legal_actions(state);require(bool(legal),'No corrected legal decisions')
    groups=[[] for _ in range(77)]
    for d in ACTION_TABLE:
        i,f=d['index'],d['family']
        if f in ('discard','choose_noble'):continue
        kind={'buy_visible':'BUY_VISIBLE','buy_reserved':'BUY_RESERVED','reserve_visible':'RESERVE_VISIBLE','reserve_deck':'RESERVE_DECK','pass':'PASS'}.get(f,'TAKE_TOKENS')
        eligible=[a for a in legal if a.action_type.name==kind]
        for key in ('tier','market_index','reserved_index'):
            if key in d:eligible=[a for a in eligible if getattr(a,key)==d[key]]
        if f.startswith('take_'):eligible=[a for a in eligible if Counter(a.take_tokens)==Counter(d['take_tokens'])]
        if f.startswith('buy_') and eligible:
            card=state.visible_tier_cards[d['tier']][d['market_index']] if f=='buy_visible' else actor.reserved_cards[d['reserved_index']]
            payment=Counter(_spend(actor,card));eligible=[a for a in eligible if Counter(a.spend_tokens)==payment]
        groups[i]=sorted([rules.serialize_action(a) for a in eligible],key=digest)
    actions=tuple(g[0] if g else None for g in groups);mask=np.asarray([bool(g) for g in groups],dtype=bool)
    require(mask.any(),'No supported corrected Dom action')
    context=phase_context(state);features=encode_phase(context)
    return dict(features=features,legal_mask=mask,source_mask=source_mask_phase(context),actions=actions,candidates=groups,
        context=context,state=state,observation_sha256=digest(observer),completion=completion,completion_sha256=digest(completion),
        completion_assignments=assigned,feature_sha256=array_digest(features),mask_sha256=array_digest(mask),
        legal_actions_sha256=digest([rules.serialize_action(a) for a in legal]),legal_count=len(legal))

def _phase(context,mask,predict):
    features=encode_phase(context);values=np.asarray(predict(features))
    require(values.shape==(77,) and values.dtype==np.float64 and np.isfinite(values).all(),'Expected77 finite float64 browser scores')
    require(mask.shape==(77,) and mask.dtype==bool and mask.any(),'Invalid phase mask')
    selected=int(np.argmax(np.where(mask,values,-np.inf)))
    return dict(phase=context['phase'],context=deepcopy(context),context_sha256=digest(context),
        features=features.tolist(),feature_sha256=array_digest(features),mask=mask.tolist(),mask_sha256=array_digest(mask),
        source_mask=source_mask_phase(context).tolist(),logits=values.tolist(),logits_sha256=array_digest(values),selected_index=selected)

def _apply_prereveal_main(prepared,index):
    context=deepcopy(prepared['context']);actor=context['players'][context['current']];d=ACTION_TABLE[index];f=d['family']
    action=prepared['candidates'][index][0];cards,_=catalogue_mapping()
    for c in action['take_tokens']:actor['tokens'][TOKENS.index(c)]+=1;context['bank'][TOKENS.index(c)]-=1
    for c in action['spend_tokens']:actor['tokens'][TOKENS.index(c)]-=1;context['bank'][TOKENS.index(c)]+=1
    card=None
    if f in ('buy_visible','reserve_visible'):
        tier,slot=d['tier'],d['market_index'];card=prepared['state'].visible_tier_cards[tier][slot]
        context['display'][tier-1][slot]=None
        if context['decks'][tier-1]>0:context['decks'][tier-1]-=1
    elif f=='buy_reserved':card=prepared['state'].players[context['current']].reserved_cards[d['reserved_index']];actor['reserved'].pop(d['reserved_index'])
    elif f=='reserve_deck':
        tier=d['tier'];card=prepared['state'].hidden_tier_decks[tier][-1];context['decks'][tier-1]-=1
    if f.startswith('reserve_'):actor['reserved'].append(cards[card.card_id]['id'])
    if f.startswith('buy_'):
        actor['owned_count']+=1;actor['bonuses'][COLORS.index(card.bonus_color)]+=1;actor['points']+=card.points
    return context

def decide(observer,predict):
    prepared=prepare(observer);phases=[_phase(prepared['context'],prepared['legal_mask'],predict)]
    selected=phases[0]['selected_index'];candidates=prepared['candidates'][selected];context=_apply_prereveal_main(prepared,selected)
    held=context['players'][context['current']]['tokens'];returned=Counter()
    while sum(held)>10:
        context['phase']='discard';mask=np.zeros(77,dtype=bool)
        for k,c in enumerate(TOKENS):mask[60+k]=held[k]>0 and any(Counter(a['return_tokens'])[c]>returned[c] and all(Counter(a['return_tokens'])[q]>=n for q,n in returned.items()) for a in candidates)
        phase=_phase(context,mask,predict);phases.append(phase);token=TOKENS[phase['selected_index']-60];returned[token]+=1
        k=TOKENS.index(token);held[k]-=1;context['bank'][k]+=1
        candidates=[a for a in candidates if all(Counter(a['return_tokens'])[q]>=n for q,n in returned.items())]
    candidates=[a for a in candidates if Counter(a['return_tokens'])==returned];require(bool(candidates),'No complete corrected action after discards')
    choices={a['noble_id'] for a in candidates};_,nobles=catalogue_mapping()
    if len(choices)>1:
        require(None not in choices,'Compulsory noble support inconsistent');context['phase']='choose_noble';mask=np.zeros(77,dtype=bool)
        for noble in choices:mask[66+nobles[noble]['id']]=True
        phase=_phase(context,mask,predict);phases.append(phase)
        chosen=next(n for n in choices if nobles[n]['id']==phase['selected_index']-66)
        candidates=[a for a in candidates if a['noble_id']==chosen]
    require(1<=len(phases)<=MAX_POLICY_CALLS,'Dom forward count exceeds whole-turn bound')
    action=min(candidates,key=digest);first=phases[0]
    return dict(player='dom',observer_sha256=prepared['observation_sha256'],legal_actions_sha256=prepared['legal_actions_sha256'],legal_count=prepared['legal_count'],action=deepcopy(action),diagnostics=dict(
        schema='dom_public_trace_v1',profile_id=PROFILE_ID,ruleset_id=rules.RULESET_ID,catalogue_sha256=rules.CATALOGUE_SHA256,turn_contract=training_rules.TURN_CONTRACT,
        action_table_sha256=ACTION_TABLE_SHA256,completion_domain=COMPLETION_DOMAIN,auxiliary_convention=AUXILIARY_CONVENTION,
        completion_sha256=prepared['completion_sha256'],completion_assignments=prepared['completion_assignments'],
        feature_sha256=first['feature_sha256'],mask_sha256=first['mask_sha256'],features=first['features'],mask=first['mask'],source_mask=first['source_mask'],
        logits=first['logits'],logits_sha256=first['logits_sha256'],selected_index=selected,policy_calls=len(phases),phases=phases))

def validate_result(result,observer):
    require(isinstance(result,dict) and isinstance(result.get('diagnostics'),dict),'Invalid Dom result')
    phases=result['diagnostics'].get('phases');require(isinstance(phases,list) and 1<=len(phases)<=MAX_POLICY_CALLS,'Invalid phase inventory')
    index=0
    def recorded(features):
        nonlocal index
        require(index<len(phases),'Missing recorded phase');phase=phases[index];index+=1
        values=phase.get('logits');require(isinstance(values,list) and len(values)==77 and all(type(v) in (int,float) and np.isfinite(v) for v in values),'Invalid recorded scores')
        return np.asarray(values,dtype=np.float64)
    expected=decide(observer,recorded)
    require(index==len(phases) and digest(result)==digest(expected),'Dom phase/context/argmax/complete-action binding differs')
    return dict(status='passed',scope='saved_phase_encoding_masks_argmax_legality_no_neural_forward',selected_index=expected['diagnostics']['selected_index'],policy_calls=index)
validate_trace=validate_result
