"""Additive root-target capture; no model is loaded by importing this module.

The supervised worker authenticates runtime rules once before initialization.
Search remains the pinned original algorithm; training evidence is separate
from its original scientific result and trace.
"""
from copy import deepcopy
import atexit
import json
import math
import struct
import subprocess
import sys
import time
from pathlib import Path

from tools.external_strength_screen_20261018 import rules
from tools.external_strength_screen_20261018.common import ROOT, digest
from tools.external_strength_screen_20261018_02 import bridge
from tools.external_strength_screen_20261018_02.splendorust_adapter import PIN, Opponent, file_hash
from tools.splendorust_search512_20261101.adapters import BaselineOpponent
from tools.future_training_rules_20261021 import RULESET_ID, CATALOGUE_SHA256, TURN_CONTRACT

SCHEMA = 'selfplay_root_capture_v1'
BRIDGE_ID = 'observer_only_root_targets_20261101_v1'
MAX_NATIVE_LINE = 4 * 1024**2
TAKE_MASKS = (1,2,4,8,16,3,5,9,17,6,10,18,12,20,24,7,11,19,13,21,25,14,22,26,28)
CARD_SLOT = (5,6,3,4,1,2,0,7,5,6,3,4,1,2,0,7,5,6,3,4,1,2,0,7,5,6,3,4,1,2,0,7,
             5,6,3,4,1,2,0,7,0,1,4,3,2,5,0,1,3,4,2,5,0,1,4,3,2,5,1,0,4,3,2,5,
             0,1,4,3,2,5,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3)
CARD_GROUP = (2,2,2,2,2,2,2,2,0,0,0,0,0,0,0,0,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,
              1,1,1,1,1,1,1,1,2,2,2,2,2,2,0,0,0,0,0,0,3,3,3,3,3,3,4,4,4,4,4,4,
              1,1,1,1,1,1,2,2,2,2,0,0,0,0,3,3,3,3,4,4,4,4,1,1,1,1)
INDEX67_TO81 = {**{mask:30+i for i,mask in enumerate(TAKE_MASKS)},
    **{32+i:55+i for i in range(5)}, **{37+i:12+i for i in range(12)},
    **{49+i:24+i for i in range(3)}, **{52+i:i for i in range(12)},
    **{64+i:27+i for i in range(3)}}


def require(condition, message):
    if not condition:
        raise ValueError('SELFPLAY_CAPTURE: '+message)


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def action_index67(action):
    kind = action['kind']
    if kind == 'Take':
        tokens = action['tokens']
        require(type(tokens) is list and len(tokens)==5 and all(type(x) is int and 0<=x<=2 for x in tokens), 'Take shape')
        if 2 in tokens:
            require(tokens.count(2)==1 and sum(tokens)==2, 'Double-token action')
            return 32+tokens.index(2)
        mask = sum((1<<i) for i,n in enumerate(tokens) if n)
        require(mask in TAKE_MASKS, 'Supported Main token subset')
        return mask
    offsets = {'ReserveVisible':(37,12,'index'), 'ReserveDeck':(49,3,'tier'),
               'BuyVisible':(52,12,'index'), 'BuyReserved':(64,3,'index')}
    require(kind in offsets, 'Auxiliary or forced pass has no policy index')
    offset,count,key = offsets[kind]; index = action[key]
    require(type(index) is int and 0<=index<count, 'Main slot index')
    return offset+index


def main_support(observer):
    actions = bridge.main_actions(observer)
    indices = [] if actions == [{'kind':'Pass'}] else [action_index67(a) for a in actions]
    require(len(indices)==len(set(indices)), 'Unique authoritative Main indices')
    mask = [False]*81
    for index in indices:
        mask[INDEX67_TO81[index]] = True
    return actions,indices,mask


def public_root_tokens(observer):
    """Independent float32 public-belief encoder, never a hidden-state decoder.

Reconstructs the runtime entity tensor directly from the lawful observer.
Unknown opponent cards stay in the public unseen pool. No sampling seed, true
deck order, neural callback or historical snapshot loader enters this function.
"""
    bridge._state_and_legal(observer)
    view=observer['observation']; actor=view['current_player']
    require(view['turn_index']<124, 'No neural encoding after late fallback threshold')
    players=[view['players'][actor],view['players'][1-actor]]
    cards=rules.corrected_cards_by_id(); nobles=rules.corrected_nobles_by_id()
    raw=[0.]*392; packed=[0.]*525
    raw[:6]=[float(view['bank_tokens'][color]) for color in bridge.TOKENS]
    raw[6]=float(view['turn_index'])
    occupied=set()
    def card_at(row, card_id):
        card=cards[card_id]
        raw[row*7:row*7+5]=[float(card.cost.get(color,0)) for color in bridge.COLORS]
        raw[(row+1)*7+bridge.COLORS.index(card.bonus_color)]=1.
        raw[(row+1)*7+6]=float(card.points)
    for tier in (1,2,3):
        for slot,card_id in enumerate(view['visible_tier_cards'][str(tier)]):
            card_at(1+2*(4*(tier-1)+slot),card_id); occupied.add(card_id)
    all_nobles=set(view['nobles'])
    for player in players:all_nobles.update(player['noble_ids'])
    require(len(all_nobles)==3, 'Three original nobles, including awarded ones')
    for slot,noble_id in enumerate(sorted(all_nobles,key=bridge.NOBLE_TO_RUST.__getitem__)):
        noble=nobles[noble_id]
        def noble_at(row):
            raw[row*7:row*7+5]=[float(noble.requirement.get(color,0)) for color in bridge.COLORS]
            raw[row*7+6]=3.
        if noble_id in view['nobles']:noble_at(31+slot)
        for p,player in enumerate(players):
            if noble_id in player['noble_ids']:noble_at(36+3*p+slot)
    unknown=[]; context=[0.]*7
    for p,player in enumerate(players):
        occupied.update(player['purchased_card_ids'])
        raw[(34+p)*7:(34+p)*7+6]=[float(player['tokens'][color]) for color in bridge.TOKENS]
        raw[(42+p)*7:(42+p)*7+5]=[float(player['bonuses'][color]) for color in bridge.COLORS]
        raw[(42+p)*7+6]=float(player['score']-3*len(player['noble_ids']))
        for slot,reserve in enumerate(player['reserves']):
            if p==1:
                context[slot]=float(reserve['source']=='deck')
                context[slot+3]=f32(reserve['tier']/3.)
                context[6]=f32(context[6]+f32(context[slot]/3.))
            if reserve['card_id'] is None:
                require(p==1 and reserve['source']=='deck', 'Only opponent private reserves are unknown')
                unknown.append((slot,reserve['tier']))
            else:
                card_at(44+6*p+2*slot,reserve['card_id']); occupied.add(reserve['card_id'])
    pool=[card_id for card_id in bridge.RUST_CARD_IDS if card_id not in occupied]
    for tier in (1,2,3):
        tier_ids=[name for name in pool if cards[name].tier==tier]
        count=len(tier_ids); deck=view['deck_counts'][str(tier)]
        require(count==deck+sum(t==tier for _,t in unknown), 'Public unseen-pool accounting')
        probability=0. if not count else deck/count
        groups=[0]*5; masks=[0]*5
        for card_id in tier_ids:
            index=bridge.CARD_TO_RUST[card_id]; group=CARD_GROUP[index]; bit=CARD_SLOT[index]
            groups[group]+=1; masks[group]|=128>>bit
            packed[392+((tier-1)*5+group)*8+7-bit]=f32(probability)
        for group in range(5):
            raw[(25+2*(tier-1))*7+group]=f32(groups[group]*probability)
            signed=masks[group] if masks[group]<128 else masks[group]-256
            raw[(26+2*(tier-1))*7+group]=f32(signed*probability)
        for slot,which in unknown:
            if which!=tier:continue
            require(count>0, 'Nonempty unknown-reserve pool')
            row=50+2*slot
            for color_index,color in enumerate(bridge.COLORS):
                raw[row*7+color_index]=f32(sum(cards[name].cost.get(color,0) for name in tier_ids)/count)
                raw[(row+1)*7+color_index]=f32(sum(cards[name].bonus_color==color for name in tier_ids)/count)
            raw[(row+1)*7+6]=f32(sum(cards[name].points for name in tier_ids)/count)
    packed[:392]=raw; packed[512:519]=context
    # Canonical corrected environment uses the existing false native-profile
    # input; it is not the separate AlphaZeroNative environment flag.
    packed[519]=0.
    normalized=[f32(value*f32(.1)) for value in raw]
    normalized[6]=f32(raw[6]/124.)
    tokens=[0.]*(31*48)
    def put(token, start, values):tokens[token*48+start:token*48+start+len(values)]=values
    put(0,0,normalized[:7]);tokens[7]=packed[519]
    for p in range(2):
        put(1+p,0,normalized[(34+p)*7:(35+p)*7]);put(1+p,7,normalized[(42+p)*7:(43+p)*7])
        put(1+p,14,normalized[(36+3*p)*7:(39+3*p)*7])
    for slot in range(12):
        put(3+slot,0,normalized[(1+2*slot)*7:(3+2*slot)*7]);tokens[(3+slot)*48+14]=f32((slot//4+1)/3.)
    for slot in range(3):put(15+slot,0,normalized[(31+slot)*7:(32+slot)*7])
    for p in range(2):
        for slot in range(3):
            t=18+3*p+slot;put(t,0,normalized[(44+6*p+2*slot)*7:(46+6*p+2*slot)*7]);tokens[t*48+14]=float(p)
            if p==1:tokens[t*48+15:t*48+17]=[context[slot],context[slot+3]]
    for tier in range(3):
        put(24+tier,0,normalized[(25+2*tier)*7:(26+2*tier)*7]);put(24+tier,7,packed[392+40*tier:432+40*tier])
        tokens[(24+tier)*48+47]=f32((tier+1)/3.)
    put(27,0,context)
    for t,row in enumerate((26,28,30)):put(28+t,0,normalized[row*7:(row+1)*7])
    return tokens


def capture_record(observer, quota, native, root_callback, main_action):
    actions,indices,mask=main_support(observer)
    reason=('forced_pass' if actions==[{'kind':'Pass'}] else
            'single_choice' if len(actions)==1 else
            'late_heuristic' if observer['observation']['turn_index']>=124 else None)
    eligible=reason is None
    base=dict(schema=SCHEMA,ruleset_id=RULESET_ID,catalogue_sha256=CATALOGUE_SHA256,turn_contract=TURN_CONTRACT,
        quota=quota,seed_decimal=str(int(digest(observer)[:16],16)),observer_sha256=digest(observer),
        actor_seat=observer['observation']['current_player'],canonical_actor_slot=0,
        eligible=eligible,exclusion_reason=reason,authoritative_actions=actions,authoritative_indices67=indices,
        legal_mask81=mask,main_action=deepcopy(main_action),native_capture=deepcopy(native),
        retained_indices67=[] if native is None else list(native['retained_indices67']),
        callback_count=0 if native is None else native['callback_count'],
        root_callback_index=None,root_tokens=None,root_input_sha256=None,root_output83=None,root_output_sha256=None,
        visits67=None,policy_target81=None)
    if eligible:
        require(native is not None and root_callback is not None, 'Expected authentic searched root capture')
        visits=native['visits67'];total=sum(visits);require(total>0,'Positive search target mass')
        target=[0.]*81
        for index,value in enumerate(visits):
            if value:require(index in INDEX67_TO81,'No target at unused policy hole');target[INDEX67_TO81[index]]=value/total
        base.update(root_callback_index=0,root_tokens=deepcopy(root_callback['tokens']),
            root_input_sha256=digest(root_callback['tokens']),root_output83=deepcopy(root_callback['outputs'][0]),
            root_output_sha256=digest(root_callback['outputs'][0]),visits67=list(visits),policy_target81=target)
    validate_capture(base,observer,expected_quota=quota)
    return base


def validate_capture(record, observer, *, expected_quota=512):
    require(type(record) is dict and record.get('schema')==SCHEMA,'Root capture schema')
    for key,value in dict(ruleset_id=RULESET_ID,catalogue_sha256=CATALOGUE_SHA256,turn_contract=TURN_CONTRACT,
        quota=expected_quota,observer_sha256=digest(observer),seed_decimal=str(int(digest(observer)[:16],16)),
        actor_seat=observer['observation']['current_player'],canonical_actor_slot=0).items():
        require(record.get(key)==value,'Capture identity '+key)
    require(type(record['quota']) is int and record['quota'] in (16,128,256,512),'Explicit fixed quota')
    actions,indices,mask=main_support(observer)
    require(record['authoritative_actions']==actions and record['authoritative_indices67']==indices and record['legal_mask81']==mask,
        'Authoritative legality before search pruning')
    require(all(type(x) is bool for x in record['legal_mask81']),'Boolean policy mask')
    reason=('forced_pass' if actions==[{'kind':'Pass'}] else 'single_choice' if len(actions)==1 else
            'late_heuristic' if observer['observation']['turn_index']>=124 else None)
    require(type(record['eligible']) is bool and record['eligible']==(reason is None) and record['exclusion_reason']==reason,'Mechanical eligibility')
    require(record['main_action'] in actions,'Captured selected Main action')
    native=record['native_capture']
    if reason=='forced_pass':
        require(native is None and record['callback_count']==0 and record['retained_indices67']==[], 'Pass has no native search')
    else:
        require(type(native) is dict and native['enabled'] is True and native['phase']=='Main' and native['quota']==expected_quota,
            'Native target source identity')
        require(set(native)=={'enabled','phase','quota','authoritative_indices67','retained_indices67','callback_count',
            'root_callback_index','root_callback_role','simulations','visits67','q67','root_value','network_value','policy67'},
            'Exact target-only native evidence fields')
        require(native['authoritative_indices67']==indices and native['retained_indices67']==record['retained_indices67'],
            'Native original and retained root support')
        retained=record['retained_indices67']
        require(type(retained) is list and retained and len(retained)==len(set(retained)) and
            all(type(i) is int and i in indices for i in retained), 'Retained support is a nonempty authoritative subset')
        require(native['callback_count']==record['callback_count'] and type(record['callback_count']) is int and record['callback_count']>=0,
            'Actual callback inventory')
    target_fields=('root_callback_index','root_tokens','root_input_sha256','root_output83','root_output_sha256','visits67','policy_target81')
    if reason is not None:
        require(all(record[key] is None for key in target_fields),'No fabricated excluded targets')
        if native is not None:require(native['visits67'] is None and native['root_callback_index'] is None and native['callback_count']==0,
            'Excluded native branch performed no inference or captured visits')
        return record
    require(native['root_callback_index']==record['root_callback_index']==0 and native['root_callback_role']=='root' and record['callback_count']>=1,
        'Explicit first fresh-root callback; never a deeper callback')
    require(record['root_tokens']==public_root_tokens(observer) and digest(record['root_tokens'])==record['root_input_sha256'],
        'Exact independently reconstructed public-belief root tensor')
    output=record['root_output83']
    require(type(output) is list and len(output)==83 and all(type(x) is float and math.isfinite(x) and f32(x)==x for x in output),
        'Finite float32 root output')
    require(all(-1<=x<=1 for x in output[81:]) and digest(output)==record['root_output_sha256'],'Signed values and root output binding')
    visits=record['visits67'];retained=set(record['retained_indices67'])
    require(type(visits) is list and len(visits)==67 and all(type(x) is int and 0<=x<=expected_quota for x in visits) and
        visits==native['visits67'] and sum(visits)==expected_quota==native['simulations'],'Complete fixed-quota visits')
    require(all(not value or index in retained for index,value in enumerate(visits)),'Visits only in retained support')
    require(native['policy67']==[f32(value/expected_quota) for value in visits],'Original native normalized root policy')
    q=native['q67']
    require(type(q) is list and len(q)==67 and all((value is None) if visits[i]==0 else
        type(value) in (int,float) and math.isfinite(value) and 0<=value<=1 for i,value in enumerate(q)),
        'Visited Q labels only; win-credit units')
    for key in ('root_value','network_value'):
        require(type(native[key]) in (int,float) and math.isfinite(native[key]) and 0<=native[key]<=1,'Native '+key+' win credits')
    require(abs(sum(visits[i]*value for i,value in enumerate(q) if value is not None)/expected_quota-native['root_value'])<=2e-6,
        'Root Q aggregate remains separate from terminal training target')
    target=[0.]*81
    for index,value in enumerate(visits):
        if value:target[INDEX67_TO81[index]]=value/expected_quota
    require(record['policy_target81']==target and abs(sum(target)-1)<1e-12 and
        all(not value or mask[i] for i,value in enumerate(target)),'Mapped normalized target and full legal loss mask')
    require(action_index67(record['main_action']) in retained and visits[action_index67(record['main_action'])]==max(visits),
        'Selected action agrees with root visits')
    return record


class CaptureOpponent(BaselineOpponent):
    """The old exact host/search facade plus separately bound training evidence."""
    def __init__(self, profile_document, *, native_artifact, shared_from, capture_targets=True):
        self.profile=deepcopy(profile_document);self.forward=None;self.capture=False;self.callbacks=[]
        require(type(capture_targets) is bool,'Explicit capture mode');self.capture_targets=capture_targets
        require(self.profile['iterations'] in (16,128,256,512) and self.profile['world_pool']==3 and self.profile['cpuct']==.4 and
            self.profile['depth']==64 and self.profile['internal_budget_seconds']==9.,'Unchanged teacher/search settings')
        for item in [*self.profile['artifacts'],native_artifact]:
            path=ROOT/item['path'];require(path.stat().st_size==item['bytes'] and file_hash(path)==item['sha256'],'Authenticated capture/model input')
        self.model,self.reference=shared_from.model,shared_from.reference
        self.native_artifact=deepcopy(native_artifact);self.last_training_capture=None;self._main_capture=None;self._root_callback=None
        self.proc=subprocess.Popen([str(ROOT/native_artifact['path'])],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=sys.stderr,
            text=True,encoding='utf-8',bufsize=1,cwd=ROOT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        atexit.register(self.close)
        try:require(json.loads(self.proc.stdout.readline(MAX_NATIVE_LINE))=={'kind':'ready','bridge':BRIDGE_ID,'source_pin':PIN},'Capture bridge READY')
        except BaseException:self.close();raise
        self.deadline=None;self.inference_calls=self.inference_rows=0

    def _begin(self):
        super()._begin()
        # Save the SAME admitted deadline; do not reset it for target encoding.
        self._capture_decision_deadline=self.active_deadline

    def exchange(self, request):
        self.boundary();self.callbacks=[]
        value=deepcopy(request)
        if value['op']=='select':value['capture_targets']=self.capture_targets
        self.proc.stdin.write(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n');self.proc.stdin.flush()
        callback_count=0;root=None
        while True:
            line=self.proc.stdout.readline(MAX_NATIVE_LINE+1)
            require(line and len(line.encode('utf-8'))<=MAX_NATIVE_LINE and line.endswith('\n'),'Bounded complete native response')
            result=json.loads(line)
            if result['kind']=='prerequisite_error':raise ValueError('ADAPTER_PREREQUISITE: '+result['detail'])
            if result['kind']=='result':break
            require(result['kind']=='infer' and result['callback_index']==callback_count,'Contiguous actual inference callbacks')
            if self.deadline is not None and time.perf_counter()>=self.deadline:raise TimeoutError('Internal fixed-work deadline exceeded')
            outputs=self._infer(result['tokens'],result['batch'])
            if result['callback_role']=='root':
                require(callback_count==0 and root is None and result['batch']==1,'One actual Main root callback')
                root=dict(tokens=deepcopy(result['tokens']),outputs=deepcopy(outputs))
            else:require(result['callback_role'] in ('leaf','parity'),'Declared callback role')
            callback_count+=1;self.inference_calls+=1;self.inference_rows+=result['batch']
            self.proc.stdin.write(json.dumps({'kind':'inference_result','outputs':outputs},separators=(',',':'),allow_nan=False)+'\n');self.proc.stdin.flush()
        native=result.pop('training_capture',None)
        if value['op']=='select':
            require(native is not None and native['enabled']==self.capture_targets and native['callback_count']==callback_count,'Native capture callback binding')
            if value['observation']['phase']['kind']=='Main':self._main_capture,self._root_callback=native,root
        self.boundary()
        result.update(neural_callback_count=len(self.callbacks),neural_callback_rows=sum(row['batch'] for row in self.callbacks),
            neural_callbacks=[row['callback_sha256'] for row in self.callbacks])
        if self.capture:result['inference']=[{key:deepcopy(value) for key,value in row.items() if key!='callback_sha256'} for row in self.callbacks]
        return result

    def choose(self, observer, *, diagnostic_capture=False):
        self.last_training_capture=None;self._main_capture=None;self._root_callback=None;self._capture_decision_deadline=None
        action,trace=super().choose(observer,diagnostic_capture=diagnostic_capture)
        if self.capture_targets:
            main={'kind':'Pass'} if trace['forced_pass'] else trace['phases'][0]['action']
            self.last_training_capture=capture_record(observer,self.profile['iterations'],self._main_capture,self._root_callback,main)
        else:
            self.last_training_capture=dict(schema='selfplay_root_capture_disabled_v1',enabled=False,
                observer_sha256=digest(observer),quota=self.profile['iterations'],training_eligible=False)
        require(self._capture_decision_deadline is not None,'Original admitted decision deadline retained')
        if time.perf_counter()>=self._capture_decision_deadline:
            raise TimeoutError('STRENGTH_INTERNAL_DEADLINE: complete target capture exceeded original9s guard')
        return action,trace

    def direct_choose(self, observer, *, diagnostic_capture=True):
        return self.choose(observer,diagnostic_capture=diagnostic_capture)
