"""Independent source-semantic reference; no external game execution.

The encoder is a line-by-line Python port of pinned encode.rs:push_card and
encode_state. Inference is the literal sequential double loop from pinned
play/template.html:forward, returning all unmasked final scores. No JS/Rust
runtime is introduced, no source engine transitions or hidden state are used.
This is a reviewed source port, not a claim that the Rust binary was executed.
"""
from functools import lru_cache
import hashlib
import itertools
import json
from pathlib import Path
import struct
import numpy as np
from .catalogue import catalogue_mapping
ASSET_ROOT=Path(__file__).resolve().parents[2]/'external/dom_splendor_426fa50'
WEIGHTS=ASSET_ROOT/'play/weights.bin'
WEIGHTS_SHA256='e80e468a7e99ead87b43277c8c17219be669e55fac54604abbd55d68b1586191'
SOURCE_HASHES={'crates/splendor-core/src/encode.rs': 'c8271cd53b1b412416ac0fa77598c7e43251e38e2fae3e75ac972347e2b3e2e6', 'crates/splendor-core/src/game.rs': 'ad4b4a016324aed24615bab3558dc0972d9d37a4d5f997a37ce7d3375e01a582', 'crates/splendor-core/src/data.rs': '596ec63ca41edd59eb926f51256bf10eed8699c8834e80e6a25c860969564be9', 'play/template.html': '80efef2f9287963303870dd7dd9be2870760ae3b93d15c71c4a77c1ee6b3f35a', 'play/weights.json': 'c6155b799d0cb9061370dabeccba9993270dcd786b862a0d34953ec94516b5a9', 'python/export_weights.py': '977cbf0880f392ec5d175c3cd8d5237d29bf1e632a4f7804228b1671eab266c2', 'python/ppo_train.py': 'fe2fb1ae9bff352fae19a29e0d8edb2d947b9329bf917c49b9723946565cec1b'}

def require(value,message):
    if not value:raise ValueError(message)

@lru_cache(maxsize=1)
def verify_sources():
    for relative,sha in SOURCE_HASHES.items():
        require(hashlib.sha256((ASSET_ROOT/relative).read_bytes()).hexdigest()==sha,'Pinned Dom source changed: '+relative)
    return True

def encode_phase(context):
    """Source lines146–213, append-order implementation independent of adapter."""
    verify_sources();cards,_=catalogue_mapping();table={x['id']:x for x in cards.values()}
    out=[]
    def fdiv(a,b):return np.float32(a)/np.float32(b)
    def push_card(identity):
        if identity is None:out.extend([np.float32(0)]*12);return
        card=table[identity];out.extend([np.float32(1),fdiv(card['points'],5)])
        out.extend(np.float32(card['bonus']==i) for i in range(5))
        out.extend(fdiv(x,7) for x in card['cost'])
    for tier in range(3):
        for slot in range(4):push_card(context['display'][tier][slot])
    out.extend(fdiv(x,40) for x in context['decks'])
    out.extend(fdiv(x,7) for x in context['bank'])
    out.extend(np.float32(i in context['nobles']) for i in range(10))
    out.extend(np.float32(context['phase']==phase) for phase in ('play','discard','choose_noble'))
    out.append(min(fdiv(context['round'],60),np.float32(1.5)))
    for offset in range(2):
        player=context['players'][(context['current']+offset)%2]
        out.extend(fdiv(x,10) for x in player['tokens']);out.extend(fdiv(x,7) for x in player['bonuses']);out.append(fdiv(player['points'],15))
        for slot in range(3):push_card(player['reserved'][slot] if slot<len(player['reserved']) else None)
        out.extend([fdiv(player['owned_count'],20),fdiv(player['noble_count'],5)])
    require(len(out)==267,'Source feature length')
    return np.asarray(out,dtype=np.float32)

def encode(observer):
    from . import adapter
    state,_,_=adapter.lawful_completion(observer)
    return encode_phase(adapter.phase_context(state))

def source_mask_phase(ctx):
    """Independent source game.rs legal_actions and encode.rs index mapping."""
    verify_sources();cards,nobles=catalogue_mapping();cs={c['id']:c for c in cards.values()};ns={n['id']:n for n in nobles.values()}
    p=ctx['players'][ctx['current']];mask=[False]*77
    if ctx['phase']=='discard':
        for g in range(6):mask[60+g]=p['tokens'][g]>0
    elif ctx['phase']=='choose_noble':
        for n in ctx['nobles']:mask[66+n]=all(p['bonuses'][i]>=ns[n]['req'][i] for i in range(5))
    else:
        combos=list(itertools.combinations(range(5),1))+list(itertools.combinations(range(5),2))+list(itertools.combinations(range(5),3))
        for i,combo in enumerate(combos):mask[i]=all(ctx['bank'][c]>0 for c in combo)
        for i in range(5):mask[25+i]=ctx['bank'][i]>=4
        def afford(card):
            total=0
            for i in range(5):total+=max(0,max(0,cs[card]['cost'][i]-p['bonuses'][i])-p['tokens'][i])
            return total<=p['tokens'][5]
        for tier in range(3):
            for slot in range(4):
                card=ctx['display'][tier][slot]
                mask[30+4*tier+slot]=card is not None and len(p['reserved'])<3
                mask[45+4*tier+slot]=card is not None and afford(card)
            mask[42+tier]=len(p['reserved'])<3 and ctx['decks'][tier]>0
        for i,card in enumerate(p['reserved']):mask[57+i]=afford(card)
        mask[76]=not any(mask)
    return np.asarray(mask,dtype=bool)

def source_mask(observer):
    from . import adapter
    state,_,_=adapter.lawful_completion(observer)
    return source_mask_phase(adapter.phase_context(state))

def load_policy():
    """Postfreeze only: raw numeric data, never pickle/code or random init."""
    verify_sources();raw=WEIGHTS.read_bytes()
    require(len(raw)==1757492 and hashlib.sha256(raw).hexdigest()==WEIGHTS_SHA256,'Pinned reference weights changed')
    flat=struct.unpack('<'+str(len(raw)//4)+'f',raw);layers=[];offset=0
    for n,m in ((267,512),(512,512),(512,77)):
        weights=tuple(flat[offset+j*n:offset+(j+1)*n] for j in range(m));offset+=n*m
        bias=flat[offset:offset+m];offset+=m;layers.append((weights,bias))
    require(offset==len(flat),'Reference weight layout');return tuple(layers)

def predict(features,policy=None):
    policy=load_policy() if policy is None else policy
    x=np.asarray(features);require(x.shape==(267,) and x.dtype==np.float32 and np.isfinite(x).all(),'Reference input')
    values=[float(v) for v in x]
    for layer,(weights,bias) in enumerate(policy):
        output=[]
        for row,b in zip(weights,bias):
            s=float(b)
            for w,value in zip(row,values):s+=float(w)*value
            output.append(float(np.float32(max(0.,s))) if layer<2 else s)
        values=output
    result=np.asarray(values,dtype=np.float64);require(result.shape==(77,) and np.isfinite(result).all(),'Reference output');return result

def validate_mapping_contract():
    from . import adapter
    verify_sources();source=(ASSET_ROOT/'crates/splendor-core/src/encode.rs').read_text()
    for text in ['NUM_ACTIONS: usize = 77','TAKE2_BASE: usize = 25','RESERVE_DISPLAY_BASE: usize = 30','RESERVE_DECK_BASE: usize = 42','BUY_DISPLAY_BASE: usize = 45','BUY_RESERVED_BASE: usize = 57','DISCARD_BASE: usize = 60','NOBLE_BASE: usize = 66','PASS_INDEX: usize = 76']:
        require(text in source,'Source action constants')
    expected=[];colors=('white','blue','green','red','black')
    for size in (1,2,3):
        for combo in itertools.combinations(colors,size):expected.append(dict(index=len(expected),family='take_distinct',take_tokens=list(combo)))
    expected.extend(dict(index=25+i,family='take_double',take_tokens=[c,c]) for i,c in enumerate(colors))
    for i in range(30,42):expected.append(dict(index=i,family='reserve_visible',tier=(i-30)//4+1,market_index=(i-30)%4))
    expected.extend(dict(index=i,family='reserve_deck',tier=i-41) for i in range(42,45))
    for i in range(45,57):expected.append(dict(index=i,family='buy_visible',tier=(i-45)//4+1,market_index=(i-45)%4))
    expected.extend(dict(index=i,family='buy_reserved',reserved_index=i-57) for i in range(57,60))
    expected.extend(dict(index=60+i,family='discard',token=c) for i,c in enumerate(colors+('gold',)))
    expected.extend(dict(index=66+i,family='choose_noble',source_noble_id=i) for i in range(10));expected.append(dict(index=76,family='pass'))
    require(tuple(expected)==adapter.ACTION_TABLE,'Source77 action descriptors')
    cm,nm=catalogue_mapping();return dict(status='passed',static_indices_checked=77,cards=len(cm),nobles=len(nm),source_encoder='independently ported pinned Rust append-order semantics',source_policy='literal pinned browser double-accumulation loops',source_engine_executed=False)
