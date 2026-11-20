"""Read-only exact source catalogue parser; corrected semantic ID bijection."""
from functools import lru_cache
from pathlib import Path
import hashlib
import re
from tools.external_strength_screen_20261018 import rules
ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'external/dom_splendor_426fa50'
DATA_SHA256 = '596ec63ca41edd59eb926f51256bf10eed8699c8834e80e6a25c860969564be9'
COLORS = ('white', 'blue', 'green', 'red', 'black')

@lru_cache(maxsize=1)
def catalogue_mapping():
    raw = (ASSETS/'crates/splendor-core/src/data.rs').read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATA_SHA256:
        raise ValueError('Pinned Dom catalogue changed')
    text = raw.decode()
    cards = re.findall(r'Card\s*\{\s*tier:\s*(\d+),\s*points:\s*(\d+),\s*bonus:\s*Color::(\w+),\s*cost:\s*\[([^]]+)\],?\s*\}', text)
    nobles = re.findall(r'Noble\s*\{\s*points:\s*(\d+),\s*req:\s*\[([^]]+)\],?\s*\}', text)
    faces = {(c.tier,c.points,c.bonus_color,tuple(c.cost.get(k,0) for k in COLORS)):c.card_id for c in rules.corrected_cards_by_id().values()}
    noble_faces = {(n.points,tuple(n.requirement.get(k,0) for k in COLORS)):n.noble_id for n in rules.corrected_nobles_by_id().values()}
    cm, nm = {}, {}
    for i,(tier,points,color,cost) in enumerate(cards):
        amounts=tuple(int(x.strip()) for x in cost.split(',') if x.strip())
        key=(int(tier),int(points),color.lower(),amounts)
        if key not in faces or faces[key] in cm: raise ValueError('Dom card face mismatch')
        cm[faces[key]]=dict(id=i,tier=int(tier),points=int(points),bonus=COLORS.index(color.lower()),cost=list(amounts))
    for i,(points,req) in enumerate(nobles):
        amounts=tuple(int(x.strip()) for x in req.split(',') if x.strip());key=(int(points),amounts)
        if key not in noble_faces or noble_faces[key] in nm: raise ValueError('Dom noble face mismatch')
        nm[noble_faces[key]]=dict(id=i,points=int(points),req=list(amounts))
    if len(cm)!=90 or len(nm)!=10: raise ValueError('Dom catalogue incomplete')
    return cm,nm
