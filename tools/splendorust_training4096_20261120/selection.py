"""Namespaced training selection from accepted metadata only.

No game, target, encoded feature or prediction is opened by this module.
In particular, fresh assessment descriptors remain metadata, not verified files.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json
from . import common as c

SOURCES=('prior_20261107','fresh_20261115')
COUNTS={'prior_20261107':(1024,57022),'fresh_20261115':(3072,170181)}
PROPOSAL_SHA='08bcd17d319703af6d35f204a5e4bfbd61d10ba0c3d8cee87f994203b130cf9b'
_LOADED=None


def identity(row): return (row['source'],row['game_index'],row['ply'])


def inventory_line(source,row):
    return [source,row['game_index'],row['ply'],row['row_sha256'],row['root_input_sha256']]


def validate_namespace(rows,*,split='train'):
    seen=set()
    for row in rows:
        c.require(row['source'] in SOURCES and type(row['game_index']) is int and row['game_index']>=0 and
            type(row['ply']) is int and row['ply']>=0 and row['split']==split,'Explicit namespaced allowed split')
        key=identity(row);c.require(key not in seen,'Duplicate namespaced root');seen.add(key)
        for name in ('row_sha256','root_input_sha256'):
            value=row[name];c.require(type(value) is str and len(value)==64 and set(value)<=set('0123456789abcdef'), 'Bound source row hash')
    return seen


def metadata(boundary=lambda:None):
    c.require(c.sha(c.PROPOSAL)==PROPOSAL_SHA,'Unchanged prepared study proposal')
    proposal=c.read(c.PROPOSAL);c.verify([proposal['source_bindings']],boundary)
    bindings=c.read(c.BINDINGS);c.verify(bindings['inputs'],boundary)
    c.verify([bindings['accepted_collection_closeout']],boundary)
    closeout=c.read(c.resolve(bindings['accepted_collection_closeout']))
    c.require(closeout.get('status')=='accepted_amended_recovered_collection',
        'Accepted collection actual post-exit closeout')
    union=c.read(c.UNION)
    c.require(union['status']=='accepted' and union['training_games']==4096 and union['fresh_assessment_games']==256 and
        union['fresh_assessment_sealed_until_future_checkpoints_committed'] is True and
        all(union[k]==v for k,v in c.IDENTITY.items()),'Corrected accepted4096 union and sealed fresh assessment')
    training_members={(x['source'],x['game_index']):x for x in union['training_members']}
    assessment_members={(x['source'],x['game_index']):x for x in union['fresh_assessment_members']}
    c.require(len(training_members)==4096 and len(assessment_members)==256 and
        not training_members.keys()&assessment_members.keys(),'Disjoint unique namespaced game membership')
    train=[];assessment=[];inventory=[]
    for source in SOURCES:
        boundary();meta=union['sources'][source]
        c.verify([meta['acceptance'],meta['catalogue']],boundary)
        acceptance=c.read(c.resolve(meta['acceptance']));catalogue=c.read(c.resolve(meta['catalogue']))
        c.require(acceptance['status']=='accepted' and acceptance['passed'] is True and
            acceptance['catalogue']==meta['catalogue'] and catalogue['targets_in_catalogue'] is False,
            'Only accepted outcome-free catalogue metadata')
        selected=[]
        for game in sorted(catalogue['games'],key=lambda g:g['spec']['game_index']):
            key=(source,game['spec']['game_index']);member=training_members.get(key);fresh=assessment_members.get(key)
            if member is None and fresh is None:continue
            item=dict(source=source,**game);spec=game['spec']
            chosen=member if member is not None else fresh
            c.require(chosen['game_id']==spec['game_id'] and chosen['split']==spec['split'],'Membership binds exact source game')
            roots=[dict(source=source,**row) for row in game['roots']]
            c.require([r['ply'] for r in roots]==sorted({r['ply'] for r in roots}) and
                all(r['game_index']==spec['game_index'] and r['split']==spec['split'] for r in roots),'Exact ordered catalogue roots')
            item['roots']=roots
            if member is not None:
                c.require(spec['split']=='train','Explicit training-only allowlist');selected.append(item);train.append(item)
                inventory.extend(roots)
            else:
                c.require(source==SOURCES[1] and spec['split'] in ('validation','offline'),'Only fresh prospective assessment');assessment.append(item)
        c.require((len(selected),sum(len(x['roots']) for x in selected))==COUNTS[source],'Exact source training game/root counts')
    validate_namespace(inventory)
    h=hashlib.sha256()
    for row in inventory:h.update((c.canonical(inventory_line(row['source'],row))+'\n').encode('ascii'))
    c.require(h.hexdigest()==bindings['canonical_training_inventory']['sha256'] and len(inventory)==227203,
        'Exact prospectively frozen canonical227203 inventory')
    c.require(Counter(x['spec']['split'] for x in assessment)=={'validation':128,'offline':128},'Both fresh128game strata complete')
    assessment_binding=dict(union=bindings['union'],members=union['fresh_assessment_members'],
        members_sha256=c.digest(union['fresh_assessment_members']),catalogue=union['sources'][SOURCES[1]]['catalogue'],
        games=256,partitions={'validation':128,'offline':128},roots=sum(len(x['roots']) for x in assessment))
    return dict(schema='training4096_namespaced_selection_v1',**c.IDENTITY,proposal=c.artifact(c.PROPOSAL),
        source_bindings=c.artifact(c.BINDINGS),union=bindings['union'],training_games=train,training_inventory=inventory,
        training_inventory_sha256=h.hexdigest(),training_rows=227203,assessment_games=assessment,
        assessment_binding=assessment_binding,assessment_protected_records_read=0,models_loaded=0)


def prepare(boundary=lambda:None):
    value=metadata(boundary)
    if c.SELECTION.exists():c.require(c.read(c.SELECTION)==value,'Existing selection exactly equals immutable metadata preparation')
    else:c.save(c.SELECTION,value)
    return value


def load(boundary=lambda:None):
    global _LOADED
    stat=c.SELECTION.stat();stamp=(stat.st_size,stat.st_mtime_ns,stat.st_ino)
    if _LOADED is None:
        descriptor=c.artifact(c.SELECTION)
        value=c.read(c.SELECTION);expected=metadata(boundary)
        c.require(value==expected,'Selection exactly reconstructs metadata-only prepared source')
        _LOADED=(descriptor,stamp,value)
    c.require(_LOADED[1]==stamp,'Selection metadata remains unchanged within worker')
    return _LOADED[2]


def fit_spec(fit_id):
    c.require(fit_id in c.FIT_IDS,'Only four fixed fits')
    return next(x for x in c.read(c.PROPOSAL)['schedules'] if x['fit_id']==fit_id)


def order(fit_id):
    import numpy as np
    spec=fit_spec(fit_id);c.verify([spec['order']]);result=np.memmap(c.resolve(spec['order']),mode='r',dtype='<u4')
    c.require(len(result)==227203 and int(result.max())< (57022 if spec['arm']=='control' else 227203),'Fixed schedule namespace range')
    c.require(len(set(map(int,result)))==spec['unique_rows'],'Fixed unique exposure count')
    return result


def schedule(fit_id):
    spec=fit_spec(fit_id);values=list(map(int,order(fit_id)));selected=load()
    return dict(spec,order=values,batches=[values[i:i+32] for i in range(0,len(values),32)],
        schedule_artifact=spec['order'],training_inventory_sha256=selected['training_inventory_sha256'])


def panel_indices():
    import numpy as np
    item=c.read(c.PROPOSAL)['training_panel']['order'];c.verify([item])
    return np.memmap(c.resolve(item),dtype='<u4',mode='r')


def verify_schedules(folder=None,boundary=lambda:None):
    selected=load(boundary);rows=selected['training_inventory'];proposal=c.read(c.PROPOSAL)
    def key(i,seed,number):
        x=rows[i];value=[proposal['order_domain'],seed,number,*identity(x)]
        return hashlib.sha256(c.canonical(value).encode('ascii')).digest(),identity(x)
    for rep,seed in enumerate(c.SEEDS):
        boundary();full=sorted(range(227203),key=lambda i:key(i,seed,1));small=[i for i in full if i<57022]
        for number in (2,3,4):small.extend(sorted(range(57022),key=lambda i:key(i,seed,number)))
        small=small[:227203]
        c.require(list(map(int,order(f'larger_r{rep}')))==full and list(map(int,order(f'control_r{rep}')))==small,
            'Exactly predetermined common ordering and concatenated matched-effort control passes')
    result=dict(status='schedules_verified',selection=c.artifact(c.SELECTION),training_inventory_sha256=selected['training_inventory_sha256'],
        schedules=[x['order'] for x in proposal['schedules']],updates_per_fit=7101,presentations_per_fit=227203,final_batch=3,
        metadata_only=True,fresh_assessment_records_read=0)
    if folder is not None:c.save(Path(folder)/'schedule_verification.json',result)
    return result


def allowed_game(game,purpose,boundary=lambda:None,*,release=None):
    selected=load(boundary)
    if purpose=='assessment':
        from . import seal
        seal.require_release(release)
        allowed=selected['assessment_games']
    else:
        c.require(purpose=='training','Explicit data role');allowed=selected['training_games']
    key=(game['source'],game['spec']['game_index'])
    matching=[x for x in allowed if (x['source'],x['spec']['game_index'])==key]
    c.require(len(matching)==1 and matching[0]==game,'Only exact role-allowed game descriptor before protected reads')
    return game


def validate_initial_inputs(items):
    """Metadata-only path exclusion, without hashing or opening a protected file."""
    selected=load();protected=set();parents=set()
    for game in selected['assessment_games']:
        for key in ('artifact','source_game'):
            path=c.resolve(game[key]);protected.add(path)
            # Source game documents live beside their move records, exclusively per game.
            if key=='source_game':parents.add(path.parent)
    for item in items:
        path=c.resolve(item)
        c.require(path not in protected and not any(path.is_relative_to(parent) for parent in parents)
            and not path.is_relative_to(c.ASSESS_CACHE.resolve()),'Fresh assessment files cannot enter precommit inputs')
    return dict(status='protected_initial_paths_excluded',checked=len(items),protected_games=len(selected['assessment_games']),protected_file_reads=0)
