"""Namespaced corrected caches, with a capability gate before fresh assessment I/O."""
from pathlib import Path
from copy import deepcopy
from collections.abc import Sequence
import hashlib,json,os,math
from . import common as c, selection
from tools.splendorust_independent_data_20261107 import corpus as source_corpus, schedule as old_schedule
from tools.splendorust_independent_data_20261107.collection import common as cc, data
from tools.splendorust_full_corpus_20261110 import targets, sampling
from tools import future_training_rules_20261021 as rules
_CACHES={}


def feature_arrays(examples):
    import numpy as np
    values=data.collate_arrays(examples,[1.]*len(examples))
    return {key:np.ascontiguousarray(values[key]) for key in ('tokens','policy','mask')}


def _write_jsonl(path,rows):
    path=Path(path);c.require(path.resolve().is_relative_to(c.OUT),'Exclusive new cache boundary')
    with path.open('xb') as stream:
        for row in rows:stream.write((c.canonical(row)+'\n').encode())
        stream.flush();os.fsync(stream.fileno())
    return c.artifact(path)


def _read_index(item,boundary):
    rows=[]
    with c.resolve(item).open('r',encoding='utf-8') as stream:
        for line in stream:
            if len(rows)%256==0:boundary()
            row=json.loads(line,object_pairs_hook=c._unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite index')))
            targets.for_arm(row,'mixed');rows.append(row)
    return rows


def _arrays(items,n,boundary):
    import numpy as np
    c.verify(items.values(),boundary)
    expected={'tokens':((n,31,48),np.dtype('float32')),'policy':((n,81),np.dtype('float32')),'mask':((n,81),np.dtype('bool'))}
    c.require(set(items)==set(expected),'Exactly three fixed feature arrays')
    out={key:np.load(c.resolve(item),allow_pickle=False,mmap_mode='r') for key,item in items.items()}
    for key,(shape,dtype) in expected.items():c.require(out[key].shape==shape and out[key].dtype==dtype,'Fixed cache shape/precision')
    return out


def _digest(rows):
    h=hashlib.sha256()
    for row in rows:
        fields={k:row[k] for k in ('source','game_index','ply','row_sha256','root_input_sha256','terminal_value','root_q_win_credit','root_q_source')}
        h.update((c.canonical(fields)+'\n').encode())
    return h.hexdigest()


def _inventory(rows):
    return [{k:x[k] for k in ('source','game_index','ply','split','row_sha256','root_input_sha256')} for x in rows]


def _old_cache(boundary=lambda:None):
    key=('old',str(c.CACHE_ACCEPTANCE))
    if key in _CACHES:return _CACHES[key]
    bindings=c.read(c.BINDINGS);c.verify([bindings['prior_cache_acceptance'],bindings['prior_cache_manifest']],boundary)
    c.require(bindings['prior_cache_acceptance']==c.artifact(c.CACHE_ACCEPTANCE) and bindings['prior_cache_manifest']==c.artifact(c.CACHE_MANIFEST),'Fixed accepted historical corrected cache')
    accepted=c.read(c.CACHE_ACCEPTANCE);manifest=c.read(c.CACHE_MANIFEST)
    c.require(accepted['status']=='independently_verified' and accepted['manifest']==bindings['prior_cache_manifest'] and accepted['rows']==64108 and
        all(accepted[x] is True for x in ('all_source_rows_checked','all_feature_bytes_checked','all_q_actor_and_terminal_targets_checked')),'Full independent old cache audit')
    c.require(manifest['rows']==64108 and manifest['root_counts']=={'train':57022,'validation':3533,'offline':3553} and all(manifest[k]==v for k,v in c.IDENTITY.items()),'Accepted corrected cache identity')
    for role,mode,keyname,item in [('corpus_build','cache','manifest',bindings['prior_cache_manifest']),('corpus_audit','cache_audit','acceptance',bindings['prior_cache_acceptance'])]:
        folder=c.ORIGINAL_OUT/role;actual=c.read(folder/'execution.json');result=c.read(folder/'worker/result.json')
        c.require(actual['status']=='passed' and actual['worker_exit_code']==0 and actual['cleanup_errors']==[] and actual['error'] is None,'Historical builder/auditor clean observed exit')
        c.require(result['status']=='passed' and result['mode']==mode and result['result'][keyname]==item,'Historical actual result binds accepted cache')
    c.verify([manifest['index']],boundary);index=_read_index(manifest['index'],boundary);arrays=_arrays(manifest['arrays'],64108,boundary)
    c.require(len(index)==64108,'Complete old cache index')
    offsets=[i for i,x in enumerate(index) if x['split']=='train'];rows=[dict(index[i],source=selection.SOURCES[0]) for i in offsets]
    selected=selection.load(boundary)
    c.require(len(rows)==57022 and _inventory(rows)==selected['training_inventory'][:57022],'Canonical old training-only cache projection')
    value=dict(index=rows,arrays=arrays,offsets=offsets,acceptance=bindings['prior_cache_acceptance'],manifest=bindings['prior_cache_manifest'])
    _CACHES[key]=value;return value


class Dataset(Sequence):
    def __init__(self,index,shards,cache_acceptance,*,release=None):
        self.index=index;self.shards=shards;self.cache_acceptance=cache_acceptance;self.release=release
        self.value_target_sources_sha256=_digest(index)
    def __len__(self):return len(self.index)
    def __getitem__(self,index):
        if isinstance(index,slice):return [self[i] for i in range(*index.indices(len(self)))]
        c.require(type(index) is int or type(index).__module__=='numpy','Integer canonical cache index')
        index=int(index);c.require(0<=index<len(self),'Canonical row within selected role')
        if self.release is not None:
            from . import seal
            seal.require_release(self.release)
        for begin,end,arrays,offsets in self.shards:
            if begin<=index<end:
                offset=index-begin if offsets is None else offsets[index-begin];row=dict(self.index[index])
                row['tokens']=arrays['tokens'][offset].reshape(-1).tolist();row['policy']=arrays['policy'][offset].tolist();row['mask']=arrays['mask'][offset].tolist()
                return targets.for_arm(row,'mixed')
        raise ValueError('Missing canonical cache shard')
    def collate(self,indices):
        examples=[self[int(i)] for i in indices]
        return sampling.collate(examples,[1.]*len(examples))


def prior_training_dataset(boundary=lambda:None):
    cache=_old_cache(boundary)
    return Dataset(cache['index'],[(0,57022,cache['arrays'],cache['offsets'])],cache['acceptance'])


def index_row(example,label,source):
    keys=('game_id','game_index','ply','split','actor','canonical_players','observer_sha256','row_sha256','root_input_sha256','route')
    row={key:deepcopy(example[key]) for key in keys};row.update(source=source,terminal_value=deepcopy(example['value']),
        root_q_win_credit=label['root_q_win_credit'],root_q_signed=label['root_q_signed'],root_q_source=deepcopy(label['source_move']))
    targets.for_arm(row,'mixed');return row


def _read_source(item,purpose,release):
    if purpose=='assessment':
        from . import seal
        seal.require_release(release)
    return data.read_artifact(item)


def decode_game(game,purpose,*,boundary=lambda:None,release=None,independent=False):
    selection.allowed_game(game,purpose,boundary,release=release)
    spec=game['spec'];parsed=source_corpus._parse_target(_read_source(game['artifact'],purpose,release),spec)
    examples=parsed['examples'];document=_read_source(game['source_game'],purpose,release)
    data._legacy.accepted.verify_identity(document)
    c.require(document['schema']==data.GAME_SCHEMA and document['spec']==spec and document['reason']=='completed','Exact completed corrected source document')
    winner=rules.require_official_terminal_target(document['final'],'completed')
    c.require(document['winner']==winner,'Official recorded terminal winner')
    roots=[dict(source=game['source'],game_index=spec['game_index'],ply=x['ply'],split=spec['split'],row_sha256=x['row_sha256'],root_input_sha256=x['root_input_sha256']) for x in examples]
    c.require(roots==game['roots'],'Every exact accepted namespaced source root')
    index=[]
    for example in examples:
        boundary();z=0. if winner is None else (1. if winner==example['actor'] else -1.)
        c.require(example['value']==[z,-z],'Official actor-relative terminal label')
        descriptor=document['rows'][example['ply']];raw=_read_source(descriptor,purpose,release)
        if independent:
            decoded=data.validate_row(raw,spec,raw['before'],example['ply']);native=raw['training_capture']['native_capture']
            credit=math.fsum(float(v)*int(n) for n,v in zip(native['visits67'],native['q67']) if v is not None)/sum(native['visits67'])
            c.require(sum(native['visits67'])==512 and abs(credit-native['root_value'])<=2e-6 and
                raw['training_capture']['actor_seat']==example['actor'] and c.digest(raw)==example['row_sha256'],'Independent actor/Q/source authentication')
            for key in ('tokens','policy','mask','canonical_players','observer_sha256','root_input_sha256'):
                c.require(c.canonical(decoded[key])==c.canonical(example[key]),'Independent corrected target '+key)
            label=dict(root_q_win_credit=native['root_value'],root_q_signed=2*native['root_value']-1,source_move=descriptor)
        else:label=targets.search_label(example,raw,descriptor,spec)
        index.append(index_row(example,label,game['source']))
    return examples,index


def _write_cache(base,games,purpose,boundary,release):
    import numpy as np
    base=Path(base);c.require(base.resolve().is_relative_to(c.OUT),'New cache path only');base.mkdir(parents=True,exist_ok=False)
    n=sum(len(g['roots']) for g in games);layout={'tokens':((n,31,48),np.float32),'policy':((n,81),np.float32),'mask':((n,81),np.bool_)}
    arrays={k:np.lib.format.open_memmap(base/(k+'.npy'),mode='w+',dtype=d,shape=shape) for k,(shape,d) in layout.items()}
    index=[];offset=0;ranges=[]
    for game_number,game in enumerate(games,1):
        boundary();examples,rows=decode_game(game,purpose,boundary=boundary,release=release);start=offset
        for begin in range(0,len(examples),32):
            subset=examples[begin:begin+32];features=feature_arrays(subset)
            for key,array in features.items():arrays[key][offset:offset+len(subset)]=array
            offset+=len(subset)
        index.extend(rows);ranges.append(dict(source=game['source'],game_index=game['spec']['game_index'],split=game['spec']['split'],start=start,stop=offset,target=game['artifact'],source_game=game['source_game']))
        c.status(base/'progress.json',dict(role=purpose+'_cache',completed_games=game_number,completed_roots=offset,total_games=len(games),total_roots=n,time=c.now()))
    c.require(offset==n,'Every selected root written once')
    for array in arrays.values():array.flush()
    del arrays
    return dict(rows=n,arrays={k:c.artifact(base/(k+'.npy')) for k in layout},index=_write_jsonl(base/'index.jsonl',index),game_ranges=ranges,target_sources_sha256=_digest(index))


def _build(purpose,folder,boundary):
    from . import seal
    release=seal.acquire_release(boundary) if purpose=='assessment' else None
    selected=selection.load(boundary);base=c.ASSESS_CACHE if purpose=='assessment' else c.TRAIN_CACHE
    games=selected['assessment_games'] if purpose=='assessment' else [g for g in selected['training_games'] if g['source']==selection.SOURCES[1]]
    prior=_old_cache(boundary) if purpose=='training' else None
    if release is not None:seal.mark_opened(release)
    manifest=_write_cache(base,games,purpose,boundary,release)
    manifest.update(schema='training4096_role_cache_v1',role=purpose,selection=c.artifact(c.SELECTION),**c.IDENTITY,
        prior_cache=prior['acceptance'] if prior else None,commitment=c.artifact(c.COMMITMENT) if release else None,
        canonical_training_inventory_sha256=selected['training_inventory_sha256'] if purpose=='training' else None,
        assessment_binding=selected['assessment_binding'] if release else None,source_rows_authenticated=manifest['rows'],models=0,optimizer_calls=0)
    item=c.save(base/'cache_manifest.json',manifest)
    return dict(status='cache_built_pending_independent_audit',manifest=item,rows=manifest['rows'],role=purpose,models=0)


def build_training(folder,boundary=lambda:None):return _build('training',folder,boundary)
def build_assessment(folder,boundary=lambda:None):return _build('assessment',folder,boundary)


def _cache(purpose,boundary,*,acceptance=True,release=None):
    from . import control,seal
    if purpose=='assessment':seal.require_release(release)
    base=c.TRAIN_CACHE if purpose=='training' else c.ASSESS_CACHE;role=purpose+'_cache'
    descriptor=c.artifact(base/'cache_manifest.json');manifest=c.read(base/'cache_manifest.json');selected=selection.load(boundary)
    _,built=control.passed_job(role,boundary)
    c.require(built['result']['manifest']==descriptor,'Actual successful builder binds manifest')
    c.require(manifest['schema']=='training4096_role_cache_v1' and manifest['role']==purpose and manifest['selection']==c.artifact(c.SELECTION) and
        all(manifest[k]==v for k,v in c.IDENTITY.items()),'Exact corrected role cache')
    if purpose=='training':
        c.require(manifest['rows']==170181 and manifest['prior_cache']==c.artifact(c.CACHE_ACCEPTANCE) and manifest['commitment'] is None and
            manifest['canonical_training_inventory_sha256']==selected['training_inventory_sha256'],'Training-only fresh shard plus accepted prior training')
        expected=[r for r in selected['training_inventory'] if r['source']==selection.SOURCES[1]]
    else:
        c.require(manifest['assessment_binding']==selected['assessment_binding'] and manifest['commitment']==c.artifact(c.COMMITMENT) and
            manifest['rows']==selected['assessment_binding']['roots'] and manifest['prior_cache'] is None,'Exact postcommit prospective assessment shard')
        expected=[r for g in selected['assessment_games'] for r in g['roots']]
    c.verify([manifest['index']],boundary);rows=_read_index(manifest['index'],boundary)
    c.require(_inventory(rows)==expected and len(rows)==manifest['rows']==manifest['source_rows_authenticated'] and _digest(rows)==manifest['target_sources_sha256'],'Cache exact namespaced row identities and target stream')
    arrays=_arrays(manifest['arrays'],manifest['rows'],boundary)
    receipt=None
    if acceptance:
        receipt=c.artifact(base/'cache_acceptance.json');proof=c.read(c.resolve(receipt));_,audited=control.passed_job(role+'_audit',boundary)
        c.require(audited['result']['acceptance']==receipt and proof['manifest']==descriptor and proof['status']=='independently_verified' and
            proof['all_source_rows_checked'] is proof['all_feature_bytes_checked'] is proof['all_q_actor_and_terminal_targets_checked'] is True and proof['rows']==manifest['rows'],'Complete independently exited role cache audit')
    return dict(index=rows,arrays=arrays,manifest=manifest,descriptor=descriptor,acceptance=receipt)


def _audit(purpose,folder,boundary):
    import numpy as np
    from . import seal
    release=seal.acquire_release(boundary) if purpose=='assessment' else None
    selected=selection.load(boundary);cache=_cache(purpose,boundary,acceptance=False,release=release)
    games=selected['assessment_games'] if purpose=='assessment' else [g for g in selected['training_games'] if g['source']==selection.SOURCES[1]]
    offset=0
    c.require(len(games)==len(cache['manifest']['game_ranges']),'Exact game range count')
    for game_number,(game,span) in enumerate(zip(games,cache['manifest']['game_ranges'],strict=True),1):
        boundary();examples,rows=decode_game(game,purpose,boundary=boundary,release=release,independent=True)
        c.require(span==dict(source=game['source'],game_index=game['spec']['game_index'],split=game['spec']['split'],start=offset,stop=offset+len(rows),target=game['artifact'],source_game=game['source_game']),'Independent exact role game ranges')
        for example,row in zip(examples,rows,strict=True):
            c.require(cache['index'][offset]==row,'Independent exact actor/terminal/Q/index')
            expected={'tokens':np.asarray(example['tokens'],dtype=np.float32).reshape(31,48),'policy':np.asarray(example['policy'],dtype=np.float32),'mask':np.asarray(example['mask'],dtype=np.bool_)}
            for key,value in expected.items():c.require(value.tobytes()==cache['arrays'][key][offset].tobytes(),'Independent feature bytes '+key)
            offset+=1
        base=c.TRAIN_CACHE if purpose=='training' else c.ASSESS_CACHE
        c.status(base/'progress.json',dict(role=purpose+'_cache_audit',completed_games=game_number,completed_roots=offset,total_games=len(games),total_roots=cache['manifest']['rows'],time=c.now()))
    c.require(offset==cache['manifest']['rows'],'Every fresh selected root independently checked')
    prior=_old_cache(boundary) if purpose=='training' else None
    proof=dict(status='independently_verified',role=purpose,manifest=cache['descriptor'],rows=offset,all_source_rows_checked=True,
        all_feature_bytes_checked=True,all_q_actor_and_terminal_targets_checked=True,prior_cache=prior['acceptance'] if prior else None,
        prior_training_rows_reused=57022 if prior else 0,models=0,optimizer_calls=0,**c.IDENTITY)
    base=c.TRAIN_CACHE if purpose=='training' else c.ASSESS_CACHE
    return dict(acceptance=c.save(base/'cache_acceptance.json',proof),**proof)


def audit_training(folder,boundary=lambda:None):return _audit('training',folder,boundary)
def audit_assessment(folder,boundary=lambda:None):return _audit('assessment',folder,boundary)


def training_dataset(boundary=lambda:None):
    key=('training',str(c.TRAIN_CACHE))
    if key not in _CACHES:
        old=_old_cache(boundary);new=_cache('training',boundary);index=old['index']+new['index']
        selection.validate_namespace(index);c.require(_inventory(index)==selection.load(boundary)['training_inventory'],'Exact combined canonical training inventory')
        _CACHES[key]=Dataset(index,[(0,57022,old['arrays'],old['offsets']),(57022,227203,new['arrays'],None)],new['acceptance'])
    return _CACHES[key]


def assessment_dataset(boundary=lambda:None):
    from . import seal
    release=seal.acquire_release(boundary);key=('assessment',str(c.ASSESS_CACHE))
    if key not in _CACHES:
        cache=_cache('assessment',boundary,release=release)
        _CACHES[key]=Dataset(cache['index'],[(0,len(cache['index']),cache['arrays'],None)],cache['acceptance'],release=release)
    return _CACHES[key]


def load_training(boundary=lambda:None):
    dataset=training_dataset(boundary)
    return dict(train_examples=dataset,train_weights=[1.]*len(dataset),corpus_cache=dataset.cache_acceptance,value_target_sources_sha256=dataset.value_target_sources_sha256)


def fixture_cache_roundtrip(examples,folder):
    import numpy as np
    folder=Path(folder);c.require(folder.resolve().is_relative_to(c.OUT),'New fixture cache boundary');folder.mkdir(parents=True,exist_ok=False)
    arrays=feature_arrays(examples);loaded={};proof=[]
    for key,array in arrays.items():
        path=folder/(key+'.npy')
        with path.open('xb') as stream:np.save(stream,array,allow_pickle=False)
        proof.append(c.artifact(path));loaded[key]=np.load(path,allow_pickle=False,mmap_mode='r')
        c.require(loaded[key].dtype==array.dtype and loaded[key].shape==array.shape and loaded[key].tobytes()==array.tobytes(),'Production fixture mmap bits')
    return loaded,proof


def consumed_inputs():
    """Engineering/old cache descriptors only; never recursive fresh descriptors."""
    gate=cc.read(cc.OUT/'qualification_gate.json');proof=cc.read(c.resolve(gate['audit']));manifest=c.read(c.CACHE_MANIFEST)
    paths=[cc.OUT/'qualification_gate.json',cc.PLAN,c.CACHE_ACCEPTANCE,c.CACHE_MANIFEST]
    for role in ('corpus_build','corpus_audit'):paths.extend([c.ORIGINAL_OUT/role/'execution.json',c.ORIGINAL_OUT/role/'worker/result.json'])
    result=[c.artifact(path) for path in paths]+[gate['audit'],manifest['index'],*manifest['arrays'].values()]
    result.extend(proof['engineering_targets']);result.extend(x['document'] for x in proof['engineering_game_proofs'])
    fixture=cc.read(cc.PLAN)['qualification']['loader'];cases=fixture['cases']+[fixture['private_variant']]
    result.extend(c.artifact(cc.OUT/'qualification/worker/cpu/loader'/f'{i:02d}/moves/000000.json') for i in range(len(cases)))
    # Engineering moves are safe prelaunch and are consumed by production target/Q qualification.
    for game in proof['engineering_game_proofs']:result.extend(data.read_artifact(game['document'])['rows'])
    return result


def qualify_training_inputs(folder,*,boundary=lambda:None):
    from tools.splendorust_independent_data_20261107.collection import qualification
    folder=Path(folder);gate=cc.read(cc.OUT/'qualification_gate.json');c.verify([gate['audit']],boundary);proof=cc.read(c.resolve(gate['audit']))
    c.require(gate['passed'] is True and gate['collection_plan']==c.artifact(cc.PLAN) and proof['status']=='passed' and proof['complete_decisions']==116 and proof['engineering_games']==2,'Actual accepted corrected collector fixture evidence')
    fixture_plan=cc.read(cc.PLAN)['qualification']['loader'];cases=fixture_plan['cases']+[fixture_plan['private_variant']]
    fixture_rows=[];eligible=[];source_rows=[]
    for i,case in enumerate(cases):
        boundary();item=c.artifact(cc.OUT/'qualification/worker/cpu/loader'/f'{i:02d}/moves/000000.json');raw=data.read_artifact(item)
        decoded=data.validate_row(raw,qualification.fixture_spec(case,i),case['snapshot'],0);fixture_rows.append(raw);source_rows.append(item)
        if decoded['eligible']:eligible.append(decoded)
    structural=data.qualify_captured_rows(fixture_plan,fixture_rows,boundary=boundary)
    c.require(structural==proof['loader_row_checks'],'Production decoder exercises all corrected counterexamples')
    cached,cache_items=fixture_cache_roundtrip(eligible,folder/'structural_cache');expected=feature_arrays(eligible)
    c.require(all(cached[k].tobytes()==expected[k].tobytes() for k in cached),'Structural fixtures preserve actual writer/mmap/collator bits')
    selected=[];raw_targets=[]
    for spec,item,game in zip(old_schedule.qualification_schedule(),proof['engineering_targets'],proof['engineering_game_proofs'],strict=True):
        parsed=data.read_artifact(item);raw_targets.append((parsed,spec));examples=source_corpus._parse_target(parsed,spec)['examples'][:16]
        document=data.read_artifact(game['document']);winner=rules.require_official_terminal_target(document['final'],'completed')
        for x in examples:
            boundary();z=0. if winner is None else (1. if winner==x['actor'] else -1.);c.require(x['value']==[z,-z],'Genuine engineering official outcome')
            descriptor=document['rows'][x['ply']];raw=data.read_artifact(descriptor);label=targets.search_label(x,raw,descriptor,spec)
            row=index_row(x,label,'qualification_excluded');row.update(tokens=x['tokens'],policy=x['policy'],mask=x['mask']);selected.append(targets.for_arm(row,'mixed'))
    c.require(len(selected)==32 and {x['terminal_value'][0] for x in selected}=={-1.,0.,1.},'Genuine excluded engineering32 win/loss/draw')
    cached,items=fixture_cache_roundtrip(selected,folder/'genuine_cache');cache_items.extend(items)
    rows=[{k:v for k,v in x.items() if k not in ('tokens','policy','mask','value','value_target_arm','value_target_contract')} for x in selected]
    dataset=Dataset(rows,[(0,32,cached,None)],None);actual=dataset.collate(list(range(32)));expected=sampling.collate(selected,[1.]*32)
    c.require(all(actual[k].tolist()==expected[k].tolist() for k in actual),'Actual production mmap Dataset and mixed target collator exactly match')
    negative=[]
    def reject(name,fn):
        try:fn()
        except (ValueError,KeyError,TypeError,AssertionError):negative.append(dict(case=name,rejected=True))
        else:raise ValueError('Production loader accepted '+name)
    raw,spec=raw_targets[0]
    for key in c.IDENTITY:
        for invalid in ('missing','wrong'):
            bad=deepcopy(raw)
            if invalid=='missing':bad.pop(key)
            else:bad[key]='historical'
            reject(invalid+'_'+key,lambda bad=bad:source_corpus._parse_target(bad,spec))
    bad=deepcopy(raw);bad['spec']['seed_decimal']='0';reject('substituted_seed',lambda:source_corpus._parse_target(bad,spec))
    bad=deepcopy(raw);bad['examples'][0]['mask'][0]=1;reject('nonbool_mask',lambda:source_corpus._parse_target(bad,spec))
    bad=deepcopy(raw);bad['examples'][0]['value']=[1,1];reject('invalid_official_value',lambda:source_corpus._parse_target(bad,spec))
    for key in ('value_target_contract','terminal_value','root_q_win_credit','root_q_signed'):
        bad=deepcopy(selected[0]);bad.pop(key);reject('mixed_missing_'+key,lambda bad=bad:sampling.collate([bad],[1.]))
    bad=deepcopy(selected[0]);bad['value']=list(reversed(bad['value']));reject('reversed_mixed_actor',lambda:sampling.collate([bad],[1.]))
    result=dict(status='actual_production_loader_qualified',collector_audit=gate['audit'],targets=proof['engineering_targets'],structural_loader_checks=structural,
        structural_source_rows=source_rows,actual_selected_loader_and_collator_exercised=True,actual_mmap_training_path_qualified=True,
        cache_feature_artifacts=cache_items,actual_target_parser_rejections=negative,qualification_examples=32,
        fresh_assessment_records_read=0,scientific_source_games_read=0,models=0,optimizer_calls=0,**c.IDENTITY)
    c.save(folder/'loader_qualification.json',result)
    return dict(proof=result,qualification_examples=selected,qualification_weights=[1.]*32,qualification_structural_examples=eligible)

qualify_inputs=qualify_training_inputs


def _verify_game_sources(game,purpose,release,boundary):
    selection.allowed_game(game,purpose,boundary,release=release)
    document=_read_source(game['source_game'],purpose,release)
    data._legacy.accepted.verify_identity(document)
    c.require(document['spec']==game['spec'] and document['reason']=='completed','Preservation binds same accepted game')
    items=[game['artifact'],game['source_game']]
    items.extend(document['rows'][root['ply']] for root in game['roots'])
    c.verify(items,boundary)
    return items


def verify_consumed_sources(folder,boundary=lambda:None):
    """Rehash only actually consumed fresh sources, after the assessment gate."""
    from . import seal
    release=seal.acquire_release(boundary)  # First: do not even inspect fresh paths if still sealed.
    selected=selection.load(boundary)
    training=[g for g in selected['training_games'] if g['source']==selection.SOURCES[1]]
    assessment=selected['assessment_games'];c.require(len(training)==3072 and len(assessment)==256,'Complete fresh consumed sources')
    digest=hashlib.sha256();files=bytecount=roots=0
    for number,(purpose,game) in enumerate([('training',g) for g in training]+[('assessment',g) for g in assessment],1):
        boundary();items=_verify_game_sources(game,purpose,release,boundary)
        for item in items:
            digest.update((c.canonical(item)+'\n').encode());files+=1;bytecount+=item['bytes']
        roots+=len(game['roots'])
        if number%16==0 or number==3328:
            c.status(Path(folder)/'source_preservation/progress.json',dict(completed_games=number,total_games=3328,completed_roots=roots,files=files,time=c.now()))
    c.require(roots==170181+selected['assessment_binding']['roots'],'Every consumed fresh root preserved')
    value=dict(status='all_consumed_fresh_sources_preserved',commitment=c.artifact(c.COMMITMENT),selection=c.artifact(c.SELECTION),
        training_games=3072,assessment_games=256,roots=roots,files=files,physical_bytes=bytecount,ordered_artifact_inventory_sha256=digest.hexdigest(),
        exact_target_game_and_selected_move_hashes=True,unconsumed_source_rows_read=False,models=0,optimizer_calls=0)
    return dict(preservation=c.save(Path(folder)/'consumed_source_preservation.json',value),**value)
