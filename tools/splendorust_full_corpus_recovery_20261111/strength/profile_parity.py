"""Exact science across authenticated relocation of identical checkpoint files.

Raw receipts are never modified. Only the explicit profile-file digest metadata
may differ after identical model bytes and all profile settings are proven.
"""
from copy import deepcopy
from . import common as c,adapters,checkpoint,scientific,lineage,analysis

_CACHE={}


def _same(a,b):return c.canonical(a)==c.canonical(b)


def relocated_candidate(source_manifest,target_manifest,source_tensors,target_tensors):
    """Pure complete descriptor comparison after separately verifying every byte."""
    left,right=source_manifest,target_manifest
    c.require(left['checkpoint']['sha256']==right['checkpoint']['sha256'] and
              left['checkpoint']['bytes']==right['checkpoint']['bytes'], 'Exactly identical relocated model bytes')
    normalized=deepcopy(left)
    normalized['checkpoint']=deepcopy(right['checkpoint']);normalized['tensor_manifest']=deepcopy(right['tensor_manifest'])
    c.require(_same(normalized,right),'Only live checkpoint/tensor descriptor locations may differ')
    tensors=deepcopy(source_tensors);tensors['checkpoint']=deepcopy(right['checkpoint'])
    c.require(_same(tensors,target_tensors),'Every tensor shape,order,offset,float32 hash and weight identity is exact')
    return True


def relocate_profile(source,target,old_manifest_item,new_manifest_item,old_manifest,new_manifest):
    """Replace three proven artifact descriptors and their explicit profile references."""
    pairs=[(old_manifest_item,new_manifest_item),(old_manifest['checkpoint'],new_manifest['checkpoint']),
           (old_manifest['tensor_manifest'],new_manifest['tensor_manifest'])]
    replacements={c.canonical(old):new for old,new in pairs}
    c.require(len(replacements)==3,'Three distinct checkpoint provenance artifacts')
    value=deepcopy(source)
    c.require(value['checkpoint']==old_manifest['checkpoint']['path'] and
              value['learned_checkpoint_manifest']==old_manifest_item and
              value['native_reference_model']==old_manifest['checkpoint'], 'Original candidate profile provenance')
    occurrences={key:0 for key in replacements};mapped=[]
    for item in value['artifacts']:
        key=c.canonical(item)
        if key in replacements:occurrences[key]+=1;mapped.append(deepcopy(replacements[key]))
        else:mapped.append(item)
    c.require(all(count==1 for count in occurrences.values()),'Exact unique provenance artifacts in profile')
    value['artifacts']=mapped;value['checkpoint']=new_manifest['checkpoint']['path']
    value['learned_checkpoint_manifest']=deepcopy(new_manifest_item)
    value['native_reference_model']=deepcopy(new_manifest['checkpoint'])
    c.require(_same(value,target),'No search,model,rule,precision,native or other profile setting may change')
    return True


def profile_map(source,target,route,*,boundary=lambda:None):
    c.require(route in c.VARIANTS,'One exact scientific route')
    binding=dict(source_profiles=source['profiles'][route],target_profiles=target['profiles'][route],
        source_binding=source['checkpoint_bindings'][route],target_binding=target['checkpoint_bindings'][route],
        source_manifest=source.get('trained_manifests',{}).get(route) if not c.is_parent_route(route) else None,
        target_manifest=target.get('trained_manifests',{}).get(route) if not c.is_parent_route(route) else None,route=route)
    key=c.canonical(binding)
    if key in _CACHE:return deepcopy(_CACHE[key])
    items=[source['profiles'][route][str(q)] for q in (16,128)]+[target['profiles'][route][str(q)] for q in (16,128)]
    c.verify(items,boundary=boundary)
    manifests=None
    if not c.is_parent_route(route):
        c.verify([source['trained_manifests'][route],target['trained_manifests'][route]],boundary=boundary)
        manifests=[c.read(c.ROOT/x['trained_manifests'][route]['path']) for x in (source,target)]
        c.verify([m[k] for m in manifests for k in ('checkpoint','tensor_manifest','model_definition')],boundary=boundary)
    c.require(source['checkpoint_bindings'][route]['weights_sha256']==target['checkpoint_bindings'][route]['weights_sha256'],
              'Same complete model weights across profile provenance')
    if not c.is_parent_route(route):
        old,old_tensors=checkpoint.candidate_sources(source['trained_manifests'][route],boundary=boundary)
        new,new_tensors=checkpoint.candidate_sources(target['trained_manifests'][route],boundary=boundary)
        c.require(_same(old,manifests[0]) and _same(new,manifests[1]),'Verified manifests bind exactly')
        for freeze,manifest in ((source,old),(target,new)):
            c.require(freeze['checkpoint_bindings'][route]['checkpoint']==manifest['checkpoint'] and
                      freeze['checkpoint_bindings'][route]['native_reference']==manifest['checkpoint'] and
                      freeze['checkpoint_bindings'][route]['weights_sha256']==manifest['metadata']['weights_sha256'],
                      'Each local freeze binds its exact imported model')
        relocated_candidate(old,new,old_tensors,new_tensors)
    else:
        c.require(_same(source['checkpoint_bindings']['control'],target['checkpoint_bindings']['control']),
                  'Original reference checkpoint and native reference are identical')
    rows={}
    for quota in (16,128):
        boundary()
        left=adapters.profile_identity(source,route,quota,boundary=boundary)
        right=adapters.profile_identity(target,route,quota,boundary=boundary)
        a,b=(c.read(c.ROOT/f['profiles'][route][str(quota)]['path']) for f in (source,target))
        if not c.is_parent_route(route):relocate_profile(a,b,source['trained_manifests'][route],target['trained_manifests'][route],old,new)
        else:c.require(_same(a,b) and left['profile_sha256']==right['profile_sha256'],'Exact unchanged reference profile')
        expected=deepcopy(left);expected['profile_sha256']=right['profile_sha256']
        c.require(_same(expected,right),'Profile execution identity differs only by proven document digest')
        rows[str(quota)]=dict(source=left['profile_sha256'],target=right['profile_sha256'],
            profile_id=left['profile_id'],wrapper_id=left['wrapper_id'])
    proof=dict(schema='authenticated_profile_relocation_v1',route=route,mapping=rows,bindings=binding,
        exact_model_bytes=True,exact_tensor_inventory=True,all_scientific_profile_settings_exact=True,
        only_normalized_field='result.profile_sha256',raw_receipts_unchanged=True)
    _CACHE[key]=deepcopy(proof)
    return proof


def result_view(result,proof):
    c.require(type(result) is dict and set(result)==scientific.RESULT_FIELDS,'Full actual scientific result envelope')
    quota=result['simulations_quota'];c.require(type(quota) is int and quota in (16,128),'Exact actual quota')
    mapped=proof['mapping'][str(quota)]
    c.require(result['profile_sha256']==mapped['source'] and result['profile_id']==mapped['profile_id'] and
              result['wrapper_id']==mapped['wrapper_id'],'Authenticated source profile digest and full identity')
    value=deepcopy(result);value['profile_sha256']=mapped['target'];return value


def compare_suite(source_result,target_result,source_freeze,target_freeze,route,*,boundary=lambda:None):
    proof=profile_map(source_freeze,target_freeze,route,boundary=boundary);changed=[];count=0
    def compare(left,right,path):
        nonlocal count
        c.require(type(left) is type(right),'Exact qualification value type: '+path)
        if type(left) is dict:
            c.require(set(left)==set(right),'Exact qualification keys: '+path)
            if 'profile_sha256' in left:
                mapped=result_view(left,proof);count+=1
                if left['profile_sha256']!=mapped['profile_sha256']:changed.append(path+'/profile_sha256')
                c.require(_same(mapped,right),'Every scientific result field exact after authenticated profile relocation: '+path)
            else:
                for key in left:compare(left[key],right[key],path+'/'+key)
        elif type(left) is list:
            c.require(len(left)==len(right),'Exact qualification list length: '+path)
            for index,(a,b) in enumerate(zip(left,right,strict=True)):compare(a,b,path+'/'+str(index))
        else:c.require(_same(left,right),'Exact qualification leaf: '+path)
    compare(source_result,target_result,'')
    c.require(count>0,'Full scientific results compared')
    return dict(status='passed',profile_binding=proof,compared_result_envelopes=count,
        normalized_profile_fields=changed,scientific_fields_exact=True,source_suite_sha256=c.digest(source_result),
        target_suite_sha256=c.digest(target_result),raw_suites_unchanged=True)


def audit_view(row,freeze,*,boundary=lambda:None):
    """A transient view for unchanged current-profile science; stored data stays original."""
    if lineage.source_of(row)=='new':return row
    boundary()
    old=lineage.old_freeze()
    from tools.splendorust_full_corpus_20261110.strength import transport as original_transport,analysis as original_analysis
    original_transport.verify_saved_record(row,old)
    if row['completed']:original_analysis.validate_completed(row)
    else:return row
    proof=profile_map(old,freeze,row['variant'],boundary=boundary)
    value=deepcopy(row);value['clock']['result']=result_view(row['clock']['result'],proof)
    return value


def audit_game(initial,rows,ending,spec,*,freeze,boundary=lambda:None):
    views=[audit_view(row,freeze,boundary=boundary) for row in rows]
    return analysis.audit_game(initial,views,ending,spec,freeze=freeze,boundary=boundary)


def operational_summary(games,rows,*,freeze,schedule=None,boundary=lambda:None):
    views=(audit_view(row,freeze,boundary=boundary) for row in rows)
    return analysis.summarize_operational(games,views,schedule=schedule,boundary=boundary)


def verify_replay(original,replayed,selection,*,freeze,boundary=lambda:None):
    left=audit_view(original,freeze,boundary=boundary)
    proof=analysis.verify_replay(left,replayed,selection)
    if lineage.source_of(original)=='old':
        proof.update(authenticated_profile_relocation=profile_map(lineage.old_freeze(),freeze,original['variant'],boundary=boundary),
            original_raw_result_sha256=c.digest(original['clock']['result']),replayed_raw_result_sha256=c.digest(replayed['clock']['result']),
            result_sha256=c.digest(original['clock']['result']),raw_receipts_unchanged=True)
    return proof
