"""Commit all seven fixed models before issuing a fresh-assessment read capability."""
from pathlib import Path
from dataclasses import dataclass
import os
from . import common as c,selection
_TOKEN=object()
_RELEASE=None


def _fit_records(boundary,*,full=False,folder=None):
    from . import control,fit
    records=[];parent_weights=set()
    for fit_id in c.FIT_IDS:
        boundary();worker=c.OUT/'jobs'/fit_id/'worker';execution,result=control.passed_job(fit_id,boundary)
        final=c.read(worker/'fit_result.json');descriptor=c.artifact(worker/'fit_result.json')
        c.require(final['fit_id']==fit_id and final['status']=='fit_complete' and final['optimizer_calls']==7101 and
            final['replay_optimizer_calls']==1 and final['record_presentations']==227203 and final['endpoint']==final['checkpoints']['7101'] and
            set(final['checkpoints'])=={'0','7100','7101'},'All four exact completed prospective endpoints')
        # The actual worker result must contain its immutable final-fit descriptor.
        c.require(descriptor in list(c.nested_artifacts(result['result'])),'Actual clean fit result binds saved fit')
        c.verify([*final['checkpoints'].values(),final['replay']],boundary)
        manifests={step:c.read(c.resolve(item)) for step,item in final['checkpoints'].items()}
        for step,manifest in manifests.items():
            metadata=manifest['metadata'];c.require(manifest['schema']=='training4096_checkpoint_manifest_v1' and metadata['checkpoint_step']==int(step) and
                metadata['fit_id']==fit_id and metadata['qualification_only'] is False and all(metadata[k]==v for k,v in c.IDENTITY.items()),'Exact corrected final checkpoint manifest')
            c.verify([manifest['checkpoint'],manifest['tensor_manifest']],boundary)
        replay=c.read(c.resolve(final['replay']))
        c.require(replay['status']=='exact_final_update_reproduced' and replay['optimizer_calls']==1 and replay['step']==7101 and
            replay['from_checkpoint']==final['checkpoints']['7100'] and replay['expected_checkpoint']==final['endpoint'] and
            replay['tree_sha256']==manifests['7101']['tree_sha256'],'Exact completed final-three replay before release')
        c.verify([replay['actual_checkpoint'],replay['expected_update']],boundary)
        actual=c.read(c.resolve(replay['actual_checkpoint']));c.verify([actual['checkpoint'],actual['tensor_manifest']],boundary)
        c.require(actual['tree_sha256']==manifests['7101']['tree_sha256'],'Replay full model/optimizer/RNG equality')
        proofpath=c.OUT/'commit_verifications'/(fit_id+'.json')
        if full:
            proof=fit.verify_fit(worker,boundary=boundary);c.save(proofpath,proof)
        proof=c.read(proofpath)
        c.require(proof['status']=='training4096_fit_saved_verified' and proof['result']==descriptor and proof['endpoint']==final['endpoint'] and
            proof['checkpoints']==final['checkpoints'] and proof['optimizer_calls']==7101 and proof['replay_optimizer_calls']==1 and proof['record_presentations']==227203,'Independently checked complete fit and exact replay')
        parent_weights.add(manifests['0']['metadata']['weights_sha256'])
        records.append(dict(fit_id=fit_id,actual_exit=execution,fit_result=descriptor,verification=c.artifact(proofpath),
            checkpoints=final['checkpoints'],endpoint=final['endpoint'],replay=final['replay'],weights_sha256=manifests['7101']['metadata']['weights_sha256']))
    c.require(len(parent_weights)==1,'All four fits start from identical EF8A weights')
    return records,parent_weights.pop()


def _historical(boundary):
    bindings=c.read(c.BINDINGS);c.verify([bindings['old_fit_commitment'],bindings['original_ef8a']],boundary)
    old=c.read(c.resolve(bindings['old_fit_commitment']));c.require(old['status']=='all_four_full_corpus_fits_committed','Accepted fixed historical references')
    models=[];receipts=[]
    for rep,bound in enumerate(bindings['historical_practical_baselines']):
        c.verify([bound['manifest'],bound['checkpoint']],boundary);manifest=c.read(c.resolve(bound['manifest']))
        reference=next(x for x in old['fits'] if x['spec']['fit_id']==bound['fit_id'])
        c.require(reference['final_checkpoint']==bound['manifest'] and manifest['checkpoint']==bound['checkpoint'] and
            manifest['metadata']['checkpoint_step']==1782 and manifest['metadata']['seed']==bound['seed'] and
            manifest['metadata']['qualification_only'] is False and all(manifest['metadata'][k]==v for k,v in c.IDENTITY.items()),'Exact fixed prior one-pass corrected warm-start reference')
        c.verify([reference['actual_exit'],reference['saved_verification'],reference['fit_result'],manifest['tensor_manifest']],boundary)
        actual=c.read(c.resolve(reference['actual_exit']));c.require(actual['status']=='passed' and actual['worker_exit_code']==0 and actual['cleanup_errors']==[] and actual['error'] is None,'Historical actual clean fit exit')
        models.append(dict(id=f'historical_r{rep}',kind='historical',checkpoint=bound['manifest'],weights_sha256=manifest['metadata']['weights_sha256']))
        receipts.append(dict(binding=bound,actual_exit=reference['actual_exit'],verification=reference['saved_verification']))
    c.require(len(models)==2,'Exactly two historical references')
    return models,receipts


def _expected(boundary,*,full=False,folder=None):
    selected=selection.load(boundary);bindings=c.read(c.BINDINGS);proposal=c.read(c.PROPOSAL)
    records,parent_weights=_fit_records(boundary,full=full,folder=folder);historical,historical_receipts=_historical(boundary)
    models=[dict(id='parent',kind='parent',checkpoint=bindings['original_ef8a'],weights_sha256=parent_weights),*historical]
    models.extend(dict(id=x['fit_id'],kind='new',checkpoint=x['endpoint'],weights_sha256=x['weights_sha256']) for x in records)
    return dict(schema='training4096_all_checkpoints_committed_v1',status='all_seven_models_committed',study_id=c.STUDY,**c.IDENTITY,
        freeze=c.artifact(c.PLAN),selection=c.artifact(c.SELECTION),proposal=c.artifact(c.PROPOSAL),source_bindings=c.artifact(c.BINDINGS),
        training_cache_acceptance=c.artifact(c.TRAIN_CACHE/'cache_acceptance.json'),models=models,fits=records,historical=historical_receipts,
        evaluation=proposal['evaluation'],training_panel=proposal['training_panel'],assessment_binding=selected['assessment_binding'],
        scientific_optimizer_calls=28404,replay_optimizer_calls=4,all_fit_exits_verified=True,all_exact_replays_verified=True,
        assessment_records_read_before_commitment=0,checkpoint_selection=False,refit_after_assessment=False)


def commit(folder,boundary=lambda:None):
    c.require(not c.COMMITMENT.exists() and not (c.OUT/'assessment_opened.json').exists() and not c.ASSESS_CACHE.exists(),'One prospective commitment before any fresh access')
    value=_expected(boundary,full=True,folder=folder);item=c.atomic_save(c.COMMITMENT,value)
    # Builder-internal verification cannot require its own future actual exit.
    validate_commitment(boundary=boundary,require_commit_exit=False)
    return dict(status='all_seven_models_committed',commitment=item)


def validate_commitment(commitment=None,*,boundary=lambda:None,require_commit_exit=True):
    from . import control
    c.require(c.COMMITMENT.exists(),'Fresh assessment sealed until commitment exists')
    item=c.artifact(c.COMMITMENT)
    if commitment is not None:c.require(commitment==item or commitment==c.read(c.COMMITMENT),'Exact fixed commitment argument')
    saved=c.read(c.COMMITMENT);expected=_expected(boundary)
    c.require(saved==expected,'All fixed endpoints, exact replays, references, splits and evaluation recipe bind commitment')
    if require_commit_exit:
        _,result=control.passed_job('commit',boundary)
        c.require(result['result']['commitment']==item,'Commit actual clean exit independently observed before fresh access')
    return saved


@dataclass(frozen=True)
class Release:
    token:object
    commitment:dict
    stamps:tuple


def _stamp(path):
    stat=Path(path).stat();return (str(Path(path).resolve()),stat.st_size,stat.st_mtime_ns,stat.st_ino)


def acquire_release(boundary=lambda:None):
    global _RELEASE
    if _RELEASE is None:
        value=validate_commitment(boundary=boundary);items=list(c.nested_artifacts(value))
        # All these are metadata or fixed models. Assessment targets are never nested in this commitment.
        paths={str(c.resolve(x)) for x in items};paths.add(str(c.COMMITMENT));paths.add(str(c.OUT/'jobs/commit/actual_exit.json'));paths.add(str(c.OUT/'jobs/commit/worker/result.json'))
        for model in value['models']:
            if model['kind']!='parent':
                manifest=c.read(c.resolve(model['checkpoint']));paths.update(str(c.resolve(x)) for x in (manifest['checkpoint'],manifest['tensor_manifest']))
        _RELEASE=Release(_TOKEN,c.artifact(c.COMMITMENT),tuple(_stamp(path) for path in sorted(paths)))
    require_release(_RELEASE);return _RELEASE


def require_release(release):
    c.require(type(release) is Release and release.token is _TOKEN and release is _RELEASE,'Fresh assessment requires validated private release capability')
    c.require(all(_stamp(stamp[0])==stamp for stamp in release.stamps),'Every fixed commitment/model/evidence remains unchanged during assessment')
    return release


def mark_opened(release):
    require_release(release);path=c.OUT/'assessment_opened.json'
    if path.exists():c.require(c.read(path)['commitment']==release.commitment,'One fixed assessment opening')
    else:c.save(path,dict(schema='training4096_assessment_opened_v1',commitment=release.commitment,time=c.now(),all_four_fits_fixed=True))

build_commitment=commit
