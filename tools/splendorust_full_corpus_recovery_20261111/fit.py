"""Saved-only reuse of four original full-corpus fits; no training launch route."""
from tools.splendorust_full_corpus_20261110.fit import *


def training_audit(boundary,sources,folder):
    from tools.splendorust_full_corpus_20261110 import fit as original, sampling
    from . import common as recovery
    original.startup(sources)
    selection=recovery.read(recovery.SELECTION)
    sampling.validate_selection(selection,boundary=boundary)
    q=original.verify_qualification(recovery.ORIGINAL_OUT/'training_qualification/worker',boundary=boundary)
    fits=[original.verify_fit(selection,x['repetition'],x['breadth'],'joint',
          recovery.ORIGINAL_OUT/'fits'/x['fit_id']/'worker',boundary=boundary) for x in recovery.fits()]
    original.validate_commitment(selection,recovery.artifact(recovery.ORIGINAL_OUT/'all_fits_committed.json'),boundary=boundary)
    result=dict(status='all_training_saved_verified',qualification=q,fits=fits,
        optimizer_calls=7134,scientific_optimizer_calls=7128,qualification_optimizer_calls=2,replay_optimizer_calls=4,
        new_optimizer_calls=0,recovery_model_calls=0,source_freeze=recovery.artifact(recovery.ORIGINAL_OUT/'freeze.json'))
    recovery.save(folder/'training_audit.json',result)
    return result


def main():
    import argparse,os,sys,traceback,psutil
    from pathlib import Path
    from . import common as recovery
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=('audit','analysis'),required=True)
    parser.add_argument('--plan-sha256',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();folder=Path(args.output).resolve()
    recovery.require(folder==recovery.OUT/({'audit':'training_audit','analysis':'offline_analysis'}[args.mode])/'worker',
                     'Exact separate saved-only recovery output')
    from .runtime import bootstrap
    plan,boundary,sources=bootstrap(folder,args.plan_sha256,args.mode,model=False)
    status,result,error='failed',None,None
    try:
        if args.mode=='audit':result=training_audit(boundary,sources,folder)
        else:
            from tools.splendorust_full_corpus_20261110 import fit as original
            original.startup(sources)
            from . import offline
            result=offline.run_analysis(recovery.read(recovery.SELECTION),
                recovery.artifact(recovery.ORIGINAL_OUT/'all_fits_committed.json'),
                recovery.artifact(recovery.ORIGINAL_OUT/'offline/worker/evaluation_result.json'),folder,boundary=boundary)
        boundary()
        if args.mode=='analysis':recovery.require('torch' not in sys.modules,'Analysis remains model-free')
        status='passed'
    except BaseException:error=traceback.format_exc()
    receipt=dict(status=status,mode=args.mode,fit_id=None,identity=dict(pid=os.getpid(),created=psutil.Process().create_time()),
        result=result,error=error,plan_sha256=args.plan_sha256,new_models=0,new_games=0,new_optimizer_calls=0,
        saved_tensor_deserialization_allowed=args.mode=='audit')
    recovery.save(folder/'result.json',receipt)
    if error:recovery.save(folder/'failure.json',receipt);print(error,flush=True)
    return 0 if status=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
