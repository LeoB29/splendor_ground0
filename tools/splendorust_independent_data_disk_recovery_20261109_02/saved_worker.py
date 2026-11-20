"""Saved-only continuation readers; no collection, fitting or model prediction route."""
import argparse
import os
from pathlib import Path
import sys
import traceback
import psutil
from . import common as c


def training_audit(boundary,sources,folder):
    from tools.splendorust_independent_data_20261107 import fit,sampling
    fit.startup(sources)
    selection=c.read(c.SELECTION);sampling.validate_selection(selection,boundary=boundary)
    q=fit.verify_qualification(c.SOURCE_OUT/'training_qualification/worker',boundary=boundary)
    fits=[fit.verify_fit(selection,x['repetition'],x['breadth'],'joint',
          c.SOURCE_OUT/'fits'/x['fit_id']/'worker',boundary=boundary) for x in c.fits()]
    fit.validate_commitment(selection,c.artifact(c.SOURCE_OUT/'all_fits_committed.json'),boundary=boundary)
    result=dict(status='all_training_saved_verified',qualification=q,fits=fits,optimizer_calls=390,
                scientific_optimizer_calls=384,qualification_optimizer_calls=2,replay_optimizer_calls=4,
                new_optimizer_calls=0,recovery_model_calls=0,source_freeze=c.artifact(c.SOURCE_PLAN))
    c.save(folder/'training_audit.json',result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('audit','analysis','final'),required=True)
    parser.add_argument('--plan-sha256',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();folder=Path(args.output).resolve()
    expected={'audit':'training_audit','analysis':'offline_analysis','final':'final_audit'}
    c.require(folder==c.OUT/expected[args.mode]/'worker','Exact saved recovery worker')
    from .runtime import bootstrap
    plan,boundary,sources=bootstrap(folder,args.plan_sha256,args.mode,model=False)
    status,error,result='failed',None,None
    try:
        c.require('torch' not in sys.modules,'Saved worker has no tensor runtime')
        if args.mode=='audit':result=training_audit(boundary,sources,folder)
        elif args.mode=='analysis':
            from tools.splendorust_independent_data_20261107 import fit
            fit.startup(sources)
            from . import offline
            result=offline.run_analysis(c.read(c.SELECTION),c.artifact(c.SOURCE_OUT/'all_fits_committed.json'),
                c.artifact(c.SOURCE_OUT/'offline/worker/evaluation_result.json'),folder,boundary=boundary)
        else:
            from .audit import final
            result=final(plan,folder,boundary)
        # The unchanged saved-fit verifier decodes checkpoint tensors with torch.load;
        # it constructs no player and executes no forward/backward/optimizer step.
        boundary()
        if args.mode!='audit':c.require('torch' not in sys.modules,'Analysis/final reader remained tensor-runtime free')
        status='passed'
    except BaseException:error=traceback.format_exc()
    receipt=dict(status=status,mode=args.mode,fit_id=None,identity=dict(pid=os.getpid(),created=psutil.Process().create_time()),
        result=result,error=error,plan_sha256=args.plan_sha256,new_models=0,new_games=0,new_optimizer_calls=0,
        saved_tensor_deserialization_allowed=args.mode=='audit')
    c.save(folder/'result.json',receipt)
    if error:c.save(folder/'failure.json',receipt);print(error,flush=True)
    return 0 if status=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
