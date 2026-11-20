"""Preserve exact historical forecasts without reapplying their old stop gate."""
from pathlib import Path
from . import common as c, continuation

RECEIPT = continuation.NEW_OUT/'prelaunch/original_forecasts_review.json'


def evaluate_original_forecasts(*, boundary=lambda: None):
    from tools.splendorust_selfplay_pilot_budget_20261102 import resources as original
    freeze=continuation.old_freeze();root=continuation.OLD_OUT;proofs=[];inputs=[]
    path=root/'collection_forecast.json';saved=c.read(path)
    result=original.source_forecast(freeze,saved['snapshot'],source_costs=[])
    result.update(passed=result['affordable'],status='passed' if result['affordable'] else 'failed')
    c.require(result==saved and result['affordable'],'Original precollection arithmetic unchanged')
    inputs.append(c.artifact(path));proofs.append(dict(path=inputs[-1],checks=result['checks'],affordable=True))
    for index in range(4):
        boundary();path=root/'block_forecasts'/f'{index:03d}.json';saved=c.read(path)
        c.verify(saved['cost_inputs']);costs=[c.read(c.ROOT/x['path']) for x in saved['cost_inputs']]
        c.require(len(costs)==8*(index+1) and saved['snapshot']['completed_games']==len(costs),
                  'Every historical forecast uses its exact contiguous completed prefix')
        result=original.source_forecast(freeze,saved['snapshot'],source_costs=costs)
        result.update(block_index=index,cost_inputs=saved['cost_inputs'])
        c.require(result==saved and result['affordable']==(index<3),'Historical forecast recomputation unchanged')
        if index==3:
            c.require(result['checks']['collection'] is False and
                      all(v for k,v in result['checks'].items() if k!='collection'),
                      'Original final forecast remains failed; never launder original acceptance')
        inputs.extend([c.artifact(path),*saved['cost_inputs']])
        proofs.append(dict(path=c.artifact(path),checks=result['checks'],affordable=result['affordable']))
    inputs.extend([c.artifact(root/'freeze.json'),freeze['resource_plan'],c.artifact(Path(original.__file__)),
                   c.artifact(Path(__file__))])
    unique={x['path']:x for x in inputs}
    return dict(schema='selfplay_original_forecast_preservation_v1',approved=True,
                original_attempt_remains_incomplete=True,original_final_forecast_remains_unaffordable=True,
                continuation_forecasts_advisory=True,models=0,games=0,
                inputs=sorted(unique.values(),key=lambda x:x['path']),forecasts=proofs)


def prepare_saved_forecasts():
    c.require(c.OUT==continuation.NEW_OUT,'New continuation receipt only')
    value=evaluate_original_forecasts()
    if RECEIPT.exists():
        c.require(c.read(RECEIPT)==value,'Existing forecast preservation proof unchanged')
        return c.artifact(RECEIPT)
    return c.save(RECEIPT,value)


def verify_saved_forecasts(descriptor, *, boundary=lambda: None):
    c.require(descriptor==c.artifact(RECEIPT),'Exact saved forecast preservation receipt')
    saved=c.read(RECEIPT);c.verify(saved['inputs'])
    c.require(saved==evaluate_original_forecasts(boundary=boundary),'Complete original forecast arithmetic preserved')
    return saved
