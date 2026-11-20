"""Recompute both preserved forecast histories without reinstating stop gates."""
from pathlib import Path
from . import common as c, continuation

RECEIPT = c.OUT/'prelaunch/original_forecasts_review.json'


def evaluate_original_forecasts(*, boundary=lambda: None):
    from tools.splendorust_selfplay_pilot_resume_20261103 import forecast_preservation as first
    from tools.splendorust_selfplay_pilot_resume_20261103 import resources as previous_resources
    original=first.evaluate_original_forecasts(boundary=boundary)
    c.require(original['approved'] and original['original_final_forecast_remains_unaffordable'],
              'Original five forecasts retain their failed final gate')
    freeze=continuation.previous_freeze();root=continuation.PREVIOUS_OUT
    proofs=list(original['forecasts']);inputs=list(original['inputs'])
    for index in (4,5):
        boundary();path=root/'block_forecasts'/f'{index:03d}.json';saved=c.read(path)
        expected=[c.artifact(continuation.prefix_game_path(i).parent/'cost.json') for i in range(8*(index+1))]
        c.require(saved['cost_inputs']==expected and saved['snapshot']['completed_games']==len(expected) and
                  saved['snapshot']['qualified_source_workers']==index+2,'Exact preserved mixed-prefix forecast inputs')
        c.verify(expected);costs=[c.read(c.ROOT/x['path']) for x in expected]
        result=previous_resources.source_forecast(freeze,saved['snapshot'],source_costs=costs)
        result.update(block_index=index,cost_inputs=expected)
        c.require(result==saved,'Previous advisory forecast arithmetic must remain exact')
        inputs.extend([c.artifact(path),*expected])
        proofs.append(dict(path=c.artifact(path),checks=result['checks'],affordable=result['affordable'],
                           policy='advisory_only_in_previous_continuation'))
    inputs.extend([c.artifact(root/'freeze.json'),freeze['resource_plan'],c.artifact(Path(previous_resources.__file__)),
                   c.artifact(Path(first.__file__)),c.artifact(Path(__file__))])
    unique={x['path']:x for x in inputs}
    return dict(schema='selfplay_original_forecast_preservation_v2',approved=True,
                original_attempt_remains_incomplete=True,previous_attempt_remains_incomplete=True,
                original_final_forecast_remains_unaffordable=True,continuation_forecasts_advisory=True,
                models=0,games=0,inputs=sorted(unique.values(),key=lambda x:x['path']),forecasts=proofs)


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
