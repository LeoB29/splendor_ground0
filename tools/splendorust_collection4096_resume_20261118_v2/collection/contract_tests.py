"""Model-free counterexamples for the new reservation and collector wrapper."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from .. import schedule, common as root
from . import collector, worker, qualification, storage, backend, data, private


def run():
    from tools.future_training_rules_20261021 import verify_runtime_sources
    sources = verify_runtime_sources()
    checks = []
    def check(condition, label):
        root.require(condition, label); checks.append(label)
    def rejects(operation, label):
        try: operation()
        except (ValueError, KeyError, TypeError): checks.append(label)
        else: raise ValueError('Expected counterexample rejection: '+label)
    rows = schedule.collection_schedule()
    counts = schedule.validate(rows)
    check(counts['collection_games'] == 3328 and counts['train_games'] == 3072 and
          counts['validation_games'] == counts['offline_games'] == 128 and
          counts['strength_games'] == counts['scientific_optimizer_updates'] == 0,
          'Only the precommitted fresh whole-game collection is authorized')
    check(rows[3071]['split'] == 'train' and rows[3072]['split'] == 'validation' and
          rows[3199]['split'] == 'validation' and rows[3200]['split'] == rows[-1]['split'] == 'offline',
          'All split boundary positions are exact')
    engineering = schedule.qualification_schedule()
    check([x['game_index'] for x in engineering] == [3330, 3331] and
          all(x['purpose'] == 'engineering_qualification_only' for x in engineering),
          'Two excluded engineering games use separate identities')
    for bad in (-1, 3328, True, 1.0, '1'):
        rejects(lambda bad=bad: schedule.collection_spec(bad), 'Reject malformed collection index '+repr(bad))
    for key, value in [('split', 'offline'), ('seed_decimal', rows[1]['seed_decimal']),
                       ('quota', 256), ('world_pool', 6), ('cpuct', .5),
                       ('start_player', True), ('game_index', False)]:
        changed = deepcopy(rows); changed[0][key] = value
        rejects(lambda changed=changed: schedule.validate(changed), 'Reject changed reservation '+key)
    rejects(lambda: schedule.validate(rows[:-1]), 'Reject omitted reserved game')
    rejects(lambda: schedule.validate(rows+[rows[0]]), 'Reject extra reserved game')
    cells = collector.cells()
    check(len(cells) == 208 and [x for cell in cells for x in cell['game_indices']] == list(range(3328)),
          '208 cells cover every source game exactly once')
    check(all(cell['specifications'] == rows[16*i:16*(i+1)] and cell['route'] == 'cpu' and
              cell['concurrency'] == 8 and cell['scheduler'] == 'original8' and
              cell['implementation'] == 'eager' for i,cell in enumerate(cells)),
          'Every cell retains the qualified CPU8 execution route')
    prior = root.read(root.ROOT/'outputs/splendorust_independent_data_20261107_01/reserved/collection_schedule.json')
    fresh_seeds = {x['seed_decimal'] for x in rows + engineering}
    check(not fresh_seeds & {x['seed_decimal'] for x in prior}, 'Fresh setups are disjoint from all accepted source games')
    previous = root.read(root.ROOT/'outputs/splendorust_gpu_collector_20261105_01/plan.json')
    plan = dict(schema='independent_corrected_production_collector_20261115_v1',
        games=collector.build_plan(),cells=cells,engineering_games=engineering,
        qualification=qualification.build_plan(),storage=storage.contract(),qualification_routes=['cpu'],
        internal_seconds=60.,outer_seconds=75.,execution_policy_id=worker.c.EXECUTION_POLICY_ID,
        total_seconds=168*3600,model_seconds=144*3600,retries=0,replacements=0,training_updates=0,
        maximum_new_games=3328,excluded_qualification_games=2,maximum_model_replays=9984,
        worker_affinity=list(range(19)),supervisor_affinity=[19],
        worker_limits=dict(qualification_seconds=6*3600,games_seconds=4*3600),
        available_ram_reserve_bytes=0,forecast_admission_gate=False,
        runtime=deepcopy(previous['runtime']),source_guard=previous['source_guard'],
        native_capture=backend.capture_provenance())
    worker.validate_plan(plan); checks.append('Actual collector entrypoint accepts only the new bounded contract')
    for key,value in [('maximum_new_games',1152),('maximum_model_replays',3456),('total_seconds',120*3600),
                      ('model_seconds',96*3600),('internal_seconds',119.),('outer_seconds',120.),
                      ('retries',1),('replacements',1),('training_updates',1),
                      ('available_ram_reserve_bytes',4*1024**3),('forecast_admission_gate',True),
                      ('qualification_routes',['cpu','gpu'])]:
        changed=deepcopy(plan);changed[key]=value
        rejects(lambda changed=changed:worker.validate_plan(changed),'Reject altered bound '+key)
    changed=deepcopy(plan);changed['cells'][-1]['specifications']=changed['cells'][-2]['specifications']
    rejects(lambda:worker.validate_plan(changed),'Reject replaced final cell')
    changed=deepcopy(plan);changed['runtime']['source_guard']={}
    rejects(lambda:worker.validate_plan(changed),'Reject unbound corrected runtime sources')
    changed=deepcopy(plan);changed['runtime']['native_capture_binary']={}
    rejects(lambda:worker.validate_plan(changed),'Reject substituted native target capture')
    check(private.SOURCES == {
        'collector':'5a0700fe0df09ac5bd06367e07756a4c52f6e8143286ecee12b6bab5d6580549',
        'data':'c38a152b24870adfa90cf6cca41f7af3a58eb33264d122831fbe4bf23f24540c',
        'qualification':'e39e5018f3fc39872ccc6a448245efbeb922047375b4f158c66d99e0dc3b5f30',
        'audit':'6a395b8a7b95dbe481ef5c6f1e2029d68de32ff6d2f672be9ef6f5dc248dfc1d'},
        'Unchanged accepted numerical collector and verifier sources remain bound')
    # Pytest may import a namespace package without the leading ``tools.``.
    # Assert the exact private identity relative to the actual imported package,
    # including its common-module object; do not accept public/old bindings.
    package=collector.__package__
    check(package.endswith('splendorust_collection4096_20261115.collection') and
          collector._legacy.__name__==package+'._private_collector' and
          data._legacy.__name__==package+'._private_data' and
          collector._legacy.c is worker.c and data._legacy.c is worker.c,
          'Only private modules receive new evidence-context bindings')
    check(root.CELLS == schedule.CELL_COUNT and root.NEW_GAMES == schedule.COLLECTION_GAMES and
          root.NEW_GAME_COUNTS == {'train':3072,'validation':128,'offline':128},
          'Root and worker fixed schedules agree')
    check('torch' not in sys.modules, 'No tensor runtime or models loaded by the wrapper checks')
    return dict(status='passed',checks=checks,count=len(checks),runtime_sources=sources,
                search_decisions=0,new_games=0,optimizer_calls=0,source_scope='new reservation and wrapper contracts only; actual-model entrypoint qualification still required')


def test_new_collection_contract_counterexamples():
    result=run()
    assert result['status']=='passed' and result['count']==40
    assert result['search_decisions']==result['new_games']==result['optimizer_calls']==0


if __name__ == '__main__':
    print(json.dumps(run(),sort_keys=True))
