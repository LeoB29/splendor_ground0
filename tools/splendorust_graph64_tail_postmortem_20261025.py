"""Read only saved benchmark evidence; no model, environment or game execution."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'outputs/splendorust_graph64_strength_20261025_01'
OUT = ROOT / 'outputs/splendorust_graph64_tail_postmortem_20261025_01'
EXPECTED = '556480a2501f4fb4cf829fe518aa4ba1ccbf10f12ee088fccf627408dfd69560'
START = time.perf_counter()
LIMIT = 300
used = {}
seal = {}


def boundary():
    if time.perf_counter() - START > LIMIT:
        raise TimeoutError('Saved-evidence analysis exceeded 300 seconds')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def read(path, sealed=False):
    boundary()
    path = Path(path)
    raw = path.read_bytes()
    rel = path.relative_to(ROOT).as_posix()
    item = dict(path=rel, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    if sealed:
        assert item == seal[rel], ('Sealed evidence differs', rel)
    used[rel] = item
    return json.loads(raw)


def save(name, value):
    boundary()
    path = OUT / name
    assert not path.exists(), ('Never overwrite an attempt', path)
    raw = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b'\n'
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) + len(raw) < 32 * 2**20
    path.write_bytes(raw)


def quantile(values, q):
    s = sorted(values)
    x = (len(s) - 1) * q
    a, b = math.floor(x), math.ceil(x)
    return s[a] + (s[b] - s[a]) * (x - a)


def stats(values):
    s = [x for x in values if x is not None]
    assert all(math.isfinite(x) for x in s)
    return dict(count=len(s), missing=len(values)-len(s), mean=statistics.mean(s),
                min=min(s), median=statistics.median(s), p95=quantile(s, .95), max=max(s)) if s else dict(count=0)


FIELDS = ('arrival_seconds', 'request_cpu_seconds', 'arrival_minus_sampled_worker_cpu',
          'post_arrival_parent_elapsed', 'inference_calls', 'phase_count', 'native_legal_max',
          'corrected_legal_count', 'canonical_result_bytes', 'observer_canonical_bytes', 'rss_bytes')


def summarize(rows):
    return dict(count=len(rows), metrics={f:stats([r.get(f) for r in rows]) for f in FIELDS},
                inference_counts=dict(Counter(str(r['inference_calls']) for r in rows)),
                phase_counts=dict(Counter(str(r['phase_count']) for r in rows)),
                statuses=dict(Counter(r['status'] for r in rows)))


def main():
    plan = read(OUT / 'plan.json')
    runfreeze = read(OUT / 'freeze.json')
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == runfreeze['script_sha256']
    assert hashlib.sha256((OUT/'plan.json').read_bytes()).hexdigest() == runfreeze['plan_sha256']
    frozen = read(SRC / 'freeze.json')
    assert used[(SRC/'freeze.json').relative_to(ROOT).as_posix()]['sha256'] == EXPECTED
    sys.path[:0] = [str(ROOT), str(ROOT/'src')]
    from tools.future_training_rules_20261021 import verify_runtime_sources
    assert verify_runtime_sources() == frozen['source_guard']
    completion, execution = read(SRC/'completion.json'), read(SRC/'execution.json')
    assert completion['status'] == 'completed_verified' and execution['exit_code'] == 0
    assert not execution['cleanup_errors']
    seal.update({x['path']:x for x in read(SRC/'completed_game_seal.json')['artifacts']})
    saved = read(SRC/'performance.json')
    schedule = read(SRC/'schedule.json')
    main_schedule = [s for s in schedule if s['stage'] == 'main']
    assert len(main_schedule) == 200
    rows, game_groups = [], {}
    for spec in main_schedule:
        g = SRC / 'games/main' / f"p{spec['pair_index']:03d}_g{spec['game_index']}"
        ending = read(g/'result.json', True)
        assert all(ending[k] == v for k,v in spec.items())
        paths = sorted((g/'moves').glob('*.json'))
        assert len(paths) == ending['plies']
        group = []
        for path in paths:
            r = read(path, True)
            c = r['clock']; t = c['telemetry']; trace = r['trace']; phases = (trace or {}).get('phases', [])
            role, seat = r['player_role'], r['seat']
            assert role in ('challenger', 'opponent')
            assert r['quota'] == (64 if role == 'challenger' else 32)
            assert r['mode'] == ('candidate' if role == 'challenger' else 'baseline')
            assert r['completed'] == (c['status'] == 'ok')
            assert r['game_index'] == spec['game_index']
            obs = r['observer']['observation']
            assert r['observer']['ruleset_id'] == 'splendor_base_qualified_rules_20261018_v1'
            assert r['observer']['catalogue_sha256'] == 'db4776918e612eee5a6f2617719c85ec5f413d592bbab0a64058babc83a70312'
            arrival, cpu = c['receipt_elapsed_seconds'], t['request_cpu_seconds']
            row = dict(id=f"g{r['game_index']}_p{r['ply']}", game_index=r['game_index'], pair_index=r['pair_index'],
                ply=r['ply'], role=role, seat=seat, phase=min(r['ply']//20,2),
                opponent_private=any(x['card_id'] is None for x in obs['players'][1-seat]['reserves']),
                own_reserves=len(obs['players'][seat]['reserves']), completed=r['completed'],
                status=c['status'], detail=c['detail'], arrival_seconds=arrival, request_cpu_seconds=cpu,
                arrival_minus_sampled_worker_cpu=arrival-cpu, post_arrival_parent_elapsed=c['elapsed_seconds']-arrival,
                inference_calls=(trace or {}).get('inference_calls'), phase_count=len(phases) if trace else None,
                native_legal_max=max([x.get('native_legal_count',0) for x in phases], default=None),
                corrected_legal_count=(c['result'] or {}).get('legal_count'),
                canonical_result_bytes=len(canonical(c['result'])) if c['result'] else None,
                observer_canonical_bytes=len(canonical(r['observer'])), rss_bytes=t['rss_bytes'],
                worker_created=next(x['created'] for x in t['process_tree'] if x['role']=='python'),
                worker_clock_file=r['worker_clock_file'], ledger=path.relative_to(ROOT).as_posix())
            rows.append(row); group.append(row)
        game_groups[spec['game_index']] = group
        if (spec['game_index']+1)%25 == 0:
            print(json.dumps(dict(games=spec['game_index']+1, decisions=len(rows), seconds=time.perf_counter()-START)),flush=True)
    assert len(rows) == 11153
    for role in ('challenger', 'opponent'):
        subset=[r for r in rows if r['role']==role]; ok=[r for r in subset if r['completed']]
        ref=saved['players'][role]
        assert len(subset)==ref['attempted_decisions'] and len(ok)==ref['completed_decisions']
        assert dict(Counter(r['status'] for r in subset))==ref['statuses']
        s=stats([r['arrival_seconds'] for r in ok])
        for key in ('mean','median','p95','min','max'):
            assert abs(s[key]-ref['completed_arrival_seconds'][key])<1e-12
    cases=[r for r in rows if r['role']=='challenger' and (not r['completed'] or r['arrival_seconds']>.7)]
    secondary=[r for r in rows if r['role']=='opponent' and (not r['completed'] or r['arrival_seconds']>.7)]
    assert len(cases)==20 and len(secondary)==1
    case_ids={r['id'] for r in cases+secondary}
    panel=[]; selected={r['id']:r for r in cases+secondary}
    for case in cases+secondary:
        links={}
        for role in ('challenger','opponent'):
            for direction in (-1,1):
                pool=[r for r in game_groups[case['game_index']] if r['role']==role and direction*(r['ply']-case['ply'])>0]
                key=f"{role}_{'before' if direction<0 else 'after'}"
                chosen=min(pool,key=lambda r:abs(r['ply']-case['ply'])) if pool else None
                links[key]=chosen['id'] if chosen else None
                if chosen: selected[chosen['id']]=chosen
        pool=[r for r in rows if r['completed'] and r['game_index']!=case['game_index']
              and all(r[k]==case[k] for k in ('role','seat','phase','opponent_private'))]
        chosen=min(pool,key=lambda r:hashlib.sha256(canonical([plan['control_domain'],case['id'],r['id']])).hexdigest()) if pool else None
        links['hash_matched_other_game']=chosen['id'] if chosen else None
        if chosen: selected[chosen['id']]=chosen
        panel.append(dict(case_id=case['id'],controls=links,matched_pool_size=len(pool)))
    # Selected worker receipts and inputs are checked against both the saved seal and ledgers.
    for row in selected.values():
        ledger=read(ROOT/row['ledger'], True)
        clock=read(ROOT/row['worker_clock_file'], True)
        assert clock==ledger['clock']
        inp=read((ROOT/row['worker_clock_file']).parent/'inputs'/Path(row['worker_clock_file']).name, True)
        assert inp['observer']==ledger['observer'] and inp['before']==ledger['before']
    groups={}
    for role in ('challenger','opponent'):
        groups[role+'_all']=summarize([r for r in rows if r['role']==role])
        groups[role+'_completed']=summarize([r for r in rows if r['role']==role and r['completed']])
    groups['graph64_slow_completed']=summarize([r for r in cases if r['completed']])
    groups['graph64_failures']=summarize([r for r in cases if not r['completed']])
    hash_ids={p['controls']['hash_matched_other_game'] for p in panel if p['case_id'] in {r['id'] for r in cases}}
    groups['graph64_hash_controls']=summarize([r for r in rows if r['id'] in hash_ids])
    local64={v for p in panel if p['case_id'] in {r['id'] for r in cases} for k,v in p['controls'].items() if k.startswith('challenger_') and v not in case_ids}
    groups['graph64_neighbor_controls']=summarize([r for r in rows if r['id'] in local64])
    blocks=[]
    for block in range(10):
        blocks.append(dict(game_indices=[20*block,20*block+19],players={role:summarize([r for r in rows if r['role']==role and 20*block<=r['game_index']<20*(block+1)]) for role in ('challenger','opponent')}))
    save('population.json',rows)
    save('case_control_panel.json',dict(cases=[r['id'] for r in cases],secondary=[r['id'] for r in secondary],
         links=panel,records=list(selected.values()),control_selection_uses_game_outcomes=False,
         hash_control_eligibility='All completed matched decisions in another game, including slow cases',
         hash_control_case_overlap=sorted(hash_ids & case_ids),
         control_ranking_uses_arrival=False,case_selection_uses_recorded_timing=True))
    save('analysis.json',dict(status='completed_saved_evidence_analysis',groups=groups,chronological_blocks=blocks,
        cases=cases,secondary=secondary,panel_unique_records=len(selected),
        case_games=sorted({r['game_index'] for r in cases}),
        caveats=['Descriptive retrospective case selection; no causal or fresh strength claim.',
        'Sampled worker-tree CPU excludes supervising parent and includes end sampling; wall minus CPU is not pure waiting.',
        'Failure traces are absent in game ledgers; late raw response is handled in the separate failure audit.',
        'Canonical payload sizes are deterministic content sizes, not raw wire bytes.',
        'No full game replay, inference, training, or rerun of the incomplete supplementary audit.']))
    save('source_manifest.json',dict(artifacts=list(used.values()),sealed_moves=11153,game_results=200,
         selected_receipt_and_input_pairs=len(selected),scope='Only bytes used in this postmortem; original full verification unchanged.'))
    save('completion.json',dict(status='completed_saved_evidence_analysis',elapsed_seconds=time.perf_counter()-START,
         utc=datetime.now(timezone.utc).isoformat(),model_calls=0,new_games=0,freeze_sha256=hashlib.sha256((OUT/'freeze.json').read_bytes()).hexdigest(),
         benchmark_freeze_sha256=EXPECTED,larger_data_deferred=True))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        (OUT/'failure.json').write_text(json.dumps(dict(status='failed',type=type(error).__name__,error=str(error),elapsed_seconds=time.perf_counter()-START),indent=2))
        raise
