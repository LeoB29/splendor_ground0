"""Immutable schedules and explicit prospective worker lifecycles."""
from . import common as c,schedule

def fixed(path,value):
    if path.exists():
        c.require(c.read(path)==value,'Prepared immutable artifact differs: '+str(path))
        return c.artifact(path)
    return c.save(path,value)

def game_worker_name(spec,role):
    return f"{spec['stage']}_{spec['candidate']}_g{spec['game_index']}_{role}"

def replay_worker_name(stage,candidate,index,route):
    return f'audit_{stage}_{candidate}_{index:02d}_{route}'

def lifecycle_inventory():
    result={}
    def add(name,route,ordinary,qualification=False):
        result[name]=dict(route=route,graph_request_id='request-00000001',
          entrypoint_request_id='request-00000002' if qualification else None,
          ordinary_first_request_id='request-00000003' if qualification else 'request-00000002',
          ordinary_requests=ordinary,startup_requests=2 if qualification else 1)
    for route in ('frozen','parity','low','high'):add('qualification_'+route,route,24,True)
    specs=schedule.make_schedule()+sum((schedule.make_schedule('confirmation',arm) for arm in schedule.CANDIDATES),[])
    for spec in specs:
        for role in ('challenger','opponent'):add(game_worker_name(spec,role),spec[role+'_mode'],2000)
    for stage in ('development','confirmation'):
        for arm in schedule.CANDIDATES:
            for index in range(48):
                for route in ('frozen',arm):add(replay_worker_name(stage,arm,index,route),route,1)
    return result

def prepare():
    schedules=dict(operational=schedule.make_schedule('operational'),development=schedule.make_schedule('development'),
      confirmation={arm:schedule.make_schedule('confirmation',arm) for arm in schedule.CANDIDATES})
    # Common setup seeds are intentional within development/operational arms;
    # disjoint phase seed domains exclude any development/confirmation reuse.
    seeds={stage:{s['seed_decimal'] for s in (schedules[stage] if stage!='confirmation' else schedules[stage]['low'])}
      for stage in ('operational','development','confirmation')}
    c.require(all(not seeds[a]&seeds[b] for a,b in [('operational','development'),('operational','confirmation'),('development','confirmation')]),'Disjoint phase setups')
    return dict(schedule=fixed(c.OUT/'schedule.json',schedules),lifecycles=fixed(c.OUT/'declared_worker_lifecycles.json',lifecycle_inventory()))


def seed_inventory():
    import subprocess
    from . import analysis
    targets={int(x['seed_decimal']) for stage in ('operational','development') for x in schedule.make_schedule(stage)}
    targets|={int(x['seed_decimal']) for x in schedule.make_schedule('confirmation','low')}
    c.require(len(targets)==253,'Exact253 distinct game setups')
    targets.add(analysis.SEED)
    found=subprocess.run(['rg','--files','--no-ignore','outputs','-g','*manifest*.json','-g','*schedule*.json','-g','*seed_registry*.json','-g','*reserved_panel*.json'],cwd=c.ROOT,capture_output=True,text=True,check=True,timeout=120)
    checked=[];skipped=[];collisions=[]
    def visit(value,name):
        if type(value) is int or isinstance(value,str) and value.isdecimal():
            if int(value) in targets:collisions.append(name)
        elif isinstance(value,list):
            for item in value:visit(item,name)
        elif isinstance(value,dict):
            for item in value.values():visit(item,name)
    for name in sorted(found.stdout.splitlines()):
        path=(c.ROOT/name).resolve()
        if path.is_relative_to(c.OUT.resolve()):continue
        if path.stat().st_size>8*c.MiB:
            skipped.append(dict(path=path.relative_to(c.ROOT).as_posix(),bytes=path.stat().st_size));continue
        visit(c.read(path),name);checked.append(c.artifact(path))
    c.require(not collisions,'Fresh registered seed collision; no replacement')
    return dict(status='passed',checked=checked,skipped_oversized=skipped,collisions=collisions,game_setups=253,statistics_seed=analysis.SEED,
      intentional_common_setups=dict(operational=3,development=50,confirmation_branches=200),
      qualification_saved_panel_reused=True,scope='Registered manifests, schedules, seed registries and reserved panels up to8MiB; excludes arbitrary raw logs and oversized registers')
