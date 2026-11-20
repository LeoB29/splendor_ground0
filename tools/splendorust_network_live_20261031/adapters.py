"""Immutable scientific adapter plus current-attempt receipt writers only."""
from tools.splendorust_search256_recovery_20261030_02.adapters import *


def validate_ready(metadata,child,freeze,*,route=None):
    from . import common as c
    from .worker import lifecycle_variant
    from tools.future_training_rules_20261021 import SOURCE_HASHES
    lifecycle_variant(child.output_dir,freeze)
    route=metadata['route'] if route is None else route;roles={}
    for row in child._sample()['process_tree']:
        if row.get('role') in ('python','search','reference'):
            require(row['role'] not in roles,'Duplicate compute role');roles[row['role']]=row['pid']
    require(set(roles)=={'python','search','reference'} and roles['python']==child.process.pid,'Exact compute roles')
    expected=ready_metadata(route,freeze,pid=child.process.pid,native_pids={k:roles[k] for k in ('reference','search')},
        sources=SOURCE_HASHES,freeze_sha256=c.file_hash(c.OUT/'freeze.json'))
    require(metadata==expected and c.read(child.output_dir/'player_identity.json')==metadata,'New execution READY identity')
    return dict(status='passed',route=route,metadata_sha256=digest(metadata),native_roles=roles)


def verify_saved_ready(folder,freeze,*,route=None):
    from . import common as c
    from types import SimpleNamespace
    folder=Path(folder);receipt=c.read(folder/'initialization.json');metadata=receipt['result']
    route=metadata['route'] if route is None else route
    require(receipt['status']=='ok' and metadata['pid']==receipt['identity']['pid'],'Saved READY receipt')
    proof=validate_ready(metadata,SimpleNamespace(output_dir=folder,process=SimpleNamespace(pid=metadata['pid']),
        _sample=lambda:receipt['telemetry']),freeze,route=route)
    identity=c.read(folder/'boundary_timing_identity.json')
    require(identity['backend']=='qualified_graph64' and
        identity['quota']==route_quota(route) and identity['metadata']==metadata and identity['execution_policy_changed'] is True,
        'Saved explicit timing-policy identity');return proof


def graph_parity(child,envelope,freeze,*,request_fn,collect_fn):
    from . import common as c
    from .worker import verify_graph_admission
    item=freeze['graph_parity_plan'];c.verify([item]);fixtures=c.read(ROOT/item['path'])['fixtures']
    payload=dict(operation='graph_parity',fixtures=fixtures)
    c.save(child.output_dir/'startup_inputs'/f'request-{child._counter+1:08d}.json',payload)
    receipt=request_fn(child,payload,envelope.timeout(60));c.save(child.output_dir/'graph_parity_clock.json',receipt)
    profiles=collect_fn(child,receipt,envelope);c.save(child.output_dir/'graph_parity_profiles.json',profiles)
    require(receipt['status']=='ok' and 'profile_capture_error' not in profiles and 'placement_error' not in profiles,'Graph startup failure')
    verify_graph_admission(child.output_dir,freeze,receipt['result'],boundary=envelope.guard);return receipt


def actual_entrypoint(child,envelope,freeze=None,*,request_fn,collect_fn):
    from . import common as c
    payload={'operation':'entrypoint_checks'}
    c.save(child.output_dir/'startup_inputs'/f'request-{child._counter+1:08d}.json',payload)
    receipt=request_fn(child,payload,envelope.timeout(300));c.save(child.output_dir/'entrypoint_clock.json',receipt)
    profiles=collect_fn(child,receipt,envelope);c.save(child.output_dir/'entrypoint_profiles.json',profiles)
    require(receipt['status']=='ok' and 'profile_capture_error' not in profiles and 'placement_error' not in profiles,'Entrypoint startup failure')
    require(receipt['result']['counts']['complete_decisions']==67,'Original52 plus15 declared same-quota direct checks');return receipt


def numerical_checks(child,envelope,freeze,*,request_fn,collect_fn):
    from . import common as c
    from .worker import verify_numerical_admission
    require(child._counter==1,'Fresh numerical qualification follows sole graph request')
    require(envelope.remaining()>=60.,'Full fresh numerical administrative clock')
    payload={'operation':'numerical_checks'}
    c.save(child.output_dir/'startup_inputs/request-00000002.json',payload)
    receipt=request_fn(child,payload,60.)
    c.save(child.output_dir/'numerical_clock.json',receipt)
    profiles=collect_fn(child,receipt,envelope);c.save(child.output_dir/'numerical_profiles.json',profiles)
    require(receipt['status']=='ok' and 'profile_capture_error' not in profiles and 'placement_error' not in profiles,
        'Fresh numerical response/sidecar/placement failure')
    verify_numerical_admission(child.output_dir,freeze,receipt['result'],boundary=envelope.guard)
    return receipt
