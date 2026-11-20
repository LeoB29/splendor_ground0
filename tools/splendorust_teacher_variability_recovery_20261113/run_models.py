"""Finite request inventories, full response clocks and exact service exits."""
import time
import traceback
import psutil
from . import common as c
from . import control


def run(role,folder,owner,plan,boundary):
    from tools.local_player_qualification_128.clock import PersistentWorker
    from tools.splendorust_teacher_variability_20261113 import panel,qualification,ownership,adapters
    frozen_panel=c.read(c.PANEL);roots={r['root_id']:r for r in frozen_panel['roots']}
    fixtures=c.read(c.resolve(plan['graph_parity_plan']))['fixtures']
    service=None;error=None;result=None;children=[];code=None;receipts=[];owned=None
    # Controller on a separate logical processor; no new deployment affinity.
    allowed=psutil.Process().cpu_affinity()
    env=control.environment();env.update(VARIABILITY_SERVICE_FOLDER=str(folder/'service'),
        VARIABILITY_FREEZE_SHA256=c.sha(c.PLAN),VARIABILITY_ROLE=role)
    started=time.monotonic()
    try:
        service=PersistentWorker(c.PACKAGE+'.service',folder/'transport',affinity=[0],
            initialization_timeout=240,env=env,max_message_bytes=16*1024**2,
            max_log_bytes=4*c.GiB,max_rss_bytes=c.MODEL_RSS,
            absolute_deadline=owner['limits']['monotonic_deadline'],resource_guard=boundary)
        if 2 in allowed:psutil.Process().cpu_affinity([2])
        initial=service.initialization
        c.require(initial['result']['identity']==service.identity and initial['result']['role']==role
            and initial['result']['freeze_sha256']==c.sha(c.PLAN),'Exact ready service identity')
        runtime=c.read(folder/'service/variability_runtime_identity.json')
        expected_binaries={}
        for name in runtime['native_pids']:
            item=plan['control_profiles']['legacy_p3_q16'] if name=='control' else plan['research_profiles']['legacy_p3_q16' if name=='reference' else 'legacy_p3_q512']
            document=c.read(c.resolve(item));expected_binaries[name]=c.artifact(c.ROOT/document['native_binary'])
            c.require(expected_binaries[name]['sha256']==document['native_binary_sha256'],'Frozen native content binding')
        startup=ownership.inspect_service_tree(service.identity,runtime['native_pids'],expected_binaries,plan['console_executable'])
        children=[dict(pid=p['pid'],created=p['created']) for p in startup]
        c.save(folder/'native_startup.json',dict(service=service.identity,rows=startup,expected_binaries=expected_binaries,
            runtime_identity=c.artifact(folder/'service/variability_runtime_identity.json')))
        graph_identity=runtime['graph_identity']
        c.require(runtime['checkpoint_sha256']==c.PARENT_CHECKPOINT_SHA and
            graph_identity['checkpoint_sha256']==c.PARENT_CHECKPOINT_SHA and
            graph_identity['weights_before_sha256']==graph_identity['weights_after_sha256'] and
            graph_identity['dtype']=='torch.float32' and graph_identity['device']=='cpu' and
            graph_identity['parameter_count']==4780883,'Unchanged model and graph identity')
        if True:
            qualified=c.read(c.ORIGINAL_OUT/'jobs/qualification/worker/service/variability_runtime_identity.json')['graph_identity']
            for field in ('checkpoint_sha256','weights_before_sha256','weights_after_sha256','torch_version','numpy_version','backend','options'):
                c.require(graph_identity[field]==qualified[field],'Qualified graph/runtime binding: '+field)
        graph=service.request(dict(operation='graph_parity',fixtures=fixtures),300)
        c.require(graph['status']=='ok' and graph['result']['passed'] is True,'Exact graph startup qualification')
        qualification.verify_graph_evidence(folder/'service/graph_parity.json',fixtures)
        if role=='admission':
            specs=c.read(c.ADMISSION)
        elif role=='diagnostic':
            specs=c.read(c.SUFFIX)
        elif role=='reproduction':
            specs=c.read(c.REPEATS)
        else:raise ValueError('Unregistered recovery model role')
        if True:
            for index,spec in enumerate(specs):
                boundary()
                c.require(owner['limits']['monotonic_deadline']-time.monotonic()>121,'Original stage has full request allowance remaining')
                if role=='diagnostic' and index==0:
                    c.require(spec==c.read(c.MATRIX)[c.PREFIX_COUNT],'Only the disclosed ambiguous boundary request may restart')
                    c.save(c.OUT/'ambiguous_retry_consumed.json',dict(request=spec,original_request_index=c.PREFIX_COUNT,
                        original_attempt_remains_incomplete=True,new_allowance_seconds=120,
                        identity=c.identity(),freeze_sha256=c.sha(c.PLAN),
                        authorization='User requested restart best way after host crash; separately disclosed amended recovery',time=c.now()))
                c.save(folder/'dispatch'/f'{index:05d}.json',dict(spec=spec,role=role,identity=c.identity(),
                    time=c.now(),monotonic_before_dispatch=time.monotonic(),deadline_seconds=120,freeze_sha256=c.sha(c.PLAN)))
                payload=dict(player='search' ,observer=roots[spec['root_id']]['observer'],
                    quota=spec['quota'],pool=spec['pool'],mode=spec['mode'],world_seed_label=spec['world_seed_label'],encoding_seed_label=spec['encoding_seed_label'],telemetry=True)
                clock=service.request(payload,120)
                receipt=c.artifact(folder/'transport'/f"{clock['request_id']}.json")
                row=dict(spec=spec,receipt=receipt);c.save(folder/'completed'/f'{index:05d}.json',row)
                c.require(clock['status']=='ok' and clock['accepted'] and clock['received_before_deadline'],
                    'Saved request failed; no retry or replacement')
                document=c.read(c.resolve(plan['research_profiles'][adapters.profile_key(spec['quota'],spec['pool'],spec['mode'])]))
                qualification.validate_result(clock['result'],payload['observer'],quota=spec['quota'],
                    pool=spec['pool'],mode=spec['mode'],world_seed_label=spec['world_seed_label'],encoding_seed_label=spec['encoding_seed_label'],profile_id=document['profile_id'],
                    profile_sha256=plan['research_profiles'][adapters.profile_key(spec['quota'],spec['pool'],spec['mode'])]['sha256'],
                    wrapper_id=document['wrapper_id'],native_binary_sha256=document['native_binary_sha256'])
                if role=='admission':
                    c.require(c.canonical(clock['result'])==c.canonical(c.read(c.resolve(spec['source_receipt']))['result']),
                        'Recovery continuity probe must exactly reproduce original complete action, callbacks, values and sampling')
                receipts.append(row)
                c.status(folder/'progress.json',dict(time=c.now(),role=role,completed=index+1,total=len(specs),
                    elapsed_seconds=time.monotonic()-started,last_request_id=spec['request_id'],
                    cumulative_completed=c.PREFIX_COUNT+index+1 if role=='diagnostic' else None,original_total=3072))
            result=dict(index=c.save(folder/'index.json',dict(requests=receipts)),requests=len(receipts))
        boundary()
        end=ownership.inspect_service_tree(service.identity,runtime['native_pids'],expected_binaries,plan['console_executable'])
        book=ownership.make_book(service.identity,runtime['native_pids'],startup,end,expected_binaries,plan['console_executable'])
        children=book['all_owned_identities']
        owned=c.save(folder/'native_owners.json',dict(book=book,startup=c.artifact(folder/'native_startup.json'),
            runtime_identity=c.artifact(folder/'service/variability_runtime_identity.json')))
        service.process.stdin.close();code=service.process.wait(timeout=20)
        c.require(code==0,'Service actual exit0')
        service_receipt=c.read(folder/'service/service_exit.json')
        c.require(service_receipt['status']=='passed' and service_receipt['error'] is None
            and service_receipt['identity']==service.identity,'Service graceful result')
    except BaseException:error=traceback.format_exc()
    finally:
        if service is not None:
            try:service.close()
            except BaseException:error=(error or '')+traceback.format_exc()
    if service is not None:
        # Windows console cleanup is asynchronous: observe exact absence with a bounded grace.
        try:
            grace=control.observe_absence([service.identity,*children],owner['limits'])
        except BaseException:
            error=(error or '')+traceback.format_exc()
            grace=dict(exact_owned_absence=not c.exact_alive(service.identity) and all(not c.exact_alive(x) for x in children),
                original_clock_failed=True)
        absence=grace['exact_owned_absence']
        actual=c.save(folder/'service_actual_exit.json',dict(status='passed' if error is None and absence and code==0 else 'failed',
            identity=service.identity,owned_native_identities=children,exit_code=code,exact_owned_absence=absence,
            error=error,cleanup_grace=grace,time=c.now()))
        if result is not None and owned is not None:result.update(service_actual_exit=actual,native_owners=owned)
    c.require(error is None and service is not None and absence and code==0,
        error or 'Model service failed actual exit or exact cleanup; receipt preserved')
    return result
