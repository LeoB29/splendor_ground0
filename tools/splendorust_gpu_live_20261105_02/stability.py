"""Prospective fixed-slot inference qualification before any new game."""
from concurrent.futures import ThreadPoolExecutor
import threading,time
from . import common as c
from tools.splendorust_gpu_probe_20261105.worker import compare_outputs,float32_bytes

WIDTH=31*48
def packing(roots,anchor_index,slot,variant):
    c.require(variant in ('zeros','forward','reverse') and 0<=slot<8,'Fixed packing case')
    rows=[]
    for position in range(8):
        if variant=='zeros':row=[0.]*WIDTH
        elif variant=='forward':row=roots[(anchor_index+position+1)%len(roots)]['tokens']
        else:row=roots[(anchor_index-position-1)%len(roots)]['tokens']
        rows.append(list(row))
    rows[slot]=list(roots[anchor_index]['tokens'])
    return [v for row in rows for v in row]

def roots_from_panel(panel):
    unique={}
    for row in panel['roots']:unique.setdefault(c.digest(row['tokens']),row)
    roots=list(unique.values());c.require(len(roots)==35,'Exactly35 authenticated roots');return roots

def qualify(plan,cpu,gpu,folder,boundary):
    from .backend import validate_output
    folder.mkdir(exist_ok=False)
    panel=c.read(c.resolve(plan['prior_panel']));roots=roots_from_panel(panel)
    cpu_rows=[];gpu_rows=[];packing_rows=[]
    for index,root in enumerate(roots):
        boundary();output=cpu.infer_raw(root['tokens'],1);validate_output(output,1)
        exact=output==[root['historical_output83']] and float32_bytes(output)==float32_bytes([root['historical_output83']])
        c.save(folder/f'cpu_root_{index:02d}.json',dict(root_id=root['root_id'],output=output,expected=[root['historical_output83']],exact=exact))
        c.require(exact,'CPU historical exact root output')
        cpu_rows.append(output)
    c.save(folder/'cpu_historical.json',dict(status='35_original_cpu_exact',outputs=cpu_rows))
    for index,root in enumerate(roots):
        baseline=None;cases=[]
        for slot in range(8):
            for variant in ('zeros','forward','reverse'):
                boundary();tokens=packing(roots,index,slot,variant)
                outputs=gpu.fixed_batch_infer(tokens);validate_output(outputs,8);anchor=[outputs[slot]]
                if baseline is None:baseline=anchor
                comparison=compare_outputs(cpu_rows[index],anchor,[root['legal_policy_indices']],plan['numerical'])
                record=dict(slot=slot,variant=variant,input_sha256=c.digest(tokens),anchor_output=anchor,
                    exact_within_gpu=float32_bytes(anchor)==float32_bytes(baseline) and anchor==baseline,comparison=comparison)
                cases.append(record)
                # Preserve the offending case before rejecting qualification.
                if not record['exact_within_gpu'] or not comparison['passed']:
                    c.save(folder/f'gpu_root_{index:02d}.json',dict(root_id=root['root_id'],cases=cases,status='failed'))
                    raise ValueError('GPU fixed-slot packing stability or CPU numerical comparison failed')
        gpu_rows.append(baseline);packing_rows.append(dict(root_id=root['root_id'],cases=cases,status='passed'))
        c.save(folder/f'gpu_root_{index:02d}.json',packing_rows[-1])
    service_rows=[]
    for runtime,expected in ((cpu,cpu_rows),(gpu,gpu_rows)):
        names=[f'packing_client_{slot}' for slot in range(8)]
        for slot,name in enumerate(names):runtime.service.register(name,slot)
        try:
            for round_index in range(5):
                boundary();barrier=threading.Barrier(8)
                def call(slot):
                    index=(8*round_index+slot)%35;barrier.wait(timeout=5.)
                    output,receipt=runtime.service.infer(names[slot],roots[index]['tokens'],1,deadline=time.perf_counter()+9.)
                    row=dict(route=runtime.route,round=round_index,slot=slot,root_index=index,output=output,receipt=receipt,
                        exact=float32_bytes(output)==float32_bytes(expected[index]) and output==expected[index])
                    return row
                with ThreadPoolExecutor(max_workers=8) as executor:rows=list(executor.map(call,range(8)))
                c.save(folder/f'service_{runtime.route}_{round_index}.json',rows)
                c.require(all(row['exact'] for row in rows),'Actual concurrent broker output/routing exact')
                service_rows.extend(rows)
        finally:
            for name in names:runtime.service.unregister(name)
    result=dict(status='actual_fixed_batch_and_concurrency_qualified',roots=35,physical_batch=8,
        packing_cases=35*8*3,actual_service_requests=len(service_rows),cpu_historical_exact=True,
        gpu_slot_companion_repeat_exact=True,cpu_concurrent_exact=True,actual_gpu_broker_exact=True,
        numerical_tolerances=plan['numerical'],no_games=True)
    c.save(folder/'summary.json',result);return result
