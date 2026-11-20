"""Execute only the new frozen GPU integration study and preserve failures."""
import argparse,os,time,traceback,json
from pathlib import Path
from . import common as c

def run(plan,folder):
    from tools.future_training_rules_20261021 import verify_runtime_sources
    sources=verify_runtime_sources()  # Exactly once at evaluator process startup.
    c.save(folder/'runtime_sources.json',dict(sources=sources,ruleset_id=plan['ruleset_id'],catalogue_sha256=plan['catalogue_sha256'],turn_contract=plan['turn_contract']))
    import psutil
    psutil.Process().cpu_affinity(plan['worker_affinity'])
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    from . import backend,qualification,games,stability
    cpu=gpu=None;error=None;cleanup=[];result=None;begun=time.monotonic();wall=time.time()
    def boundary():
        c.require(time.monotonic()-begun<plan['worker_seconds'] and time.time()-wall<plan['worker_seconds'],'Actual evaluator dual-clock bound')
    def stage(name):
        boundary();c.atomic_json(folder/'stage.json',dict(time=c.now(),stage=name,pid=os.getpid()))
    try:
        stage('initialize_cpu');cpu=backend.build_runtime(plan,'cpu',folder/'cpu',runtime_sources=sources)
        stage('initialize_gpu');gpu=backend.build_runtime(plan,'gpu',folder/'gpu',runtime_sources=sources)
        stage('packing_and_concurrency_qualification');packing=stability.qualify(plan,cpu,gpu,folder/'stability',boundary)
        stage('cpu_actual_entrypoint_and_fixed_roots');cpu_qualification=qualification.run_qualification(plan,cpu,folder/'qualification_cpu',boundary)
        stage('gpu_actual_entrypoint_and_fixed_roots');gpu_qualification=qualification.run_qualification(plan,gpu,folder/'qualification_gpu',boundary)
        stage('compare_fixed_root_decisions');comparison=qualification.compare_qualification(cpu_qualification,gpu_qualification,plan)
        c.save(folder/'decision_comparison.json',comparison)
        stage('eight_concurrent_cpu_games_and_replays');cpu_games=games.run_games(plan,cpu,folder/'games_cpu',boundary)
        stage('eight_concurrent_gpu_games_and_replays');gpu_games=games.run_games(plan,gpu,folder/'games_gpu',boundary)
        stage('throughput_summary');throughput=games.compare_game_rates(cpu_games,gpu_games)
        c.save(folder/'throughput_comparison.json',throughput)
        result=dict(status='completed_live_qualification_and_games',packing=packing,decision_comparison=comparison,
            cpu=cpu_games,gpu=gpu_games,throughput=throughput,training_updates=0,reference_weights_unchanged=True,
            no_automatic_promotion=True,larger_data_deferred=True)
    except BaseException:
        error=traceback.format_exc();c.save(folder/'worker_failure.json',dict(time=c.now(),error=error,no_retry=True))
        for runtime in (cpu,gpu):
            if runtime:
                try:runtime.abort()
                except BaseException as e:cleanup.append(repr(e))
    finally:
        for runtime in (cpu,gpu):
            if runtime:
                try:
                    runtime.close();c.save(folder/runtime.route/'final_identity.json',dict(time=c.now(),identity=runtime.identity,
                        weights_verified_unchanged=True,service_closed=True,active_operations=runtime.active,
                        completed_requests=runtime.service.completed_requests,physical_batches=runtime.service.batch_count))
                except BaseException as e:cleanup.append(repr(e))
        c.save(folder/'cleanup.json',dict(time=c.now(),errors=cleanup,success=not cleanup))
    if error or cleanup:raise RuntimeError('New GPU integration attempt failed; evidence preserved')
    boundary();c.verify(plan['inputs']);c.save(folder/'worker_result.json',result);stage('completed_pending_independent_acceptance')

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    c.require(c.sha(a.plan)==a.plan_sha256,'Exact frozen plan before model imports');plan=c.read(a.plan);c.verify(plan['inputs'])
    c.require(plan['schema']=='splendorust_gpu_live_fixed512_v1' and plan['gpu_physical_batch']==8 and plan['live_callback_batch']==1 and
        plan['maximum_batch_wait_seconds']==.001 and plan['internal_seconds']==9. and plan['outer_seconds']==10.,'Fixed scientific/backend contract')
    run(plan,Path(a.output))
if __name__=='__main__':main()
