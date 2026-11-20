"""Outcome-blind preparation timing on fixed TRAIN batches; no retained fitted model."""
from pathlib import Path
import argparse
import gc
import io
import math
import shutil
import time

import torch

from . import common as c, learning, training_source


def memory_snapshot():
    """Read process/system memory when the host's optional psutil is available."""
    try:
        import psutil
        current=psutil.Process().memory_info()
        system=psutil.virtual_memory()
        return {"rss_bytes":current.rss,"peak_working_set_bytes":getattr(current,"peak_wset",None),
                "available_system_bytes":system.available,"total_system_bytes":system.total}
    except ImportError:
        return {"available":False}


def batch_indices(prefix_length, count=32):
    c.require(prefix_length >= count >= 2, "Invalid resource probe sample")
    return [i*(prefix_length-1)//(count-1) for i in range(count)]


def projection(arm_seconds, checkpoint_bytes, checkpoint_count, free_bytes):
    c.require(set(arm_seconds) == set(c.LEARNING_ARMS), "Both arms must be timed")
    c.require(all(math.isfinite(v) and v > 0 for v in arm_seconds.values()), "Invalid timing")
    c.require(all(type(x) is int and x > 0 for x in (checkpoint_bytes, checkpoint_count, free_bytes)),
              "Invalid storage measurement")
    # Two complete passes: fitting and independent reproduction. Budget a 50%
    # allowance for sampling/timing variation, plus diagnostics/features/I/O.
    learning_seconds = max(arm_seconds.values()) * c.CONFIG["updates"] * len(c.FITS) * len(c.LEARNING_ARMS) * 2
    learning_seconds = learning_seconds * 1.5 / c.CONFIG["fit_workers"] + 2*3600
    total_seconds = learning_seconds + c.CONFIG["pre_evaluation_reserve_seconds"]
    projected_bytes = int(checkpoint_bytes * checkpoint_count * len(c.FITS)*len(c.LEARNING_ARMS)*1.25) + 5*1024**3
    return {"reasonable": total_seconds <= c.CONFIG["maximum_runtime_hours"]*3600
                           and free_bytes-projected_bytes >= c.CONFIG["disk_reserve_bytes"],
            "projected_learning_hours": learning_seconds/3600,
            "evaluation_reserve_hours": c.CONFIG["pre_evaluation_reserve_seconds"]/3600,
            "projected_total_hours": total_seconds/3600,
            "projected_additional_bytes": projected_bytes, "free_bytes":free_bytes,
            "allowance_multiplier":1.5, "includes_independent_full_learning_replay":True,
            "fresh_evaluation_requires_separate_measured_probe":True}


def run(folder):
    from . import experiment
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=False)
    before=experiment.identity()
    parent=training_source.closed_parent()
    data=c.verified(parent["data"])
    started=time.perf_counter()
    pools={kind:torch.load(c.verify_hash(data["files"][kind]),map_location="cpu",weights_only=True)
           for kind in ("old","added")}
    fit=c.FITS[0]
    plan,_=learning._plans(pools,parent["partition"],fit,learning.CONFIG)
    indices=batch_indices(learning.CONFIG["prefix_config"]["updates"])
    def padded_work(index):
        batch=plan[index]
        return len(batch)*max(len(pools[kind][row]["features"]) for kind,row in batch)
    stress_index=max(range(learning.CONFIG["prefix_config"]["updates"]),key=padded_work)
    memory_loaded=memory_snapshot()
    timings={}
    size=0
    for arm in c.LEARNING_ARMS:
        net,params,opt,frozen,names=learning._model(parent["models"][f"anchor_{fit}"],arm,
                            learning.CONFIG["train_seeds"][str(fit)],learning.CONFIG)
        measurements=[]
        for step in [indices[0],indices[0],*indices,stress_index]:
            batch=[pools[kind][index] for kind,index in plan[step]]
            tick=time.perf_counter()
            opt.zero_grad(set_to_none=True)
            logits,_=net(*learning.collate(batch))
            loss=learning.objective(logits,batch,learning.CONFIG["target_temperature"])[0].mean()
            c.require(bool(torch.isfinite(loss)),"Nonfinite resource-probe objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params,learning.CONFIG["gradient_norm"],error_if_nonfinite=True)
            opt.step()
            learning._check_frozen(net,frozen)
            c.require(all(bool(torch.isfinite(v).all()) for v in net.state_dict().values()),"Nonfinite probe model")
            measurements.append(time.perf_counter()-tick)
        payload={"model_state_dict":net.state_dict(),"optimizer_state_dict":opt.state_dict(),
                 "preclip_gradient_norms":[0.]*c.CONFIG["updates"],"torch_rng_state":torch.get_rng_state()}
        stream=io.BytesIO();torch.save(payload,stream);size=max(size,stream.tell())
        measured=measurements[2:-1]
        timings[arm]={"step_seconds":measured,"mean_seconds":sum(measured)/len(indices),
                      "warmups":2,"updates":len(indices),"trainable_parameters":sum(p.numel() for p in params),
                      "memory_stress_index":stress_index,"memory_stress_padded_actions":padded_work(stress_index),
                      "memory_stress_seconds":measurements[-1],"memory":memory_snapshot()}
        del net,params,opt,frozen,payload,stream,logits,loss
        gc.collect()
    c.require(experiment.identity()==before,"Code changed during resource probe")
    forecast=projection({a:v["mean_seconds"] for a,v in timings.items()},size,
                        len(learning._progress_updates(learning.CONFIG)),shutil.disk_usage(c.ROOT).free)
    report={"time":c.old.now(),"identity":before,"fit":fit,"arms":list(c.LEARNING_ARMS),
            "data":parent["data"],"batch_indices":indices,"timings":timings,
            "checkpoint_payload_bytes":size,"seconds":time.perf_counter()-started,
            "memory_after_data_load":memory_loaded,
            "forecast":forecast,"assessment_read":False,"validation_read":False,
            "training_metrics_used_for_selection":False,"model_retained":False,
            "new_simulations":0,"production_updates_completed":0}
    c.old.save_new(folder/"resource_probe.json",report)
    print({"resource_probe":str(folder/"resource_probe.json"),"forecast":forecast},flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    run(parser.parse_args().folder)
