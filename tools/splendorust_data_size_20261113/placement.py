"""Separate physical cores; authenticate, never alter the calibration process."""
from copy import deepcopy
from . import common as c


def validate_topology(topology):
    c.require(topology['api']=='GetLogicalProcessorInformationEx(RelationProcessorCore)','Physical Windows core inventory')
    owners={}
    for row in topology['cores']:
        for cpu in row['logical_processors']:
            c.require(cpu not in owners,'Unique processor topology');owners[cpu]=row
    cpus=(0,2,c.TRAIN_CPU,c.SUPERVISOR_CPU)
    c.require(len(set(cpus))==4 and all(cpu in owners for cpu in cpus),'All four declared processor roles present')
    c.require(len({owners[cpu]['core_index'] for cpu in cpus})==4,'Search, supervision, training and training-owner cores physically distinct')
    c.require(len({owners[cpu]['efficiency_class'] for cpu in cpus})==1 and
        owners[0]['efficiency_class']==max(x['efficiency_class'] for x in topology['cores']),
        'Same performance-core class for numerical CPU comparison')
    return dict(training_cpu=c.TRAIN_CPU,supervisor_cpu=c.SUPERVISOR_CPU,
        calibration_search_cpu=0,calibration_supervisor_cpu=2,
        core_rows={str(cpu):deepcopy(owners[cpu]) for cpu in cpus},
        physical_cores_distinct=True,active_processes_modified=False)


def snapshot():
    from tools.local_player_qualification_128.clock import processor_topology
    return validate_topology(processor_topology())


def set_own(cpu):
    import psutil
    c.require(type(cpu) is int and cpu in (c.TRAIN_CPU,c.SUPERVISOR_CPU),'Only new-study processor assignments')
    process=psutil.Process();process.cpu_affinity([cpu]);process.nice(psutil.NORMAL_PRIORITY_CLASS)
    c.require(process.cpu_affinity()==[cpu],'Exact new process affinity')
    return dict(identity=c.identity(),affinity=[cpu],priority='NORMAL_PRIORITY_CLASS')


def preserve_calibration(*,boundary=lambda:None):
    path=c.ACTIVE_PROXY/'freeze.json'
    c.require(c.sha(path)==c.ACTIVE_PROXY_SHA,'Unchanged active calibration freeze')
    plan=c.read(path);c.verify(plan['worker_inputs'],boundary=boundary);c.verify(plan['data_inputs'],boundary=boundary)
    return dict(freeze=c.artifact(path),source_inputs=len(plan['worker_inputs']),data_inputs=len(plan['data_inputs']),
        status='frozen_calibration_inputs_preserved',processes_modified=False,
        scientific_results_inspected=False,time=c.now())
