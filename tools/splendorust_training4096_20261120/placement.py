"""Only this study's own affinity; two distinct physical performance cores."""
from copy import deepcopy
from . import common as c

def validate_topology(topology):
    c.require(topology['api']=='GetLogicalProcessorInformationEx(RelationProcessorCore)','Physical Windows core inventory')
    owners={}
    for row in topology['cores']:
        for cpu in row['logical_processors']:
            c.require(cpu not in owners,'Unique topology');owners[cpu]=row
    cpus=(c.TRAIN_CPU,c.SUPERVISOR_CPU)
    c.require(len(set(cpus))==2 and all(cpu in owners for cpu in cpus),'Both declared CPUs present')
    c.require(owners[cpus[0]]['core_index']!=owners[cpus[1]]['core_index'],'Distinct physical model/supervisor cores')
    c.require(all(owners[cpu]['efficiency_class']==max(x['efficiency_class'] for x in topology['cores']) for cpu in cpus),'Performance core class')
    return dict(training_cpu=cpus[0],supervisor_cpu=cpus[1],core_rows={str(cpu):deepcopy(owners[cpu]) for cpu in cpus},physical_cores_distinct=True)

def snapshot():
    from tools.local_player_qualification_128.clock import processor_topology
    return validate_topology(processor_topology())

def set_own(cpu):
    import psutil
    c.require(type(cpu) is int and cpu in (c.TRAIN_CPU,c.SUPERVISOR_CPU),'Only declared own CPU')
    process=psutil.Process();process.cpu_affinity([cpu]);process.nice(psutil.NORMAL_PRIORITY_CLASS)
    c.require(process.cpu_affinity()==[cpu],'Exact own process affinity')
    return dict(identity=c.identity(),affinity=[cpu],priority='NORMAL_PRIORITY_CLASS')
