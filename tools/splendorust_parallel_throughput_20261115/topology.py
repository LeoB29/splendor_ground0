"""Read-only physical topology; explicit affinity lifecycle for future workers.

Importing this module neither loads a model nor changes affinity. Windows' two
supported topology APIs must agree on core membership and efficiency class.
The highest class denotes performance cores, per Microsoft's
PROCESSOR_RELATIONSHIP documentation. Core labels from the two APIs need not
be equal; only membership is compared.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
import threading

import psutil

from tools.local_player_qualification_128.clock import processor_topology
from tools.simulation_backend_diagnosis_128.topology import cpu_sets


API = 'GetLogicalProcessorInformationEx(RelationProcessorCore)'
CPU_SET_API = 'GetSystemCpuSetInformation'
_SPAWN_AFFINITY_LOCK = threading.RLock()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _cpus(value, label):
    _require(isinstance(value, (list, tuple)) and bool(value), label + ' is empty')
    _require(all(type(cpu) is int and 0 <= cpu < 64 for cpu in value),
             label + ' must contain group-zero logical IDs')
    _require(len(value) == len(set(value)), label + ' contains duplicate CPUs')
    return sorted(value)


def validate_topology(topology):
    """Validate supplied mappings without any system call or state change."""
    _require(topology.get('api') == API, 'Windows physical-core API required')
    _require(topology.get('cpu_set_api') == CPU_SET_API, 'CPU-set cross-check required')
    cores = topology.get('cores', [])
    rows = topology.get('cpu_sets', [])
    _require(bool(cores) and bool(rows), 'Both topology inventories are required')
    core_members = {}
    classes = set()
    seen = set()
    for core in cores:
        logical = tuple(_cpus(core['logical_processors'], 'Core mask'))
        _require(not (seen & set(logical)), 'Overlapping physical core masks')
        seen.update(logical)
        efficiency = core['efficiency_class']
        _require(type(efficiency) is int and 0 <= efficiency <= 255,
                 'Invalid physical efficiency class')
        _require(bool(core['flags'] & 1) == (len(logical) > 1),
                 'SMT flag disagrees with physical membership')
        core_members[logical] = efficiency
        classes.add(efficiency)
    _require(len(classes) >= 2, 'Heterogeneous core classes are required')
    by_set_core = {}
    set_logical = set()
    ids = set()
    for row in rows:
        _require(row['group'] == 0, 'Multi-group topology is not supported by this pilot')
        logical = row['logical_processor_index']
        _cpus([logical], 'CPU-set logical ID')
        _require(logical not in set_logical and row['id'] not in ids,
                 'Duplicate CPU-set logical processor or ID')
        set_logical.add(logical)
        ids.add(row['id'])
        key = (row['group'], row['core_index'])
        by_set_core.setdefault(key, []).append(row)
    set_members = {}
    for rows_for_core in by_set_core.values():
        efficiencies = {row['efficiency_class'] for row in rows_for_core}
        _require(len(efficiencies) == 1, 'CPU-set siblings have different classes')
        logical = tuple(sorted(row['logical_processor_index'] for row in rows_for_core))
        set_members[logical] = efficiencies.pop()
    _require(set_members == core_members, 'Windows topology APIs disagree')
    allowed = _cpus(topology['allowed_logical_processors'], 'Original allowed mask')
    _require(set(allowed) <= seen, 'Allowed mask contains an unmapped logical processor')
    return {'physical_core_count': len(cores), 'logical_processor_count': len(seen),
            'performance_class': max(classes), 'efficiency_class': min(classes),
            'physical_membership_cross_checked': True, 'single_processor_group': True}


def _signature(topology):
    """Bind physical membership and classes, not unrelated CPU-set ordinal IDs."""
    canonical = sorted((sorted(core['logical_processors']), core['efficiency_class'],
                        core['flags']) for core in topology['cores'])
    return hashlib.sha256(json.dumps(canonical, separators=(',', ':')).encode()).hexdigest()


def read_topology():
    """Read both Windows APIs and this process's allowed mask; never mutate it."""
    if os.name != 'nt':
        raise OSError('Parallel-throughput placement requires Windows')
    physical = processor_topology()
    observed = {'api': physical['api'], 'cpu_set_api': CPU_SET_API,
                'cores': physical['cores'], 'cpu_sets': cpu_sets(),
                'allowed_logical_processors': physical['allowed_logical_processors'],
                'selection_rule': 'four distinct highest-class physical cores; fifth for audit; low-class controller'}
    observed['validation'] = validate_topology(observed)
    observed['physical_signature_sha256'] = _signature(observed)
    return observed


def choose_placement(topology):
    """Pick four model cores, one separate audit core, and lower-class controller."""
    checked = validate_topology(topology)
    allowed = set(topology['allowed_logical_processors'])
    performance = sorted((core for core in topology['cores']
                          if core['efficiency_class'] == checked['performance_class']
                          and allowed.intersection(core['logical_processors'])),
                         key=lambda core: min(allowed.intersection(core['logical_processors'])))
    _require(len(performance) >= 5, 'Five distinct allowed performance cores required')
    pins = [min(allowed.intersection(core['logical_processors'])) for core in performance[:5]]
    controller = sorted(cpu for core in topology['cores']
                        if core['efficiency_class'] == checked['efficiency_class']
                        for cpu in core['logical_processors'] if cpu in allowed)
    _require(controller, 'At least one allowed efficiency core is required')
    placement = {'model_cpus': pins[:4], 'audit_cpus': pins[4:],
                 'controller_cpus': controller, 'original_allowed': sorted(allowed),
                 'physical_signature_sha256': _signature(topology)}
    placement['validation'] = validate_placement(placement, topology)
    return placement


def validate_placement(placement, topology):
    checked = validate_topology(topology)
    _require(placement['physical_signature_sha256'] == _signature(topology),
             'Physical topology differs from the recorded placement')
    models = _cpus(placement['model_cpus'], 'Model pins')
    audit = _cpus(placement['audit_cpus'], 'Audit pins')
    controller = _cpus(placement['controller_cpus'], 'Controller mask')
    original = _cpus(placement['original_allowed'], 'Recorded original allowed mask')
    _require(len(models) == 4 and len(audit) == 1, 'Pilot requires four model pins and one audit pin')
    _require(set(original) <= set(topology['allowed_logical_processors']),
             'Recorded original mask is no longer allowed; validate before narrowing affinity')
    _require(set(models + audit + controller) <= set(original), 'Placement is outside original mask')
    members = {cpu: core for core in topology['cores'] for cpu in core['logical_processors']}
    assigned = [tuple(members[cpu]['logical_processors']) for cpu in models + audit]
    _require(len(set(assigned)) == 5, 'Model/audit pins share a physical core or SMT sibling')
    _require(all(members[cpu]['efficiency_class'] == checked['performance_class']
                 for cpu in models + audit), 'Model/audit pin is not a performance core')
    _require(all(members[cpu]['efficiency_class'] == checked['efficiency_class']
                 for cpu in controller), 'Controller must use efficiency cores')
    _require(not set(models + audit).intersection(controller), 'Controller overlaps model/audit pins')
    return {'physical_cores_distinct': True, 'model_physical_masks': [list(x) for x in assigned[:4]],
            'audit_physical_mask': list(assigned[4]), 'controller_off_model_cores': True}


def verify_startup(expected_topology, placement):
    """Call at future worker startup, before narrowing its inherited affinity."""
    observed = read_topology()
    validate_topology(expected_topology)
    _require(_signature(expected_topology) == _signature(observed), 'Host core map changed since preparation')
    validate_placement(placement, observed)
    return observed


@contextmanager
def service_launch_affinity(original_allowed, controller_cpus, service_cpus, process=None):
    """Temporarily restore only our controller's saved mask for child admission.

    Construction must be serial within this process. The trusted child clock
    still pins each new live or replay service itself. Restoration happens even
    if construction raises; an inability to restore is an explicit failure.
    Supplying ``process`` supports model-free lifecycle fixtures.
    """
    original = _cpus(original_allowed, 'Original allowed mask')
    controller = _cpus(controller_cpus, 'Controller mask')
    service = _cpus(service_cpus, 'Service mask')
    _require(set(controller + service) <= set(original), 'Launch mask outside original allowance')
    _require(not set(controller).intersection(service), 'Service and controller masks overlap')
    process = psutil.Process() if process is None else process
    if isinstance(process, psutil.Process):
        _require(process.pid == os.getpid(), 'Only the calling process affinity may be changed')
    with _SPAWN_AFFINITY_LOCK:
        _require(set(process.cpu_affinity()) <= set(original), 'Current mask escaped original allowance')
        try:
            process.cpu_affinity(original)
            _require(sorted(process.cpu_affinity()) == original, 'Original mask did not restore for admission')
            yield
        finally:
            process.cpu_affinity(controller)
            _require(sorted(process.cpu_affinity()) == controller, 'Controller affinity restoration failed')
