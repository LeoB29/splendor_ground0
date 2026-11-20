"""Screen identity, strict artifact binding and descendant-aware supervision."""
import json
import os
from pathlib import Path
import sys

import psutil

from tools.external_strength_screen_20261018.common import serialize_action, deserialize_action, legal_digest
from tools.local_player_qualification_128.common import (
    ROOT, artifact, boundary, canonical, digest, directory_bytes, execution_lease,
    file_hash, frozen_source_check, now, read, save_new, ResourceHold,
)
from tools.local_player_qualification_128.clock import PersistentWorker

OPPONENT_ID = 'splendorust'
PROFILE_SCHEMA = 'external_opponent_profile_v1'
RAW035_SHA256 = '4422a82b593340eecd02af3a0941f24f0318cc7453e6a0739fd9f524539af2cd'


def verify(items):
    seen = set()
    for item in items:
        path = (ROOT / item['path']).resolve()
        if not path.is_relative_to(ROOT) or path in seen:
            raise ValueError('Invalid/duplicate bound artifact: ' + item['path'])
        seen.add(path)
        if path.stat().st_size != item['bytes'] or file_hash(path) != item['sha256']:
            raise ValueError('Changed bound artifact: ' + item['path'])


def load_profile(path, expected_sha256=None):
    path = Path(path).resolve()
    if expected_sha256 is not None and file_hash(path) != expected_sha256:
        raise ValueError('Opponent profile hash changed')
    profile = read(path)
    if (profile['schema'] != PROFILE_SCHEMA or profile['opponent_id'] != OPPONENT_ID or
            not isinstance(profile['profile_id'], str) or not profile['profile_id'] or
            profile['deterministic_fixed_work'] is not True or
            type(profile['internal_budget_seconds']) not in (int, float) or
            not 0 < profile['internal_budget_seconds'] <= .9 or not profile['artifacts']):
        raise ValueError('Invalid deterministic opponent profile')
    verify(profile['artifacts'])
    return profile


class ProcessTreeWorker(PersistentWorker):
    """Keep the frozen request clock; account for Python and its Rust child.

The Windows job already owns all descendants for termination. CPU totals retain
the last sample for exited descendants; RSS is summed over currently live ones.
Sampling does not measure peaks between polls, and is labelled accordingly.
"""
    def __init__(self, *args, **kwargs):
        self._tree_cpu = {}
        self._tree_identities = set()
        self._tree_roles = {}
        self._role_owners = {}
        self._verified_executables = {}
        self._console_startup_affinity = {}
        self._initializing = True
        self._python_executable = Path(sys.executable).resolve()
        environment = kwargs.get('env') or {}
        profile = load_profile(environment['SPLENDOR_OPPONENT_PROFILE'],
                               environment['SPLENDOR_OPPONENT_PROFILE_SHA256'])
        self._native_executable = (ROOT / profile['native_binary']).resolve()
        native_items = [v for v in profile['artifacts'] if (ROOT / v['path']).resolve() == self._native_executable]
        if len(native_items) != 1:
            raise ValueError('Native executable must have exactly one authenticated artifact')
        self._expected_executables = {
            self._python_executable: file_hash(self._python_executable),
            self._native_executable: native_items[0]['sha256'],
        }
        self._console_executable = (Path(os.environ['SystemRoot']) / 'System32/conhost.exe').resolve() if os.name == 'nt' else None
        if self._console_executable is not None:
            self._expected_executables[self._console_executable] = file_hash(self._console_executable)
        try:
            super().__init__(*args, **kwargs)
        finally:
            self._initializing = False

    def _sample(self):
        if self._ps is None:
            return dict(self._last_telemetry)
        try:
            processes = [self._ps] + self._ps.children(recursive=True)
        except psutil.NoSuchProcess:
            return dict(self._last_telemetry)
        except psutil.AccessDenied as error:
            self._failure, self._failure_detail = 'process_tree_unobservable', str(error)
            return dict(self._last_telemetry)
        live, rss, threads, process_by_identity = [], 0, 0, {}
        for process in processes:
            try:
                identity = (process.pid, process.create_time())
                process_by_identity[identity] = process
                times = process.cpu_times()
                affinity = process.cpu_affinity()
                memory = process.memory_info().rss
                count = process.num_threads()
                executable, name, parent_pid = Path(process.exe()).resolve(), process.name(), process.ppid()
                if executable not in self._verified_executables:
                    expected = self._expected_executables.get(executable)
                    if expected is None or file_hash(executable) != expected:
                        self._failure, self._failure_detail = 'process_tree_limit', 'Unexpected or changed descendant executable: ' + str(executable)
                    else:
                        self._verified_executables[executable] = expected
                self._tree_cpu[identity] = max(self._tree_cpu.get(identity, 0.), times.user + times.system)
                self._tree_identities.add(identity)
                live.append({'pid': identity[0], 'created': identity[1], 'rss_bytes': memory,
                             'cpu_seconds': self._tree_cpu[identity], 'affinity': affinity,
                             'observed_threads': count, 'executable': str(executable), 'name': name,
                             'parent_pid': parent_pid, 'executable_sha256': self._verified_executables.get(executable)})
                rss += memory
                threads += count
            except psutil.NoSuchProcess:
                continue
            except psutil.AccessDenied as error:
                self._failure, self._failure_detail = 'process_tree_unobservable', str(error)
        # First establish the two computation processes; console helpers can be
        # enumerated before their owner and are classified in the second pass.
        for row in live:
            executable = Path(row['executable'])
            role = ('python' if row['pid'] == self._ps.pid and executable == self._python_executable else
                    'rust' if executable == self._native_executable and row['parent_pid'] == self._ps.pid else None)
            if role is not None:
                identity = (row['pid'], row['created'])
                if role in self._role_owners and self._role_owners[role] != identity:
                    self._failure, self._failure_detail = 'process_tree_limit', 'Computation process replacement is forbidden'
                self._role_owners[role] = identity
                self._tree_roles[identity] = role
        for row in live:
            identity = (row['pid'], row['created'])
            role = self._tree_roles.get(identity)
            if role is None and self._console_executable is not None and Path(row['executable']) == self._console_executable and row['name'].lower() == 'conhost.exe':
                owners = [k for k, v in self._role_owners.items() if k in ('python', 'rust') and v[0] == row['parent_pid']]
                if len(owners) == 1:
                    role = 'console:' + owners[0]
                    if role in self._role_owners and self._role_owners[role] != identity:
                        self._failure, self._failure_detail = 'process_tree_limit', 'Console helper replacement is forbidden'
                    self._role_owners[role] = identity
                    self._tree_roles[identity] = role
            row['role'] = role
            if role is None:
                self._failure, self._failure_detail = 'process_tree_limit', 'Unrecognized worker descendant: ' + row['executable']
            # Windows can create a console host before the Python parent is
            # pinned. Only authenticated OS helpers get this startup repair.
            if (role in ('console:python', 'console:rust') and self._initializing and
                    identity not in self._console_startup_affinity and row['executable_sha256'] is not None):
                before = list(row['affinity'])
                try:
                    if before != self.affinity:
                        process_by_identity[identity].cpu_affinity(self.affinity)
                    row['affinity'] = process_by_identity[identity].cpu_affinity()
                    self._console_startup_affinity[identity] = {
                        'startup_affinity_before': before, 'startup_affinity_after': list(row['affinity']),
                        'startup_affinity_adjusted': before != row['affinity'],
                        'startup_affinity_adjustment_phase': 'initialization'}
                except (psutil.NoSuchProcess, psutil.AccessDenied) as error:
                    self._failure, self._failure_detail = 'process_tree_unobservable', str(error)
            row.update(self._console_startup_affinity.get(identity, {}))
            if row['affinity'] != self.affinity:
                self._failure, self._failure_detail = 'affinity_changed', 'A worker process left its assigned core'
        self._peak_rss = max(self._peak_rss, rss)
        if len(self._tree_identities) > 4:
            self._failure, self._failure_detail = 'process_tree_limit', 'Only Python, Rust and one OS console helper per owner are permitted'
        self._last_telemetry = {'cpu_seconds': sum(self._tree_cpu.values()), 'rss_bytes': rss,
            'peak_sampled_rss_bytes': self._peak_rss, 'affinity': self.affinity,
            'observed_threads': threads, 'process_tree': live,
            'processes_observed': len(self._tree_identities), 'resource_scope': 'supervised_worker_and_all_descendants'}
        if rss > self.max_rss_bytes:
            self._failure, self._failure_detail = 'rss_limit', 'Combined worker/descendant RSS exceeded its bound'
        return dict(self._last_telemetry)
