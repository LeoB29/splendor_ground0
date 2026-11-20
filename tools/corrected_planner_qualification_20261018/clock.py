"""Reuse the accepted receipt clock and tree sampler for a Python-only worker."""
import os
from pathlib import Path
import sys

from tools.local_player_qualification_128.clock import PersistentWorker
from tools.external_strength_screen_20261018_02.common import ProcessTreeWorker
from tools.corrected_planner_qualification_20261018.common import file_hash


class PlannerWorker(ProcessTreeWorker):
    """One Python computation process, plus at most its authenticated OS helper."""
    def __init__(self, *args, **kwargs):
        self._tree_cpu = {}
        self._tree_identities = set()
        self._tree_roles = {}
        self._role_owners = {}
        self._verified_executables = {}
        self._console_startup_affinity = {}
        self._initializing = True
        self._python_executable = Path(sys.executable).resolve()
        self._native_executable = None  # No native computation child is permitted.
        self._expected_executables = {self._python_executable: file_hash(self._python_executable)}
        self._console_executable = (Path(os.environ['SystemRoot']) / 'System32/conhost.exe').resolve() if os.name == 'nt' else None
        if self._console_executable is not None:
            self._expected_executables[self._console_executable] = file_hash(self._console_executable)
        try:
            PersistentWorker.__init__(self, *args, **kwargs)
        finally:
            self._initializing = False

    def _sample(self):
        value = super()._sample()
        if len(self._tree_identities) > 2 or any(
                role not in ('python', 'console:python') for role in self._tree_roles.values()):
            self._failure, self._failure_detail = 'process_tree_limit', 'Only Python and its exact OS console helper are permitted'
        return value
