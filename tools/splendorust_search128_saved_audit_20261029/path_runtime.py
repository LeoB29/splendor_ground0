"""Narrow absolute-runtime dispatch for an authenticated historical receipt.

Project-relative verification and every physical-file operation are inherited.
The allowlist authorizes exact descriptors, not arbitrary files or aliases.
"""
import json
from pathlib import Path, PureWindowsPath
import re
from types import MappingProxyType

from tools.splendorust_search128_strength_20261028.review_runtime import ReviewRuntime, require

CLOSURE_PATH = 'outputs/splendorust_search128_closure_check_20261029_01/closure_check.json'
CLOSURE_SHA256 = '647f06bdca298e4b1fcc4b4bdff205906c422707e4a1684f23e2fe409dc17dad'
ABSOLUTE_RUNTIME_COUNT = 95
RUNTIME_INVENTORY_COUNT = 171
_DRIVE_ROOTED = re.compile(r'^[A-Za-z]:[\\/]')
_SHA256 = re.compile(r'[0-9a-f]{64}')


def _descriptor(item):
    require(type(item) is dict and set(item) == {'path', 'bytes', 'sha256'} and
            type(item['path']) is str and type(item['bytes']) is int and item['bytes'] >= 0 and
            type(item['sha256']) is str and _SHA256.fullmatch(item['sha256']) is not None,
            'Exact saved-runtime artifact descriptor')


def _absolute_path(name):
    """Accept drive-rooted syntax only; aliases still require exact allowlisting."""
    require(type(name) is str and _DRIVE_ROOTED.match(name) is not None and
            '\x00' not in name, 'Drive-rooted runtime path required')
    parts = re.split(r'[\\/]', name[3:])
    require(parts and all(part and part not in ('.', '..') and ':' not in part and
                         not part.endswith((' ', '.')) for part in parts),
            'No runtime traversal, empty component, stream or ambiguous component')
    path = PureWindowsPath(name)
    require(path.is_absolute() and len(path.drive) == 2 and path.drive[1] == ':' and
            path.root == '\\', 'No UNC, device or drive-relative runtime path')
    return Path(name)


class AuthorizedRuntimeReview(ReviewRuntime):
    """Only95 exact saved runtime descriptors extend the original verifier.

    Constructor authentication is charged to the unchanged inherited guard.
    No deadline, output path, model setting or original module global changes.
    """
    def __init__(self, root, out, default_seconds=3600., *, seconds=None):
        super().__init__(root, out, default_seconds, seconds=seconds)
        pin, raw = self.pin(self.physical(CLOSURE_PATH), keep_bytes=True)
        require(pin['sha256'] == CLOSURE_SHA256, 'Exact closure receipt must authenticate before runtime authorization')
        value = json.loads(raw)
        require(value['status'] == 'qualified_saved_closure_checker_only' and
                value['benchmark_accepted'] is False and value['new_model_calls'] == value['new_games'] == 0,
                'Historical closure receipt remains saved-only and unaccepted')
        items = value['imported_frozen_sources_and_runtime']
        require(type(items) is list and len(items) == RUNTIME_INVENTORY_COUNT, 'Exact saved runtime/source inventory')
        authorized = {}; physical_keys = set()
        for item in items:
            self.guard(); _descriptor(item)
            name = item['path']
            if _DRIVE_ROOTED.match(name):
                path = _absolute_path(name); key = self._key(path)
                require(name not in authorized and key not in physical_keys, 'No duplicate or aliased runtime authority')
                authorized[name] = (item['bytes'], item['sha256'])
                physical_keys.add(key)
            else:
                # This validates the unchanged relative syntax without reading
                # any listed source or learning from an unauthenticated path.
                self.physical(name)
        require(len(authorized) == ABSOLUTE_RUNTIME_COUNT, 'Exactly95 authenticated runtime descriptors')
        self._authorized_runtime = MappingProxyType(authorized)
        self.closure_receipt = dict(pin)
        self.guard()

    @property
    def absolute_runtime_descriptors(self):
        return tuple(dict(path=path, bytes=value[0], sha256=value[1])
                     for path, value in self._authorized_runtime.items())

    def dispatch_descriptor(self, item):
        """Resolve a descriptor without hashing its target; useful for preflight."""
        _descriptor(item)
        name = item['path']
        if _DRIVE_ROOTED.match(name):
            path = _absolute_path(name)
            require(self._authorized_runtime.get(name) == (item['bytes'], item['sha256']),
                    'Absolute runtime descriptor not authorized by exact closure receipt')
            return path
        require(all(part and part not in ('.', '..') for part in name.split('/')),
                'Canonical project-relative components')
        return self.physical(name)

    def verify(self, item):
        path = self.dispatch_descriptor(item)
        if not _DRIVE_ROOTED.match(item['path']):
            return super().verify(item)
        # pin normalizes its returned path separators. Compare the authenticated
        # bytes/hash, retaining the exact original descriptor as the authority.
        self._expect(path, item)
        got = self.pin(path)
        require((got['bytes'], got['sha256']) == (item['bytes'], item['sha256']),
                'Pinned absolute runtime bytes differ: '+item['path'])
