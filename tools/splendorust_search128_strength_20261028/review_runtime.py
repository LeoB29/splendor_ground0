"""Bounded parallel byte authentication only; no scientific or model imports."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import shutil
import threading
import time
import psutil


def require(value, message):
    if not value:
        raise ValueError('SEARCH128_REVIEW: ' + message)


def identity(value):
    return (value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns, value.st_mode,
            getattr(value, 'st_file_attributes', 0))


class ReviewRuntime:
    """At most eight reads in flight; every parsed JSON authenticates its bytes.

    Cached hashes are reusable only under the complete unchanged lstat identity.
    Directory identities/reparse status and every cached file are checked again
    before a passing component can be written. Contradictory expected digests or
    observations fail, including races between authentication threads.
    """
    def __init__(self, root, out, default_seconds=3600., *, seconds=None):
        self.root, self.out = Path(root).resolve(), Path(out).resolve()
        self.start, self.wall = time.monotonic(), time.time()
        self.seconds = default_seconds if seconds is None else seconds
        self.shared_deadlines = None
        self.consumed, self.counts = {}, {}
        self._cache, self._directories, self._expected = {}, {}, {}
        self._lock = threading.RLock()
        self._cancel = threading.Event()
        self._resource_process = psutil.Process()
        self._last_resource_check = 0.
        self._peak_rss = 0
        self.metrics = dict(hashed_reads=0, cached_verifications=0,
                            actual_bytes_hashed=0, parallel_workers=8)

    def guard(self):
        require(not self._cancel.is_set(), 'Authentication cancelled after a thread failure')
        require(max(time.monotonic()-self.start, time.time()-self.wall) < self.seconds,
                'Independent saved-only review deadline; no acceptance on exhaustion')
        if self.shared_deadlines is not None:
            mono, wall = self.shared_deadlines
            require(time.monotonic() < mono and time.time() < wall, 'Original global/final shared deadline exhausted')
        now = time.monotonic()
        if now-self._last_resource_check >= 2.:
            with self._lock:
                if now-self._last_resource_check >= 2.:
                    rss = self._resource_process.memory_info().rss
                    self._peak_rss = max(self._peak_rss,rss)
                    require(rss <= 2*1024**3, 'Review RSS exceeds2GiB')
                    require(shutil.disk_usage(self.root).free >= 5*1024**3, 'Review free-disk reserve below5GiB')
                    self._last_resource_check = now

    def physical(self, name):
        require(type(name) is str and '\\' not in name, 'Canonical project-relative path')
        path = PurePosixPath(name)
        require(not path.is_absolute() and bool(path.parts) and
                all(x not in ('', '.', '..') and ':' not in x for x in path.parts), 'Safe path')
        return self.root.joinpath(*path.parts)

    @staticmethod
    def _key(path):
        return os.path.normcase(os.path.abspath(path))

    @staticmethod
    def _stat(path, directory=False):
        value = os.stat(path, follow_symlinks=False)
        require(not stat.S_ISLNK(value.st_mode) and
                not (getattr(value, 'st_file_attributes', 0) & 0x400),
                'No symlink/reparse redirection: '+str(path))
        require(stat.S_ISDIR(value.st_mode) if directory else stat.S_ISREG(value.st_mode),
                'Expected directory' if directory else 'Expected regular file: '+str(path))
        return value

    def _parents(self, path):
        # Keep these identities separate: directory size/mtime may change when
        # a sibling postrun receipt is added, so only directory ownership and
        # redirection identity are invariant. File checks use every stat field.
        for parent in path.parents:
            key = self._key(parent)
            with self._lock:
                if key in self._directories:
                    continue
            value = self._stat(parent, directory=True)
            marker = (value.st_dev, value.st_ino, value.st_mode,
                      getattr(value, 'st_file_attributes', 0))
            with self._lock:
                old = self._directories.get(key)
                require(old is None or old[1] == marker, 'Parent changed during authentication')
                self._directories[key] = (str(parent), marker)

    def pin(self, path, *, keep_bytes=False):
        self.guard()
        path = Path(os.path.abspath(path))
        self._parents(path)
        key = self._key(path)
        before = self._stat(path)
        marker = identity(before)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                require(cached['identity'] == marker, 'Cached file identity changed: '+str(path))
                if not keep_bytes:
                    self.metrics['cached_verifications'] += 1
                    return dict(cached['item'])
        hashed, chunks = hashlib.sha256(), []
        with path.open('rb') as stream:
            opened = os.fstat(stream.fileno())
            # Windows path-stat and handle-stat timestamps can use distinct
            # API conversions. Compare device/file ID/size across APIs, and
            # compare ALL timestamp fields only within the same API.
            require((opened.st_dev,opened.st_ino,opened.st_size)==
                    (before.st_dev,before.st_ino,before.st_size), 'Opened handle identity differs')
            handle_marker = identity(opened)
            while True:
                self.guard()
                chunk = stream.read(1024*1024)
                if not chunk:
                    break
                hashed.update(chunk)
                if keep_bytes:
                    chunks.append(chunk)
            require(identity(os.fstat(stream.fileno())) == handle_marker, 'Open file changed during read')
        require(identity(self._stat(path)) == marker, 'File changed during read')
        try:
            name = path.relative_to(self.root).as_posix()
        except ValueError:
            name = path.as_posix()
        item = dict(path=name, bytes=before.st_size, sha256=hashed.hexdigest())
        with self._lock:
            old = self._cache.get(key)
            require(old is None or (old['identity'] == marker and old['item'] == item),
                    'Contradictory cached bytes/identity: '+name)
            require(name not in self.consumed or self.consumed[name] == item,
                    'Consumed file changed: '+name)
            self._cache[key] = dict(path=str(path), identity=marker, item=item)
            self.consumed[name] = item
            self.metrics['hashed_reads'] += 1
            self.metrics['actual_bytes_hashed'] += before.st_size
        return (dict(item), b''.join(chunks)) if keep_bytes else dict(item)

    def verify(self, item):
        require(type(item) is dict and set(item) == {'path', 'bytes', 'sha256'} and
                type(item['bytes']) is int and item['bytes'] >= 0 and
                type(item['sha256']) is str and len(item['sha256']) == 64,
                'Artifact descriptor')
        path = self.physical(item['path'])
        self._expect(path, item)
        require(self.pin(path) == item, 'Pinned bytes differ: '+item['path'])

    def _expect(self, path, item):
        key = self._key(path)
        expected = (item['bytes'], item['sha256'])
        with self._lock:
            require(key not in self._expected or self._expected[key] == expected,
                    'Contradictory expected artifact pins: '+str(path))
            self._expected[key] = expected

    def verify_many(self, items):
        """Authenticate a finite path/descriptor iterable with only16 futures."""
        unique = {}
        for path, item in items:
            self.guard(); self._expect(path, item)
            key = self._key(path)
            if key not in unique:
                unique[key] = (Path(path), item)
        def check(job):
            path, expected = job
            got = self.pin(path)
            require((got['bytes'], got['sha256']) == (expected['bytes'], expected['sha256']),
                    'Parallel artifact bytes differ: '+str(path))
            return got
        iterator = iter(unique.values())
        pending = set()
        executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix='saved-hash')
        try:
            for _ in range(min(16, len(unique))):
                pending.add(executor.submit(check, next(iterator)))
            while pending:
                self.guard()
                done, pending = wait(pending, timeout=.25, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                    try:
                        job = next(iterator)
                    except StopIteration:
                        continue
                    pending.add(executor.submit(check, job))
        except BaseException:
            self._cancel.set()
            for future in pending:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        self.guard()

    def revalidate_identities(self):
        with self._lock:
            files = list(self._cache.values())
            directories = list(self._directories.values())
        for item in files:
            self.guard()
            require(identity(self._stat(item['path'])) == item['identity'],
                    'Authenticated file changed before acceptance: '+item['path'])
        for path, expected in directories:
            self.guard(); value = self._stat(path, directory=True)
            require((value.st_dev, value.st_ino, value.st_mode,
                     getattr(value, 'st_file_attributes', 0)) == expected,
                    'Authenticated directory was replaced or redirected: '+path)
        return dict(files=len(files), directories=len(directories), status='passed')

    def read_path(self, path):
        _, raw = self.pin(path, keep_bytes=True)
        return json.loads(raw)

    def read(self, name):
        return self.read_path(self.out/name)

    def exact_tree(self, folder, expected):
        found = set()
        for base, dirs, names in os.walk(folder, followlinks=False):
            self.guard()
            for name in dirs:
                self._stat(Path(base)/name, directory=True)
            for name in names:
                self._stat(Path(base)/name)
                found.add((Path(base)/name).relative_to(self.root).as_posix())
        require(found == set(expected), 'Exact preserved tree inventory: '+str(folder))
        return len(found)

    def absent(self, identity):
        self.guard()
        require(type(identity['pid']) is int and identity['pid'] > 0 and
                type(identity['created']) in (int, float), 'Owned PID/creation identity')
        try:
            current = psutil.Process(identity['pid'])
        except psutil.NoSuchProcess:
            return dict(identity, state='absent')
        require(abs(current.create_time()-identity['created']) > .0001,
                'Original owned process still alive')
        return dict(identity, state='pid_reused_original_absent')
