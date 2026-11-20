"""One durable ascending queue shared by eight clients in one or four processes."""
from pathlib import Path
import json
import sqlite3
import threading
import time


class SharedStop:
    def __init__(self, path):
        self.path = Path(path)

    def is_set(self):
        return self.path.exists()

    def set(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open('x', encoding='utf8') as stream:
                json.dump({'stopped_unix': time.time(), 'no_retry': True}, stream)
        except FileExistsError:
            pass


def create_queue(path, specifications):
    path = Path(path)
    if path.exists() or len(specifications) != 16:
        raise ValueError('Fresh fixed sixteen-game queue required')
    ids = [x['game_id'] for x in specifications]
    if len(set(ids)) != 16:
        raise ValueError('Unique game identities required')
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE games(position INTEGER PRIMARY KEY,spec TEXT NOT NULL,assignment TEXT)')
        db.executemany('INSERT INTO games VALUES(?,?,NULL)', [(i, json.dumps(s, sort_keys=True)) for i,s in enumerate(specifications)])


def take_next(path, stopped, *, process_index, client_index, clients):
    if clients not in (2,8) or process_index not in range(4 if clients==2 else 1) or client_index not in range(clients):
        raise ValueError('Only one CPU8 or four CPU2 services')
    with sqlite3.connect(path, timeout=5) as db:
        db.execute('BEGIN IMMEDIATE')
        if stopped.is_set():
            return None
        found = db.execute('SELECT position,spec FROM games WHERE assignment IS NULL ORDER BY position LIMIT 1').fetchone()
        if found is None:
            return None
        order, raw = found
        spec = json.loads(raw)
        slot = client_index
        assignment = dict(queue_index=order, game_id=spec['game_id'], worker_index=process_index*clients+client_index,
            process_index=process_index, client_id=f'worker_{client_index}', slot=slot,
            assigned_perf_counter=time.perf_counter(), assigned_unix=time.time())
        db.execute('UPDATE games SET assignment=? WHERE position=? AND assignment IS NULL',
                   (json.dumps(assignment, sort_keys=True), order))
        return spec, assignment


def assignments(path):
    with sqlite3.connect(path) as db:
        return [json.loads(row[0]) for row in db.execute('SELECT assignment FROM games WHERE assignment IS NOT NULL ORDER BY position')]


def inventory(path):
    """Read the closed committed queue without creating or changing it."""
    path=Path(path).resolve()
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
        return [dict(position=i,spec=json.loads(s),assignment=None if a is None else json.loads(a))
                for i,s,a in db.execute('SELECT position,spec,assignment FROM games ORDER BY position')]


def run_clients(clients, take, play, stopped, boundary=lambda:None, abort=lambda:None):
    """The actual thread loop also runs with fake clients in model-free tests.

    Failure stops new assignments, drains already active decisions under their
    original clocks, and never starts a replacement. The owner remains the hard
    process deadline/resource supervisor.
    """
    if clients not in (2,8):
        raise ValueError('Fixed two/eight clients')
    rows, errors, lock = [], [], threading.Lock()
    barrier = threading.Barrier(clients)
    def work(index):
        try:
            barrier.wait(timeout=75.)
            while not stopped.is_set():
                boundary()
                item = take(index)
                if item is None:
                    return
                result = play(index, *item)
                with lock:
                    rows.append(result)
                if result['status'] != 'completed':
                    stopped.set()
                    return
        except BaseException as error:
            stopped.set()
            with lock:
                errors.append({'client_index':index,'type':type(error).__name__,'error':str(error)})
    threads = [threading.Thread(target=work, args=(i,), name=f'pilot-collector-{i}', daemon=True) for i in range(clients)]
    for thread in threads:
        thread.start()
    try:
        while any(thread.is_alive() for thread in threads):
            boundary()
            for thread in threads:
                thread.join(.025)
    except BaseException:
        stopped.set(); abort()
        deadline = time.monotonic()+80.
        for thread in threads:
            thread.join(max(0.,deadline-time.monotonic()))
        if any(thread.is_alive() for thread in threads):
            raise RuntimeError('Owner must terminate undrained collector; preserve partial evidence')
        raise
    return dict(status='passed' if not stopped.is_set() and not errors else 'failed', outcomes=rows,
                errors=errors, drained=True, retries=0, replacements=0)
