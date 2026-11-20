"""New evidence boundary, immutable identities and actual resource ceilings."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/splendorust_learning_diagnostic_20261106_01'
PLAN = OUT / 'plan.json'
SELECTION = OUT / 'selection.json'
PROTOCOL = ROOT / 'docs/engineering/splendorust_learning_diagnostic_execution_20261106.md'
PREPARATION = ROOT / 'docs/engineering/splendorust_learning_diagnostic_preparation_20261106.md'
PACKAGE = 'tools.splendorust_learning_diagnostic_20261106'
TOTAL_SECONDS = 12 * 3600
MODEL_SECONDS = 8 * 3600
QUALIFICATION_SECONDS = 3600
FIT_SECONDS = 3600
EVALUATION_SECONDS = 2 * 3600
AUDIT_SECONDS = 4 * 3600
MAX_RSS = 4 * 1024**3
MAX_OUTPUT = 4 * 1024**3
MIN_DISK = 1024**3
PARENT_SHA256 = 'ef8a4521cd6c03c7075f15efee23f4cde6ec94d1b5398765cf0c24a09ad745cb'


def require(value, message):
    if not value:
        raise ValueError('LEARNING_DIAGNOSTIC: ' + message)


def now(): return datetime.now(timezone.utc).isoformat()
def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
def digest(value): return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return {'path': path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
            'bytes': path.stat().st_size, 'sha256': sha(path)}


def resolve(item): return (ROOT / item['path']).resolve()


def verify(items, boundary=lambda: None):
    for item in items:
        boundary()
        path = resolve(item)
        require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'],
                'Frozen artifact changed: ' + item['path'])


def save(path, value):
    path = Path(path).resolve()
    require(path.is_relative_to(OUT), 'Write only within new evidence root')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(canonical(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    return artifact(path)


def atomic_json(path, value):
    path = Path(path).resolve()
    require(path.is_relative_to(OUT) and path.name in ('stage.json', 'progress.json'), 'Only mutable status')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        stream.write(canonical(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    from tools.splendorust_gpu_live_20261105_03.status_io import atomic_replace_status
    atomic_replace_status(temporary, path)


def nested_artifacts(value):
    if isinstance(value, dict):
        if {'path', 'bytes', 'sha256'} <= set(value):
            yield {key: value[key] for key in ('path', 'bytes', 'sha256')}
        else:
            for child in value.values(): yield from nested_artifacts(child)
    elif isinstance(value, list):
        for child in value: yield from nested_artifacts(child)


def fits():
    first = [('dense', 'joint'), ('broad', 'joint'), ('dense', 'policy'), ('broad', 'policy')]
    return [{'fit_id': f'r{rep}_{breadth}_{scope}', 'repetition': rep, 'breadth': breadth, 'scope': scope}
            for rep in (0, 1) for breadth, scope in (first if rep == 0 else list(reversed(first)))]
