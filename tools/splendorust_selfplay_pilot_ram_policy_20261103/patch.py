"""Pure correction for the two exact frozen worker source files.

Apply this before a future separately reviewed namespace migration. This module
does not import a worker, change its globals, write a file, or launch a process.
The output retains its original imports and is therefore preparation material,
not an independently runnable continuation. A new owner/output/clock plan must
be bound before either corrected worker is used.
"""
import hashlib


FROZEN_WORKERS = {
    'source': {
        'path': 'tools/splendorust_selfplay_pilot_resume_20261103/worker.py',
        'sha256': '580de71be96c3f0012ddb5f17b05f9fbd05c9c9f2cafd55df3bbec605339f95f',
    },
    'assessment': {
        'path': 'tools/splendorust_selfplay_pilot_resume_20261103/assessment/worker.py',
        'sha256': '34e1910e176124ceebe42c4c4948b9e38811fa75068fed017d01c1a51363de0a',
    },
}
OLD_LINE = b"            require(psutil.virtual_memory().available>=2*1024**3,'Qualification available RAM below2GiB')"
NEW_LINE = b'            return None  # Available host RAM is advisory; external actual guards remain.'


def corrected_worker_source(role, original):
    """Return exactly one byte-level line replacement on authenticated input."""
    if role not in FROZEN_WORKERS or type(original) is not bytes:
        raise ValueError('Exact source or assessment worker bytes required')
    if hashlib.sha256(original).hexdigest() != FROZEN_WORKERS[role]['sha256']:
        raise ValueError('Frozen worker source identity mismatch')
    if original.count(OLD_LINE) != 1 or NEW_LINE in original:
        raise ValueError('Exactly one original available-RAM rejection required')
    result = original.replace(OLD_LINE, NEW_LINE, 1)
    if result.replace(NEW_LINE, OLD_LINE, 1) != original:
        raise ValueError('Correction must preserve every other source byte')
    return result
