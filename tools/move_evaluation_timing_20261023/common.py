"""New evidence boundary; all scientific dependencies remain immutable."""
import json
import os
from pathlib import Path

from tools.move_evaluation_20261023.common import (
    ROOT, PREP, BUILD, GiB, StudyWorker, require, canonical, digest,
    read, now, file_hash, artifact, verify, selector_payload)

OUT = ROOT/'outputs/move_evaluation_timing_20261023_01'
OLD = ROOT/'outputs/move_evaluation_20261023_01'
PREVIOUS = ROOT/'outputs/move_evaluation_profile_20261023_01'
PROTOCOL = ROOT/'docs/engineering/move_evaluation_timing_20261023.md'
OLD_FREEZE_SHA = 'd059b8b2dbf5ebebc0543a4df8cbbe94b1a92bc5526a5356d15c3dcf98e30d8e'
PREVIOUS_FREEZE_SHA = '6f3757aa2ac045a3263e683fe4ec675c616e434fb9351cc88cadf7f9e5d9fd68'
ROOT_INDICES = (0, 6, 12)
JOB_SECONDS = 3600
FINAL_RESERVE = 300
REQUEST_SECONDS = 600


def save(path, value):
    p = Path(path).resolve()
    require(p.is_relative_to(OUT.resolve()), 'Timing output boundary')
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return artifact(p)


def static_artifact(value):
    return {k: value[k] for k in ('path', 'sha256', 'bytes')}
