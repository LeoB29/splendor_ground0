"""Additive evidence boundary for the actual-model inference diagnosis."""
import json
import os
from pathlib import Path

from tools.move_evaluation_20261023.common import (
    ROOT, PREP, BUILD, GiB, StudyWorker, require, canonical, digest,
    read, now, file_hash, artifact, verify, selector_payload)

OUT = ROOT/'outputs/move_evaluation_inference_20261023_01'
STATUS = ROOT/'outputs/move_evaluation_inference_20261023_status_01'
ORIGINAL = OLD = ROOT/'outputs/move_evaluation_20261023_01'
PREVIOUS = ROOT/'outputs/move_evaluation_reuse_20261023_01'
PREVIOUS_FREEZE_SHA = 'd1cca323f022ab0531217fe7bdf2175bd329e1f23c348dea951762ca62dd103d'
PROTOCOL = ROOT/'docs/engineering/move_evaluation_inference_20261023.md'
IMPLEMENTATION_ID = 'entity_inference_scopes_cpu_float32_20261023_v1'
ROOT_INDICES = (0, 6, 12)
JOB_SECONDS = 900
FINAL_RESERVE = 120
REQUEST_SECONDS = 120
PANEL_REPETITIONS = 8


def save(path, value):
    p = Path(path).resolve()
    require(p.is_relative_to(OUT.resolve()), 'Inference output boundary')
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return artifact(p)


def static_artifact(value):
    return {k: value[k] for k in ('path', 'sha256', 'bytes')}
