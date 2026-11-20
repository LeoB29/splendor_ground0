"""Separate Stage B boundary inheriting the original shared study clock."""
import json
import os
from pathlib import Path

from tools.move_evaluation_20261023.common import (
    ROOT, PREP, BUILD, GiB, StudyWorker, require, canonical, digest,
    read, now, file_hash, artifact, verify, selector_payload)

OUT = ROOT/'outputs/move_evaluation_inference_opt_20261023_01'
SOURCE = ROOT/'outputs/move_evaluation_inference_20261023_01'
STAGE_A = ROOT/'outputs/move_evaluation_inference_recovery_20261023_01'
STAGE_A_STATUS = ROOT/'outputs/move_evaluation_inference_recovery_20261023_status_01'
STAGE_A_FREEZE_SHA = 'c39c8bc648febf67af84c4322e0d42c24e401d498e0bef3ef33f88a88cd5ede7'
STAGE_A_CLOSEOUT_SHA = '5e9b0c73a550156aa57d9b7ada1dc8ee09fe52f9318d07301eb30ad53edac7c1'
PLAN_PATH = SOURCE/'study_plan.json'
PLAN_SHA = '45078eb59079b7d7178029df28d9796f7ea7983124129501e5390bf710477ab4'
ORIGINAL = OLD = ROOT/'outputs/move_evaluation_20261023_01'
PREVIOUS = ROOT/'outputs/move_evaluation_reuse_20261023_01'
PREVIOUS_FREEZE_SHA = 'd1cca323f022ab0531217fe7bdf2175bd329e1f23c348dea951762ca62dd103d'
PROTOCOL = ROOT/'docs/engineering/move_evaluation_inference_opt_20261023.md'
STAGE_A_REPORT = ROOT/'docs/engineering/move_evaluation_inference_20261023_result.md'
IMPLEMENTATION_ID = 'entity_prebound_fused_transformer_cpu_float32_20261023_v1'
ROOT_INDICES = (0, 6, 12)
JOB_SECONDS = 3600
FINAL_RESERVE = 300
REQUEST_SECONDS = 600
PANEL_REPETITIONS = 8


def save(path, value):
    p = Path(path).resolve()
    require(p.is_relative_to(OUT.resolve()), 'Inference optimization output boundary')
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return artifact(p)


def static_artifact(value):
    return {k: value[k] for k in ('path', 'sha256', 'bytes')}
