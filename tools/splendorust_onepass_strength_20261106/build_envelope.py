"""One-time source construction; no models or process launches."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
source = ROOT / 'tools/splendorust_policy_head_20261104/qualification_control.py'
text = source.read_text(encoding='utf-8')
tree = ast.parse(text)
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Envelope')
body = ast.get_source_segment(text, node).replace('class Envelope:', 'class _AccountingEnvelope:', 1)
header = '''"""Qualified resource accounting with immutable inherited phase clocks.

The base accounting class is a source copy of the accepted policy-head envelope.
Only the separate SegmentEnvelope binds a predetermined owner phase instead of
starting/resetting phases. No model execution is implemented here.
"""
import shutil
import time
from pathlib import Path
import psutil
from . import common as c


def bytes_used():
    return sum(p.stat().st_size for p in c.OUT.rglob('*') if p.is_file())


'''
tail = '''


class SegmentEnvelope(_AccountingEnvelope):
    def __init__(self, claim_descriptor, phase_descriptor, segment_descriptor):
        c.verify([claim_descriptor, phase_descriptor, segment_descriptor])
        claim, phase, segment = (c.read(c.ROOT / x['path']) for x in
                                 (claim_descriptor, phase_descriptor, segment_descriptor))
        c.require(segment['claim'] == claim_descriptor and segment['phase_clock'] == phase_descriptor,
                  'Exact original claim and phase binding')
        super().__init__(claim)
        self.claim = claim_descriptor
        self.phase_receipt = phase_descriptor
        self.segment_claim = segment_descriptor
        self.segment_id = segment['segment_id']
        self.stage = phase['stage']
        c.require(self.stage in ('qualification', 'operational', 'main', 'replay', 'final'), 'Fixed root phase')
        c.require(phase['claim'] == claim_descriptor, 'Phase belongs to original owner')
        cap = c.STAGES[self.stage]
        cutoff = c.JOB_SECONDS if self.stage == 'final' else c.MODEL_CUTOFF_SECONDS
        c.require(phase['cap_seconds'] == cap and
                  phase['monotonic_deadline'] == min(phase['monotonic'] + cap, self.start + cutoff, self.end) and
                  phase['unix_deadline'] == min(phase['unix_time'] + cap, self.wall_start + cutoff,
                                                self.wall_start + c.JOB_SECONDS), 'Unreset original phase clocks')
        self.deadline, self.wall_deadline = phase['monotonic_deadline'], phase['unix_deadline']
        self.prior_worker_count = sum(1 for path in c.OUT.glob('r*/assessment/workers/*') if path.is_dir())
        self.enable_evidence_accounting()
        self.guard()

    def begin(self, stage):
        raise ValueError('Only the original root owner creates phase clocks')

    def admit(self):
        self.guard()
        if not self.admitted:
            c.save(c.OUT / 'memory_admissions' / (self.segment_id + '.json'),
                   dict(time=c.now(), observed_available_bytes=psutil.virtual_memory().available,
                        ram_reserve_policy='advisory_only', wait_seconds=0., cumulative_wait_seconds=0.,
                        stable_seconds=0., claim=self.claim, phase_clock=self.phase_receipt))
        self.admitted = True
        self.admissions += 1

    def finish(self):
        c.require(not self.resident, 'All segment workers closed before completion')
        self.last_resource_guard = -float('inf')
        self.guard()
        actual = bytes_used()
        c.require(actual <= self.accounted_bytes(), 'Unaccounted segment evidence')
        return dict(actual_evidence_bytes=actual, accounted_bytes=self.accounted_bytes(),
                    peak_combined_inference_rss=self.peak_combined_rss,
                    new_worker_lifecycles=len(self.worker_accounting),
                    prior_worker_lifecycles=self.prior_worker_count,
                    original_claim=self.claim, original_phase=self.phase_receipt)
'''
destination = Path(__file__).with_name('envelope.py')
with destination.open('x', encoding='utf-8', newline='\n') as stream:
    stream.write(header + body + tail)
print(destination)
