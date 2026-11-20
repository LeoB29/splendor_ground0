"""Small bounded, read-only evidence context for the learning postmortem."""
import hashlib
import json
from pathlib import Path
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/splendorust_selfplay_postmortem_20261104_01'
PLAN = OUT / 'plan.json'
PLAN_SHA = '7db5bd6c33da26a61114faf2c13a6c4215bc6a2064a25e2e51bb8b294f38ccca'


def artifact(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size, sha256=sha)


def require(value, message):
    if not value:
        raise ValueError(message)


class Context:
    def __init__(self):
        self.started = time.monotonic()
        self.inputs = {}
        self.accepted = {}
        self.inventory_ready = False
        self.peak_rss = 0
        require(artifact(PLAN)['sha256'] == PLAN_SHA, 'Exact bounded postmortem plan')
        self.plan = self.read(PLAN)
        self.source = ROOT / self.plan['source']
        for key in ('source_acceptance', 'source_independent_review', 'source_freeze', 'source_result'):
            descriptor = self.plan[key]
            require(self.bind(ROOT / descriptor['path']) == descriptor, 'Accepted input identity: '+key)
        require(self.read(self.source/'final_closeout.json')['status'] == 'accepted_completed_verified', 'Accepted pilot')
        review = self.read(self.source/'independent_review.json')
        require(review['approved'] is True, 'Independent acceptance')
        seal = review['seal']
        require(self.bind(ROOT/seal['path']) == seal, 'Accepted saved-evidence seal')
        self.accepted = {item['path']:item for item in self.read(ROOT/seal['path'])['artifacts']}
        freeze = self.read(self.source/'freeze.json')
        for item in freeze['inputs']:
            previous = self.accepted.setdefault(item['path'], item)
            require(previous == item, 'Consistent accepted source and evidence inventories')
        self.trusted_late = {self.plan[key]['path']:self.plan[key] for key in
                            ('source_acceptance', 'source_independent_review', 'source_freeze', 'source_result')}
        self.trusted_late[seal['path']] = seal
        self.inventory_ready = True
        for descriptor in self.inputs.values():
            self.verify_accepted(descriptor)
        from tools.future_training_rules_20261021 import verify_runtime_sources
        self.runtime_sources = verify_runtime_sources()
        self.check()

    def bind(self, path):
        descriptor = artifact(path)
        self.verify_accepted(descriptor)
        previous = self.inputs.setdefault(descriptor['path'], descriptor)
        require(previous == descriptor, 'Input changed during diagnosis')
        return descriptor

    def verify_accepted(self, descriptor):
        if not self.inventory_ready:
            return
        expected = self.accepted.get(descriptor['path']) or self.trusted_late.get(descriptor['path'])
        if expected is not None:
            require(descriptor == expected, 'Input matches independently accepted inventory')
        elif descriptor['path'].startswith(self.plan['source']+'/'):
            require(False, 'Unsealed saved evidence is not an analysis input')

    def read(self, path):
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        self.bind(path)
        return json.loads(path.read_text(encoding='utf-8'))

    def check(self):
        require(time.monotonic()-self.started < self.plan['component_seconds'], 'Bounded component time')
        require(time.time() < self.plan['start_unix']+self.plan['wall_seconds'], 'Bounded overall postmortem time')
        self.peak_rss = max(self.peak_rss, psutil.Process().memory_info().rss)
        require(self.peak_rss < self.plan['maximum_rss_bytes'], 'Actual analysis memory limit')
        require('torch' not in sys.modules, 'No model runtime in saved-record postmortem')

    def finish(self, name, result):
        self.check()
        for descriptor in self.inputs.values():
            require(artifact(ROOT/descriptor['path']) == descriptor, 'Preserved analysis input')
        result = dict(result)
        result.update(schema='saved_learning_postmortem_component_v1', plan=artifact(PLAN),
                      inputs=sorted(self.inputs.values(), key=lambda x:x['path']),
                      runtime_sources=self.runtime_sources, seconds=time.monotonic()-self.started,
                      peak_rss_bytes=self.peak_rss, new_games=0, model_forward_calls=0,
                      training_updates=0, descriptive_only=True)
        path = OUT/name
        require(path.parent == OUT and path.suffix == '.json', 'Separate component output')
        content = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n'
        used = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
        require(used+len(content.encode()) < self.plan['maximum_output_bytes'], 'Actual output size bound')
        with path.open('x', encoding='utf-8') as stream:
            stream.write(content)
        return artifact(path)
