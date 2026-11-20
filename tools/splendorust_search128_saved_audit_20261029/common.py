"""Separate saved-evidence recovery; no change to any frozen experiment."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/splendorust_search128_saved_audit_20261029_01'
SOURCE = ROOT / 'outputs/splendorust_search128_resume_20261029_01'
SOURCE_SHA = '0e4eb49a3f32bf4e52480cd24d6472b7bbe42e131c85c289cc396ee8d4ff5aaf'
PACKAGE = 'tools.splendorust_search128_saved_audit_20261029'
PROTOCOL = ROOT / 'docs/engineering/splendorust_search128_saved_audit_20261029.md'
ROOTS = [ROOT / ('outputs/' + name) for name in (
    'splendorust_search128_strength_20261028_01',
    'splendorust_search128_recovery_20261028_01',
    'splendorust_search128_closure_check_20261029_01',
    'splendorust_search128_resume_20261029_01')]
GiB = 1024**3
LIMIT = 3600.
# The unchanged imported cleanup helper reconstructs its deadline using86400.
# This compatibility constant is not the new audit allowance (LIMIT above).
JOB_SECONDS = 86400.
MAX_RSS_BYTES = 2*GiB
MAX_OUTPUT_BYTES = 256*1024**2
ORIGINAL_MONO, ORIGINAL_UNIX = 474768.187, 1793218912.6661534
OWNER_EXIT = ROOT / 'outputs/splendorust_search128_resume_20261029_continuity/root_owner_exit_observation.json'


def require(value, message):
    if not value: raise ValueError('SAVED_AUDIT: '+message)


def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def now(): return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def artifact(path):
    path = Path(path)
    return dict(path=path.resolve().relative_to(ROOT).as_posix(), bytes=path.stat().st_size, sha256=file_hash(path))


def output_size(): return sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())


def save(path, value):
    path = Path(path)
    require(path.resolve().is_relative_to(OUT.resolve()), 'Publication must stay in separate audit output')
    path.parent.mkdir(parents=True, exist_ok=True)
    raw=(canonical(value)+'\n').encode()
    require(output_size()+len(raw)<MAX_OUTPUT_BYTES, 'Separate audit output cap')
    with path.open('xb') as stream: stream.write(raw)
    return artifact(path)


atomic_save = save


def verify_pins(items):
    for item in items:
        path=ROOT/item['path']
        require(artifact(path)==item, 'Pinned input changed: '+item['path'])


def deadline_pair(mono, wall, final):
    return (min(mono+LIMIT, final['monotonic_deadline'], ORIGINAL_MONO+86400),
            min(wall+LIMIT, final['unix_deadline'], ORIGINAL_UNIX+86400))


def guard_clock(clock):
    require(time.monotonic()<clock['deadline_monotonic'] and time.time()<clock['deadline_unix'],
            'One recovery audit clock exhausted; no acceptance')


def historical_failure(r):
    """Authenticate the exact known non-scientific failure and old real exits."""
    failure=r.read('postrun/independent_review_failure.json')
    terminal=r.read('postrun/independent_review_os_exit.json')
    owner=r.read('owner_failure.json'); invalid=r.read('postrun/acceptance_invalidated.json')
    completion=r.read('completion.json'); original_clock=r.read('postrun/review_owner_clock.json')
    final=r.read('stages/final_start.json'); launch=r.read('owner_launch.json')
    require(failure['status']=='failed_no_acceptance' and
        failure['error']=="ValueError('SEARCH128_REVIEW: Canonical project-relative path')" and
        'review_previous_closure' in failure['traceback'] and failure['games_run'] is False and
        failure['models_run'] is False, 'Exact saved path-schema failure only')
    require(terminal['exit_code']==1 and terminal['observed_by']=='owned Process handle' and
        terminal['cleanup_reason'] is None and terminal['component_sha256'] is None and
        owner['status']=='not_accepted' and owner['error']==invalid['reason']==
        "ValueError('Independent reader actual exit must pass')" and invalid['accepted'] is False,
        'Failed independent reader and old owner remain failed')
    require(completion['status']=='amended_recovered_verified' and completion['failure'] is None and
        terminal['freeze_sha256']==owner['freeze_sha256']==completion['freeze_sha256']==SOURCE_SHA,
        'Exact completed source study with failed independent acceptance')
    require(not (SOURCE/'owner_completion.json').exists() and
        not (SOURCE/'postrun/independent_completion_review.json').exists(), 'Never relabel old acceptance')
    proof=r.read_path(OWNER_EXIT)
    require(proof['actual_owner_exit_code']==1 and proof['actual_exit_observed'] is True and
        proof['session_closed'] is True and proof['actual_tool_session']==43971,
        'Observed actual failed launch-owner exit')
    observed={x['source']:x for x in proof['identities']}
    for name,pid,created in [('owner_launch.json',launch['pid'],launch['created']),
        ('postrun/independent_review_os_exit.json',terminal['reader_pid'],terminal['reader_created'])]:
        item=observed.get(name,{})
        require(item.get('pid')==pid and item.get('created')==created and item.get('original_identity_absent') is True,
                'Observed old exit identity binds exact owner and reader')
    require(final['cap_seconds']==14210 and
        final['monotonic_deadline']==min(final['monotonic']+14210,ORIGINAL_MONO+86400) and
        final['unix_deadline']==min(final['unix_time']+14210,ORIGINAL_UNIX+86400), 'Original final clock unchanged')
    require(original_clock['deadline_monotonic']==min(original_clock['review_monotonic_origin']+3600,final['monotonic_deadline'],ORIGINAL_MONO+86400) and
        original_clock['deadline_unix']==min(original_clock['review_unix_origin']+3600,final['unix_deadline'],ORIGINAL_UNIX+86400),
        'Historical expired review clock preserved')
    require(original_clock['guardian_claim_sha256']==r.pin(SOURCE/'guardian_claim.json')['sha256'] and
        original_clock['completion_sha256']==r.pin(SOURCE/'completion.json')['sha256'], 'Historical clock byte binding')
    absent=[r.absent(dict(pid=terminal['reader_pid'],created=terminal['reader_created'])),
            r.absent(dict(pid=launch['pid'],created=launch['created']))]
    return dict(status='authenticated_original_failure_preserved',original_review_clock=original_clock,
        original_final_clock=final,reader_exit_code=1,owner_exit_code=1,owned_identities_absent=absent,
        original_acceptance=False,source_attempt=artifact(SOURCE/'freeze.json'))
