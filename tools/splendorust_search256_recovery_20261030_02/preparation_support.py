"""Read-only identity, clean-stop and resource provenance; no model imports."""
from datetime import datetime
import hashlib
import math
from pathlib import Path
import time
from tools.splendorust_search256_recovery_20261030 import preparation_support as original

ROOT = Path(__file__).resolve().parents[2]
OLD_OUT = ROOT/'outputs/splendorust_search256_strength_20261029_02'
PREVIOUS_OUT = ROOT/'outputs/splendorust_search256_recovery_20261030_01'
DIAGNOSIS_OUT = ROOT/'outputs/splendorust_search256_recovery_20261030_01_postfailure'
PREVIOUS_SHA = 'af5174b5659914751776d636bf49cd13d7818cfaed88b4b161daa685da522a52'
BOUNDARY_SHA = '3b725f7c59d957a66ec243d9338900aca58e257ca95d0af8f916f10ee02bf8bd'
STATUS_SHA = 'c257e247bce8479bdca62c86cd8d5155048b234f2ea29ae47357f7f508bd1b82'
WORKERS = ('main_research_g294_challenger', 'main_research_g294_opponent')
_require, _read, _identity = original._require, original._read, original._identity


def _hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _pair(identity):
    return identity['pid'], identity['created']


def previous_exit_provenance():
    """Authenticate failed owner exits without turning failure into success."""
    _require(_hash(PREVIOUS_OUT/'freeze.json') == PREVIOUS_SHA, 'Exact previous recovery freeze')
    _require(_hash(DIAGNOSIS_OUT/'boundary_01.json') == BOUNDARY_SHA and
             _hash(DIAGNOSIS_OUT/'status_01.json') == STATUS_SHA, 'Exact saved clean-boundary diagnosis')
    status, boundary = _read(DIAGNOSIS_OUT/'status_01.json'), _read(DIAGNOSIS_OUT/'boundary_01.json')
    _require(status['guardian_actual_exit_code'] == 1 and status['guardian_exec_session'] == 42201 and
             status['guardian_exit_observation_chunk'] == 'ed1d4f', 'Actual retained guardian exit observation')
    for name, expected in (('execution.json', 'runner_failed'), ('watchdog_execution.json', 'watchdog_failed')):
        receipt = _read(PREVIOUS_OUT/name)
        _require(receipt['freeze_sha256'] == PREVIOUS_SHA and receipt['status'] == expected and
                 receipt['exit_code'] == 1 and not receipt['cleanup_errors'] and
                 not receipt['cleanup_receipt']['errors'] and not receipt['cleanup_receipt']['unconfirmed'],
                 'Previous failed owner exit with confirmed cleanup')
    for name, digest in status['evidence'].items():
        _require(Path(name).name == name and _hash(PREVIOUS_OUT/name) == digest,
                 'Saved status binds exact failure/completion/exit receipts')
    completion, failure = _read(PREVIOUS_OUT/'completion.json'), _read(PREVIOUS_OUT/'failure.json')
    _require(completion['status'] == 'incomplete' and completion['failure'] == failure['error'] and
             failure['stage'] == 'main' and 'Cumulative frozen RAM wait exceeded' in failure['error'],
             'Original resource stop remains incomplete')
    _require(boundary['freeze_sha256'] == PREVIOUS_SHA and boundary['completed_main_games'] == 294 and
             boundary['accepted_current_moves'] == 50 and
             boundary['bounded_reverification']['actual_exit_code'] == 0 and
             boundary['bounded_reverification']['model_calls'] == 0 and
             boundary['next_move']['ply'] == 50 and boundary['next_move']['quota'] == 128 and
             boundary['next_move']['role'] == 'opponent' and
             all(boundary['next_move'][key] is False for key in
                 ('input_created', 'request_dispatched', 'move_clock_started', 'additional_retry_required')),
             'Exact clean between-move continuation, no second retry')
    return dict(status='verified_failed_owner_exits_and_clean_boundary', runner_exit_code=1,
        watchdog_exit_code=1, guardian_exit_code=1, original_attempt_status='incomplete',
        completed_main_games=294, prefix_moves=50, next_move_started=False,
        new_failed_decision_retry=False, boundary_sha256=BOUNDARY_SHA, status_sha256=STATUS_SHA)


def recorded_previous_identities():
    previous_exit_provenance()
    guardian, watchdog, runner = (_read(PREVIOUS_OUT/name) for name in
        ('guardian_claim.json', 'watchdog_claim.json', 'launch.json'))
    owner = _read(PREVIOUS_OUT/'guardian_owner.json')
    for receipt in (guardian, watchdog, runner):
        _require(receipt['freeze_sha256'] == PREVIOUS_SHA, 'Previous owner freeze identity')
    g = _identity(guardian['guardian_pid'], guardian['guardian_created'])
    w = _identity(watchdog['watchdog_pid'], watchdog['watchdog_created'])
    r = _identity(runner['pid'], runner['created'])
    _require(owner['guardian'] == guardian and owner['watchdog'] == w and
             _identity(watchdog['guardian_pid'], watchdog['guardian_created']) == g,
             'Previous owner chain identities agree')
    result = [g, w, r]
    for name, route, quota in ((WORKERS[0], 'research', 256), (WORKERS[1], 'frozen', 128)):
        folder = PREVIOUS_OUT/'workers'/name
        initial, latest = _read(folder/'initialization.json'), _read(folder/'request-00000026.json')
        _require(initial['status'] == latest['status'] == 'ok' and initial['accepted'] is latest['accepted'] is True and
                 latest['request_id'] == 'request-00000026', 'Accepted latest previous worker receipt')
        tree = original._process_tree(initial, startup=True)
        _require(tree == original._process_tree(latest), 'Previous startup/latest process identities agree')
        startup, placement = initial['result'], _read(folder/'placement_admission.json')
        _require(startup['freeze_sha256'] == PREVIOUS_SHA and startup['route'] == route and
                 startup['request_quotas'] == [quota] and startup['pid'] == tree['python']['pid'] and
                 startup['native_pids'] == {role: tree[role]['pid'] for role in ('reference', 'search')},
                 'Previous actual player/native identity')
        _require(tree['python']['parent_pid'] == r['pid'] and
                 _identity(placement['player_pid'], placement['player_created']) == initial['identity'] and
                 _identity(placement['supervisor']['pid'], placement['supervisor']['created']) == r,
                 'Previous game workers owned by previous runner')
        cleanup = _read(folder/'cleanup.json')
        members = [_identity(tree[role]['pid'], tree[role]['created']) for role in sorted(tree)]
        _require(cleanup['status'] == 'passed' and not cleanup['alive'] and not cleanup['errors'] and
                 {_pair(item) for item in cleanup['confirmed']} == {_pair(item) for item in members},
                 'All six previous worker/helper exits were confirmed')
        _require(not any((folder/sub/name).exists() for sub in ('', 'inputs', 'parent_profiles', 'placement_requests')
            for name in ('request-00000027.json', 'request-00000027_child_profile.json')),
            'No input, receipt or sidecar after the clean stop')
        result.extend(members)
    boundary = _read(DIAGNOSIS_OUT/'boundary_01.json')
    _require({_pair(item) for item in result[3:]} == {_pair(item) for item in boundary['current_worker_helper_identities']},
             'Same twelve independently diagnosed previous worker/helper identities')
    _require(len(result) == len({_pair(item) for item in result}) == 15, 'Exactly fifteen previous identities')
    return result


def recorded_old_identities():
    identities = original.recorded_old_identities() + recorded_previous_identities()
    _require(len(identities) == len({_pair(item) for item in identities}) == 30,
             'Both prior ownership trees retain thirty exact identities')
    return identities


def old_owner_identities():
    return [original._absent(identity) for identity in recorded_old_identities()]


def verify_old_absence():
    return old_owner_identities()


def old_tree_counts():
    original_counts = original.old_tree_counts()
    previous = original._tree_count(PREVIOUS_OUT)
    _require(previous == dict(qualification=0, operational=0, main=120, worker_trees=120,
        graph_builds=120, graph_parities=120, cleanup_receipts=120, unclosed_workers=[]),
        'Exact previous recovery lifecycle counts')
    return dict(original_attempts=original_counts, previous_recovery=previous,
        used_worker_trees=608, used_graph_builds=608, completed_graph_parities=607,
        recovery_replacement_trees=2, remaining_complete_game_trees=210,
        reserved_replay_trees=48, maximum_new_worker_trees=260, maximum_combined_worker_trees=868)


def ram_wait_accounting():
    """Overcharge all elapsed time after the final durable RAM receipt.

    Fast admissions accrue unsaved time too. The interval deliberately includes
    later decisions, verification and cleanup, rather than omitting that time.
    """
    admissions = sorted((PREVIOUS_OUT/'memory_admissions').glob('*.json'))
    _require(len(admissions) == 6, 'Six preserved completed RAM admissions')
    last = _read(admissions[-1]); execution = _read(PREVIOUS_OUT/'execution.json')
    _require(last['cumulative_wait_seconds'] == 1722.90699999989, 'Last durable cumulative RAM wait')
    stamp = datetime.fromisoformat(last['time']).timestamp()
    elapsed = execution['unix_end'] - stamp
    _require(0 < elapsed < 120, 'Finite final unrecorded admission interval')
    bound = last['cumulative_wait_seconds'] + elapsed
    charged = math.ceil(bound)
    _require(charged == 1821, 'Conservative cumulative RAM carryforward')
    return dict(status='conservative_elapsed_upper_bound', last_durable_wait_seconds=last['cumulative_wait_seconds'],
        final_unrecorded_elapsed_upper_seconds=elapsed, cumulative_upper_bound_seconds=bound,
        charged_prior_ram_wait_seconds=charged, exact_actual_wait_seconds_known=False,
        includes_intervening_decisions_verification_cleanup=True, fast_admissions_not_silently_omitted=True,
        counters_reset=False)


def prior_evidence_census():
    """Metadata-only census; the owning preparation separately hashes files."""
    deadline = time.monotonic() + 60
    counts = []
    for folder, expected in ((PREVIOUS_OUT, (23163, 1043690409)), (DIAGNOSIS_OUT, (2, 12655))):
        files = size = 0
        for path in folder.rglob('*'):
            _require(time.monotonic() < deadline and not path.is_symlink(), 'Bounded regular prior evidence inventory')
            if path.is_file():
                files += 1; size += path.stat().st_size
                _require(files <= 24000, 'Bounded prior evidence file count')
        _require((files, size) == expected, 'Exact preserved previous evidence metadata')
        counts.append(dict(path=folder.relative_to(ROOT).as_posix(), files=files, bytes=size))
    return dict(trees=counts, earlier_evidence_bytes=4072159505,
        total_prior_evidence_bytes=4072159505 + sum(item['bytes'] for item in counts), model_calls=0)
