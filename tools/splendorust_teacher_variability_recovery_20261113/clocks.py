"""Rebase watchdog clocks after reboot without refunding any study time.

Old monotonic readings are retained as evidence only. Absolute UTC deadlines
remain authoritative and are mapped once onto the new boot's monotonic epoch.
Per-decision clocks are never resumed or rebased by this module.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_OUT = ROOT / 'outputs/splendorust_teacher_variability_20261113_01'
ORIGINAL_FREEZE_SHA = '60239d1f7221aa8f25557a6ebae5b18218d24c521fd086dceb0f796ed6aada54'
TOTAL_SECONDS = 24 * 3600
MODEL_SECONDS = 18 * 3600
ROLE_CAPS = dict(admission=2*3600, diagnostic=12*3600, reproduction=2*3600,
                 analysis=4*3600, acceptance=4*3600, fixture=30)
MODEL_ROLES = ('admission', 'diagnostic', 'reproduction')
# Windows monotonic reads have finite resolution. A backwards wall movement
# exceeding this sampling allowance fails closed; no amount extends a bound.
CLOCK_SAMPLING_TOLERANCE = .25
BOOT_ESTIMATE_TOLERANCE = 2.
SCHEMA = 'teacher_variability_rebased_clock_v1'


def require(condition, message):
    if not condition:
        raise ValueError('TEACHER_VARIABILITY_RECOVERY_CLOCK: ' + message)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _read_bound(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'Duplicate original claim key')
            result[key] = value
        return result
    def reject(value):
        raise ValueError('Nonfinite original clock JSON: ' + value)
    path = Path(path).resolve()
    require(path.is_relative_to(ORIGINAL_OUT.resolve()), 'Original claim path boundary')
    before = path.stat(); raw = path.read_bytes(); after = path.stat()
    require(0 < len(raw) <= 4*1024**2 and
            (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Stable bounded original clock receipt')
    item = dict(path=path.relative_to(ROOT).as_posix(), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    return json.loads(raw, object_pairs_hook=unique, parse_constant=reject), item


def _old_limits(value):
    require(type(value) is dict and set(value) == {'monotonic_origin', 'unix_origin', 'monotonic_deadline', 'unix_deadline'},
            'Original four-field dual clock')
    require(all(_finite(x) for x in value.values()), 'Finite original clocks')
    require(value['monotonic_origin'] < value['monotonic_deadline'] and value['unix_origin'] < value['unix_deadline'],
            'Original positive clock span')
    require(abs((value['monotonic_deadline'] - value['monotonic_origin']) -
                (value['unix_deadline'] - value['unix_origin'])) <= 1e-5, 'Original dual durations agree')


def original_deadlines():
    """Read and bind the immutable original study and active-stage claims."""
    freeze, frozen = _read_bound(ORIGINAL_OUT/'freeze.json')
    require(frozen['sha256'] == ORIGINAL_FREEZE_SHA, 'Exact original freeze')
    launcher, launch_item = _read_bound(ORIGINAL_OUT/'launcher_claim.json')
    root, root_item = _read_bound(ORIGINAL_OUT/'claim.json')
    diagnostic, diagnostic_item = _read_bound(ORIGINAL_OUT/'jobs/diagnostic/owner.json')
    require(launcher['freeze'] == root['freeze'] == frozen, 'Original claims bind exact freeze')
    require(launcher['limits'] == root['limits'], 'Original launcher/root share clocks')
    require(diagnostic['freeze_sha256'] == ORIGINAL_FREEZE_SHA and diagnostic['role'] == 'diagnostic'
            and diagnostic['windows_job_assigned'] is True and diagnostic['parent'] == root['identity'],
            'Original diagnostic ownership and freeze')
    global_limits, stage = launcher['limits'], diagnostic['limits']
    _old_limits(global_limits); _old_limits(stage)
    design = freeze['design']
    require(design['total_seconds'] == TOTAL_SECONDS and design['model_seconds'] == MODEL_SECONDS
            and design['job_caps']['diagnostic'] == ROLE_CAPS['diagnostic'], 'Original frozen time allowances')
    start = global_limits['unix_origin']
    require(abs(global_limits['unix_deadline'] - (start + TOTAL_SECONDS)) <= 1e-6, 'Unmodified original global UTC cutoff')
    model = start + MODEL_SECONDS
    require(start <= stage['unix_origin'] < stage['unix_deadline'] and
            abs(stage['unix_deadline'] - min(stage['unix_origin'] + ROLE_CAPS['diagnostic'], model,
                                           global_limits['unix_deadline'])) <= 1e-6, 'Unmodified original diagnostic UTC cutoff')
    value = dict(schema='teacher_variability_original_deadlines_v1', original_freeze_sha256=ORIGINAL_FREEZE_SHA,
                 original_unix_origin=start, global_unix_deadline=global_limits['unix_deadline'],
                 model_unix_deadline=model, diagnostic_unix_deadline=stage['unix_deadline'],
                 original_diagnostic_unix_origin=stage['unix_origin'],
                 original_monotonic_origin=global_limits['monotonic_origin'],
                 original_monotonic_deadline=global_limits['monotonic_deadline'],
                 original_diagnostic_monotonic_origin=stage['monotonic_origin'],
                 original_diagnostic_monotonic_deadline=stage['monotonic_deadline'],
                 original_bindings=[frozen, launch_item, root_item, diagnostic_item])
    _validate_original(value)
    return value


def _validate_original(value):
    require(type(value) is dict and value.get('schema') == 'teacher_variability_original_deadlines_v1'
            and value.get('original_freeze_sha256') == ORIGINAL_FREEZE_SHA, 'Original deadline contract')
    names = ('original_unix_origin', 'global_unix_deadline', 'model_unix_deadline', 'diagnostic_unix_deadline',
             'original_diagnostic_unix_origin', 'original_monotonic_origin', 'original_monotonic_deadline',
             'original_diagnostic_monotonic_origin', 'original_diagnostic_monotonic_deadline')
    require(all(_finite(value.get(k)) for k in names), 'Finite original deadline metadata')
    start = value['original_unix_origin']
    require(abs(value['global_unix_deadline'] - start - TOTAL_SECONDS) <= 1e-6
            and abs(value['model_unix_deadline'] - start - MODEL_SECONDS) <= 1e-6, 'Original model/global allowances unchanged')
    require(start <= value['original_diagnostic_unix_origin'] < value['diagnostic_unix_deadline'] <= value['model_unix_deadline'],
            'Original diagnostic within model/global cutoff')
    require(abs(value['diagnostic_unix_deadline'] - min(value['original_diagnostic_unix_origin'] + ROLE_CAPS['diagnostic'],
                 value['model_unix_deadline'], value['global_unix_deadline'])) <= 1e-6, 'Original diagnostic duration unchanged')
    require(type(value.get('original_bindings')) is list and len(value['original_bindings']) == 4
            and len({r['path'] for r in value['original_bindings']}) == 4, 'Four distinct original clock artifacts')
    for row in value['original_bindings']:
        require(set(row) == {'path', 'bytes', 'sha256'} and type(row['path']) is str and type(row['bytes']) is int
                and row['bytes'] > 0 and type(row['sha256']) is str and len(row['sha256']) == 64,
                'Original clock artifact descriptor')


def _now(now_mono=None, now_unix=None, boot_unix=None):
    require((now_mono is None) == (now_unix is None), 'Both clock readings supplied together')
    if now_mono is None:
        now_mono, now_unix = time.monotonic(), time.time()
    if boot_unix is None:
        import psutil
        boot_unix = psutil.boot_time()
    require(all(_finite(x) for x in (now_mono, now_unix, boot_unix)) and now_mono >= 0
            and 0 < boot_unix <= now_unix, 'Finite new-boot clock readings')
    return float(now_mono), float(now_unix), float(boot_unix)


def rebase(unix_deadline, *, now_mono, now_unix):
    """Map one unchanged UTC deadline to this boot; an expired bound is rejected."""
    require(all(_finite(x) for x in (unix_deadline, now_mono, now_unix)) and now_mono >= 0, 'Finite rebase values')
    require(now_unix < unix_deadline, 'Original absolute deadline already expired')
    return now_mono + (unix_deadline - now_unix)


def _scope():
    return dict(original_utc_deadlines_preserved=True, original_deadlines_extended=False, downtime_charged=True,
                old_monotonic_epoch_used_for_enforcement=False, per_decision_clocks_rebased=False,
                frozen_resource_limits_changed=False, recovery_only=True)


def fresh_global_limits(original=None, *, now_mono=None, now_unix=None, boot_unix=None):
    bound = original_deadlines()
    require(original is None or original == bound, 'Supplied original deadlines equal bound claims')
    original = deepcopy(bound)
    mono, wall, boot = _now(now_mono, now_unix, boot_unix)
    require(wall >= original['original_diagnostic_unix_origin'] and boot > original['original_unix_origin'],
            'New boot follows the original launched study; no wall-origin rollback')
    deadline = rebase(original['global_unix_deadline'], now_mono=mono, now_unix=wall)
    anchor = dict(monotonic=mono, unix=wall, boot_unix_estimate=boot,
                  clock_sampling_tolerance_seconds=CLOCK_SAMPLING_TOLERANCE,
                  boot_estimate_tolerance_seconds=BOOT_ESTIMATE_TOLERANCE)
    return dict(schema=SCHEMA, role='root', monotonic_origin=mono, unix_origin=wall,
                monotonic_deadline=deadline, unix_deadline=original['global_unix_deadline'],
                original_unix_origin=original['original_unix_origin'], original_deadlines=original,
                original_deadlines_sha256=_digest(original), recovery_anchor=anchor,
                recovery_anchor_sha256=_digest(anchor), maximum_seconds=original['global_unix_deadline'] - wall,
                scope=_scope())


def _validate_limits(limits):
    require(type(limits) is dict and limits.get('schema') == SCHEMA and limits.get('role') in ('root', *ROLE_CAPS),
            'Recovery dual-clock contract and role')
    require(limits.get('scope') == _scope(), 'No reset/extension or resource policy change')
    original = limits['original_deadlines']; _validate_original(original)
    require(limits['original_deadlines_sha256'] == _digest(original)
            and limits['original_unix_origin'] == original['original_unix_origin'], 'Bound original deadline metadata')
    anchor = limits['recovery_anchor']
    require(type(anchor) is dict and set(anchor) == {'monotonic', 'unix', 'boot_unix_estimate',
            'clock_sampling_tolerance_seconds', 'boot_estimate_tolerance_seconds'} and
            limits['recovery_anchor_sha256'] == _digest(anchor), 'Immutable new-boot anchor')
    require(all(_finite(anchor.get(k)) for k in ('monotonic', 'unix', 'boot_unix_estimate'))
            and anchor['monotonic'] >= 0 and original['original_unix_origin'] < anchor['boot_unix_estimate'] <= anchor['unix'],
            'New-boot anchor follows original launch')
    require(anchor['clock_sampling_tolerance_seconds'] == CLOCK_SAMPLING_TOLERANCE
            and anchor['boot_estimate_tolerance_seconds'] == BOOT_ESTIMATE_TOLERANCE, 'Fixed clock sampling tolerances')
    require(all(_finite(limits.get(k)) for k in ('monotonic_origin', 'unix_origin', 'monotonic_deadline', 'unix_deadline', 'maximum_seconds')),
            'Finite enforced clocks')
    require(anchor['monotonic'] <= limits['monotonic_origin'] < limits['monotonic_deadline']
            and anchor['unix'] <= limits['unix_origin'] < limits['unix_deadline'] <= original['global_unix_deadline'],
            'Recovery clocks within original global cutoff')
    # This upper bound is tied to the single original rebase, not each stage.
    require(limits['monotonic_deadline'] <= anchor['monotonic'] + original['global_unix_deadline'] - anchor['unix'] + 1e-6,
            'Monotonic global cutoff never extends')
    role = limits['role']
    if role in MODEL_ROLES:
        require(limits['unix_deadline'] <= original['model_unix_deadline']
                and limits['monotonic_deadline'] <= anchor['monotonic'] + original['model_unix_deadline'] - anchor['unix'] + 1e-6,
                'Original model UTC and anchored monotonic cutoff')
    if role == 'diagnostic':
        require(limits['unix_deadline'] <= original['diagnostic_unix_deadline']
                and limits['monotonic_deadline'] <= anchor['monotonic'] + original['diagnostic_unix_deadline'] - anchor['unix'] + 1e-6,
                'Original diagnostic UTC and anchored monotonic cutoff')
    if role != 'root':
        require(0 < limits['maximum_seconds'] <= ROLE_CAPS[role]
                and limits['unix_deadline'] <= limits['unix_origin'] + limits['maximum_seconds'] + 1e-6
                and limits['monotonic_deadline'] <= limits['monotonic_origin'] + limits['maximum_seconds'] + 1e-6,
                'No new stage-cap extension')


def check_deadlines(limits, *, now_mono=None, now_unix=None, boot_unix=None):
    _validate_limits(limits)
    mono, wall, boot = _now(now_mono, now_unix, boot_unix)
    anchor = limits['recovery_anchor']
    require(limits['monotonic_origin'] <= mono < limits['monotonic_deadline']
            and limits['unix_origin'] <= wall < limits['unix_deadline'], 'Original rebased dual deadline expired or origin reversed')
    require(abs(boot - anchor['boot_unix_estimate']) <= BOOT_ESTIMATE_TOLERANCE, 'Boot changed after recovery anchor')
    require(wall - anchor['unix'] >= mono - anchor['monotonic'] - CLOCK_SAMPLING_TOLERANCE,
            'Wall clock moved backwards relative to monotonic recovery anchor')
    return True


def stage_limits(parent, seconds, role, *, now_mono=None, now_unix=None, boot_unix=None):
    require(role in ROLE_CAPS and _finite(seconds) and 0 < seconds <= ROLE_CAPS[role], 'Explicit bounded recovery stage')
    mono, wall, boot = _now(now_mono, now_unix, boot_unix)
    check_deadlines(parent, now_mono=mono, now_unix=wall, boot_unix=boot)
    original = parent['original_deadlines']
    require(original == original_deadlines(), 'Original claims unchanged before stage starts')
    cutoff = original['global_unix_deadline']
    if role in MODEL_ROLES: cutoff = min(cutoff, original['model_unix_deadline'])
    if role == 'diagnostic': cutoff = min(cutoff, original['diagnostic_unix_deadline'])
    wall_deadline = min(wall + seconds, parent['unix_deadline'], cutoff)
    anchor = parent['recovery_anchor']
    mono_deadline = min(mono + seconds, parent['monotonic_deadline'],
                        rebase(wall_deadline, now_mono=mono, now_unix=wall),
                        anchor['monotonic'] + cutoff - anchor['unix'])
    value = dict(parent, role=role, monotonic_origin=mono, unix_origin=wall,
                 monotonic_deadline=mono_deadline, unix_deadline=wall_deadline, maximum_seconds=seconds)
    check_deadlines(value, now_mono=mono, now_unix=wall, boot_unix=boot)
    return value


def fixture_limits(seconds=30, *, now_mono=None, now_unix=None, boot_unix=None):
    """Short detached process fixture, still inside the original global deadline."""
    mono, wall, boot = _now(now_mono, now_unix, boot_unix)
    parent = fresh_global_limits(now_mono=mono, now_unix=wall, boot_unix=boot)
    return stage_limits(parent, seconds, 'fixture', now_mono=mono, now_unix=wall, boot_unix=boot)
