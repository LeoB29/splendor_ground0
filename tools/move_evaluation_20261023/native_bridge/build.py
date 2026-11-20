"""Offline, additive native build. Never invokes a model or the new executable."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'outputs/external_opponent_search_20261018_02'
OLD = SOURCE / 'native_build_v2'
BUILD = ROOT / 'outputs/move_evaluation_20261023_build_01'
LOCAL = Path(__file__).resolve().parent
PIN = '1828507b8d2f0e02bc03d102b9c6d29a54fa8d57'


def artifact(path):
    path = Path(path)
    with path.open('rb') as handle:
        sha = hashlib.file_digest(handle, 'sha256').hexdigest()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha}


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def main(resume=False):
    # Authenticate the exact already-qualified source files before copying them.
    profile = json.loads((ROOT / 'outputs/external_strength_screen_20261018_02/opponent_profile.json').read_text())
    expected = {item['path']: item for item in profile['artifacts']}
    copies = []
    upstream = SOURCE / 'splendorust'
    for path in sorted((OLD / 'crates').rglob('*')):
        if path.is_file():
            original = upstream / path.relative_to(OLD)
            actual = artifact(original)
            if actual != expected[actual['path']] or path.read_bytes() != original.read_bytes():
                raise ValueError('Pinned crate changed: ' + str(path))
            copies.append(actual)
    if resume:
        if not BUILD.is_dir() or (BUILD / 'build_receipt.json').exists():
            raise ValueError('Only an unaccepted build can resume')
        attempt_number = 1 + len(list(BUILD.glob('build_attempt_*.json')))
        shutil.copytree(BUILD / 'bridge/src', BUILD / f'failed_source_{attempt_number - 1:02d}')
        for path in sorted((BUILD / 'crates').rglob('*')):
            if path.is_file() and path.read_bytes() != (OLD / path.relative_to(BUILD)).read_bytes():
                raise ValueError('Unchanged crate copy mismatch')
    else:
        attempt_number = 1
        BUILD.mkdir(exist_ok=False)
        shutil.copytree(OLD / 'crates', BUILD / 'crates')
        shutil.copytree(OLD / 'cargo_home', BUILD / 'cargo_home')
        shutil.copyfile(OLD / 'Cargo.toml', BUILD / 'Cargo.toml')
        shutil.copyfile(OLD / 'Cargo.lock', BUILD / 'Cargo.lock')
        (BUILD / 'bridge/src').mkdir(parents=True)
        shutil.copyfile(OLD / 'bridge/Cargo.toml', BUILD / 'bridge/Cargo.toml')
    for name in ('main.rs', 'wire.rs'):
        shutil.copyfile(LOCAL / 'src' / name, BUILD / 'bridge/src' / name)
    toolchain = SOURCE / 'runtime/toolchain'
    cargo = toolchain / 'bin/cargo.exe'
    overrides = {'RUSTC': str(toolchain / 'bin/rustc.exe'), 'CARGO_HOME': str(BUILD / 'cargo_home'),
                 'CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER': str(toolchain / 'lib/rustlib/x86_64-pc-windows-gnu/bin/rust-lld.exe'),
                 'RUSTFLAGS': '-C linker-flavor=ld.lld -C link-self-contained=yes',
                 'CARGO_BUILD_JOBS': '1', 'CARGO_INCREMENTAL': '0'}
    argv = [str(cargo), 'build', '--release', '--offline', '--locked', '--target',
            'x86_64-pc-windows-gnu', '-p', 'splendor-observer-bridge']
    started = time.monotonic()
    suffix = '' if attempt_number == 1 else f'_{attempt_number:02d}'
    with (BUILD / f'build_stdout{suffix}.log').open('xb') as stdout, (BUILD / f'build_stderr{suffix}.log').open('xb') as stderr:
        completed = subprocess.run(argv, cwd=BUILD, env={**os.environ, **overrides},
                                   stdout=stdout, stderr=stderr, timeout=600,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    attempt = {'schema': 'move_evaluation_greedy_compile_v1', 'argv': argv,
               'environment': overrides, 'returncode': completed.returncode,
               'elapsed_seconds': time.monotonic() - started, 'model_inference_executed': False}
    attempt_path = BUILD / f'build_attempt_{attempt_number:02d}.json'
    save(attempt_path, attempt)
    if completed.returncode:
        raise RuntimeError('Compile failed; preserved build logs')
    runtime = [artifact(p) for p in sorted((BUILD / 'crates').rglob('*')) if p.is_file()]
    runtime += [artifact(LOCAL / 'src' / n) for n in ('main.rs', 'wire.rs')]
    runtime += [artifact(BUILD / 'bridge/src' / n) for n in ('main.rs', 'wire.rs')]
    runtime += [expected[name] for name in (
        'tools/external_strength_screen_20261018_02/bridge.py',
        'tools/external_strength_screen_20261018_02/splendorust_adapter.py',
        'outputs/external_opponent_search_20261018_02/splendorust/research/architecture_pivots/models.py',
        'outputs/external_opponent_search_20261018_02/splendorust_model.pt')]
    receipt = {'schema': 'move_evaluation_greedy_build_v1', 'status': 'compiled',
               'created_utc': datetime.now(timezone.utc).isoformat(), 'source_pin': PIN,
               'executable': artifact(BUILD / 'target/x86_64-pc-windows-gnu/release/splendor-observer-bridge.exe'),
               'runtime_inputs': runtime, 'upstream_inputs': copies,
               'compiler': artifact(toolchain / 'bin/rustc.exe'), 'cargo': artifact(cargo),
               'attempt': artifact(attempt_path),
               'lock': artifact(BUILD / 'Cargo.lock'), 'model_inference_executed': False,
               'old_sources_modified': False}
    save(BUILD / 'build_receipt.json', receipt)
    print(json.dumps({'status': 'compiled', 'receipt': artifact(BUILD / 'build_receipt.json'),
                      'executable': receipt['executable']}))


if __name__ == '__main__':
    main(resume='--resume' in sys.argv[1:])
