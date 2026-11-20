"""Offline additive bridge build. No learned model or search is executed."""
from __future__ import annotations
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'outputs/external_opponent_search_20261018_02/native_build_v2'
BUILD = ROOT/'outputs/splendorust_sampling_stability_build_20261027_01'
TOOLCHAIN = ROOT/'outputs/external_opponent_search_20261018_02/runtime/toolchain'
ORIGINAL_BRIDGE_SHA = '76d1ae52d9c8982a432bfe313a1519f5bb4c609366640a466e42f58c77db28f1'

def artifact(path):
    p = Path(path)
    return dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())

def bridge_source(source):
    """Bridge-only configuration/capture; core, encoder and RNG remain copied."""
    pool_before = '        agent.world_pool=3;agent.chance_universes=3;agent.inference_batch_size=1;'
    pool_after = ('        let world_pool=match v.get("world_pool") {None=>3,Some(value)=>match value.as_u64() {'
                  'Some(3)=>3,Some(6)=>6,_=>panic!("world_pool must be integer3or6")}};\n'
                  '        agent.world_pool=world_pool;agent.chance_universes=3;agent.inference_batch_size=1;')
    before = '        let selected=agent.select_action(&o,&choices);assert!(choices.contains(&selected));'
    after = ('        agent.capture_search_targets=v["capture_search_targets"].as_bool().unwrap_or(false);\n'
             + before + '\n'
             '        let targets=agent.search_targets.as_ref().map(|t|json!({"visits":t.visits.to_vec(),"q":t.q.to_vec(),"root_value":t.root_value,"network_value":t.network_value,"selected_edge_value":agent.value_target()}));')
    ending = '"heuristic_phase":o.phase!=Phase::Main || choices.len()==1 || o.turns>=124}'
    replacement = ending[:-1] + ',"search_targets":targets}'
    if source.count(before) != 1 or source.count(ending) != 1 or source.count(pool_before) != 1:
        raise ValueError('Pinned bridge substitution sites changed')
    return source.replace(pool_before,pool_after).replace(before, after).replace(ending, replacement)

def build():
    if BUILD.exists():
        raise ValueError('Build area already exists; preserve every attempt')
    BUILD.mkdir()
    originals = []
    for group in ('crates', 'bridge'):
        for source in sorted((SOURCE/group).rglob('*')):
            if source.is_file():
                originals.append(artifact(source))
                target = BUILD/source.relative_to(SOURCE)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
    for name in ('Cargo.toml', 'Cargo.lock', 'LICENSE'):
        originals.append(artifact(SOURCE/name)); shutil.copyfile(SOURCE/name, BUILD/name)
    source = SOURCE/'bridge/src/main.rs'
    if artifact(source)['sha256'] != ORIGINAL_BRIDGE_SHA:
        raise ValueError('Original bridge source changed')
    target = BUILD/'bridge/src/main.rs'
    target.write_text(bridge_source(source.read_text(encoding='utf-8')), encoding='utf-8', newline='\n')
    # A separate cache prevents Cargo from touching the historical build tree.
    shutil.copytree(SOURCE/'cargo_home', BUILD/'cargo_home')
    inputs = [artifact(p) for p in sorted(BUILD.rglob('*')) if p.is_file()]
    save(BUILD/'source_preservation_before.json', originals)
    save(BUILD/'source_inventory.json', inputs)
    cargo = TOOLCHAIN/'bin/cargo.exe'
    env = dict(os.environ)
    overrides = dict(RUSTC=str(TOOLCHAIN/'bin/rustc.exe'), CARGO_HOME=str(BUILD/'cargo_home'),
        CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=str(TOOLCHAIN/'lib/rustlib/x86_64-pc-windows-gnu/bin/rust-lld.exe'),
        RUSTFLAGS='-C linker-flavor=ld.lld -C link-self-contained=yes', CARGO_BUILD_JOBS='1', CARGO_INCREMENTAL='0')
    env.update(overrides)
    command = [str(cargo), 'build', '--release', '--offline', '--locked', '--target',
               'x86_64-pc-windows-gnu', '-p', 'splendor-observer-bridge']
    save(BUILD/'build_plan.json', dict(command=command, environment_overrides=overrides,
        timeout_seconds=600, learned_model_execution=False, native_search_execution=False,
        original_bridge=artifact(source), cargo=artifact(cargo), rustc=artifact(TOOLCHAIN/'bin/rustc.exe')))
    started = time.monotonic()
    with (BUILD/'build_stdout.log').open('xb') as out, (BUILD/'build_stderr.log').open('xb') as err:
        try:
            completed = subprocess.run(command, cwd=BUILD, env=env, stdout=out, stderr=err,
                timeout=600, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            outcome = dict(returncode=completed.returncode, error=None)
        except BaseException as error:
            outcome = dict(returncode=None, error=repr(error))
    save(BUILD/'build_attempt.json', dict(outcome, elapsed_seconds=time.monotonic()-started))
    if outcome != dict(returncode=0, error=None):
        raise RuntimeError('Offline native build failed; logs preserved')
    executable = BUILD/'target/x86_64-pc-windows-gnu/release/splendor-observer-bridge.exe'
    native = BUILD/'native'; native.mkdir()
    for role in ('search', 'reference'):
        shutil.copyfile(executable, native/(role+'_bridge.exe'))
    for item in originals:
        if artifact(ROOT/item['path']) != item:
            raise ValueError('Historical source changed during additive build')
    save(BUILD/'source_preservation_after.json', originals)
    receipt = dict(schema='sampling_stability_native_build_v1', status='compiled_not_model_qualified',
        time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        executable=artifact(executable), natives={role:artifact(native/(role+'_bridge.exe')) for role in ('search','reference')},
        source_inventory=artifact(BUILD/'source_inventory.json'),
        prior_source_preservation=artifact(BUILD/'source_preservation_after.json'),
        copied_core_agents_unchanged=True, bridge_changes='optional strict world_pool3or6 default3; capture flag before select; extract existing targets after select',
        world_pool_default=3,world_pool_allowed=[3,6],chance_universes_setting=3,
        rng_architecture_unchanged=True,
        source_pin='1828507b8d2f0e02bc03d102b9c6d29a54fa8d57',
        original_bridge=artifact(source), new_bridge=artifact(target),
        build_attempt=artifact(BUILD/'build_attempt.json'), build_plan=artifact(BUILD/'build_plan.json'),
        stdout=artifact(BUILD/'build_stdout.log'), stderr=artifact(BUILD/'build_stderr.log'),
        model_inference_executed=False, search_executed=False, native_smoke_executed=False,
        original_sources_modified=False)
    save(BUILD/'build_receipt.json', receipt)
    return receipt

if __name__ == '__main__':
    print(json.dumps(build(), sort_keys=True))
