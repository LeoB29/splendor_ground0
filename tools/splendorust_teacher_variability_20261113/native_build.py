"""Additive native diagnostic build. No historical source or cache is modified.

Legacy mode preserves the original RNG draws. Split mode gives only the root
world sampler its own RNG; the original RNG still supplies neural encodings.
The diagnostic exposes full hypothetical worlds for independent identity checks.
Building this package executes no model, search, or game.
"""
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
BUILD = ROOT/'outputs/splendorust_teacher_variability_build_20261113_01'
TOOLCHAIN = ROOT/'outputs/external_opponent_search_20261018_02/runtime/toolchain'
ORIGINAL_BRIDGE_SHA = '76d1ae52d9c8982a432bfe313a1519f5bb4c609366640a466e42f58c77db28f1'
ORIGINAL_SEARCH_SHA = '99eac73ef6cf5ff629b8ff6918f4b48429cf81268d4e2b83981602795abb9676'


def artifact(path):
    p = Path(path)
    return dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError('Pinned source substitution site changed: ' + before[:70])
    return source.replace(before, after)


def search_source(source):
    source = replace_once(source, '    rng: Rng,\n', '''    rng: Rng,
    /// Separate saved-position diagnostic only; None retains legacy draw order.
    pub diagnostic_world_seed: Option<u64>,
    pub diagnostic_root_worlds: Vec<String>,
    pub diagnostic_encoding_before_worlds: Option<u64>,
    pub diagnostic_encoding_after_worlds: Option<u64>,
''')
    source = replace_once(source, '            rng: Rng::new(seed),\n', '''            rng: Rng::new(seed),
            diagnostic_world_seed: None,
            diagnostic_root_worlds: Vec::new(),
            diagnostic_encoding_before_worlds: None,
            diagnostic_encoding_after_worlds: None,
''')
    before = '''        let worlds: Vec<_> = (0..self.world_pool)
            .map(|_| o.determinize(&mut self.rng).expect("valid observation"))
            .collect();'''
    after = '''        self.diagnostic_encoding_before_worlds = Some(self.rng.0);
        let worlds: Vec<_> = if let Some(seed) = self.diagnostic_world_seed {
            let mut world_rng = Rng::new(seed);
            (0..self.world_pool)
                .map(|_| o.determinize(&mut world_rng).expect("valid observation"))
                .collect()
        } else {
            (0..self.world_pool)
                .map(|_| o.determinize(&mut self.rng).expect("valid observation"))
                .collect()
        };
        self.diagnostic_encoding_after_worlds = Some(self.rng.0);
        self.diagnostic_root_worlds = worlds.iter().map(|w| format!("{w:?}")).collect();'''
    return replace_once(source, before, after)


def bridge_source(source):
    pool_before = '        agent.world_pool=3;agent.chance_universes=3;agent.inference_batch_size=1;'
    pool_after = '''        let world_pool=match v["world_pool"].as_u64() {Some(3)=>3,Some(6)=>6,_=>panic!("world_pool must be integer3or6")};
        let mode=v["sampling_mode"].as_str().expect("sampling_mode string");
        let world_seed=v["world_seed"].as_u64().expect("world_seed u64");
        assert!(mode=="legacy" || mode=="split", "sampling mode allowlist");
        if mode=="legacy" {assert_eq!(world_seed,seed,"legacy has one shared seed");}
        agent.diagnostic_world_seed=if mode=="split" {Some(world_seed)} else {None};
        agent.world_pool=world_pool;agent.chance_universes=3;agent.inference_batch_size=1;'''
    source = replace_once(source, pool_before, pool_after)
    before = '        let selected=agent.select_action(&o,&choices);assert!(choices.contains(&selected));'
    after = '''        agent.capture_search_targets=v["capture_search_targets"].as_bool().unwrap_or(false);
''' + before + '''
        let targets=agent.search_targets.as_ref().map(|t|json!({"visits":t.visits.to_vec(),"q":t.q.to_vec(),"root_value":t.root_value,"network_value":t.network_value,"selected_edge_value":agent.value_target()}));
        let sampling=json!({"schema":"teacher_variability_sampling_v1","mode":mode,"world_pool":world_pool,"world_seed":world_seed,"encoding_seed":seed,"root_world_debug":agent.diagnostic_root_worlds,"encoding_rng_before_worlds":agent.diagnostic_encoding_before_worlds,"encoding_rng_after_worlds":agent.diagnostic_encoding_after_worlds});'''
    source = replace_once(source, before, after)
    ending = '"heuristic_phase":o.phase!=Phase::Main || choices.len()==1 || o.turns>=124}'
    return replace_once(source, ending, ending[:-1] + ',"search_targets":targets,"sampling":sampling}')


def build():
    if BUILD.exists():
        raise ValueError('Build area already exists; preserve every attempt')
    bridge_original = SOURCE/'bridge/src/main.rs'
    search_original = SOURCE/'crates/splendor-agents/src/neural_search.rs'
    if artifact(bridge_original)['sha256'] != ORIGINAL_BRIDGE_SHA or artifact(search_original)['sha256'] != ORIGINAL_SEARCH_SHA:
        raise ValueError('Pinned native source identity changed')
    # Transform in memory before creating any new evidence directory.
    transformed_bridge = bridge_source(bridge_original.read_text(encoding='utf-8'))
    transformed_search = search_source(search_original.read_text(encoding='utf-8'))
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
    bridge_target = BUILD/'bridge/src/main.rs'
    search_target = BUILD/'crates/splendor-agents/src/neural_search.rs'
    bridge_target.write_text(transformed_bridge, encoding='utf-8', newline='\n')
    search_target.write_text(transformed_search, encoding='utf-8', newline='\n')
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
        original_bridge=artifact(bridge_original), original_search=artifact(search_original),
        cargo=artifact(cargo), rustc=artifact(TOOLCHAIN/'bin/rustc.exe')))
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
    shutil.copyfile(executable, native/'diagnostic_bridge.exe')
    for item in originals:
        if artifact(ROOT/item['path']) != item:
            raise ValueError('Historical source changed during additive build')
    save(BUILD/'source_preservation_after.json', originals)
    receipt = dict(schema='teacher_variability_native_build_v1', status='compiled_not_model_qualified',
        time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        executable=artifact(executable), native=artifact(native/'diagnostic_bridge.exe'),
        source_inventory=artifact(BUILD/'source_inventory.json'),
        prior_source_preservation=artifact(BUILD/'source_preservation_after.json'),
        copied_core_unchanged=True, copied_encoder_unchanged=True,
        search_changes='optional independent root world RNG and passive world/RNG evidence only',
        legacy_rng_draw_order_unchanged_by_construction=True,
        bridge_changes='strict pool3or6, legacy/split mode and explicit seeds; existing targets and sampling evidence',
        world_pool_allowed=[3,6], sampling_modes=['legacy','split'], chance_universes_setting=3,
        source_pin='1828507b8d2f0e02bc03d102b9c6d29a54fa8d57',
        original_bridge=artifact(bridge_original), original_search=artifact(search_original),
        new_bridge=artifact(bridge_target), new_search=artifact(search_target),
        build_attempt=artifact(BUILD/'build_attempt.json'), build_plan=artifact(BUILD/'build_plan.json'),
        stdout=artifact(BUILD/'build_stdout.log'), stderr=artifact(BUILD/'build_stderr.log'),
        model_inference_executed=False, search_executed=False, native_smoke_executed=False,
        original_sources_modified=False)
    save(BUILD/'build_receipt.json', receipt)
    return receipt


if __name__ == '__main__':
    print(json.dumps(build(), sort_keys=True))
