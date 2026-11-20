"""Bounded model-free build of an isolated observer/target bridge."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import time

from . import common as c

OLD = c.ROOT/'outputs/external_opponent_search_20261018_02/native_build_v2'
TOOLCHAIN = c.ROOT/'outputs/external_opponent_search_20261018_02/runtime/toolchain'
SOURCE = c.ROOT/'tools/splendorust_selfplay_pilot_resume_20261103/native_bridge/src/main.rs'
NATIVE = c.OUT/'native'


def validate_build(path=None):
    path=NATIVE/'build_receipt.json' if path is None else Path(path)
    value=c.read(path)
    c.require(value['status']=='passed' and value['actual_exit_code']==0 and value['model_calls']==0,'Successful model-free native build')
    c.verify(value['inputs']+[value['binary'],value['stdout'],value['stderr']])
    for item in value['copies']:
        c.verify([item['original'],item['copy']])
        c.require(item['original']['sha256']==item['copy']['sha256'],'Unmodified copied core/agent sources')
    return value


def prepare_native():
    """Compile once, offline/locked/one job/180s; never replace an old attempt."""
    receipt=NATIVE/'build_receipt.json'
    if receipt.exists():return validate_build()
    c.require(not NATIVE.exists(),'Separate build directory must be absent; preserve failed builds')
    NATIVE.mkdir(parents=True)
    freeze=c.read(c.ACCEPTED_OUT/'freeze.json')
    c.require(c.file_hash(c.ACCEPTED_OUT/'freeze.json')==c.ACCEPTED_FREEZE_SHA,'Accepted512 source freeze')
    original_receipt=c.read(OLD/'source_receipt_v2.json')
    source_inputs=[];copies=[]
    for item in original_receipt['source_files']:
        relative=Path(item['copy']);source=OLD/relative
        if not relative.as_posix().startswith('crates/'):continue
        c.require(c.file_hash(source)==item['sha256'] and source.stat().st_size==item['bytes'],'Pinned original corrected native source')
        target=NATIVE/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        copies.append(dict(original=c.artifact(source),copy=c.artifact(target)))
    for relative in ('Cargo.toml','Cargo.lock','bridge/Cargo.toml'):
        source=OLD/relative;target=NATIVE/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        copies.append(dict(original=c.artifact(source),copy=c.artifact(target)))
    target=NATIVE/'bridge/src/main.rs';target.parent.mkdir(parents=True);shutil.copyfile(SOURCE,target)
    source_inputs=[c.artifact(SOURCE),c.artifact(target),c.artifact(OLD/'source_receipt_v2.json'),c.artifact(c.ACCEPTED_OUT/'freeze.json')]
    # An isolated cache prevents Cargo locks/cache metadata touching any frozen
    # cache. Cached crates are copied; offline/locked forbids network resolution.
    shutil.copytree(OLD/'cargo_home/registry',NATIVE/'cargo_home/registry')
    cargo=TOOLCHAIN/'bin/cargo.exe';rustc=TOOLCHAIN/'bin/rustc.exe'
    env=os.environ.copy();env.update(RUSTC=str(rustc),CARGO_HOME=str(NATIVE/'cargo_home'),
        CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER=str(TOOLCHAIN/'lib/rustlib/x86_64-pc-windows-gnu/bin/rust-lld.exe'),
        RUSTFLAGS='-C linker-flavor=ld.lld -C link-self-contained=yes',CARGO_BUILD_JOBS='1',CARGO_INCREMENTAL='0')
    argv=[str(cargo),'build','--release','--offline','--locked','--target','x86_64-pc-windows-gnu','-p','splendor-observer-bridge']
    started=time.monotonic();code=None;error=None
    with (NATIVE/'build_stdout.log').open('xb') as out,(NATIVE/'build_stderr.log').open('xb') as err:
        try:
            process=subprocess.run(argv,cwd=NATIVE,env=env,stdout=out,stderr=err,timeout=180,
                                   creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            code=process.returncode
        except BaseException as caught:error=repr(caught)
    attempt=dict(argv=argv,actual_exit_code=code,error=error,elapsed_seconds=time.monotonic()-started,
        maximum_seconds=180,model_calls=0,inputs=source_inputs,copies=copies,
        stdout=c.artifact(NATIVE/'build_stdout.log'),stderr=c.artifact(NATIVE/'build_stderr.log'))
    c.save(NATIVE/'build_attempt.json',attempt)
    c.require(code==0 and error is None,'Isolated native build failed; evidence preserved')
    binary=NATIVE/'target/x86_64-pc-windows-gnu/release/splendor-observer-bridge.exe'
    c.save(receipt,dict(attempt,status='passed',schema='selfplay_capture_build_v1',binary=c.artifact(binary),
        algorithm_sources_unchanged=True,only_bridge_capture_added=True))
    return validate_build()


if __name__=='__main__':
    result=prepare_native()
    print(json.dumps({'status':result['status'],'binary':result['binary'],'receipt':c.artifact(NATIVE/'build_receipt.json')}))
