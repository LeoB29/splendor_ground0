"""Bounded parallel hashing of stopped evidence; never edits source trees."""
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import os
from pathlib import Path
import time
from . import common as c


def files(folder):
    pending=[Path(folder).resolve()]
    while pending:
        with os.scandir(pending.pop()) as entries: rows=sorted(entries,key=lambda x:x.name)
        for entry in rows:
            c.require(not entry.is_symlink(),'No linked preservation input')
            if entry.is_dir(follow_symlinks=False):pending.append(Path(entry.path))
            elif entry.is_file(follow_symlinks=False):yield Path(entry.path)


def stable_artifact(path):
    before=path.stat();item=c.artifact(path);after=path.stat()
    c.require((before.st_size,before.st_mtime_ns,before.st_ino)==
              (after.st_size,after.st_mtime_ns,after.st_ino),'Stopped evidence stable during hashing')
    return item


def inventory(folder,destination,*,workers=4):
    folder=Path(folder).resolve();destination=Path(destination).resolve()
    c.require(workers==4 and destination.is_relative_to(c.OUT/'prelaunch') and
              not destination.is_relative_to(folder),'Four bounded hashing workers; separate new output')
    destination.parent.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();count=total=0;source=iter(files(folder))
    with destination.open('x',encoding='utf-8',newline='\n') as stream, ThreadPoolExecutor(max_workers=workers) as pool:
        while batch:=list(islice(source,256)):
            for item in pool.map(stable_artifact,batch):
                stream.write(c.canonical(item)+'\n');count+=1;total+=item['bytes']
        stream.flush();os.fsync(stream.fileno())
    return dict(status='preserved_inventory',inventory=c.artifact(destination),files=count,bytes=total,
        original_attempt_status='incomplete',elapsed_seconds=time.monotonic()-started,time=c.now(),
        models=0,games=0,hash_workers=workers,maximum_queued_files=256)


def main():
    from tools.future_training_rules_20261021 import verify_runtime_sources
    sources=verify_runtime_sources()
    c.require(sources==c.read(c.ORIGINAL_OUT/'freeze.json')['corrected_source_receipt'],'Corrected source identity')
    result=inventory(c.PREVIOUS_OUT,c.OUT/'prelaunch/previous_preservation_inventory_parallel.jsonl')
    c.save(c.OUT/'prelaunch/previous_preservation.json',result)
    print(c.canonical(result),flush=True)


if __name__=='__main__':main()
