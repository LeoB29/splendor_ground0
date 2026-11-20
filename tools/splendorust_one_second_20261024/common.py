"""Study identities and two-native descendant supervision; no model import."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import psutil
from tools.local_player_qualification_128.clock import PersistentWorker

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/splendorust_one_second_20261024_01'
PREP = ROOT/'outputs/splendorust_roadmap_20261024_preparation_01'
PROFILE = ROOT/'outputs/external_strength_screen_20261018_02/opponent_profile.json'
PROFILE_SHA = 'cd915908df1a7c6f39e55cfc9df59d88002a2b5ce167ce9512803e7c63d64424'
BUILD = ROOT/'outputs/move_evaluation_20261023_build_01/build_receipt.json'
PROTOCOL = ROOT/'docs/engineering/splendorust_one_second_comparison_20261024.md'
DOMAIN = 'splendorust_policy_puct16_1s_20261024_v1'
GiB, MiB = 1024**3, 1024**2
JOB_SECONDS = 21600
STAGES = {'qualification':3600, 'operational':1800, 'main':14400, 'final':1800}
WRITE_OBSERVER = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def now():
    return datetime.now(timezone.utc).isoformat()


def file_hash(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def artifact(path):
    p = Path(path).resolve()
    return {'path': p.relative_to(ROOT).as_posix(), 'sha256': file_hash(p), 'bytes': p.stat().st_size}


def verify(items):
    for item in items:
        require(artifact(ROOT/item['path']) == item, 'Changed frozen input: '+item['path'])


def save(path, value):
    p = Path(path)
    require(p.resolve().is_relative_to(OUT.resolve()), 'Output outside study directory')
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        f.write(canonical(value)+'\n'); f.flush(); os.fsync(f.fileno())
    if WRITE_OBSERVER is not None: WRITE_OBSERVER(p,p.stat().st_size)
    return artifact(p)


def make_schedule():
    rows=[]
    for stage, pairs in [('operational',3), ('main',100)]:
        for pair in range(pairs):
            seed=int.from_bytes(hashlib.sha256(canonical([DOMAIN,stage,pair]).encode()).digest(),'big')
            for game in range(2):
                rows.append(dict(stage=stage,pair_index=pair,game_index=2*pair+game,
                                 search_seat=(pair+game)%2,seed_decimal=str(seed)))
    return rows


class StudyWorker(PersistentWorker):
    """Unchanged trusted arrival clock, separately authenticated two-native tree."""
    def __init__(self, *args, native_artifacts, **kwargs):
        self._tree_cpu={}; self._roles={}; self._owners={}; self._verified={}
        self._initializing=True
        self._python=Path(sys.executable).resolve()
        self._natives={ (ROOT/a['path']).resolve(): ('search' if 'native_build_v2' in a['path'] else 'greedy')
                       for a in native_artifacts }
        verify(native_artifacts)
        self._expected={self._python:file_hash(self._python)}
        self._expected.update({(ROOT/a['path']).resolve():a['sha256'] for a in native_artifacts})
        self._console=(Path(os.environ['SystemRoot'])/'System32/conhost.exe').resolve() if os.name=='nt' else None
        if self._console: self._expected[self._console]=file_hash(self._console)
        try:
            super().__init__(*args, **kwargs)
        finally:
            self._initializing=False

    def _sample(self):
        if self._ps is None: return dict(self._last_telemetry)
        try: processes=[self._ps]+self._ps.children(recursive=True)
        except psutil.NoSuchProcess: return dict(self._last_telemetry)
        except psutil.AccessDenied as error:
            self._failure,self._failure_detail='process_tree_unobservable',str(error)
            return dict(self._last_telemetry)
        live=[]; rss=0; threads=0
        for proc in processes:
            try:
                identity=(proc.pid,proc.create_time()); exe=Path(proc.exe()).resolve(); ppid=proc.ppid()
                if exe not in self._verified:
                    require(exe in self._expected and file_hash(exe)==self._expected[exe], 'Unexpected descendant executable')
                    self._verified[exe]=self._expected[exe]
                role=('python' if proc.pid==self._ps.pid and exe==self._python else
                      self._natives.get(exe) if ppid==self._ps.pid else None)
                if exe==self._console:
                    owners=[r for r,i in self._owners.items() if i[0]==ppid and r in ('python','search','greedy')]
                    role='console:'+owners[0] if len(owners)==1 else None
                # A helper can be observed before its parent; classify in a second pass.
                memory=proc.memory_info().rss; cpu=proc.cpu_times(); count=proc.num_threads()
                self._tree_cpu[identity]=max(self._tree_cpu.get(identity,0),cpu.user+cpu.system)
                live.append(dict(pid=proc.pid,created=identity[1],parent_pid=ppid,role=role,
                    executable=str(exe),rss_bytes=memory,cpu_seconds=self._tree_cpu[identity],
                    affinity=proc.cpu_affinity(),observed_threads=count))
                rss+=memory; threads+=count
                if role is not None:
                    require(role not in self._owners or self._owners[role]==identity,'Computation/helper replacement')
                    self._owners[role]=identity
            except psutil.NoSuchProcess: continue
            except (ValueError,psutil.AccessDenied) as error:
                self._failure,self._failure_detail='process_tree_limit',str(error)
        for row in live:
            identity=(row['pid'],row['created'])
            if row['role'] is None and Path(row['executable'])==self._console:
                owners=[r for r,i in self._owners.items() if i[0]==row['parent_pid'] and r in ('python','search','greedy')]
                if len(owners)==1:
                    row['role']='console:'+owners[0]
                    if row['role'] in self._owners and self._owners[row['role']]!=identity:
                        self._failure,self._failure_detail='process_tree_limit','Console replacement'
                    self._owners[row['role']]=identity
            if row['role'] is None:
                self._failure,self._failure_detail='process_tree_limit','Unrecognized descendant'
            if row['affinity']!=self.affinity:
                if self._initializing and str(row['role']).startswith('console:'):
                    try:
                        psutil.Process(row['pid']).cpu_affinity(self.affinity)
                        row['startup_affinity_before']=row['affinity']; row['affinity']=psutil.Process(row['pid']).cpu_affinity()
                    except (psutil.NoSuchProcess,psutil.AccessDenied): pass
                if row['affinity']!=self.affinity:
                    self._failure,self._failure_detail='affinity_changed','Worker descendant changed assigned CPU'
        if len(self._owners)>6:
            self._failure,self._failure_detail='process_tree_limit','More than three compute processes plus OS helpers'
        self._peak_rss=max(self._peak_rss,rss)
        self._last_telemetry=dict(cpu_seconds=sum(self._tree_cpu.values()),rss_bytes=rss,
            peak_sampled_rss_bytes=self._peak_rss,affinity=self.affinity,observed_threads=threads,
            process_tree=live,resource_scope='sampled_worker_and_all_descendants')
        if rss>self.max_rss_bytes:
            self._failure,self._failure_detail='rss_limit','Combined worker/helper RSS exceeded 2GiB'
        return dict(self._last_telemetry)

    def close(self):
        if getattr(self,'_closed',False): return
        identities=set(getattr(self,'_tree_cpu',{}))
        if getattr(self,'_ps',None) is not None:
            try:
                identities.update((p.pid,p.create_time()) for p in [self._ps]+self._ps.children(recursive=True))
            except (psutil.NoSuchProcess,psutil.AccessDenied): pass
        super().close()
        alive=[]
        for pid,created in sorted(identities):
            try:
                p=psutil.Process(pid)
                if p.create_time()==created and p.is_running(): alive.append(dict(pid=pid,created=created))
            except psutil.NoSuchProcess: pass
        if hasattr(self,'output_dir'):
            path=self.output_dir/'cleanup.json'
            if not path.exists():
                with path.open('x',encoding='utf-8') as stream:
                    json.dump(dict(time=now(),status='passed' if not alive else 'failed',
                        observed_owned_identities=len(identities),alive=alive),stream,sort_keys=True)
        callback=getattr(self,'study_after_close',None)
        if callback is not None: callback()
        if alive: raise RuntimeError('Owned worker descendants remain after cleanup')
