"""Fixed engineering-only setups; whole-registry collision checks, no redraw."""
import hashlib, re, subprocess, sys, time
from pathlib import Path
from . import common as c

DOMAIN='splendorust_parallel_throughput_20261115_v1'

def seed(*parts):return str(int.from_bytes(hashlib.sha256(c.canonical([DOMAIN,*parts]).encode()).digest(),'big'))

def collection_specs():
 return [dict(game_id=f'g{i:04d}',game_index=i,cell_index=i//16,split='engineering',
  seed_decimal=seed('collection',i),start_player=i%2,quota=512,world_pool=3,cpuct=.4,
  purpose='excluded_parallel_throughput_engineering_only') for i in range(32)]

def qualification_specs():
 return [dict(game_id=f'qualification_g{i}',game_index=34+i,split='engineering',
  seed_decimal=seed('qualification',i),start_player=i,quota=512,world_pool=3,cpuct=.4,
  purpose='excluded_parallel_throughput_qualification_only') for i in range(2)]

def seeds_in(value):
 if type(value) is int or (isinstance(value,str) and value.isdecimal()):yield int(value)
 elif isinstance(value,dict):
  for item in value.values():yield from seeds_in(item)
 elif isinstance(value,list):
  for item in value:yield from seeds_in(item)

def scan_registered(path,targets,boundary=lambda:None,block_bytes=1024*1024):
 """Stream exact decimal literals and hashes; decode unusual JSON encodings.

 Whole JSON parsing of4GiB of ordinary ASCII receipts was needlessly costly.
 Decimal strings with leading zeros are included. Any Unicode escape or
 non-ASCII byte triggers the full existing semantic decoder as a fallback.
 False positives are safe refusals; no registered target can be skipped.
 """
 path=c.normalized(path);before=path.stat();digest=hashlib.sha256();carry=b'';found=False;fallback=False
 tokens=[str(x).encode('ascii') for x in sorted(targets)]
 # Extract the rare long decimal identifiers once, instead of trying fifty
 # alternative256-bit seeds at every byte of multi-gigabyte metadata.
 token_set=set(tokens)
 pattern=re.compile(rb'(?<![0-9])[0-9]{'+str(min(map(len,tokens))).encode()+rb',}')
 overlap=max(map(len,tokens))+4
 with path.open('rb') as stream:
  while True:
   boundary();chunk=stream.read(block_bytes)
   if not chunk:break
   digest.update(chunk);raw=carry+chunk
   fallback|=not raw.isascii() or b'\\u' in raw or b'\x00' in raw
   found|=any(match.group().lstrip(b'0') in token_set for match in pattern.finditer(raw))
   carry=raw[-overlap:]
 after=path.stat();c.require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Registered evidence changed while scanning')
 descriptor=dict(path=path.relative_to(c.ROOT).as_posix(),bytes=after.st_size,sha256=digest.hexdigest())
 if fallback:
  found|=bool(targets.intersection(seeds_in(c.read(path))))
  c.require(c.artifact(path)==descriptor,'Semantic fallback uses the same registered bytes')
 return found,descriptor,fallback

def registry_inventory(targets,boundary=lambda:None):
 """All discovered registered JSONs; oversized evidence is not silently skipped."""
 result=subprocess.run(['rg','--files','--no-ignore','outputs','-g','*manifest*.json',
  '-g','*schedule*.json','-g','*seed_registry*.json','-g','*reserved_panel*.json',
  '-g','*freeze*.json','-g','*selection*.json','-g','*reservation*.json'],
  cwd=c.ROOT,capture_output=True,text=True,check=True,timeout=120)
 checked=[];collisions=[];excluded=[];decoded=[]
 for name in sorted(result.stdout.splitlines()):
  boundary();path=(c.ROOT/name).resolve()
  if path.is_relative_to(c.OUT) or path.is_relative_to(c.PREP):excluded.append(name);continue
  found,item,fallback=scan_registered(path,targets,boundary)
  if found:collisions.append(name)
  if fallback:decoded.append(name)
  checked.append(item)
 active=c.ACTIVE_OUT/'reserved/collection_schedule.json';engineering=c.ACTIVE_OUT/'reserved/qualification_schedule.json'
 for path,count in [(active,3328),(engineering,2)]:
  rows=c.read(path);c.require(len(rows)==count and len({x['seed_decimal'] for x in rows})==count,'Complete active collection reservation')
  c.require(not targets.intersection(int(x['seed_decimal']) for x in rows),'Collision with active collection')
  c.require(c.artifact(path) in checked,'Active collection covered by registry scan')
 c.require(not collisions,'Setup collision; never redraw')
 return dict(status='passed',checked=checked,collisions=collisions,own_outputs_excluded=excluded,
  skipped_oversized=[],unicode_semantic_fallbacks=decoded,scanner='Streaming decimal literal matching and SHA256; full semantic decode on Unicode escapes/non-ASCII',scope='Every discovered registered manifest/schedule/seed_registry/reserved_panel/freeze/selection/reservation JSON; not arbitrary unregistered raw logs',
  target_count=len(targets),time=c.now())

def initial(spec):
 from tools.splendorust_collection4096_20261115.schedule import _initial
 return _initial(spec['seed_decimal'],spec.get('start_player',0))

def prepare():
 from .live.schedule import expected_spec
 from tools.future_training_rules_20261021 import verify_runtime_sources
 sources=verify_runtime_sources();live=[expected_spec(i) for i in range(16)]
 collection=collection_specs();qualification=qualification_specs();all_rows=live+collection+qualification
 targets={int(x['seed_decimal']) for x in all_rows};c.require(len(targets)==50,'Fifty unique engineering starts')
 start=time.monotonic()
 def boundary():c.require(time.monotonic()-start<600,'Ten-minute model-free reservation bound')
 inventory=registry_inventory(targets,boundary)
 reservation=dict(live=c.save(c.OUT/'reserved/live_schedule.json',live),
  collection_schedule=c.save(c.OUT/'reserved/collection_schedule.json',collection),
  qualification_schedule=c.save(c.OUT/'reserved/qualification_schedule.json',qualification),
  collection_initials=c.save(c.OUT/'reserved/collection_initial_states.json',[dict(spec=s,initial=initial(s)) for s in collection]),
  qualification_initials=c.save(c.OUT/'reserved/qualification_initial_states.json',[dict(spec=s,initial=initial(s)) for s in qualification]),
  seed_inventory=c.save(c.OUT/'reserved/seed_inventory.json',inventory))
 c.require('torch' not in sys.modules,'No learned model during reservation')
 c.save(c.OUT/'reserved/reservation_receipt.json',dict(status='reserved',domain=DOMAIN,artifacts=reservation,
  corrected_source_receipt=sources,model_calls=0,games_played=0,all_excluded_from_corpora=True))
 return reservation
