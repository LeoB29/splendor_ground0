"""Distinct bounded type for the complete, score-free preservation inventory.

This inventory is naturally larger than small control receipts. The existing
8MiB control reader remains unchanged. Only the exact separately bound metadata
inventory is admitted here, never arbitrary result payloads or larger controls.
"""
from datetime import datetime
from pathlib import PurePosixPath
import json
from tools.splendorust_administrative_stop_20261115 import cancel as stop

MAX_INVENTORY_BYTES=256*1024**2
MAX_INVENTORY_ROWS=1_000_000
PATH=stop.OUT/'preservation_metadata.json'
MODE='read-only file inventory; prior immutable evidence hashes retained in original receipts'


def unique(pairs):
 result={}
 for key,value in pairs:
  stop.require(key not in result,'Duplicate preservation JSON key');result[key]=value
 return result


def validate_inventory(value):
 stop.require(type(value) is dict and set(value)=={'files','complete','deferred_while_original_writer_alive','mode','time'},
  'Exact preservation inventory schema')
 stop.require(value['complete'] is True and value['deferred_while_original_writer_alive'] is False
  and value['mode']==MODE,'Only completed closed-evidence metadata inventory')
 stop.require(type(value['time']) is str and datetime.fromisoformat(value['time']).utcoffset() is not None,
  'Explicit inventory timestamp timezone')
 rows=value['files'];stop.require(type(rows) is list and 0<len(rows)<=MAX_INVENTORY_ROWS,
  'Bounded nonempty preservation metadata rows')
 seen=set()
 for row in rows:
  stop.require(type(row) is dict and set(row)=={'path','bytes','mtime_ns'},'Exact preservation row fields')
  name=row['path']
  stop.require(type(name) is str and 0<len(name)<=4096 and '\\' not in name and ':' not in name,
   'Canonical relative preservation path text')
  parts=PurePosixPath(name)
  stop.require(not parts.is_absolute() and str(parts)==name and not {'..','.'}.intersection(parts.parts),
   'Normalized relative preservation path')
  path=(stop.ROOT/name).resolve()
  stop.require(path.is_relative_to(stop.ACTIVE.resolve()) and path.relative_to(stop.ROOT).as_posix()==name,
   'Every preserved file stays inside the original study')
  stop.require(type(row['bytes']) is int and row['bytes']>=0
   and type(row['mtime_ns']) is int and row['mtime_ns']>0,'Strict finite file size and timestamp')
  stop.require(name not in seen,'Unique preserved file path');seen.add(name)
 return value


def read_inventory(item):
 stop.require(type(item) is dict and set(item)=={'path','bytes','sha256'},'Exact bound inventory descriptor')
 stop.require(type(item['path']) is str and stop.resolve(item)==PATH.resolve()
  and item['path']==PATH.relative_to(stop.ROOT).as_posix(),'Only the original preservation metadata inventory')
 stop.require(type(item['bytes']) is int and 0<item['bytes']<=MAX_INVENTORY_BYTES,
  'Preservation inventory byte allowance')
 stop.require(type(item['sha256']) is str and len(item['sha256'])==64
  and set(item['sha256'])<=set('0123456789abcdef'),'Complete inventory hash')
 before=PATH.stat();stop.require(before.st_size==item['bytes'],'Exact inventory physical size before parse')
 with PATH.open('rb') as stream:raw=stream.read(MAX_INVENTORY_BYTES+1)
 stop.require(len(raw)==item['bytes'] and len(raw)<=MAX_INVENTORY_BYTES,'Bounded complete inventory bytes')
 stop.require(__import__('hashlib').sha256(raw).hexdigest()==item['sha256'],'Preserved inventory hash before parse')
 value=json.loads(raw,object_pairs_hook=unique,
  parse_constant=lambda s:(_ for _ in ()).throw(ValueError('Nonfinite preservation JSON')))
 after=PATH.stat()
 stop.require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Inventory immutable while parsed')
 return validate_inventory(value)
