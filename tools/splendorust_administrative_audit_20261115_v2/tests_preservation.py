"""Reproduce the real large-inventory refusal without reading a score payload."""
from copy import deepcopy
from pathlib import Path
import ast,json
import pytest
from . import audit as a,preservation as p


def synthetic():
 return dict(files=[dict(path=(p.stop.ACTIVE/'example.bin').relative_to(p.stop.ROOT).as_posix(),bytes=1,mtime_ns=1)],
  complete=True,deferred_while_original_writer_alive=False,mode=p.MODE,time='2026-11-15T00:00:00+00:00')


def test_saved_failure_reproduced_and_dedicated_reader_accepts():
 refusal=p.stop.read(a.FAILED_OUT/'launch_refusal.json')
 item=refusal['preservation_inventory']
 assert item['bytes']==15_846_579 and item['bytes']>8*1024**2
 with pytest.raises(ValueError,match='Bounded control JSON'):p.stop.read(p.PATH)
 value=p.read_inventory(item)
 assert value['complete'] and 0<len(value['files'])<=p.MAX_INVENTORY_ROWS


@pytest.mark.parametrize('fault',['extra','missing','incomplete','deferred','mode','naive_time','not_list','empty',
 'duplicate','row_extra','row_missing','bool_size','negative_size','float_size','bool_time','zero_time',
 'absolute','outside','parent','slash','drive','double_slash','dot','nonstring'])
def test_strict_inventory_shapes_and_paths(fault):
 value=synthetic();row=value['files'][0]
 if fault=='extra':value['unexpected']=True
 if fault=='missing':value.pop('time')
 if fault=='incomplete':value['complete']=False
 if fault=='deferred':value['deferred_while_original_writer_alive']=True
 if fault=='mode':value['mode']='arbitrary control'
 if fault=='naive_time':value['time']='2026-11-15T00:00:00'
 if fault=='not_list':value['files']={}
 if fault=='empty':value['files']=[]
 if fault=='duplicate':value['files'].append(dict(row))
 if fault=='row_extra':row['sha256']='0'*64
 if fault=='row_missing':row.pop('bytes')
 if fault=='bool_size':row['bytes']=True
 if fault=='negative_size':row['bytes']=-1
 if fault=='float_size':row['bytes']=1.0
 if fault=='bool_time':row['mtime_ns']=True
 if fault=='zero_time':row['mtime_ns']=0
 if fault=='absolute':row['path']='/outputs/a'
 if fault=='outside':row['path']='outputs/other/file.json'
 if fault=='parent':row['path']+='/'+'../other'
 if fault=='slash':row['path']=row['path'].replace('/','\\')
 if fault=='drive':row['path']='C:/'+row['path']
 if fault=='double_slash':row['path']=row['path'].replace('outputs/','outputs//')
 if fault=='dot':row['path']=row['path'].replace('outputs/','outputs/./')
 if fault=='nonstring':row['path']=123
 with pytest.raises((ValueError,TypeError)):p.validate_inventory(value)


def test_row_bound_is_fixed_and_enforced(monkeypatch):
 assert p.MAX_INVENTORY_ROWS==1_000_000 and p.MAX_INVENTORY_BYTES==256*1024**2
 monkeypatch.setattr(p,'MAX_INVENTORY_ROWS',1)
 value=synthetic();value['files'].append({**value['files'][0],'path':value['files'][0]['path']+'2'})
 with pytest.raises(ValueError,match='metadata rows'):p.validate_inventory(value)


@pytest.mark.parametrize('fault',['oversized','bool_size','wrong_path','hash_shape','extra','changed_bytes'])
def test_descriptor_rejections_precede_large_read(fault):
 item=p.stop.artifact(p.PATH)
 if fault=='oversized':item['bytes']=p.MAX_INVENTORY_BYTES+1
 if fault=='bool_size':item['bytes']=True
 if fault=='wrong_path':item['path']='outputs/arbitrary.json'
 if fault=='hash_shape':item['sha256']='BAD'
 if fault=='extra':item['extra']=1
 if fault=='changed_bytes':item['bytes']-=1
 with pytest.raises(ValueError):p.read_inventory(item)


def test_duplicate_json_key_rejected():
 with pytest.raises(ValueError,match='Duplicate'):p.unique([('files',[]),('files',[])])


def test_saved_refusal_remains_no_workers():
 proof=a.prior_preservation()
 assert proof['original_worker_launches']==proof['models']==proof['new_games']==0
 assert proof['refusal']==a.artifact(a.FAILED_OUT/'launch_refusal.json')


def test_unchanged_scientific_verification_and_watchdog_source():
 old=a.ROOT/'tools/splendorust_administrative_audit_20261115'
 def functions(path):
  return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text(encoding='utf8')).body
   if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
 previous,current=functions(old/'audit.py'),functions(Path(a.__file__))
 changed={'freeze','admit','inventory','prior_preservation'}
 assert all(previous[name]==current[name] for name in previous if name not in changed)
 assert (old/'launcher.py').read_bytes()==(Path(a.__file__).parent/'launcher.py').read_bytes()
 # The only preflight change discovers this additional regression file.
 assert functions(old/'preflight.py')['record']==functions(Path(a.__file__).parent/'preflight.py')['record']
 assert a.binding_contract()==__import__('tools.splendorust_administrative_audit_20261115.audit',fromlist=['binding_contract']).binding_contract()

