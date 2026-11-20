"""Model-free saved-failure counterexamples for administrative finalization."""
from copy import deepcopy
from pathlib import Path
import json
import pytest
from . import audit as a


class FakeCustody:
 def __init__(self,values):self.values=values
 def read(self,item):return self.values[item]


def game(seat=0,winner=0,reason='natural'):
 spec=dict(pair_index=0,seed_decimal='9',candidate_seat=seat)
 points=1 if winner is None else 2 if winner==seat else 0
 row=dict(spec=spec,game='saved'+str(seat),candidate_points2=points,failed_decisions=int(reason=='forfeit'))
 saved=dict(spec=spec,complete=True,reason=reason,winner=winner,candidate_points2=points,
  failure=None if reason=='natural' else dict(seat=1-winner),
  final=dict(state=dict(terminal=reason=='natural',winner=winner if reason=='natural' else None)))
 return row,spec,saved


@pytest.mark.parametrize('winner,seat,points',[(0,0,2),(1,0,0),(None,0,1),(0,1,0),(1,1,2),(None,1,1)])
def test_independent_points(winner,seat,points):
 row,spec,saved=game(seat,winner)
 assert a.validated_game_points(row,spec,FakeCustody({row['game']:saved}))==points


@pytest.mark.parametrize('change',['credit','complete','winner','terminal','reason','forfeit_count','spec'])
def test_rejects_corrupted_game_receipt(change):
 row,spec,saved=game()
 if change=='credit':saved['candidate_points2']=0
 if change=='complete':saved['complete']=False
 if change=='winner':saved['winner']=True
 if change=='terminal':saved['final']['state']['terminal']=False
 if change=='reason':saved['reason']='administrative_interruption'
 if change=='forfeit_count':row['failed_decisions']=1
 if change=='spec':saved['spec']={**spec,'candidate_seat':1}
 with pytest.raises(ValueError):a.validated_game_points(row,spec,FakeCustody({row['game']:saved}))


def test_real_saved_forfeit_preserved():
 row,spec,saved=game(0,1,'forfeit')
 assert a.validated_game_points(row,spec,FakeCustody({row['game']:saved}))==0


def test_pairs_use_complete_swapped_seats():
 left,ls,lv=game(0,0);right,rs,rv=game(1,None)
 assert a.pair_points([left,right],[ls,rs],FakeCustody({left['game']:lv,right['game']:rv}))==[.75]
 with pytest.raises(ValueError):a.pair_points([left],[ls],FakeCustody({left['game']:lv}))
 with pytest.raises(ValueError):a.pair_points([left,right],[ls,{**rs,'seed_decimal':'10'}],FakeCustody({left['game']:lv,right['game']:rv}))


def test_output_is_exclusive_and_separate(tmp_path,monkeypatch):
 monkeypatch.setattr(a,'OUT',tmp_path/'new')
 p=a.OUT/'item.json';a.save(p,dict(a=1))
 with pytest.raises(FileExistsError):a.save(p,dict(a=2))
 with pytest.raises(ValueError):a.save(tmp_path/'old.json',dict(a=3))


def test_bound_source_change_detected(tmp_path):
 p=tmp_path/'source.bin';p.write_bytes(b'original');item=a.artifact(p);p.write_bytes(b'changed')
 with pytest.raises(ValueError):a.Custody(lambda:None).verify(item)


def test_recursive_control_custody_without_parsing_packets(tmp_path,monkeypatch):
 monkeypatch.setattr(a.original,'OUT',tmp_path)
 folder=tmp_path/'jobs/block_01/worker/live/services/life_0000'
 raw=folder/'transport/request-00000001.json';raw.parent.mkdir(parents=True);raw.write_bytes(b'opaque-accepted-raw-packet')
 linked=folder/'link.json';linked.write_text(json.dumps({'source':a.artifact(raw)}))
 custody=a.Custody(lambda:None);custody.verify(a.artifact(linked))
 assert set(custody.items)=={a.artifact(raw)['path'],a.artifact(linked)['path']}
 assert a.artifact(raw)['path'] not in custody.parsed


def test_conflicting_descriptor_rejected(tmp_path):
 p=tmp_path/'file';p.write_bytes(b'data');item=a.artifact(p);custody=a.Custody(lambda:None)
 custody.verify(item)
 with pytest.raises(ValueError):custody.verify({**item,'sha256':'0'*64})


@pytest.mark.parametrize('clock',['worker_mono','worker_wall','original'])
def test_real_clock_bounds(clock,monkeypatch):
 times={'mono':100.,'wall':200.}
 monkeypatch.setattr(a.time,'monotonic',lambda:times['mono'])
 monkeypatch.setattr(a.time,'time',lambda:times['wall'])
 boundary=a.Boundary(dict(monotonic_origin=50.,monotonic_deadline=10000.,unix_origin=150.,unix_deadline=20000.))
 if clock=='worker_mono':times['mono']+=a.CAP_SECONDS
 if clock=='worker_wall':times['wall']+=a.CAP_SECONDS
 if clock=='original':boundary.limits['monotonic_deadline']=100.
 with pytest.raises(ValueError):boundary()


def test_contract_never_promotes_or_retries():
 row=a.binding_contract()
 assert row['models']==row['new_games']==0 and not row['original_attempt_complete']
 assert row['no_training'] and row['no_retries'] and row['maximum_workers']==2


def test_watchdog_enforces_actual_output_cap(monkeypatch):
 from . import launcher
 monkeypatch.setattr(launcher,'output_bytes',lambda:a.MAX_OUTPUT)
 monkeypatch.setattr(a.original,'check_deadlines',lambda _:None)
 with pytest.raises(ValueError,match='512MiB'):launcher.guard({},a.time.monotonic())


def test_watchdog_enforces_real_clock(monkeypatch):
 from . import launcher
 monkeypatch.setattr(launcher,'output_bytes',lambda:0)
 monkeypatch.setattr(a.original,'check_deadlines',lambda _:None)
 with pytest.raises(ValueError,match='per-pass'):launcher.guard({},a.time.monotonic()-a.CAP_SECONDS)


def test_no_score_read_at_import():
 import sys
 assert 'torch' not in sys.modules


def test_admission_rejects_incomplete_cancellation_before_score_read(tmp_path,monkeypatch):
 freeze={'contract':a.binding_contract(),'inputs':[],'original_freeze':{},'cancellation_freeze':{},'previous_attempt':{}}
 receipt={'status':'administrative_cleanup_incomplete','exact_owned_absence':False,
  'original_frozen_attempt_complete':False,'retried_decisions':0}
 monkeypatch.setattr(a,'sha',lambda _: 'frozen')
 monkeypatch.setattr(a,'read',lambda _:freeze)
 monkeypatch.setattr(a.stop,'verify',lambda _:None)
 monkeypatch.setattr(a,'prior_preservation',lambda:{})
 monkeypatch.setattr(a,'artifact',lambda _:dict(path='cancellation',bytes=1,sha256='0'*64))
 monkeypatch.setattr(a,'resolve',lambda _:tmp_path/'file')
 called=[]
 def fake_read(path):
  if Path(path)==tmp_path/'file':
   called.append(path)
   return {'source_inputs':[]} if len(called)==1 else receipt
  raise AssertionError('Unexpected evidence opened')
 monkeypatch.setattr(a.stop,'read',fake_read)
 with pytest.raises(ValueError,match='completed disclosed administrative'):a.admit('frozen')

