"""No real processes are signalled by this model-free regression suite."""
from pathlib import Path
import json
import pytest
from . import cancel as c

def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(c.canonical(value),encoding='utf-8');return c.artifact(path)

@pytest.fixture
def workspace(tmp_path,monkeypatch):
 monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'ACTIVE',tmp_path/'active')
 monkeypatch.setattr(c,'OUT',tmp_path/'administrative');monkeypatch.setattr(c,'PREP',tmp_path/'preparation')
 monkeypatch.setattr(c,'POLICY',c.PREP/'policy.json');monkeypatch.setattr(c,'FREEZE',c.OUT/'freeze.json')
 source=write(tmp_path/'source.json',{'source':'fixture'})
 active=write(c.ACTIVE/'freeze.json',{'active':'fixture'});monkeypatch.setattr(c,'ACTIVE_SHA',active['sha256'])
 policy=write(c.POLICY,dict(active_freeze_sha256=c.ACTIVE_SHA,new_interim_scores_opened_for_this_decision=False,
  readiness_requirements=['protocol','qualification','review','recipe','cancellation']))
 readiness=dict(schema='splendor_collection_readiness_20261115_v1',status='launch_ready',qualified_preparation=True,
  policy=policy,teacher=dict(checkpoint_sha256=c.TEACHER_SHA,pool=3,simulations=512),source_inputs=[source],
  requirements=[dict(requirement=k,passed=True,evidence=[source]) for k in ['protocol','qualification','review','recipe','cancellation']])
 ready=write(c.PREP/'readiness.json',readiness)
 frozen=write(c.FREEZE,dict(source_inputs=[source],policy=policy,active_freeze=active))
 return dict(source=source,policy=policy,ready=ready,readiness=readiness,freeze=frozen)

def test_interim_payload_is_never_parsed(workspace):
 path=c.ACTIVE/'interims/look_000400.json';write(path,{'secret_score':.99})
 with pytest.raises(ValueError,match='forbidden'):c.read(path)
 assert c.artifact(path)['bytes']>0

def test_valid_bound_readiness(workspace):
 assert c.validate_readiness(c.resolve(workspace['ready']),workspace['ready']['sha256'],workspace['policy'])['status']=='launch_ready'

@pytest.mark.parametrize('change',[
 {'status':'draft'},{'qualified_preparation':False},
 {'teacher':{'checkpoint_sha256':c.TEACHER_SHA,'pool':6,'simulations':512}},
 {'source_inputs':[]},{'requirements':[]},
])
def test_incomplete_or_changed_readiness_cannot_cancel(workspace,change):
 value=dict(workspace['readiness'],**change);item=write(c.PREP/'bad.json',value)
 with pytest.raises(ValueError):c.validate_readiness(c.resolve(item),item['sha256'],workspace['policy'])

def test_changed_readiness_evidence_rejected(workspace):
 c.resolve(workspace['source']).write_text('{}')
 with pytest.raises(ValueError):c.validate_readiness(c.resolve(workspace['ready']),workspace['ready']['sha256'],workspace['policy'])

def test_missing_requirement_evidence_rejected(workspace):
 value=workspace['readiness'];value['requirements'][1]['evidence']=[];item=write(c.PREP/'bad.json',value)
 with pytest.raises(ValueError):c.validate_readiness(c.resolve(item),item['sha256'],workspace['policy'])

def test_frozen_output_writes_are_forbidden(workspace):
 with pytest.raises(ValueError):c.save(c.ACTIVE/'new_receipt.json',{})

@pytest.mark.parametrize('stage',['analysis','acceptance','complete','failed'])
def test_existing_scientific_stop_is_never_cancelled(workspace,stage):
 write(c.ACTIVE/'stage.json',{'stage':stage,'status':'running'})
 assert c.stop_state()['action']=='already_stopped'

def test_latest_verified_look_selected_by_index_without_score_read(workspace):
 for i in (1,2):
  role=f'block_{i:02d}';ident={'pid':i,'created':float(i)}
  result=write(c.ACTIVE/'jobs'/role/'worker/result.json',dict(status='passed',error=None,identity=ident,role=role,freeze_sha256=c.ACTIVE_SHA))
  write(c.ACTIVE/'jobs'/role/'actual_exit.json',dict(status='passed',exit_code=0,actual_exit_observed=True,exact_owned_absence=True,
   cleanup_errors=[],error=None,freeze_sha256=c.ACTIVE_SHA,result_artifact=result,identity=ident))
  write(c.ACTIVE/'interims'/f'look_{i*400:06d}.json',{'secret_score':.99 if i==1 else .01})
 inventory=c.scheduled_inventory()
 assert inventory['latest_games']==800 and inventory['scores_opened'] is False
 assert [r['games'] for r in inventory['looks']]==[400,800]

def test_unverified_later_look_is_not_selected(workspace):
 write(c.ACTIVE/'interims/look_000400.json',{'secret_score':.99})
 assert c.scheduled_inventory()['latest_games']==0

def pair_fixture(count,*,following=False,pending=False):
 folder=c.ACTIVE/'jobs/block_01/worker'
 write(folder/'arena/progress.json',{'games_completed':count})
 write(folder/'arena/games/g0001/result.json',dict(complete=True,spec={'game_index':1},secret_score=.9))
 if following:(folder/'arena/games/g0002').mkdir()
 life=folder/'live/services/life_0000'
 write(life/'links/request-00000001.json',{})
 if pending:
  p=life/'inputs/request-00000002.json.gz';p.parent.mkdir();p.write_bytes(b'pending')
 return folder

def test_pair_gap_detected_before_progress_advances(workspace):
 pair_fixture(1);gap=c.pair_audit_gap('block_01',c.RequestTracker())
 assert gap['observed'] and gap['completed_games_in_block']==2 and not gap['atomic_boundary_guaranteed']

def test_next_game_or_pending_request_closes_preferred_gap(workspace):
 pair_fixture(1,following=True)
 assert not c.pair_audit_gap('block_01',c.RequestTracker())['observed']

def test_replay_pending_request_also_closes_gap(workspace):
 folder=pair_fixture(1)
 p=folder/'replay/services/life_0000/inputs/request-00000001.json.gz';p.parent.mkdir(parents=True);p.write_bytes(b'pending')
 assert not c.pair_audit_gap('block_01',c.RequestTracker())['observed']

def test_partial_terminal_json_is_not_a_boundary(workspace):
 folder=pair_fixture(1);(folder/'arena/games/g0001/result.json').write_text('{')
 assert not c.pair_audit_gap('block_01',c.RequestTracker())['observed']

def test_known_next_game_race_is_not_called_an_idle_gap(workspace):
 folder=pair_fixture(1)
 class RacingTracker:
  def snapshot(self,folder):
   (folder/'arena/games/g0002').mkdir(exist_ok=True)
   return dict(pending=[],lifetimes=[])
 assert not c.pair_audit_gap('block_01',RacingTracker())['observed']

def test_pending_request_tracker_advances_without_payload_reads(workspace):
 folder=pair_fixture(1,pending=True);tracker=c.RequestTracker()
 assert len(tracker.snapshot(folder)['pending'])==1
 write(folder/'live/services/life_0000/links/request-00000002.json',{'secret':'never read'})
 assert tracker.snapshot(folder)['pending']==[]

def test_reused_pid_is_never_terminated(workspace,monkeypatch):
 class Process:
  killed=False
  def create_time(self):return 2.
  def is_running(self):return True
  def kill(self):self.killed=True
 process=Process();monkeypatch.setattr(c.psutil,'Process',lambda pid:process)
 assert c.terminate_exact({'pid':123,'created':1.}) is False and not process.killed
 assert c.terminate_exact({'pid':123,'created':2.}) is True and process.killed

def test_already_stopped_execution_sends_no_signal(workspace,monkeypatch):
 write(c.ACTIVE/'stage.json',dict(stage='analysis',status='running'))
 monkeypatch.setattr(c,'terminate_exact',lambda identity:pytest.fail('No signal permitted'))
 item=c.execute(workspace['freeze']['sha256'],c.resolve(workspace['ready']),workspace['ready']['sha256'])
 assert c.read(c.resolve(item))['status']=='not_cancelled_already_stopped'

def test_cancellation_intent_precedes_exact_stage_signal(workspace,monkeypatch):
 write(c.ACTIVE/'stage.json',dict(stage='block_01',status='running'))
 owners=dict(launcher={'pid':1,'created':1.},root={'pid':2,'created':2.},stage={'pid':3,'created':3.},role='block_01')
 monkeypatch.setattr(c,'authenticate_owners',lambda role:owners)
 live=[True];monkeypatch.setattr(c,'exact_alive',lambda item:live[0])
 monkeypatch.setattr(c,'process_inventory',lambda own:[own['launcher'],own['root'],own['stage']])
 monkeypatch.setattr(c,'pair_audit_gap',lambda *a:dict(observed=True,atomic_boundary_guaranteed=False))
 def terminate(item):
  assert item==owners['stage'] and (c.OUT/'cancellation_intent.json').exists()
  live[0]=False;return True
 monkeypatch.setattr(c,'terminate_exact',terminate)
 result=c.read(c.resolve(c.execute(workspace['freeze']['sha256'],c.resolve(workspace['ready']),workspace['ready']['sha256'])))
 assert result['status']=='administratively_cancelled' and result['exact_owned_absence']
 assert not result['original_frozen_attempt_complete'] and result['retried_decisions']==0
 assert result['interrupted_tail_is_not_a_forfeit'] and (c.ACTIVE/'freeze.json').exists()

def test_bounded_fallback_is_explicit_and_never_invents_a_loss(workspace,monkeypatch):
 write(c.ACTIVE/'stage.json',dict(stage='block_01',status='running'))
 owners=dict(launcher={'pid':1,'created':1.},root={'pid':2,'created':2.},stage={'pid':3,'created':3.},role='block_01')
 monkeypatch.setattr(c,'authenticate_owners',lambda role:owners)
 live=[True];monkeypatch.setattr(c,'exact_alive',lambda item:live[0])
 monkeypatch.setattr(c,'process_inventory',lambda own:[own['launcher'],own['root'],own['stage']])
 monkeypatch.setattr(c,'pair_audit_gap',lambda *a:dict(observed=False,requests={'pending':[{'request_id':'unfinished'}]}))
 tick=[0]
 def monotonic():tick[0]+=901;return tick[0]
 monkeypatch.setattr(c.time,'monotonic',monotonic)
 def terminate(item):live[0]=False;return True
 monkeypatch.setattr(c,'terminate_exact',terminate)
 result=c.read(c.resolve(c.execute(workspace['freeze']['sha256'],c.resolve(workspace['ready']),workspace['ready']['sha256'])))
 intent=c.read(c.resolve(result['intent']))
 assert intent['bounded_fallback'] and intent['wait_elapsed_seconds']>=900
 assert not intent['active_move_clock_reset'] and result['retried_decisions']==0
 assert result['interrupted_tail_is_not_a_forfeit'] and not list(c.ACTIVE.rglob('deadline_failure.json'))

def test_incomplete_parent_cleanup_blocks_replacement_launch(workspace,monkeypatch):
 write(c.ACTIVE/'stage.json',dict(stage='block_01',status='running'))
 owners=dict(launcher={'pid':1,'created':1.},root={'pid':2,'created':2.},stage={'pid':3,'created':3.},role='block_01')
 monkeypatch.setattr(c,'authenticate_owners',lambda role:owners)
 monkeypatch.setattr(c,'exact_alive',lambda item:True)
 monkeypatch.setattr(c,'process_inventory',lambda own:[own['launcher'],own['root'],own['stage']])
 monkeypatch.setattr(c,'pair_audit_gap',lambda *a:dict(observed=True))
 monkeypatch.setattr(c,'terminate_exact',lambda item:True)
 tick=[0]
 def monotonic():tick[0]+=61;return tick[0]
 monkeypatch.setattr(c.time,'monotonic',monotonic)
 with pytest.raises(ValueError,match='cleanup incomplete'):
  c.execute(workspace['freeze']['sha256'],c.resolve(workspace['ready']),workspace['ready']['sha256'])
 result=c.read(c.OUT/'cancellation.json')
 assert result['status']=='administrative_cleanup_incomplete' and not result['exact_owned_absence']
