"""Model-free counterexamples for phase isolation and prospective admission."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from tools.splendorust_pool6_strengthfirst_20261028 import preparation as prep


@pytest.mark.parametrize('phase,games,workers,seconds,cutoff,storage,final',[
    ('qualification',0,62,7200,3600,4,1),
    ('assessment',400,848,86400,72000,12,4),
])
def test_fresh_process_binds_correct_phase_and_research_limits(phase,games,workers,seconds,cutoff,storage,final):
    env=dict(os.environ,SPLENDOR_POOL6_STRENGTH_PHASE=phase,PYTHONDONTWRITEBYTECODE='1')
    code=('import json;from tools.splendorust_pool6_strengthfirst_20261028 import preparation as p,common as c;'
          'print(json.dumps(dict(contract=p.contract(),out=str(c.OUT),qual=str(c.QUAL_OUT),assess=str(c.ASSESS_OUT))))')
    result=subprocess.run([sys.executable,'-B','-c',code],env=env,text=True,capture_output=True,timeout=30,check=True)
    doc=json.loads(result.stdout);value=doc['contract']
    assert value['phase']==phase and value['main_games']==games and value['max_workers']==workers
    assert value['job_seconds']==seconds and value['collection_cutoff_seconds']==cutoff
    assert value['max_evidence_bytes']==storage*1024**3 and value['final_reserve_bytes']==final*1024**3
    assert value['seconds_per_move']==value['research_seconds_per_move']==10.
    assert value['internal_seconds']==value['research_internal_seconds']==9.
    assert value['minimum_available_ram_bytes']==4*1024**3
    assert value['startup_minimum_available_ram_bytes']==2*1024**3
    assert value['late_completed_moves_accepted'] and value['timing_thresholds_do_not_gate_admission']
    assert doc['qual']!=doc['assess'] and doc['out']==doc['qual' if phase=='qualification' else 'assess']


@pytest.mark.parametrize('seconds,internal,soft',[(1.,.9,True),(10.,.9,True),(1.,9.,True),(10.,9.,False)])
def test_old_or_mixed_clock_acceptance_cannot_authorize_new_assessment(monkeypatch,tmp_path,seconds,internal,soft):
    monkeypatch.setattr(prep.c,'QUAL_OUT',tmp_path)
    monkeypatch.setattr(prep.c,'file_hash',lambda _: 'bound')
    documents={
        'independent_completion_review.json':dict(accepted=True,freeze_sha256='bound'),
        'freeze.json':dict(seconds_per_move=seconds,internal_seconds=internal,
                           timing_thresholds_do_not_gate_admission=soft),
    }
    monkeypatch.setattr(prep.c,'read',lambda p:documents[Path(p).name])
    with pytest.raises(ValueError,match='Wrong accepted timing policy'):
        prep.accepted_admission()


def test_seed_inventory_rejects_registered_collision_without_replacement(monkeypatch,tmp_path):
    monkeypatch.setattr(prep.c,'ROOT',tmp_path)
    monkeypatch.setattr(prep.c,'QUAL_OUT',tmp_path/'newqual')
    monkeypatch.setattr(prep.c,'ASSESS_OUT',tmp_path/'newmain')
    monkeypatch.setattr(prep.design,'PANEL',tmp_path/'panel.json')
    panel=dict(positions=[dict(seed_decimal=str(10**80+i)) for i in range(18)])
    historical=tmp_path/'history_schedule.json';historical.write_text('{}')
    collision=prep.schedule.stage_schedule('main')[11]['seed_decimal']
    def read(path):
        return panel if Path(path)==prep.design.PANEL else [dict(seed_decimal=collision)]
    monkeypatch.setattr(prep.c,'read',read)
    monkeypatch.setattr(prep.c,'artifact',lambda p:dict(path=Path(p).name,sha256='fake',bytes=2))
    monkeypatch.setattr(prep.subprocess,'run',lambda *a,**kw:subprocess.CompletedProcess([],0,'history_schedule.json\n',''))
    with pytest.raises(ValueError,match='Fresh seed collision; no replacement'):
        prep.seed_inventory()


def test_guardian_launches_only_new_version():
    root=Path(__file__).resolve().parents[1]
    source=(root/'tools/splendorust_pool6_strengthfirst_20261028/guardian.py').read_text()
    assert "'tools.splendorust_pool6_strengthfirst_20261028.runner'" in source
    assert "'tools.splendorust_pool6_strengthfirst_20261028.guardian'" in source
    assert "'tools.splendorust_pool6_live_20261027.runner'" not in source
    assert "'tools.splendorust_pool6_live_20261027.guardian'" not in source
