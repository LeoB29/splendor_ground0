"""Actual request/publication/collection seam without invoking a player."""
from copy import deepcopy
from types import SimpleNamespace
import threading
import pytest
from tools.splendorust_selfplay_pilot_v2_20261102 import common as root
from tools.splendorust_selfplay_pilot_v2_20261102.accounting import WorkerAccounting,LOG_NAMES
from tools.splendorust_selfplay_pilot_v2_20261102.assessment import transport as t,common as c
from tools.splendorust_boundary_timing_20261025.child_profile import publish_profile


@pytest.mark.parametrize('tamper',[False,True])
def test_actual_transport_publication_binds_learned128_graph_and_one_gib(monkeypatch,tmp_path,tamper):
    monkeypatch.setattr(root,'ROOT',tmp_path);monkeypatch.setattr(root,'OUT',tmp_path/'scope')
    monkeypatch.setattr(root,'WRITE_OBSERVER',None)
    monkeypatch.setattr(c,'ROOT',tmp_path);monkeypatch.setattr(c,'OUT',root.OUT/'assessment')
    folder=c.OUT/'workers/operational_g000_candidate'
    account=WorkerAccounting(folder);folder.mkdir(parents=True)
    env=SimpleNamespace(time_guard=lambda:None,guard=lambda:None,remaining=lambda:10.)
    child=SimpleNamespace(output_dir=folder,_log_lock=threading.Lock(),_logs={x:None for x in LOG_NAMES},
        max_log_bytes=root.GiB,_closed=False,_counter=2,process=SimpleNamespace(pid=123),
        challenger_quota=128,placement_mode='isolated',_placement_freeze={},recovery_accounting=account,
        recovery_envelope=env,initialization={'result':{'graph_identity_sha256':'a'*64}})
    account.bind_child(child);account.finish_initialization()
    monkeypatch.setattr(t.placement,'verify_runtime',lambda *args,**kwargs:{'passed':True})
    def reply(payload,timeout,validate=None):
        assert timeout==10. and account.state=='request'
        profile=deepcopy(account._publication['expected'])
        assert profile['quota']==128 and profile['graph_identity_sha256']=='a'*64
        if tamper:profile['graph_identity_sha256']='b'*64
        publish_profile(folder,profile['request_id'],profile,batch=False)
        child._counter+=1;child.last_parent_profile={'request_id':profile['request_id']}
        return {'request_id':profile['request_id'],'status':'ok'}
    child.request=reply
    receipt=t.request(child,{'player':'search','observer':{'public':'fixture'}},10.)
    result=t.collect_profiles(child,receipt,env)
    assert receipt=={'request_id':'request-00000003','status':'ok'}
    if tamper:
        assert result['profile_capture_status']=='evidence_error'
        with pytest.raises(ValueError):t._require_profiles(result)
    else:
        t._require_profiles(result)
        assert result['profile_capture_status']=='complete' and account.state=='idle'
        assert result['child_profile']['graph_identity_sha256']=='a'*64
    assert (folder/'placement_requests/request-00000003.json').is_file()
