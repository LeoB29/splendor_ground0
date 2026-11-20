"""Synthetic profile/trace files; no engine, checkpoints or model imports."""
from copy import deepcopy
import cProfile
import gzip
import hashlib
import json
import math
import zlib

import pytest

from tools.move_evaluation_profile_20261023 import analyze as a


def artifact(path,root):
    return {'path':path.relative_to(root).as_posix(),'bytes':path.stat().st_size,
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def profile_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(a,'ROOT',tmp_path)
    path=tmp_path/'synthetic.prof'
    profiler=cProfile.Profile()
    profiler.runcall(lambda:sum(math.sqrt(i) for i in range(100)))
    profiler.dump_stats(str(path))
    return path,artifact(path,tmp_path)


def test_profile_exclusive_partition_and_separate_nonadditive_rows(tmp_path,monkeypatch):
    path,item=profile_fixture(tmp_path,monkeypatch)
    result=a.summarize_profile(path,item)
    groups=result['exclusive_additive_categories']
    assert math.isclose(sum(g['self_seconds'] for g in groups),result['profiled_python_observed_seconds'],abs_tol=1e-9)
    assert math.isclose(sum(g['profiled_self_time_fraction'] for g in groups),1.)
    assert result['functions']==len(result['function_call_counts'])
    assert result['interpretation']['cumulative_rows_overlap']
    assert not result['interpretation']['native_child_internals_profiled']
    assert result['top_self_time'] and result['top_cumulative_time_nonadditive']
    assert result['total_calls']>=result['primitive_calls']>0


@pytest.mark.parametrize('damage',['hash','size','path','absolute','bool_size','extra'])
def test_artifact_is_checked_before_profiler_load(tmp_path,monkeypatch,damage):
    path,item=profile_fixture(tmp_path,monkeypatch)
    if damage=='hash':item['sha256']='0'*64
    elif damage=='size':item['bytes']+=1
    elif damage=='path':item['path']='../synthetic.prof'
    elif damage=='absolute':item['path']=str(path)
    elif damage=='bool_size':item['bytes']=True
    else:item['extra']='bad'
    monkeypatch.setattr(a.pstats,'Stats',lambda *_:pytest.fail('Must authenticate before loading'))
    with pytest.raises(ValueError):a.summarize_profile(path,item)


def row(index,support=1):
    wi,ci=divmod(index,support)
    return {'world_index':wi,'candidate_index':ci,'seed':f'seed{wi}',
            'steps':[{'policy':'forced','forced':True,'legal_count':4},
                     {'policy':'raw035','forced':False,'legal_count':9},
                     {'policy':'greedy_entity','forced':False,'legal_count':7,
                      'policy_trace':{'inference_calls':1,'phases':[{'branch':'neural_greedy'},{'branch':'auxiliary_heuristic'}]}}]}


def encoded(rows):
    return ''.join(a.canonical(r)+'\n' for r in rows).encode()


def runs(tmp_path,monkeypatch,support=1):
    monkeypatch.setattr(a,'ROOT',tmp_path)
    rows=[row(i,support) for i in range(128*support)]
    raw=gzip.compress(encoded(rows),mtime=0)
    paths=[tmp_path/'plain.gz',tmp_path/'profiled.gz']
    for path in paths:path.write_bytes(raw)
    base={'decision':{'action':{'kind':'fake'},'record':{'candidates':{'actions':[{'fake':i} for i in range(support)]},
                                                      'worlds':[{'seed':f'seed{i}'} for i in range(128)]}},
          'profile_id':'frozen','observer_sha256':'a'*64,'ruleset_id':'corrected','catalogue_sha256':'b'*64}
    values=[{**deepcopy(base),'trace':{**artifact(p,tmp_path),'records':128*support}} for p in paths]
    return values,paths,rows


def test_scientific_equality_ignores_location_only_and_counts_work(tmp_path,monkeypatch):
    values,_,_=runs(tmp_path,monkeypatch,4); before=deepcopy(values)
    got=a.compare_scientific_runs(*values)
    assert values==before
    assert got['status']=='exact_scientific_match' and got['trajectory_records']==512 and got['moves']==1536
    assert got['policy_moves']=={'forced':512,'raw035':512,'greedy_entity':512}
    assert got['entity_inference_calls']==512
    assert got['entity_phase_branches']=={'neural_greedy':512,'auxiliary_heuristic':512}
    assert got['legal_actions_considered_by_policy']=={'forced':2048,'raw035':4608,'greedy_entity':3584}
    assert not got['independent_policy_transition_replay_performed_here']


@pytest.mark.parametrize('damage',['decision','extra','profile','trace_hash','trace_size','count','record_bool','seed','policy','order','truncated'])
def test_scientific_mismatch_rejected(tmp_path,monkeypatch,damage):
    values,paths,rows=runs(tmp_path,monkeypatch)
    if damage=='decision':values[1]['decision']['action']={'kind':'different'}
    elif damage=='extra':values[1]['timing']=1.
    elif damage=='profile':values[1]['profile_id']='other'
    elif damage=='trace_hash':values[1]['trace']['sha256']='0'*64
    elif damage=='trace_size':values[1]['trace']['bytes']+=1
    elif damage=='count':values[1]['trace']['records']=127
    elif damage=='record_bool':values[1]['trace']['records']=True
    else:
        if damage=='seed':rows[0]['seed']='changed'
        elif damage=='policy':rows[0]['steps'][0]['policy']='raw035'
        elif damage=='order':rows[1]['world_index']=0
        raw=gzip.compress(encoded(rows),mtime=0)
        if damage=='truncated':raw=raw[:-8]
        for i,path in enumerate(paths):
            path.write_bytes(raw); values[i]['trace'].update(artifact(path,tmp_path))
    with pytest.raises((ValueError,EOFError,gzip.BadGzipFile)):a.compare_scientific_runs(*values)


def prefix_files(tmp_path,monkeypatch,count=296,tail=True):
    monkeypatch.setattr(a,'ROOT',tmp_path)
    rows=[row(i,4) for i in range(512)]
    text=encoded(rows[:count])+(encoded([rows[count]])[:23] if tail else b'')
    compressor=zlib.compressobj(wbits=31)
    old=tmp_path/'old.gz'; old.write_bytes(compressor.compress(text)+compressor.flush(zlib.Z_SYNC_FLUSH))
    new=tmp_path/'new.gz';new.write_bytes(gzip.compress(encoded(rows),mtime=0))
    return old,new,rows


def test_truncated_gzip_prefix_only_complete296_records_are_compared(tmp_path,monkeypatch):
    old,new,_=prefix_files(tmp_path,monkeypatch)
    got=a.inspect_old_prefix(old,new)
    assert got['compared_complete_records']==296 and got['compared_moves']==888
    assert got['last_complete_old_record']=={'world_index':73,'candidate_index':3}
    assert got['new_complete_records']==512 and got['incomplete_tail_bytes']==23 and got['incomplete_gzip']
    assert got['complete_prefix_bytes_equal'] and not got['gzip_eof']
    assert not got['partial_tail_used_as_scientific_evidence'] and not got['original_timed_out_request_reclassified']


@pytest.mark.parametrize('count',[0,295,297])
def test_unexpected_old_complete_count_rejected(tmp_path,monkeypatch,count):
    old,new,_=prefix_files(tmp_path,monkeypatch,count=count)
    with pytest.raises(ValueError):a.inspect_old_prefix(old,new)


def test_old_complete_gzip_cannot_masquerade_as_interrupted_prefix(tmp_path,monkeypatch):
    old,new,rows=prefix_files(tmp_path,monkeypatch)
    old.write_bytes(gzip.compress(encoded(rows[:296]),mtime=0))
    with pytest.raises(ValueError,match='incomplete'):a.inspect_old_prefix(old,new)


@pytest.mark.parametrize('damage',['mismatch','short','truncated','order','extra_stream'])
def test_old_prefix_and_new_completeness_tampering(tmp_path,monkeypatch,damage):
    old,new,rows=prefix_files(tmp_path,monkeypatch)
    if damage=='mismatch':rows[100]['steps'][1]['legal_count']=12
    elif damage=='short':rows=rows[:295]
    elif damage=='order':rows[7]['candidate_index']=0
    raw=gzip.compress(encoded(rows),mtime=0)
    if damage=='truncated':raw=raw[:-8]
    if damage=='extra_stream':old.write_bytes(old.read_bytes()+b'garbage')
    new.write_bytes(raw)
    with pytest.raises((ValueError,EOFError,gzip.BadGzipFile,zlib.error)):a.inspect_old_prefix(old,new)


def test_categories_are_conservative_and_disjoint():
    assert a._category(('C:/repo/tools/external_strength_screen_20261018_02/bridge.py',10,'encode_observation'))=='opponent_bridge_orchestration'
    assert a._category(('C:/repo/tools/external_strength_screen_20261018/rules.py',100,'legal_actions'))=='legal_action_enumeration'
    assert a._category(('C:/repo/src/splendor_ai/engine/env.py',180,'step'))=='rule_transition_wrappers'
    assert a._category(('C:/Python/Lib/site-packages/torch/nn/module.py',20,'forward'))=='torch_and_identified_tensor_operations'
    assert a._category(('~',0,'<built-in method read of _io.BufferedReader>'))=='io_and_wait_shared_call_sites'


def test_cprofile_caller_edges_have_total_before_primitive(tmp_path,monkeypatch):
    from types import SimpleNamespace
    path,item=profile_fixture(tmp_path,monkeypatch)
    caller=('synthetic.py',3,'recursive_parent')
    callee=('~',0,'<built-in method read of _io.BufferedReader>')
    fake=SimpleNamespace(stats={callee:(2,5,.5,.7,{caller:(5,2,.5,.7)})},total_tt=.5)
    monkeypatch.setattr(a.pstats,'Stats',lambda _:fake)
    got=a.summarize_profile(path,item)['io_caller_edges_nonadditive'][0]
    assert got['total_calls']==5 and got['primitive_calls']==2


def test_profile_changed_during_parse_is_rejected(tmp_path,monkeypatch):
    path,item=profile_fixture(tmp_path,monkeypatch); original=a.pstats.Stats
    def changed(p):
        stats=original(p)
        with path.open('ab') as f:f.write(b'changed')
        return stats
    monkeypatch.setattr(a.pstats,'Stats',changed)
    with pytest.raises(ValueError,match='mismatch'):a.summarize_profile(path,item)
