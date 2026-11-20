"""Preserved partial streams, exact segmented seam, and disclosed sensitivity."""
from copy import deepcopy
from pathlib import Path
import gzip
import zlib
import pytest
from tools.splendorust_search256_recovery_20261030 import analysis,audit,verification,schedule,common as c


def test_incomplete_gzip_is_preserved_without_fabricated_closure(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path)
    payload=b'{"response":1}\n'*100
    compressor=zlib.compressobj(wbits=31)
    stored=compressor.compress(payload)+compressor.flush(zlib.Z_SYNC_FLUSH)
    path=tmp_path/'stdout.log.gz';path.write_bytes(stored);pin=c.artifact(path)
    raw,proof=audit.decode_interrupted_gzip(path,pin)
    assert raw==payload and not proof['gzip_complete'] and not proof['normal_closure_proven']
    assert path.read_bytes()==stored and c.artifact(path)==pin


def test_interrupted_stream_changed_hash_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path);path=tmp_path/'events.gz'
    path.write_bytes(gzip.compress(b'one\n'));pin=c.artifact(path);path.write_bytes(gzip.compress(b'two\n'))
    with pytest.raises(ValueError):audit.decode_interrupted_gzip(path,pin)


def test_multimember_stream_not_silently_dropped(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'ROOT',tmp_path);path=tmp_path/'events.gz'
    path.write_bytes(gzip.compress(b'one\n')+gzip.compress(b'two\n'))
    with pytest.raises(ValueError,match='extra gzip'):audit.decode_interrupted_gzip(path,c.artifact(path))


def seam_rows():
    prefix=[dict(ply=i,after={'turn':i+1}) for i in range(30)]
    first=dict(ply=30,before={'turn':30},player_role='challenger',route='research',mode='research',quota=256,
        clock={'request_id':'request-00000003'})
    return prefix,prefix+[first],schedule.make_schedule('main')[235]


def test_exact_seam_preserves_all30_moves():
    prefix,rows,spec=seam_rows();proof=audit.validate_segment_seam(rows,prefix,spec)
    assert proof['retained_moves']==30 and proof['authorized_fresh_move_clocks']==1


@pytest.mark.parametrize('change',('prefix','ply','snapshot','role','quota','request','game'))
def test_seam_rejects_changed_prefix_or_retry(change):
    prefix,rows,spec=seam_rows();rows=deepcopy(rows);spec=deepcopy(spec)
    if change=='prefix':rows[1]['after']={'changed':True}
    if change=='ply':rows[30]['ply']=31
    if change=='snapshot':rows[30]['before']={}
    if change=='role':rows[30]['player_role']='opponent'
    if change=='quota':rows[30]['quota']=128
    if change=='request':rows[30]['clock']['request_id']='request-00000002'
    if change=='game':spec['game_index']=236
    with pytest.raises(ValueError):audit.validate_segment_seam(rows,prefix,spec)


def test_source_dispatch_retains_old_prefix_and_235_completed_games():
    specs=schedule.make_schedule('main')
    for game,ply,old in ((234,80,True),(235,29,True),(235,30,False),(236,0,False)):
        path=verification.source_move_path(specs[game],ply)
        assert path.is_relative_to(verification.OLD_OUT) is old
    assert verification.source_move_path(schedule.make_schedule('operational')[0],0).is_relative_to(verification.OLD_OUT)


def test_normal_later_game_does_not_require_interrupted_prefix(monkeypatch):
    monkeypatch.setattr(audit,'audit_game',lambda *args,**kwargs:{'ordinary':True})
    assert verification.audit_merged_game({},[],{},schedule.make_schedule('main')[236],{}, {})=={'ordinary':True}


def artificial_games():
    return [dict(s,complete=True,reason='natural',winner=s['challenger_seat'] if s['pair_index']<120 else 1-s['challenger_seat'])
        for s in schedule.make_schedule('main')]


def test_retry_sensitivity_uses_full_sample_without_changing_outcomes(monkeypatch):
    monkeypatch.setattr(analysis,'REPLICATES',1000)
    games=artificial_games();before=deepcopy(games)
    official=analysis.analyze_main(games,True)
    report=analysis.analyze_recovery_sensitivity(games,official)
    assert report['recovery_conservative']['mean_score']==239/400 and official['official']['mean_score']==.6
    assert report['recovery_conservative']['games']==400 and games==before
    proof=analysis.verify_recovery_sensitivity(games,official,report)
    assert proof['independent_integer_arithmetic']
    report['recovery_conservative']['numerator']+=1
    with pytest.raises(ValueError):analysis.verify_recovery_sensitivity(games,official,report)


def test_retry_plus_admin_sensitivity_does_not_double_deduct(monkeypatch):
    monkeypatch.setattr(analysis,'REPLICATES',100)
    games=artificial_games();row=games[235];loser=1-row['challenger_seat']
    row.update(reason='forfeit',loser=loser,winner=1-loser,failed_seat=loser,failed_role='opponent',
        administrative_forfeit=True,failure_policy=analysis.FAILURE_POLICY,failure='research_outer_hard_guard')
    official=analysis.analyze_main(games,True);result=analysis.analyze_recovery_sensitivity(games,official)
    assert result['recovery_conservative']==official['conservative']


def test_prefix_changes_cannot_be_laundered_as_clean_games(monkeypatch):
    prefix,rows,spec=seam_rows();rows[30]['clock']['request_id']='request-00000017'
    with pytest.raises(ValueError):verification.audit_merged_game({},rows,{},spec,{'prefix':{'initial':{},'rows':prefix}}, {})


def test_original_directory_extra_file_rejected(tmp_path,monkeypatch):
    monkeypatch.setattr(verification,'OLD_OUT',tmp_path);monkeypatch.setattr(c,'ROOT',tmp_path)
    (tmp_path/'extra').write_text('preserve me')
    with pytest.raises(ValueError,match='omitted, added'):
        verification.verify_prior_preservation({'inventory_map':{}},type('Guard',(),{'guard':lambda s:None})())


def clock_fixture(monkeypatch):
    origin=c.PRIOR_ORIGIN;wall=c.PRIOR_WALL_START
    def start(offset,deadline):
        return dict(monotonic=origin+offset,unix_time=wall+offset,
            monotonic_deadline=origin+deadline,unix_deadline=wall+deadline)
    def end(offset):return dict(monotonic=origin+offset,unix_time=wall+offset)
    items={str(verification.OLD_OUT/'stages/main_start.json'):dict(monotonic=origin+2000,unix_time=wall+2000,
        monotonic_deadline=c.ORIGINAL_MAIN_MONOTONIC_DEADLINE,unix_deadline=c.ORIGINAL_MAIN_UNIX_DEADLINE),
        str(c.OUT/'guardian_claim.json'):dict(owner_started_monotonic=origin+40000,owner_started_unix_time=wall+40000),
        str(c.OUT/'stages/recovery_qualification_start.json'):start(40010,47200),
        str(c.OUT/'stages/recovery_qualification_end.json'):end(40100),
        str(c.OUT/'stages/main_start.json'):start(40101,100000),
        str(c.OUT/'stages/main_end.json'):end(50000),
        str(c.OUT/'stages/main_verification_start.json'):start(50001,71601),
        str(c.OUT/'stages/main_verification_end.json'):end(51000),
        str(c.OUT/'stages/final_start.json'):start(51001,94201)}
    monkeypatch.setattr(c,'read',lambda path:items[str(path)])
    return items,type('Guard',(),dict(start=origin,wall_start=wall))()


def test_recovery_stage_clocks_keep_original_origin_and_owner_allowance(monkeypatch):
    items,envelope=clock_fixture(monkeypatch)
    assert verification.verify_stage_clocks(envelope)['status']=='passed'


def test_recovery_stage_cannot_reset_owner_qualification_clock(monkeypatch):
    items,envelope=clock_fixture(monkeypatch)
    items[str(c.OUT/'stages/recovery_qualification_start.json')]['monotonic_deadline']+=1
    with pytest.raises(ValueError,match='owning preparation'):verification.verify_stage_clocks(envelope)


def test_resumed_main_cannot_extend_original_stage_deadline(monkeypatch):
    items,envelope=clock_fixture(monkeypatch)
    items[str(c.OUT/'stages/main_end.json')]['monotonic']=c.ORIGINAL_MAIN_MONOTONIC_DEADLINE
    with pytest.raises(ValueError,match='entire interruption'):verification.verify_stage_clocks(envelope)
