"""Model-free saved timing extraction regressions; no game fixtures constructed."""
import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from tools import splendorust_search128_saved_timing_20261029 as m


SAVED = 'outputs/splendorust_search128_strength_20261028_01/games/main/p016_g33/moves/0017.json'


def saved():
    return json.loads((m.ROOT/SAVED).read_text())


def test_real_saved_row_clock_payload_sidecars_and_partition():
    row = saved()
    reader = m.Reader()
    paths = [SAVED, row['worker_clock_file'], row['parent_profile_file'], row['child_profile_file'],
             row['worker_folder']+'/inputs/'+row['clock']['request_id']+'.json',
             row['worker_folder']+'/placement_requests/'+row['clock']['request_id']+'.json']
    for path in paths:
        reader.add(m.descriptor(m.ROOT/path))
    selected = {k: row[k] for k in ('stage','game_index','pair_index','ply','seat','quota')}
    selected.update(row=reader.pins[SAVED], role='challenger', receipt_elapsed_seconds=2.0121378)
    bound = m.bound_row(reader, selected)
    result = m.extract(bound)
    assert result['partition_total_ns'] == 2012137800
    assert sum(result['partition_ns'].values()) == 2012137800
    assert result['partition_ns']['whole_handler'] == 1887434100
    assert result['sampled_request_tree_cpu_seconds'] == .765625
    assert result['inference_calls'] == 129
    assert 'native_search_wall_time' in result['unavailable']
    # Legitimately overlapping dispatch/child read remains signed and is not
    # added to the exclusive trusted-arrival partition.
    assert result['overlapping_diagnostics_ns']['child_read_minus_parent_dispatch'] < 0
    reader.recheck()


@pytest.mark.parametrize('path', ['../x.json', 'a/../x.json', '/tmp/x.json', 'C:/x.json', 'a\\x.json'])
def test_path_escape_rejected(tmp_path, path):
    with pytest.raises(ValueError, match='path escape|noncanonical'):
        m.Reader(tmp_path).add(dict(path=path, bytes=1, sha256='0'*64))


def test_hash_tamper_and_conflicting_pin(tmp_path):
    p = tmp_path/'data.json'
    p.write_text('{}')
    pin = m.descriptor(p, tmp_path)
    reader = m.Reader(tmp_path)
    reader.add(pin)
    p.write_text('[]')  # same byte count
    with pytest.raises(ValueError, match='hash mismatch'):
        reader.read(pin)
    with pytest.raises(ValueError, match='conflicting'):
        reader.add(dict(pin, sha256='0'*64))


def test_consumed_input_mutation_rejected(tmp_path):
    p = tmp_path/'data.json'
    p.write_text('{}')
    reader = m.Reader(tmp_path)
    reader.read(m.descriptor(p, tmp_path))
    p.write_text('[]')
    with pytest.raises(ValueError, match='consumed input changed'):
        reader.recheck()


def test_read_access_time_change_allowed_but_content_tamper_rejected(tmp_path, monkeypatch):
    p=tmp_path/'data.json'
    p.write_text('{}')
    reader=m.Reader(tmp_path)
    reader.read(m.descriptor(p,tmp_path))
    original_open=Path.open
    before=p.stat()
    @contextmanager
    def open_with_atime_change(path,*args,**kwargs):
        with original_open(path,*args,**kwargs) as stream:
            yield stream
        if path == p:
            st=p.stat()
            m.os.utime(p, ns=(st.st_atime_ns+2_000_000_000, st.st_mtime_ns))
    monkeypatch.setattr(Path,'open',open_with_atime_change)
    reader.recheck()
    after=p.stat()
    assert after.st_atime_ns != before.st_atime_ns
    assert (after.st_ino,after.st_size,after.st_mtime_ns)==(before.st_ino,before.st_size,before.st_mtime_ns)
    with original_open(p,'w') as f:
        f.write('[]')
    # Even restoration of the original modification timestamp cannot hide a
    # same-size content mutation because the SHA must still agree.
    m.os.utime(p,ns=(after.st_atime_ns,before.st_mtime_ns))
    with pytest.raises(ValueError,match='consumed input changed'):
        reader.recheck()


def test_missing_marker_is_not_imputed():
    row = saved()
    row['child_profile']['profiles'][0].pop('handle_end_ns')
    with pytest.raises(ValueError, match='missing/invalid ordered markers'):
        m.extract(row)


def test_partition_disagreement_and_unordered_marker_rejected():
    row = saved()
    row['clock']['receipt_elapsed_seconds'] += .001
    with pytest.raises(ValueError, match='partition disagrees'):
        m.extract(row)
    row = saved()
    row['child_profile']['pre_write_ns'] = row['parent_profile']['receipt_arrival_ns']+1
    with pytest.raises(ValueError, match='unordered'):
        m.extract(row)


def test_wrong_quota_and_sidecar_rejected():
    row = saved()
    selected = {k:row[k] for k in ('stage','game_index','pair_index','ply','seat','quota')}
    selected.update(row={}, role='challenger', receipt_elapsed_seconds=2.0121378)
    class MemoryReader:
        def read(self, target):
            if target == {}:
                return row
            return {'c':row['clock'], 'p':row['parent_profile'], 's':row['child_profile']}[target]
    row.update(worker_clock_file='c', parent_profile_file='p', child_profile_file='s')
    row['quota'] = 64
    with pytest.raises(ValueError, match='quota'):
        m.bound_row(MemoryReader(), selected)
    row['quota'] = 128
    row['child_profile']['quota'] = 64
    with pytest.raises(ValueError, match='child result identity'):
        m.bound_row(MemoryReader(), selected)


def candidate(game, ply, seconds, quota=128):
    return dict(stage='main', game_index=game, pair_index=game//2, ply=ply, seat=ply%2,
        quota=quota, role='challenger' if quota==128 else 'opponent',
        row=dict(path=f'game{game}/moves/{ply:04}.json', bytes=1, sha256='0'*64),
        source_freeze_sha256='a'*64, receipt_elapsed_seconds=seconds)


def control_fixture():
    rows = [candidate(0, 2*i, 2.-i*.01) for i in range(10)]
    rows += [candidate(0, 20, .9), candidate(0, 22, .8)]
    rows += [candidate(0, i, .4, 64) for i in range(1, 22, 2)]
    for game in m.RULE['systematic_games']:
        rows += [candidate(game-1, i*2, .5) for i in range(12)]
    return rows


def test_fixed_selection_reuse_missing_and_systematic_ignores_timing():
    rows = control_fixture()
    panel = m.select_panel(rows)
    assert len(panel['slots']) == 50 and panel['distinct_rows'] <= 50
    assert panel['reused_slots'] > 0
    assert panel['missing_slots'] == 1  # no preceding opponent at ply-1
    assert all(x['ply'] == 18 for x in panel['rows'] if
               any(g['group']=='systematic128' for g in x['memberships']))
    assert all(s['row']=='game0/moves/0020.json' for s in panel['slots']
               if s['group']=='same_game_128_control')
    assert not any(r['stage']=='operational' for r in panel['rows'])


def test_tie_earlier_and_missing_matches_not_replaced():
    rows = control_fixture()
    rows += [candidate(0, 24, 2.1), candidate(0, 26, .8)]
    panel = m.select_panel(rows)
    slow1control = next(s for s in panel['slots'] if s['group']=='same_game_128_control' and s['target']==1)
    assert slow1control['row']=='game0/moves/0022.json'
    rows = [r for r in rows if not (r['game_index']==0 and r['quota']==128 and r['receipt_elapsed_seconds']<=1)]
    panel = m.select_panel(rows)
    assert all(s['row'] is None for s in panel['slots'] if s['group']=='same_game_128_control')


def test_descriptive_thresholds_exact_and_missing_cpu():
    result = m.distribution([.7,.9,1.,1.01])
    assert (result['above_700ms'],result['above_900ms'],result['above_1s'])==(3,2,1)
    row=saved()
    row['clock']['telemetry'].pop('request_cpu_seconds')
    assert m.extract(row)['sampled_request_tree_cpu_seconds'] is None


def test_population_segments_are_descriptive_and_reconcile():
    rows=[candidate(0,0,1.2),candidate(0,1,.4,64),candidate(39,0,.8)]
    rows[-1]['source_freeze_sha256']='b'*64
    groups=m.summarize_population(rows)
    group=next(g for g in groups if g['stage']=='main' and g['quota']==128)
    assert group['count']==2 and group['above_1s']==1
    assert sum(s['count'] for s in group['source_segments'])==2
    assert sum(s['count'] for s in group['fixed_game_blocks'])==2
    assert sum(g['above_1s'] for g in group['games'])==1


def test_deadline_cannot_reset():
    guard=m.Guard.__new__(m.Guard)
    guard.start=m.time.monotonic()-1
    guard.start_unix=m.time.time()-1
    guard.seconds=0
    with pytest.raises(ValueError, match='deadline'):
        guard.check()


@pytest.mark.skipif(m.os.name != 'nt', reason='Windows native resource API')
def test_actual_initial_resource_probe(tmp_path):
    guard=m.Guard(tmp_path)
    assert 0 < guard.peak_rss <= m.LIMITS['maximum_rss_bytes']
    guard.check()


def test_fresh_reader_restores_real_inventory_for_compact_population():
    # Real saved schema regression: a new process starts from finite consumed
    # descriptors, not prepare's in-memory159151-entry authority. No model/game
    # execution or full historical file rehash occurs.
    reader=m.Reader()
    reader.add(m.ANCHOR)
    m.authority(reader)
    rows, bridges=m.population(reader)
    assert len(rows)==22619 and len(bridges)==1
    assert len(m.select_panel(rows)['slots'])==50
    assert len(reader.consumed)<1000

