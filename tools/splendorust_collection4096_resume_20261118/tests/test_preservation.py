import json
import pytest
from tools.splendorust_collection4096_resume_20261118 import preservation as p


def test_parallel_hash_inventory_has_every_original_byte_and_no_input_writes(tmp_path,monkeypatch):
    source=tmp_path/'old';source.mkdir();(source/'nested').mkdir()
    (source/'a').write_bytes(b'original failure');(source/'nested/b').write_bytes(b'completed game')
    out=tmp_path/'new';monkeypatch.setattr(p.c,'OUT',out)
    before=[p.c.artifact(x) for x in p.files(source)]
    result=p.inventory(source,out/'prelaunch/inventory.jsonl')
    rows=[json.loads(x) for x in (out/'prelaunch/inventory.jsonl').read_text().splitlines()]
    assert rows==before==[p.c.artifact(x) for x in p.files(source)]
    assert result['files']==2 and result['bytes']==len(b'original failurecompleted game')
    assert result['models']==result['games']==0


def test_changed_evidence_during_hash_refused(tmp_path,monkeypatch):
    path=tmp_path/'record';path.write_bytes(b'old');artifact=p.c.artifact
    def altered(path):
        result=artifact(path);path.write_bytes(b'changed bytes');return result
    monkeypatch.setattr(p.c,'artifact',altered)
    with pytest.raises(ValueError,match='stable'):p.stable_artifact(path)


def test_cannot_inventory_into_original_tree(tmp_path,monkeypatch):
    monkeypatch.setattr(p.c,'OUT',tmp_path)
    with pytest.raises(ValueError):p.inventory(tmp_path,tmp_path/'prelaunch/inventory.jsonl')
