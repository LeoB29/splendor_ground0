import pytest
from . import common as c


def test_output_namespace_protects_current_study(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        c.new_json(c.STUDY_ROOT / "not_allowed.json", {})
    assert not (c.STUDY_ROOT / "not_allowed.json").exists()


def test_presence_reads_no_file_contents(monkeypatch, tmp_path):
    monkeypatch.setattr(c, "STUDY_ROOT", tmp_path)
    (tmp_path / "main").mkdir()
    (tmp_path / "final_status.json").write_text("not json", encoding="utf-8")
    result = c.blind_presence()
    assert result["main"] is True and result["final_status.json"] is True
    assert result["probe"] is False


def test_hash_rejects_mutation_and_path_escape(monkeypatch, tmp_path):
    monkeypatch.setattr(c, "ROOT", tmp_path)
    path = tmp_path / "data.json"
    path.write_text('{"train": true}', encoding="utf-8")
    ref = c.artifact(path)
    assert c.verified(ref) == {"train": True}
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        c.verify_hash(ref)
    with pytest.raises(ValueError, match="escapes"):
        c.verify_hash({"path": "../elsewhere", "sha256": "0"})


def test_registration_requires_pre_result_presence(monkeypatch, tmp_path):
    monkeypatch.setattr(c, "OUTPUT", tmp_path)
    monkeypatch.setattr(c, "blind_presence", lambda: {"main": True})
    with pytest.raises(ValueError, match="materialized"):
        c.register(tmp_path / "absent.json")
