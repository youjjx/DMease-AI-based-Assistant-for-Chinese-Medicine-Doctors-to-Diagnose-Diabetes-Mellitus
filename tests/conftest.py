"""Small explicit test fixtures; never installed as application knowledge."""
import json
import pytest
import torch

from dmease.evidence_policy import canonical_hash


@pytest.fixture
def evidence_path(tmp_path):
    torch.set_num_threads(1)
    symptoms = [{"id": f"s{i}", "name": name, "aliases": aliases, "source_url": "https://example.invalid/test-fixture"}
                for i, (name, aliases) in enumerate([
                    ("口渴", ["口渴多饮", "口干"]), ("乏力", ["疲劳", "容易疲劳"]),
                    ("肢体麻木", ["手脚麻木"]), ("尿频", ["尿多"]),
                ])]
    herbs = [{"id": f"h{i}", "name": name, "source_url": "https://example.invalid/test-fixture"}
             for i, name in enumerate(["黄芪", "麦冬", "山药"])]
    def edge(a, b, eligible=True):
        return {"source": a, "target": b, "eligible": eligible, "evidence": {"fixture": True}, "source_url": "https://example.invalid/test-fixture"}
    data = {"schema": "dmease-public-evidence-v1", "data_scope": "test-fixture-not-public-data",
            "symptoms": symptoms, "herbs": herbs, "syndromes": {"z1": {"id": "z1", "name": "测试证候"}},
            "targets": {"t1": {"id": "t1", "name": "T1"}, "t2": {"id": "t2", "name": "T2"}},
            "symptom_herb": [edge("s0", "h0"), edge("s0", "h1"), edge("s1", "h2")],
            "symptom_syndrome": [edge("s0", "z1"), edge("s1", "z1")],
            "symptom_target": [edge("s0", "t1"), edge("s1", "t2")],
            "herb_target": [edge("h0", "t1"), edge("h1", "t1"), edge("h2", "t2")],
            "ppi": [{"source": "t1", "target": "t2", "score": .9}], "files": []}
    data["fingerprint"] = canonical_hash({k: v for k, v in data.items() if k != "files"})
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf8")
    return path
