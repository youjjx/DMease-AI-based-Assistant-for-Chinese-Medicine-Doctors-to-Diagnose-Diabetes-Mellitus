import json

import numpy as np
import pytest

from dmease.local_extraction import validate_output, set_counts, summarize_counts
from dmease.affinity_research import pk_from_nm, scaffold_split, scaffold_bootstrap


def test_grounding_rejects_invented_quote_and_string_boolean():
    rows = [{"term": "口渴", "evidence": "患者口渴", "present": True, "canonical": "口渴"},
            {"term": "乏力", "evidence": "患者乏力", "present": True, "canonical": "乏力"},
            {"term": "口渴", "evidence": "患者口渴", "present": "false", "canonical": "口渴"},
            {"term": "口渴", "evidence": "患者口渴", "present": True, "canonical": "unknown"}]
    result = validate_output(json.dumps({"symptoms": rows}, ensure_ascii=False), "患者口渴，无头痛。", {"口渴", "乏力"})
    assert len(result["items"]) == 1
    assert result["rejected"] == {"ungrounded": 1, "schema": 1, "unknown_canonical": 1}
    assert result["proposed_items"] == 4


def test_malformed_response_is_not_silently_repaired():
    assert validate_output('```json\n{"symptoms":[]}\n```', "", set())["invalid_json"]
    assert validate_output('{"symptoms": [{"term": "口渴"}]}', "口渴", {"口渴"})["rejected"] == {"schema": 1}


def test_empty_reference_false_positive_remains_in_denominator():
    report = summarize_counts([set_counts({"a"}, {"a"}), set_counts(set(), {"b"})])
    assert report["precision"] == .5 and report["recall"] == 1
    assert report["f1"] == pytest.approx(2/3)
    assert report["records"] == 2 and report["exact_set_rate"] == .5


def test_affinity_units_and_invalid_values():
    assert pk_from_nm(1000) == 6
    assert pk_from_nm(10) == 8
    for invalid in [0, -1, float("inf"), float("nan")]:
        with pytest.raises(ValueError):
            pk_from_nm(invalid)


def test_scaffolds_keep_analogs_and_salts_together():
    pytest.importorskip("rdkit")
    from dmease.affinity_research import standardized_structure
    a, sa = standardized_structure("Cc1ccccc1")
    b, sb = standardized_structure("Oc1ccccc1")
    assert a != b and sa == sb and scaffold_split(sa) == scaffold_split(sb)
    neutral, sn = standardized_structure("Nc1ccccc1")
    salt, ss = standardized_structure("[NH3+]c1ccccc1.[Cl-]")
    assert neutral == salt and sn == ss
    assert standardized_structure("CCCC")[1] == "ACYCLIC"


def test_cluster_bootstrap_preserves_pairing():
    result = scaffold_bootstrap(np.array([1.,2.,3.]), np.array([1.,2.,3.]), np.array([2.,3.,4.]), ["a","a","b"], repeats=100)
    assert result["MAE_selected_minus_mean"] == -1
    assert result["ci95"] == [-1, -1] and result["scaffolds"] == 2


def test_format_adapter_preserves_content_and_rejects_partial_objects():
    from dmease.grounded_response import parse_grounded_response
    raw = '{"symptoms":[{"term":"口渴","evidence":"口渴","present":true,"canonical":"口渴"}]}'
    clean = parse_grounded_response(raw, "口渴", {"口渴"})
    quote = parse_grounded_response(raw+'"', "口渴", {"口渴"})
    fence = parse_grounded_response('```json\n'+raw+'\n```', "口渴", {"口渴"})
    assert clean["items"] == quote["items"] == fence["items"]
    assert quote["format_adjustment"] == "trailing_quote_after_complete_object"
    assert fence["format_adjustment"] == "outer_json_fence"
    for broken in (raw[:-1], raw+' extra advice', raw+'{}', raw+'""',raw+'{"symptoms":['):
        result = parse_grounded_response(broken, "口渴", {"口渴"})
        assert result["invalid_json"] and result["items"] == []
