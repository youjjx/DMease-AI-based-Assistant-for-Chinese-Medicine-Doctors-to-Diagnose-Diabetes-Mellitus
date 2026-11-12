import importlib.util
from pathlib import Path

import pytest

ROOT=Path(__file__).parents[1]


def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/f"scripts/{name}.py")
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


def test_external_denominator_keeps_out_of_scope_and_abstention():
    audit=module("external_syndrome_audit")
    rows=[{"gold":["A"],"predicted":["A"]},{"gold":["outside"],"predicted":[]},{"gold":["B","A"],"predicted":["B"]}]
    result=audit.metrics(rows,{"A","B"})
    assert result["records"]==3 and result["abstentions"]==1
    assert result["top1_any_gold_hit"]==pytest.approx(2/3)
    assert result["single_label_records"]==2
    assert result["single_label_top1_accuracy"]==.5
    assert result["multilabel_micro_f1"]==pytest.approx(4/6)


def test_benchmark_text_identity_and_topk():
    benchmark=module("tcm_sd_benchmark")
    assert benchmark.digest("口渴\n乏力")==benchmark.digest("口渴 乏力")
    result=benchmark.scores(["A","B"],[["A"],["A","B"]],["A","B"])
    assert result["accuracy"]==.5 and result["top3_hit"]==1
