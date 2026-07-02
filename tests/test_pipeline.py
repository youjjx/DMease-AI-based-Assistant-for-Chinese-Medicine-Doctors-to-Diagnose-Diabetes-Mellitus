from __future__ import annotations

from dmease import DMeasePipeline


def test_example_inference_returns_prescription() -> None:
    pipeline = DMeasePipeline.from_config("configs/dmease.yaml")
    result = pipeline.infer("P2025-0001")

    assert result.patient_id == "P2025-0001"
    assert result.syndrome.syndrome == "气阴两虚证"
    assert result.targets
    assert result.prescription
    assert result.target_consistency_score > 0


def test_patient_constraints_remove_allergic_herb() -> None:
    pipeline = DMeasePipeline.from_config("configs/dmease.yaml")
    result = pipeline.infer("P2025-0002")

    prescribed = {step.herb for step in result.prescription}
    assert "黄连" not in prescribed
