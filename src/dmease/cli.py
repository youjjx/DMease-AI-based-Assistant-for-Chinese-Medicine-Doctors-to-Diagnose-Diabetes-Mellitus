from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from dmease.pipeline import DMeasePipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DMease command line interface")
    parser.add_argument("--config", default="configs/dmease.yaml", help="Path to DMease YAML config")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("patients", help="List patient IDs in PatientDB")

    infer_parser = subparsers.add_parser("infer", help="Run single-patient inference")
    infer_parser.add_argument("--patient-id", required=True)
    infer_parser.add_argument("--output", default="", help="Optional JSON output path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pipeline = DMeasePipeline.from_config(args.config)

    if args.command == "patients":
        for patient_id in pipeline.patient_db.list_ids():
            print(patient_id)
        return

    if args.command == "infer":
        result = pipeline.infer(args.patient_id)
        payload = json.dumps(asdict(result), ensure_ascii=False, indent=2)
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload, encoding="utf-8")
        print(payload)


if __name__ == "__main__":
    main()

