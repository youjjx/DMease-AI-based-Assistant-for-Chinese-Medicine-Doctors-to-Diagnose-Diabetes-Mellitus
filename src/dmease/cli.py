from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from dmease.pipeline import DMeasePipeline
from dmease.service import DMeaseService
from dmease.types import PatientProfile
import tempfile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DMease command line interface")
    parser.add_argument("--config", default="configs/dmease.yaml", help="Path to DMease YAML config")
    parser.add_argument("--legacy-nonclinical", action="store_true", help="Explicitly enable historical synthetic-data commands")

    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze_parser = subparsers.add_parser("analyze", help="Analyze using the verified public evidence bundle")
    analyze_parser.add_argument("--symptoms", required=True)
    analyze_parser.add_argument("--bundle", default=None, help="Local full bundle or published planning feature snapshot")
    analyze_parser.add_argument("--checkpoint", default="models/public_evidence_kan_ppo.pt")
    analyze_parser.add_argument("--exclude", nargs="*", default=[])
    analyze_parser.add_argument("--top", type=int, default=8)
    analyze_parser.add_argument("--strategy", choices=["greedy", "kan", "molecular_greedy", "molecular_kan", "molecular_guarded"], default=None)
    subparsers.add_parser("patients", help="List patient IDs in PatientDB")

    infer_parser = subparsers.add_parser("infer", help="Run single-patient inference")
    infer_parser.add_argument("--patient-id", required=True)
    infer_parser.add_argument("--output", default="", help="Optional JSON output path")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "analyze":
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(__file__).resolve().parents[2]
            default = root / "data/processed/evidence_bundle.json"
            published = root / "experiments/v5/planning_features.json"
            bundle = Path(args.bundle) if args.bundle else (default if default.exists() else published)
            if bundle.resolve() == published.resolve():
                from .molecular_feature_graph import published_service
                service = published_service(bundle, Path(temporary) / "analysis.db", args.checkpoint)
            else:
                service = DMeaseService(bundle, Path(temporary) / "analysis.db", args.checkpoint)
            patient = PatientProfile(name="CLI查询", complaint=args.symptoms, forbidden_herbs=args.exclude)
            if args.strategy and args.strategy.startswith("molecular_"):
                from .molecular_runtime import analyze
                result = analyze(service, patient, [], args.top, root, args.strategy)
            else:
                result = service.analyze(patient, top_herbs=args.top, strategy=args.strategy)
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return
    if not args.legacy_nonclinical:
        parser.error("Historical patient examples require --legacy-nonclinical. Use analyze for public evidence.")
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

