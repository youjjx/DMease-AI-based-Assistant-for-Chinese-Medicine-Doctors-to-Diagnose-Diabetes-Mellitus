from __future__ import annotations

from dataclasses import asdict
import json

from dmease import DMeasePipeline


def main() -> None:
    pipeline = DMeasePipeline.from_config("configs/dmease.yaml")
    result = pipeline.infer("P2025-0001")
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
