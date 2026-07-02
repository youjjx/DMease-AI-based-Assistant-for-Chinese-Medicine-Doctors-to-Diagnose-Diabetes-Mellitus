from __future__ import annotations

import argparse
import json
from pathlib import Path

from dmease.io import write_jsonl
from dmease.semantic_parser import SemanticParser


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DMease triples from LLM JSON output files.")
    parser.add_argument("--input", required=True, help="Directory containing JSON files from the LLM extractor")
    parser.add_argument("--output", default="data/processed/triples.jsonl")
    args = parser.parse_args()

    parser_ = SemanticParser()
    triples = []
    for path in sorted(Path(args.input).glob("*.json")):
        raw = path.read_text(encoding="utf-8")
        for triple in parser_.extract_triples_from_llm_json(raw):
            row = triple.to_dict()
            row["source_file"] = path.name
            triples.append(row)

    write_jsonl(args.output, triples)
    print(f"wrote {len(triples)} triples to {args.output}")


if __name__ == "__main__":
    main()

