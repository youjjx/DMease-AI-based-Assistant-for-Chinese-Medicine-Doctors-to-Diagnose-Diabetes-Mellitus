"""Rebuild the frozen affinity study from its licensed, published data snapshot.

Use a separate checkout to retain the released numerical files. This wrapper
handles a fresh checkout with published selection records but no local weights.
It never substitutes newly downloaded data for the frozen measured snapshot.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"experiments/v3"
PRIVATE = ROOT/"outputs/research_v3/affinity"


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def main():
    PRIVATE.mkdir(parents=True, exist_ok=True)
    reference = PRIVATE/"published_reference"; reference.mkdir(exist_ok=True)
    for path in OUT.glob("affinity*.json"):
        saved = reference/path.name
        if not saved.exists():
            saved.write_bytes(path.read_bytes())
    manifest = read(OUT/"affinity_sources.json")
    text = (OUT/"affinity_curated.json").read_text(encoding="utf8")
    choices = [text.encode(), text.replace("\n", "\r\n").encode()]
    content = next((c for c in choices if hashlib.sha256(c).hexdigest() == manifest["curated_sha256"]), None)
    if content is None:
        raise ValueError("Published measured snapshot checksum mismatch")
    destination = PRIVATE/"curated.json"
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != manifest["curated_sha256"]:
        raise ValueError("Existing private snapshot differs; use a separate checkout")
    destination.write_bytes(content)
    # Restore the original protocol newline representation after a Windows checkout.
    protocol = OUT/"protocol.json"
    expected = read(OUT/"affinity.json")["provenance"]["protocol_sha256"]
    normalized = protocol.read_text(encoding="utf8").encode()
    assert hashlib.sha256(normalized).hexdigest() == expected
    protocol.write_bytes(normalized)
    for selection_path in OUT.glob("affinity_selection_*.json"):
        selection = read(selection_path)
        if not all((PRIVATE/c["checkpoint"]).exists() for c in selection["candidates"]):
            # The published reference above remains intact; the core runner will
            # fit the same candidates and select on the same validation partition.
            selection_path.unlink()
    subprocess.run([sys.executable, str(ROOT/"scripts/affinity_benchmark.py")], cwd=ROOT, check=True)
    current, original = read(OUT/"affinity.json"), read(reference/"affinity.json")
    comparisons = []
    for name,result in current["strata"].items():
        old = original["strata"][name]
        assert result["counts"] == old["counts"] and result["status"] == old["status"]
        if result["status"] == "evaluated":
            differences = [abs(a["MAE_pK"]-b["MAE_pK"]) for a,b in zip(result["test"],old["test"])]
            comparisons.append({"stratum": name, "same_selected_index": result["selection"]["selected_index"] == old["selection"]["selected_index"],
                                "maximum_candidate_test_MAE_difference": max(differences)})
    print(json.dumps({"reproduction_comparison": comparisons}, indent=2))


if __name__ == "__main__":
    main()
