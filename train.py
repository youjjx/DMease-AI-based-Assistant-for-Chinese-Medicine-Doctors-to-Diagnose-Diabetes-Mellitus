"""Default training entry point for the public-evidence policy."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "scripts/train_evidence_policy.py"), run_name="__main__")
