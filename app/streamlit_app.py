"""Compatibility entry point for the unified authenticated interface."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[1] / "app.py"), run_name="__main__")
