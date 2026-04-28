"""Phase 2 A4: dry-run reports clean coverage on the F1 dense fixture."""

from __future__ import annotations

import io
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "tiny-dense"
BUILDER = REPO_ROOT / "tests" / "fixtures" / "build_tiny_triplet.py"


def _ensure_fixture() -> None:
    if (FIXTURE_DIR / "llm" / "config.json").exists():
        return
    subprocess.run([sys.executable, str(BUILDER), "--variant", "dense"], check=True, cwd=REPO_ROOT)


def test_dry_run_clean_dense() -> None:
    _ensure_fixture()
    from transformers_impl.merge_ov2.cli import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "dry-run",
                "--variant",
                "dense",
                "--vit",
                str(FIXTURE_DIR / "vit"),
                "--adapter",
                str(FIXTURE_DIR / "adapter"),
                "--llm",
                str(FIXTURE_DIR / "llm"),
                "--processor",
                str(FIXTURE_DIR / "processor"),
            ]
        )
    out = buf.getvalue()
    assert rc == 0, f"dry-run exit {rc}\n{out}"
    assert "missing_in_model: 0" in out, out
    assert "shape_mismatch: 0" in out, out
    assert "uncovered_model_params: 0" in out, out
