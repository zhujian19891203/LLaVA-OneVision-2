"""Phase 7 A4-moe: MoE dry-run on F1-moe reports zero anomalies."""

from __future__ import annotations

import io
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "tiny-moe"
BUILDER = REPO_ROOT / "tests" / "fixtures" / "build_tiny_triplet.py"


def _ensure_fixture():
    if (FIXTURE / "llm" / "config.json").exists():
        return
    subprocess.run([sys.executable, str(BUILDER), "--variant", "moe"], check=True, cwd=REPO_ROOT)


def test_moe_dry_run_clean():
    from transformers_impl.merge_ov2.cli import main

    _ensure_fixture()
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(
            [
                "dry-run",
                "--variant",
                "moe",
                "--vit",
                str(FIXTURE / "vit"),
                "--adapter",
                str(FIXTURE / "adapter"),
                "--llm",
                str(FIXTURE / "llm"),
                "--processor",
                str(FIXTURE / "processor"),
            ]
        )
    assert rc == 0
    out = buf.getvalue()
    assert "missing_in_model: 0" in out
    assert "shape_mismatch: 0" in out
    assert "uncovered_model_params: 0" in out
