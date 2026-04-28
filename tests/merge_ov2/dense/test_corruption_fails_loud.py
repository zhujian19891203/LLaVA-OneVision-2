"""Phase 2 A5: corrupted ViT key triggers loud, named RuntimeError."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from safetensors.torch import load_file, save_file


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "tiny-dense"
BUILDER = REPO_ROOT / "tests" / "fixtures" / "build_tiny_triplet.py"
TYPO_KEY = "model.visual.bogus_typo.weight"


def _ensure_fixture() -> None:
    if (FIXTURE_DIR / "llm" / "config.json").exists():
        return
    subprocess.run([sys.executable, str(BUILDER), "--variant", "dense"], check=True, cwd=REPO_ROOT)


def test_corrupted_vit_key_fails_loud(tmp_path: Path) -> None:
    _ensure_fixture()

    work = tmp_path / "fix"
    shutil.copytree(FIXTURE_DIR, work)
    vit_st = work / "vit" / "model.safetensors"

    sd = load_file(str(vit_st))
    first_real = next(iter(sd.keys()))
    sd[TYPO_KEY] = sd.pop(first_real)
    save_file(sd, str(vit_st))

    from transformers_impl.merge_ov2.cli import main

    out_dir = tmp_path / "out"
    with pytest.raises((RuntimeError, KeyError)) as excinfo:
        main(
            [
                "merge",
                "--variant",
                "dense",
                "--vit",
                str(work / "vit"),
                "--adapter",
                str(work / "adapter"),
                "--llm",
                str(work / "llm"),
                "--processor",
                str(work / "processor"),
                "--out",
                str(out_dir),
                "--validate-skip",
                "vit",
                "--validate-skip",
                "llm",
                "--validate-skip",
                "e2e",
            ]
        )

    assert not out_dir.exists() or not any(out_dir.iterdir()), "merge should not have produced output"
    err_text = str(excinfo.value)
    assert "ViT" in err_text or "visual" in err_text.lower(), f"expected ViT context in error: {err_text}"
