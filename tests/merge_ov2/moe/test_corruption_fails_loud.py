"""Phase 7 A5-moe: corrupted ViT key in MoE merge raises with 'ViT' label."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from safetensors.torch import load_file, save_file


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "tiny-moe"
BUILDER = REPO_ROOT / "tests" / "fixtures" / "build_tiny_triplet.py"


def _ensure_fixture():
    if (FIXTURE / "llm" / "config.json").exists():
        return
    subprocess.run([sys.executable, str(BUILDER), "--variant", "moe"], check=True, cwd=REPO_ROOT)


def test_moe_corrupted_vit_key_fails_loud(tmp_path):
    from transformers_impl.merge_ov2.cli import main

    _ensure_fixture()
    bad_fixture = tmp_path / "bad-moe"
    shutil.copytree(FIXTURE, bad_fixture)
    vit_st = bad_fixture / "vit" / "model.safetensors"
    sd = load_file(str(vit_st))
    first_key = next(iter(sd))
    sd["bogus_typo.weight"] = sd.pop(first_key)
    save_file(sd, str(vit_st))

    out_dir = tmp_path / "out"
    with pytest.raises(RuntimeError) as exc:
        main(
            [
                "merge",
                "--variant",
                "moe",
                "--vit",
                str(bad_fixture / "vit"),
                "--adapter",
                str(bad_fixture / "adapter"),
                "--llm",
                str(bad_fixture / "llm"),
                "--processor",
                str(bad_fixture / "processor"),
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
    msg = str(exc.value)
    assert "ViT" in msg or "visual" in msg
    assert not out_dir.exists()
