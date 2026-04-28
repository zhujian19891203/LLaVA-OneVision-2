"""Phase 5 e2e: validate subcommand on F2-dense pre-staged ckpt."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests._shared.fixture_f2 import f2_paths


@pytest.fixture(scope="module")
def staged_ckpt(tmp_path_factory):
    paths = f2_paths("dense")
    from transformers_impl.merge_ov2.cli import main

    out = tmp_path_factory.mktemp("ov2_p5_ckpt")
    rc = main(
        [
            "merge",
            "--variant",
            "dense",
            "--vit",
            paths["vit"],
            "--llm",
            paths["llm"],
            "--processor",
            paths["processor"],
            "--out",
            str(out),
            "--validate-skip",
            "vit",
            "--validate-skip",
            "llm",
            "--validate-skip",
            "e2e",
        ]
    )
    assert rc == 0
    return Path(out), paths


def test_validate_subcommand_e2e(staged_ckpt):
    from transformers_impl.merge_ov2.cli import main

    ckpt, paths = staged_ckpt
    rc = main(
        [
            "validate",
            "--variant",
            "dense",
            "--ckpt",
            str(ckpt),
            "--vit",
            paths["vit"],
            "--llm",
            paths["llm"],
            "--processor",
            paths["processor"],
            "--qwen-processor",
            paths["qwen_processor"],
            "--img",
            paths["img"],
            "--sample-text",
            paths["sample_text"],
        ]
    )
    assert rc == 0
