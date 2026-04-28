"""Phase 4 A6: granular --validate-skip controls which validators run."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from tests._shared.fixture_f2 import f2_paths


def _run_validate(skip: list[str], paths: dict[str, str], ckpt: Path, caplog) -> None:
    from transformers_impl.merge_ov2.cli import main

    argv = [
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
    for s in skip:
        argv += ["--validate-skip", s]
    with caplog.at_level(logging.INFO):
        rc = main(argv)
    assert rc == 0


@pytest.fixture(scope="module")
def staged_ckpt(tmp_path_factory) -> tuple[Path, dict[str, str]]:
    paths = f2_paths("dense")
    from transformers_impl.merge_ov2.cli import main

    out = tmp_path_factory.mktemp("ov2_p4_ckpt")
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
    return out, paths


def test_skip_vit_llm_only_e2e_runs(staged_ckpt, caplog):
    ckpt, paths = staged_ckpt
    _run_validate(["vit", "llm"], paths, ckpt, caplog)
    msgs = caplog.text
    assert "VALIDATE: e2e" in msgs
    assert "VALIDATE: vit" not in msgs
    assert "VALIDATE: llm" not in msgs


def test_skip_all_no_validator_runs(staged_ckpt, caplog):
    ckpt, paths = staged_ckpt
    _run_validate(["vit", "llm", "e2e"], paths, ckpt, caplog)
    assert "VALIDATE:" not in caplog.text


def test_no_skip_all_run(staged_ckpt, caplog):
    ckpt, paths = staged_ckpt
    _run_validate([], paths, ckpt, caplog)
    msgs = caplog.text
    assert "VALIDATE: vit" in msgs
    assert "VALIDATE: llm" in msgs
    assert "VALIDATE: e2e" in msgs
