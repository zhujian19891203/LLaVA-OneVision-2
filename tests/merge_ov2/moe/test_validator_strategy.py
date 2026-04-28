"""Phase 4 A9: --llm-validator-strategy=parallel + --variant=moe must exit 2."""

from __future__ import annotations

import pytest


def test_moe_parallel_strategy_rejected():
    from transformers_impl.merge_ov2.cli import main

    argv = [
        "merge",
        "--variant",
        "moe",
        "--vit",
        "/nonexistent/vit",
        "--llm",
        "/nonexistent/llm",
        "--processor",
        "/nonexistent/proc",
        "--out",
        "/tmp/should_not_be_created",
        "--llm-validator-strategy",
        "parallel",
        "--qwen-processor",
        "/nonexistent/qproc",
        "--img",
        "/nonexistent/img",
        "--sample-text",
        "x",
    ]
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
