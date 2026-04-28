"""Phase 7 A9 full: MoE validator dispatch and override semantics.

Three sub-tests cover the MoE validator wiring contract:
1. moe_default_strategies: real F2-moe end-to-end run, asserts default
   strategies (layerwise + sequential) actually execute via call counts.
2. moe_dispatch_supported_override: same run with `--vit-validator-strategy
   blockorder --llm-validator-strategy sequential`, asserts only the chosen
   runners are invoked.
3. moe_rejects_unsafe_override: `--llm-validator-strategy parallel` with
   `--variant moe` exits 2 (already covered in test_moe_validator_strategy.py
   for the negative case; this sub-test asserts the positive boundary stays
   correct after the dispatch refactor).
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest import mock

import pytest

from tests._shared.fixture_f2 import f2_paths


def _common_args(paths: dict[str, str], out_dir: Path) -> list[str]:
    return [
        "merge",
        "--variant",
        "moe",
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
        "--out",
        str(out_dir),
        "--target-dtype",
        "bf16",
    ]


def _patch_runners():
    return (
        mock.patch("transformers_impl.merge_ov2.validators.vit_blockorder.run"),
        mock.patch("transformers_impl.merge_ov2.validators.vit_layerwise.run"),
        mock.patch("transformers_impl.merge_ov2.validators.llm_parallel.run"),
        mock.patch("transformers_impl.merge_ov2.validators.llm_sequential.run"),
        mock.patch("transformers_impl.merge_ov2.validators.e2e.run"),
    )


def test_moe_default_strategies_dispatch(tmp_path: Path) -> None:
    paths = f2_paths("moe")
    out_dir = tmp_path / "default"
    try:
        from transformers_impl.merge_ov2.cli import main

        patches = _patch_runners()
        with patches[0] as m_vb, patches[1] as m_vl, patches[2] as m_lp, patches[3] as m_ls, patches[4] as m_e2e:
            rc = main(_common_args(paths, out_dir))
            assert rc == 0
            assert m_vl.call_count == 1, f"layerwise (default for moe) not invoked: {m_vl.call_count}"
            assert m_vb.call_count == 0, f"blockorder must NOT run by default for moe: {m_vb.call_count}"
            assert m_ls.call_count == 1, f"sequential (default for moe) not invoked: {m_ls.call_count}"
            assert m_lp.call_count == 0, f"parallel must NOT run by default for moe: {m_lp.call_count}"
            assert m_e2e.call_count == 1, f"e2e not invoked: {m_e2e.call_count}"
    finally:
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)


def test_moe_dispatch_supported_override(tmp_path: Path) -> None:
    paths = f2_paths("moe")
    out_dir = tmp_path / "override"
    try:
        from transformers_impl.merge_ov2.cli import main

        argv = _common_args(paths, out_dir) + [
            "--vit-validator-strategy",
            "blockorder",
            "--llm-validator-strategy",
            "sequential",
        ]
        patches = _patch_runners()
        with patches[0] as m_vb, patches[1] as m_vl, patches[2] as m_lp, patches[3] as m_ls, patches[4] as m_e2e:
            rc = main(argv)
            assert rc == 0
            assert m_vb.call_count == 1, f"blockorder (overridden) not invoked: {m_vb.call_count}"
            assert m_vl.call_count == 0, f"layerwise must NOT run when overridden: {m_vl.call_count}"
            assert m_ls.call_count == 1, f"sequential (overridden) not invoked: {m_ls.call_count}"
            assert m_lp.call_count == 0, f"parallel must NOT run: {m_lp.call_count}"
            assert m_e2e.call_count == 1
    finally:
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)


def test_moe_rejects_parallel_override():
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
        "--qwen-processor",
        "/nonexistent/qproc",
        "--img",
        "/nonexistent/img",
        "--sample-text",
        "x",
        "--out",
        "/tmp/should_not_be_created",
        "--llm-validator-strategy",
        "parallel",
    ]
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
