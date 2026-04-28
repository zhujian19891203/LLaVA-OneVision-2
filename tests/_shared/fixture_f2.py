"""F2 real-triplet fixture resolver.

Returns local paths for dense or MoE F2 fixtures, or skips the calling test
when the gating env vars or local paths are missing. Centralizes the
opt-in policy from plan section "Fixture F2".
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


_F2_DENSE_DEFAULTS = {
    "vit": "/ov2/pretrain_models/onevision-encoder-large",
    "adapter": "",
    "llm": "/ov2/pretrain_models/Qwen3-1.7B-Base",
    "processor": "/ov2/pretrain_models/lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
    "qwen_processor": "/ov2/pretrain_models/Qwen2.5-VL-7B-Instruct-processor",
    "img": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
    "sample_text": "Hello, my dog is cute",
}

_F2_MOE_DEFAULTS = {
    **_F2_DENSE_DEFAULTS,
    "llm": "/mnt/publicdataset/Qwen/Qwen3-30B-A3B-Instruct-2507",
}


def _check_local(paths: dict[str, str]) -> str | None:
    for key in ("vit", "llm", "processor", "qwen_processor"):
        p = paths[key]
        if p and not Path(p).exists():
            return f"missing local path for {key}: {p}"
    return None


def f2_paths(variant: str) -> dict[str, str]:
    if variant not in ("dense", "moe"):
        raise ValueError(f"unknown variant: {variant}")
    if not os.environ.get("OV2_REAL_FIXTURE"):
        pytest.skip(f"F2-{variant} unavailable: OV2_REAL_FIXTURE unset")
    if variant == "moe" and not os.environ.get("OV2_REAL_FIXTURE_MOE"):
        pytest.skip("F2-moe unavailable: OV2_REAL_FIXTURE_MOE unset")

    paths = dict(_F2_MOE_DEFAULTS if variant == "moe" else _F2_DENSE_DEFAULTS)
    if override := os.environ.get("OV2_F2_PROCESSOR"):
        paths["processor"] = override

    if reason := _check_local(paths):
        pytest.skip(f"F2-{variant} unavailable: {reason}")
    return paths
