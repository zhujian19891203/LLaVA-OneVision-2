import time
from contextlib import contextmanager
from io import BytesIO

import requests
import torch
import torch.nn.functional as F
from PIL import Image

from transformers import logging


logger = logging.get_logger(__name__)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    return float(F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0))


def load_image(path: str) -> Image.Image:
    if path.startswith("http"):
        return Image.open(BytesIO(requests.get(path).content)).convert("RGB")
    return Image.open(path).convert("RGB")


def build_patch_positions(grid_thw: torch.Tensor, device: torch.device) -> torch.Tensor:
    parts = []
    for t, h, w in grid_thw.tolist():
        ti = torch.arange(t, device=device, dtype=torch.float32)
        hi = torch.arange(h, device=device, dtype=torch.float32)
        wi = torch.arange(w, device=device, dtype=torch.float32)
        mt, mh, mw = torch.meshgrid(ti, hi, wi, indexing="ij")
        parts.append(torch.stack([mt, mh, mw], dim=-1).reshape(-1, 3))
    return torch.cat(parts, dim=0)


def rowmajor_to_block(features: torch.Tensor, t: int, h: int, w: int, sms: int) -> torch.Tensor:
    """Reorder [t*h*w, d] features from row-major to sms x sms block layout."""
    if sms == 1:
        return features
    d = features.shape[-1]
    assert h % sms == 0 and w % sms == 0, f"({h},{w}) not divisible by sms={sms}"
    return features.view(t, h // sms, sms, w // sms, sms, d).permute(0, 1, 3, 2, 4, 5).contiguous().view(t * h * w, d)


@contextmanager
def log_stage(name: str):
    bar = "=" * 6
    logger.info(f"{bar} {name} {bar}")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        logger.info(f"{bar} {name} done in {dt:.2f}s {bar}")
