import os
from collections.abc import Iterator

import torch
from safetensors import safe_open
from safetensors.torch import load_file


def load_safetensors(path: str) -> dict:
    if os.path.isdir(path):
        out: dict[str, torch.Tensor] = {}
        for fn in sorted(os.listdir(path)):
            if fn.endswith(".safetensors"):
                out.update(load_file(os.path.join(path, fn)))
        return out
    if path.endswith(".safetensors"):
        return load_file(path)
    sd = torch.load(path, map_location="cpu")
    return sd.get("state_dict", sd)


def _iter_one_file(path: str) -> Iterator[tuple[str, torch.Tensor]]:
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            yield key, f.get_tensor(key)


def iter_safetensors(path: str) -> Iterator[tuple[str, torch.Tensor]]:
    if os.path.isdir(path):
        for fn in sorted(os.listdir(path)):
            if fn.endswith(".safetensors"):
                yield from _iter_one_file(os.path.join(path, fn))
        return
    if path.endswith(".safetensors"):
        yield from _iter_one_file(path)
        return
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("state_dict", sd)
    for k, v in sd.items():
        yield k, v
