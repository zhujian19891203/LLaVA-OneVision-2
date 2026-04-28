from .dense import DenseVariant
from .moe import MoeVariant


def get_variant(name: str):
    if name == "dense":
        return DenseVariant()
    if name == "moe":
        return MoeVariant()
    raise ValueError(f"unknown variant: {name!r} (expected 'dense' or 'moe')")
