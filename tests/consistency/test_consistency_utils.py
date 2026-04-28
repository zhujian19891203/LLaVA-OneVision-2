import os
from typing import Any

import numpy as np
import torch
from PIL import Image


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten(), b.flatten()
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if norm_a == 0 or norm_b == 0 else float(np.dot(a, b) / (norm_a * norm_b))


def align_rotary_debug_tensors(
    hf_output: torch.Tensor, megatron_output: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    hf_tensor = hf_output.float().cpu()
    megatron_tensor = megatron_output.float().cpu()

    if hf_tensor.dim() == 3 and hf_tensor.shape[0] == 1:
        hf_tensor = hf_tensor.squeeze(0)

    if hf_tensor.dim() == 2 and megatron_tensor.dim() == 2:
        if hf_tensor.shape[0] == megatron_tensor.shape[0] and hf_tensor.shape[1] == megatron_tensor.shape[1] * 2:
            megatron_tensor = torch.cat([megatron_tensor, megatron_tensor], dim=-1)

    return hf_tensor.numpy(), megatron_tensor.numpy()


def align_encoder_debug_tensors(
    hf_output: torch.Tensor, megatron_output: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    hf_tensor = hf_output.float().cpu()
    megatron_tensor = megatron_output.float().cpu()

    if hf_tensor.dim() == 3 and hf_tensor.shape[0] == 1:
        hf_tensor = hf_tensor.squeeze(0)

    if megatron_tensor.dim() == 3 and megatron_tensor.shape[1] == 1:
        megatron_tensor = megatron_tensor.squeeze(1)

    return hf_tensor.numpy(), megatron_tensor.numpy()


def compare_arrays(hf_array: np.ndarray, mcore_array: np.ndarray, threshold: float = 0.99) -> dict[str, Any]:
    hf_flat = hf_array.flatten()
    mcore_flat = mcore_array.flatten()
    min_len = min(len(hf_flat), len(mcore_flat))
    diff = np.abs(hf_flat[:min_len] - mcore_flat[:min_len])
    similarity = cosine_similarity(hf_array, mcore_array)
    return {
        "similarity": float(similarity),
        "max_diff": float(np.max(diff)) if min_len > 0 else 0.0,
        "hf_shape": list(hf_array.shape),
        "mcore_shape": list(mcore_array.shape),
        "status": "match" if similarity > threshold else "mismatch",
    }


def summarize_named_results(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    mismatches = {key: value for key, value in results.items() if value.get("status") != "match"}
    return {
        "total": len(results),
        "matched": len(results) - len(mismatches),
        "mismatched": len(mismatches),
        "mismatches": mismatches,
    }


def summarize_layer_results(layer_comparisons: dict[str, dict[str, Any]]) -> dict[str, Any]:
    mismatched_layers = {}
    for layer_name, layer_result in layer_comparisons.items():
        layer_mismatches = {
            key: value
            for key, value in layer_result.items()
            if isinstance(value, dict) and value.get("status") == "mismatch"
        }
        if layer_mismatches:
            mismatched_layers[layer_name] = layer_mismatches

    return {
        "total_layers": len(layer_comparisons),
        "mismatched_layers": len(mismatched_layers),
        "mismatch_details": mismatched_layers,
    }


def convert_hf_qkv_to_mcore_layout(hf_weight: np.ndarray, num_heads: int, is_bias: bool = False) -> np.ndarray:
    if hf_weight is None:
        raise ValueError("hf_weight cannot be None")
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")

    if is_bias:
        if hf_weight.ndim != 1:
            raise ValueError(f"Expected 1D tensor for bias, got shape {hf_weight.shape}")
        total_size = hf_weight.shape[0]
        if total_size % 3 != 0:
            raise ValueError(f"Bias size {total_size} is not divisible by 3")
        hidden_size = total_size // 3
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} is not divisible by num_heads {num_heads}")
        head_dim = hidden_size // num_heads

        q = hf_weight[:hidden_size]
        k = hf_weight[hidden_size : 2 * hidden_size]
        v = hf_weight[2 * hidden_size :]

        q = q.reshape(num_heads, head_dim)
        k = k.reshape(num_heads, head_dim)
        v = v.reshape(num_heads, head_dim)

        mcore_bias = np.stack([q, k, v], axis=1)
        mcore_bias = mcore_bias.reshape(-1)
        return mcore_bias
    else:
        if hf_weight.ndim != 2:
            raise ValueError(f"Expected 2D tensor for weight, got shape {hf_weight.shape}")
        out_features, in_features = hf_weight.shape
        if out_features % 3 != 0:
            raise ValueError(f"out_features {out_features} is not divisible by 3")
        hidden_size = out_features // 3
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} is not divisible by num_heads {num_heads}")
        head_dim = hidden_size // num_heads

        q_weight = hf_weight[:hidden_size, :]
        k_weight = hf_weight[hidden_size : 2 * hidden_size, :]
        v_weight = hf_weight[2 * hidden_size :, :]

        q_weight = q_weight.reshape(num_heads, head_dim, in_features)
        k_weight = k_weight.reshape(num_heads, head_dim, in_features)
        v_weight = v_weight.reshape(num_heads, head_dim, in_features)

        mcore_weight = np.stack([q_weight, k_weight, v_weight], axis=1)
        mcore_weight = mcore_weight.reshape(out_features, in_features)
        return mcore_weight


def load_and_resize_image(image_path: str, image_size: int = 336) -> Image.Image:
    if image_path.startswith("http://") or image_path.startswith("https://"):
        raise ValueError(
            "Remote test_image_path is disabled for consistency tests. Please provide a local image path."
        )
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Test image does not exist: {image_path}")
    img = Image.open(image_path)
    img = img.resize((image_size, image_size)).convert("RGB")
    return img


def convert_mcore_pixel_values_to_hf_format(
    pixel_values_mcore: torch.Tensor,
    image_grid_thw: torch.Tensor,
    patch_size: int = 14,
    temporal_patch_size: int = 1,
    spatial_merge_size: int = 2,
) -> torch.Tensor:
    C = 3
    num_patches, patch_dim = pixel_values_mcore.shape
    t, h_patches, w_patches = image_grid_thw[0].tolist()

    expected_patch_dim = C * temporal_patch_size * patch_size * patch_size
    assert patch_dim == expected_patch_dim, f"Expected patch_dim={expected_patch_dim}, got {patch_dim}"
    expected_num_patches = t * h_patches * w_patches
    assert num_patches == expected_num_patches

    h_merged = h_patches // spatial_merge_size
    w_merged = w_patches // spatial_merge_size

    patches = pixel_values_mcore.view(num_patches, C, temporal_patch_size, patch_size, patch_size)
    if temporal_patch_size == 1:
        patches = patches.squeeze(2)

    patches = patches.view(h_merged, w_merged, spatial_merge_size, spatial_merge_size, C, patch_size, patch_size)
    patches = patches.permute(0, 2, 1, 3, 4, 5, 6)
    patches = patches.contiguous().view(h_patches, w_patches, C, patch_size, patch_size)
    patches = patches.permute(2, 0, 3, 1, 4)
    H = h_patches * patch_size
    W = w_patches * patch_size
    image = patches.contiguous().view(C, H, W)
    image = image.unsqueeze(0)
    return image


def convert_hf_output_to_mcore_format(
    hf_output: torch.Tensor,
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int = 2,
) -> torch.Tensor:
    t, h_patches, w_patches = image_grid_thw[0].tolist()
    shape = hf_output.shape

    if len(shape) == 2:
        num_patches, hidden_dim = shape
    elif len(shape) == 3:
        if shape[1] == 1:
            num_patches = shape[0]
            hidden_dim = shape[2]
            hf_output = hf_output.squeeze(1)
        elif shape[0] == 1:
            num_patches = shape[1]
            hidden_dim = shape[2]
            hf_output = hf_output.squeeze(0)
        else:
            return hf_output
    else:
        return hf_output

    expected_num_patches = h_patches * w_patches
    if num_patches != expected_num_patches:
        return hf_output

    h_merged = h_patches // spatial_merge_size
    w_merged = w_patches // spatial_merge_size

    patches = hf_output.view(h_patches, w_patches, hidden_dim)
    sms = spatial_merge_size
    patches = patches.view(h_merged, sms, w_merged, sms, hidden_dim)
    patches = patches.permute(0, 2, 1, 3, 4).contiguous()
    patches = patches.view(num_patches, hidden_dim)
    return patches


def test_cosine_similarity():
    """Test cosine similarity with known vectors."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    assert abs(cosine_similarity(a, b) - 1.0) < 1e-6

    c = np.array([1.0, 0.0, 0.0])
    d = np.array([0.0, 1.0, 0.0])
    assert abs(cosine_similarity(c, d) - 0.0) < 1e-6


def test_compare_arrays():
    """Test array comparison with threshold."""
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = compare_arrays(a, b, threshold=0.99)
    assert result["status"] == "match"
    assert result["similarity"] > 0.99


def test_convert_hf_qkv_to_mcore_layout_bias():
    """Test QKV bias layout conversion."""
    num_heads = 4
    head_dim = 8
    hidden_size = num_heads * head_dim  # 32

    hf_bias = np.arange(3 * hidden_size, dtype=np.float32)
    mcore_bias = convert_hf_qkv_to_mcore_layout(hf_bias, num_heads, is_bias=True)
    assert mcore_bias.shape == (3 * hidden_size,)

    expected_first_head = np.concatenate(
        [
            hf_bias[0:head_dim],
            hf_bias[hidden_size : hidden_size + head_dim],
            hf_bias[2 * hidden_size : 2 * hidden_size + head_dim],
        ]
    )
    np.testing.assert_array_equal(mcore_bias[: 3 * head_dim], expected_first_head)


def test_convert_hf_qkv_to_mcore_layout_weight():
    """Test QKV weight layout conversion."""
    num_heads = 2
    head_dim = 4
    hidden_size = num_heads * head_dim  # 8

    hf_weight = np.arange(3 * hidden_size * hidden_size, dtype=np.float32).reshape(3 * hidden_size, hidden_size)
    mcore_weight = convert_hf_qkv_to_mcore_layout(hf_weight, num_heads, is_bias=False)
    assert mcore_weight.shape == (3 * hidden_size, hidden_size)


def test_align_rotary_debug_tensors():
    """Test rotary tensor alignment."""
    hf_tensor = torch.randn(1, 100, 64)
    mcore_tensor = torch.randn(100, 32)
    hf_np, mcore_np = align_rotary_debug_tensors(hf_tensor, mcore_tensor)
    assert hf_np.shape == (100, 64)
    assert mcore_np.shape == (100, 64)


def test_align_encoder_debug_tensors():
    """Test encoder tensor alignment."""
    hf_tensor = torch.randn(1, 100, 1024)
    mcore_tensor = torch.randn(100, 1, 1024)
    hf_np, mcore_np = align_encoder_debug_tensors(hf_tensor, mcore_tensor)
    assert hf_np.shape == (100, 1024)
    assert mcore_np.shape == (100, 1024)


def test_load_and_resize_image():
    """Test image loading with real asset."""
    # Use relative path that works both in container and on host
    img = load_and_resize_image("asset/performance.png", image_size=336)
    assert img.size == (336, 336)
    assert img.mode == "RGB"


def test_convert_mcore_pixel_values_to_hf_format():
    """Test pixel value format conversion."""
    patch_size = 14
    spatial_merge_size = 2
    h_patches = 48
    w_patches = 48
    t = 1
    C = 3

    num_patches = t * h_patches * w_patches
    patch_dim = C * t * patch_size * patch_size

    pixel_values_mcore = torch.randn(num_patches, patch_dim)
    image_grid_thw = torch.tensor([[t, h_patches, w_patches]])

    hf_format = convert_mcore_pixel_values_to_hf_format(
        pixel_values_mcore, image_grid_thw, patch_size, t, spatial_merge_size
    )

    expected_H = h_patches * patch_size
    expected_W = w_patches * patch_size
    assert hf_format.shape == (1, C, expected_H, expected_W)


def test_convert_hf_output_to_mcore_format():
    """Test HF output to mcore format conversion."""
    spatial_merge_size = 2
    h_patches = 48
    w_patches = 48
    hidden_dim = 1024

    num_patches = h_patches * w_patches
    hf_output = torch.randn(num_patches, hidden_dim)
    image_grid_thw = torch.tensor([[1, h_patches, w_patches]])

    mcore_format = convert_hf_output_to_mcore_format(hf_output, image_grid_thw, spatial_merge_size)
    assert mcore_format.shape == (num_patches, hidden_dim)


def test_summarize_named_results():
    """Test result summarization."""
    results = {
        "key1": {"status": "match", "similarity": 0.99},
        "key2": {"status": "mismatch", "similarity": 0.85},
        "key3": {"status": "match", "similarity": 1.0},
    }
    summary = summarize_named_results(results)
    assert summary["total"] == 3
    assert summary["matched"] == 2
    assert summary["mismatched"] == 1
    assert "key2" in summary["mismatches"]


def test_summarize_layer_results():
    """Test layer result summarization."""
    layer_comparisons = {
        "layer_0": {
            "attn_output": {"status": "match", "similarity": 0.99},
            "mlp_output": {"status": "match", "similarity": 0.98},
        },
        "layer_1": {
            "attn_output": {"status": "mismatch", "similarity": 0.85},
            "mlp_output": {"status": "match", "similarity": 0.99},
        },
    }
    summary = summarize_layer_results(layer_comparisons)
    assert summary["total_layers"] == 2
    assert summary["mismatched_layers"] == 1
    assert "layer_1" in summary["mismatch_details"]
