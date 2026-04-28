import torch

from transformers import logging

from ..utils import build_patch_positions, load_image


logger = logging.get_logger(__name__)


def run(model, processor, tokenizer, img_path: str, device: torch.device):
    from qwen_vl_utils import process_vision_info

    image = load_image(img_path)
    model = model.to(device=device).eval()

    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image in detail."},
            ],
        }
    ]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    img_in, vid_in = process_vision_info(msgs)
    inputs = processor(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    patch_pos = build_patch_positions(inputs["image_grid_thw"], device)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        gen = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            patch_positions=patch_pos,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    out = tokenizer.decode(gen[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    logger.info(f"Generated: {out!r}")
    if out.strip():
        logger.info("end-to-end OK")
    else:
        logger.warning("empty generation (untrained model?)")
    model.to("cpu")
    torch.cuda.empty_cache()
    return out
