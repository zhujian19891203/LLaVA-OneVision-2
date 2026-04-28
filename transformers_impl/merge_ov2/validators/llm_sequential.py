import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, logging

from ..utils import cosine_similarity


logger = logging.get_logger(__name__)


def run(model, llm_path: str, sample_text: str):
    device = next(model.model.language_model.parameters()).device
    dtype = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(llm_path, use_fast=True, trust_remote_code=True)
    inputs = tok(sample_text, return_tensors="pt")

    with torch.no_grad():
        merged_lm = model.model.language_model.to(dtype=dtype, device=device).eval()
        merged_head = model.lm_head.to(dtype=dtype, device=device)
        gpu_in = {k: v.to(device) for k, v in inputs.items()}
        merged_h = merged_lm(
            input_ids=gpu_in["input_ids"], attention_mask=gpu_in.get("attention_mask")
        ).last_hidden_state
        merged_logits = merged_head(merged_h).cpu()
        model.model.language_model = merged_lm.to("cpu")
        model.lm_head = merged_head.to("cpu")
        del merged_lm, merged_head, gpu_in
        torch.cuda.empty_cache()

        orig = AutoModelForCausalLM.from_pretrained(
            llm_path, torch_dtype=dtype, device_map=device, trust_remote_code=True, low_cpu_mem_usage=True
        ).eval()
        gpu_in = {k: v.to(device) for k, v in inputs.items()}
        orig_logits = orig(**gpu_in).logits.cpu()
        del orig, gpu_in
        torch.cuda.empty_cache()

    sim = cosine_similarity(merged_logits.flatten(0, 1), orig_logits.flatten(0, 1))
    diff = (merged_logits.float() - orig_logits.float()).abs().mean().item()
    logger.info(f"LLM diff={diff:.6f} cos={sim:.6f}")
    if not (sim > 0.99 and diff < 5e-2):
        raise ValueError(f"LLM sequential mismatch (diff={diff}, cos={sim})")
    logger.info("LLM sequential consistency OK")
