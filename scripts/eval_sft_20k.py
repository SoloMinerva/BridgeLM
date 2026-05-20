"""Minimal inference script for sft_20k evaluation."""
import json
import time
from pathlib import Path

import torch

from bridgelm.model import TransformerLM
from bridgelm.tokenizer import BPETokenizer
from bridgelm.training import build_generation_prompt

DEVICE = "cuda"
DTYPE = torch.bfloat16
EOS = "<|endoftext|>"
CKPT = Path("outputs/sft_20k/ckpt_final.pt")
PROMPTS = Path("eval/prompts_v2.json")
OUT = Path("results/eval_v2/eval_results.json")
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.8
TOP_P = 0.9

# Load tokenizer
tokenizer = BPETokenizer.from_files(
    "outputs/tokenizer_full_clean/vocab.json",
    "outputs/tokenizer_full_clean/merge.txt",
    special_tokens=[EOS],
)
eos_id = tokenizer.vocab_to_id[EOS.encode()]
print("Tokenizer ok, eos_id:", eos_id)

# Load model
with open(CKPT.parent / "model_config.json", encoding="utf-8") as f:
    cfg = json.load(f)

ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
cleaned = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}

model = TransformerLM(
    vocab_size=cleaned["token_embeddings.weight"].shape[0],
    context_length=int(cfg["context_length"]),
    d_model=int(cfg["d_model"]),
    num_layers=int(cfg["num_layers"]),
    num_heads=int(cfg["num_heads"]),
    d_ff=int(cfg["d_ff"]),
    rope_theta=float(cfg.get("rope_theta", 1000000.0)),
    use_rms_norm=True,
    norm_mode="pre",
    ffn_type="swiglu",
    device=DEVICE,
    dtype=DTYPE,
).to(DEVICE)
model.load_state_dict(cleaned, strict=True)
model.eval()
print("Model loaded")

# Load prompts
with open(PROMPTS, encoding="utf-8") as f:
    data = json.load(f)
prompts = data["prompts"] if "prompts" in data else data

results = []
for i, item in enumerate(prompts):
    if "conversations" in item:
        prompt_text = build_generation_prompt(item["conversations"], eos_token=EOS)
    else:
        prompt_text = item["prompt_text"]

    ids = tokenizer.encode(prompt_text)
    input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    t0 = time.time()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=DTYPE):
        gen_ids = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            eos_token_id=eos_id,
        )
    elapsed = time.time() - t0

    new_ids = gen_ids[0, input_ids.shape[1]:].tolist()
    output_text = tokenizer.decode(new_ids)
    results.append({
        "id": item["id"],
        "category": item["category"],
        "input": prompt_text,
        "output": output_text,
        "output_tokens": len(new_ids),
        "latency_s": elapsed,
    })
    print(f"[{i+1}/{len(prompts)}] {item['id']}: {len(new_ids)} tokens, {elapsed:.1f}s", flush=True)

# Merge into existing eval_results.json
if OUT.exists():
    with open(OUT, encoding="utf-8") as f:
        all_results = json.load(f)
else:
    all_results = {}

all_results["sft_20k"] = {
    "checkpoint": str(CKPT),
    "generation_params": {"max_new_tokens": MAX_NEW_TOKENS, "temperature": TEMPERATURE, "top_p": TOP_P},
    "results": results,
    "total_time_s": sum(r["latency_s"] for r in results),
    "avg_latency_s": sum(r["latency_s"] for r in results) / len(results),
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print(f"\nSaved to {OUT}")
