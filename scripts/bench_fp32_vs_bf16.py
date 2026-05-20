import time
import torch
from bridgelm.model import TransformerLM
from bridgelm.training import AdamW, cross_entropy

DEVICE = "cuda"
VOCAB = 6400
SEQ = 512
BATCH = 24
D_MODEL = 512
LAYERS = 8
HEADS = 8
D_FF = 1344
WARMUP = 5
ITERS = 20


def build_model():
    m = TransformerLM(
        vocab_size=VOCAB,
        context_length=SEQ,
        d_model=D_MODEL,
        num_layers=LAYERS,
        num_heads=HEADS,
        d_ff=D_FF,
        rope_theta=1000000.0,
        use_rms_norm=True,
        norm_mode="pre",
        ffn_type="swiglu",
        device=DEVICE,
    ).to(DEVICE)
    return m


def make_batch():
    x = torch.randint(0, VOCAB, (BATCH, SEQ), device=DEVICE)
    y = torch.randint(0, VOCAB, (BATCH, SEQ), device=DEVICE)
    return x, y


def run(label, use_bf16):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_model()
    opt = AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    # warmup
    for _ in range(WARMUP):
        x, y = make_batch()
        opt.zero_grad()
        if use_bf16:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = cross_entropy(logits, y)
        else:
            logits = model(x)
            loss = cross_entropy(logits, y)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()

    # timed
    t0 = time.perf_counter()
    for _ in range(ITERS):
        x, y = make_batch()
        opt.zero_grad()
        if use_bf16:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = cross_entropy(logits, y)
        else:
            logits = model(x)
            loss = cross_entropy(logits, y)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    sec_per_iter = dt / ITERS
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    full_run_h = sec_per_iter * 20000 / 3600
    print(f"[{label}]  {sec_per_iter*1000:.0f} ms/iter   peak_vram={peak_mem:.2f} GB   "
          f"20000 iter ≈ {full_run_h:.2f} h")
    del model, opt
    torch.cuda.empty_cache()


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"batch={BATCH} seq={SEQ} d_model={D_MODEL} layers={LAYERS}  warmup={WARMUP} iters={ITERS}")
    print()
    run("FP32       ", use_bf16=False)
    run("BF16 amp   ", use_bf16=True)


if __name__ == "__main__":
    main()
