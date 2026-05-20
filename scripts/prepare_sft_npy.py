"""Pre-tokenize SFT JSONL into numpy arrays for fast DataLoader loading.

Usage:
    python scripts/prepare_sft_npy.py --config configs/sft_improved.json --out-dir data/sft_npy
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

from bridgelm.tokenizer import BPETokenizer
from bridgelm.training.sft import (
    ROLE_MARKERS,
    build_loss_labels,
    maybe_add_system_prompt,
    normalize_conversations,
    render_chat_prompt,
)


def count_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def process_jsonl(
    jsonl_path: Path,
    tokenizer: BPETokenizer,
    max_length: int,
    system_prompt_ratio: float,
    eos_token: str,
    seed: int,
    desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    assistant_header_ids = tokenizer.encode(ROLE_MARKERS["assistant"])
    eos_boundary_ids = tokenizer.encode(f"{eos_token}\n")
    pad_token_id = tokenizer.vocab_to_id[eos_token.encode("utf-8")]

    print(f"Counting samples in {jsonl_path} ...")
    n_lines = count_lines(jsonl_path)
    print(f"  {n_lines:,} lines")

    # Pre-allocate to avoid huge Python lists
    input_ids_arr = np.zeros((n_lines, max_length), dtype=np.int16)
    labels_arr = np.full((n_lines, max_length), fill_value=-100, dtype=np.int16)

    skipped = 0
    written = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for raw_idx, line in enumerate(tqdm(f, total=n_lines, desc=desc)):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                conversations = sample.get("conversations")
                if not isinstance(conversations, list):
                    skipped += 1
                    continue

                normalized = normalize_conversations(conversations)
                rng = random.Random(seed + raw_idx)
                convs = maybe_add_system_prompt(
                    normalized, rng=rng, system_prompt_ratio=system_prompt_ratio
                )
                rendered = render_chat_prompt(convs, eos_token=eos_token, add_generation_prompt=False)

                ids = tokenizer.encode(rendered)[:max_length]
                ids += [pad_token_id] * (max_length - len(ids))

                lbls = build_loss_labels(
                    input_ids=ids,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    assistant_header_ids=assistant_header_ids,
                    eos_boundary_ids=eos_boundary_ids,
                    pad_token_id=pad_token_id,
                )

                input_ids_arr[written] = ids
                labels_arr[written] = lbls
                written += 1
            except Exception:
                skipped += 1

    # Trim to actual written rows
    input_ids_arr = input_ids_arr[:written]
    labels_arr = labels_arr[:written]
    print(f"  Done: {written:,} samples, {skipped} skipped")
    return input_ids_arr, labels_arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-tokenize SFT JSONL to npy arrays")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    tok_cfg = cfg["tokenizer"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    tokenizer = BPETokenizer.from_files(
        tok_cfg["vocab_path"],
        tok_cfg["merges_path"],
        special_tokens=tok_cfg.get("special_tokens", ["<|endoftext|>"]),
    )

    max_length: int = model_cfg["context_length"]
    system_prompt_ratio: float = data_cfg.get("system_prompt_ratio", 0.0)
    eos_token: str = data_cfg.get("eos_token", "<|endoftext|>")
    seed: int = train_cfg.get("seed", 42)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- Train ---
    print("\n=== Train ===")
    train_ids, train_lbls = process_jsonl(
        Path(data_cfg["train_data_path"]), tokenizer, max_length,
        system_prompt_ratio, eos_token, seed, "Train",
    )
    np.save(args.out_dir / "train_input_ids.npy", train_ids)
    np.save(args.out_dir / "train_labels.npy", train_lbls)
    print(f"  Saved: {train_ids.shape}  {train_ids.nbytes / 1024**2:.0f} MB each")

    # --- Valid ---
    print("\n=== Valid ===")
    valid_ids, valid_lbls = process_jsonl(
        Path(data_cfg["valid_data_path"]), tokenizer, max_length,
        0.0, eos_token, seed, "Valid",
    )
    np.save(args.out_dir / "valid_input_ids.npy", valid_ids)
    np.save(args.out_dir / "valid_labels.npy", valid_lbls)
    print(f"  Saved: {valid_ids.shape}  {valid_ids.nbytes / 1024**2:.0f} MB each")

    print(f"\nAll done. Point train_data_path to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
