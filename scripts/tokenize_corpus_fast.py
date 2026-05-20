from __future__ import annotations

import argparse
import json
import os
import multiprocessing as mp
from pathlib import Path

import numpy as np
from tqdm import tqdm

from bridgelm.tokenizer import BPETokenizer


def split_bounds(path: Path, n_parts: int) -> list[tuple[int, int]]:
    size = path.stat().st_size
    if size == 0:
        return [(0, 0)]
    step = max(1, size // n_parts)
    bounds = [0]
    with path.open("rb") as f:
        for i in range(1, n_parts):
            f.seek(i * step)
            f.readline()
            pos = f.tell()
            if pos > bounds[-1] and pos < size:
                bounds.append(pos)
    bounds.append(size)
    return list(zip(bounds[:-1], bounds[1:]))


_TOKENIZER: BPETokenizer | None = None


def _init_worker(vocab_path: str, merges_path: str, special_tokens: list[str]) -> None:
    global _TOKENIZER
    _TOKENIZER = BPETokenizer.from_files(vocab_path, merges_path, special_tokens=special_tokens)


def _encode_part(job) -> tuple[int, str, int, int]:
    idx, text_path, start, end, out_dir = job
    assert _TOKENIZER is not None
    with open(text_path, "rb") as f:
        f.seek(start)
        data = f.read(end - start)
    text = data.decode("utf-8", errors="replace")

    SUB = 2 * 1024 * 1024
    pieces: list[np.ndarray] = []
    pos = 0
    while pos < len(text):
        nxt = text.find("\n", pos + SUB)
        if nxt == -1:
            nxt = len(text)
        else:
            nxt += 1
        seg = text[pos:nxt]
        ids = _TOKENIZER.encode(seg)
        if ids:
            pieces.append(np.asarray(ids, dtype=np.uint16))
        pos = nxt

    arr = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.uint16)
    out_path = os.path.join(out_dir, f"_part_{idx:04d}.npy")
    np.save(out_path, arr)
    return idx, out_path, int(arr.size), int(arr.max()) if arr.size else -1


def encode_dataset(
    text_path: Path,
    out_path: Path,
    vocab_path: str,
    merges_path: str,
    special_tokens: list[str],
    n_workers: int,
    n_parts: int,
    label: str,
) -> tuple[int, int]:
    tmp_dir = out_path.parent / f"_tmp_{label}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    bounds = split_bounds(text_path, n_parts)
    jobs = [(i, str(text_path), s, e, str(tmp_dir)) for i, (s, e) in enumerate(bounds)]

    size_mb = text_path.stat().st_size / 1e6
    print(f"[{label}] file={size_mb:.1f}MB  parts={len(jobs)}  workers={n_workers}", flush=True)

    results: list[tuple[int, str, int, int]] = []
    with mp.Pool(n_workers, initializer=_init_worker, initargs=(vocab_path, merges_path, special_tokens)) as pool:
        for r in tqdm(
            pool.imap_unordered(_encode_part, jobs),
            total=len(jobs),
            desc=f"[{label}] encode",
            ncols=80,
        ):
            results.append(r)

    results.sort(key=lambda x: x[0])
    total = sum(r[2] for r in results)
    max_id = max((r[3] for r in results if r[3] >= 0), default=-1)

    print(f"[{label}] writing {total:,} tokens -> {out_path}", flush=True)
    out_arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint16, shape=(total,))
    offset = 0
    for _, ppath, count, _ in results:
        if count == 0:
            continue
        part = np.load(ppath, mmap_mode="r")
        out_arr[offset : offset + count] = part
        offset += count
        del part
    out_arr.flush()

    for _, ppath, _, _ in results:
        try:
            os.remove(ppath)
        except OSError:
            pass
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    return total, max_id


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--parts", type=int, default=32)
    args = p.parse_args()

    cfg = load_config(args.config)
    tk = cfg["tokenizer"]
    data = cfg["data"]
    out = cfg["output"]

    vocab_path = tk["vocab_path"]
    merges_path = tk["merges_path"]
    special_tokens = tk.get("special_tokens", ["<|endoftext|>"])
    train_path = Path(data["train_path"])
    valid_path = Path(data["valid_path"])
    out_dir = Path(out["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    probe = BPETokenizer.from_files(vocab_path, merges_path, special_tokens=special_tokens)
    vocab_size = len(probe.id_to_vocab)
    del probe
    if vocab_size > np.iinfo(np.uint16).max + 1:
        raise ValueError(f"vocab {vocab_size} too big for uint16")

    metadata: dict = {
        "dtype": "uint16",
        "tokenizer_vocab_path": vocab_path,
        "tokenizer_merges_path": merges_path,
        "special_tokens": special_tokens,
        "vocab_size": vocab_size,
        "datasets": {},
    }

    for label, src in [("train", train_path), ("valid", valid_path)]:
        out_path = out_dir / f"{label}_ids.npy"
        if label == "train":
            n_parts = args.parts
            n_workers = args.workers
        else:
            n_parts = max(2, args.workers)
            n_workers = min(args.workers, n_parts)
        total, max_id = encode_dataset(
            src, out_path, vocab_path, merges_path, special_tokens,
            n_workers=n_workers, n_parts=n_parts, label=label,
        )
        metadata["datasets"][label] = {
            "source_path": str(src),
            "token_ids_path": str(out_path),
            "num_tokens": total,
            "max_token_id": max_id,
        }

    meta_path = out_dir / "metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"saved metadata to {meta_path}", flush=True)


if __name__ == "__main__":
    main()
