#!/usr/bin/env python3
# eval_t5_middle.py

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
SEED = 42
MODEL_NAME = "google/flan-t5-xl"

MACHINE_TYPE = "laptop"
PROCESS_TYPE = "singleprocess"

DATASET_NAME = "ultrachat"
SENTENCE_SCOPE = "middlesentences"
FRAMEWORK = "llamacpp"
TARGET_MODEL = "phi"
K = 64

CHECKPOINT_TAG = "final"

BATCH_SIZE = 256
NUM_WORKERS = 0

MAX_NEW_TOKENS = 128
NUM_BEAMS = 4
DO_SAMPLE = False
EARLY_STOPPING = True
LENGTH_PENALTY = 1.0
NO_REPEAT_NGRAM_SIZE = 3

GEN_CFG = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "num_beams": NUM_BEAMS,
    "do_sample": DO_SAMPLE,
    "early_stopping": EARLY_STOPPING,
    "length_penalty": LENGTH_PENALTY,
    "no_repeat_ngram_size": NO_REPEAT_NGRAM_SIZE,
}

EVAL_DIR = Path("./eval")
EMPTY_SYMBOL_FALLBACK_ID = 1

CONTEXT_JSONL = "context_text.jsonl"
SEGMENT_META_JSONL = "segment_meta.jsonl"

PROMPT_PREFIX = "Translate the following trace symbols using the following context.\n"
CONTEXT_PREFIX = "Context:\n"
SYMBOLS_PREFIX = "\nTrace Symbols:"
OUTPUT_PREFIX = "\nOutput:"


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_eval_tag(machine_type: str, process_type: str, framework: str, target_model: str) -> str:
    return (
        f"dataset-{DATASET_NAME}_"
        f"{process_type}_"
        f"{machine_type}_"
        f"{SENTENCE_SCOPE}_"
        f"framework-{framework}_"
        f"target-{target_model}"
    )


def build_symbols_dir(machine_type: str, process_type: str, framework: str, target_model: str, k: int) -> Path:
    return Path(
        f"/outpath/{framework}/ultrachat_cluster/"
        f"{machine_type}_{process_type}_{target_model}/K_{k}/middle"
    )


def build_run_dir(machine_type: str, process_type: str, framework: str, target_model: str, k: int) -> Path:
    return Path(
        f"/outpath/{framework}/ultrachat_cluster/"
        f"{machine_type}_{process_type}_{target_model}/flan_t5_xl_ultrachat_K{k}_runs/run_middle_sentences"
    )


def load_context_jsonl(path: Path, n_expected: int) -> np.ndarray:
    ctx: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ctx.append(str(json.loads(line).get("context_text", "")))
    if len(ctx) != n_expected:
        raise RuntimeError(f"{path.name} lines {len(ctx)} != n_expected {n_expected}")
    return np.asarray(ctx, dtype=object)


def load_segment_meta_jsonl(path: Path, n_expected: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "paragraph_id" in obj:
                paragraph_id = str(obj["paragraph_id"])
            elif "group_id" in obj:
                paragraph_id = str(obj["group_id"])
            else:
                raise RuntimeError(f"{path.name} line {i+1}: missing paragraph_id/group_id")
            if "segment_idx" not in obj:
                raise RuntimeError(f"{path.name} line {i+1}: missing segment_idx")
            out = dict(obj)
            out["paragraph_id"] = paragraph_id
            out["segment_idx"] = int(obj["segment_idx"])
            rows.append(out)
    if len(rows) != n_expected:
        raise RuntimeError(f"{path.name} lines {len(rows)} != n_expected {n_expected}")
    return rows


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────
class HeldoutMiddleTestDataset(Dataset):
    def __init__(self, symbols_dir: Path):
        meta = json.loads((symbols_dir / "meta.json").read_text())
        self.pad_sym = int(meta.get("symbols_pad", 0))
        self.t5_pad  = int(meta.get("t5_pad", 0))

        self.symbols = np.lib.format.open_memmap(symbols_dir / "symbols_heldout.npy", mode="r")
        self.targets = np.lib.format.open_memmap(symbols_dir / "t5_labels.npy",       mode="r")
        if self.symbols.shape[0] != self.targets.shape[0]:
            raise RuntimeError("symbols_heldout.npy and t5_labels.npy row count mismatch")

        ctx_path = symbols_dir / CONTEXT_JSONL
        if not ctx_path.exists():
            raise FileNotFoundError(f"Missing: {ctx_path}")
        self.context_text = load_context_jsonl(ctx_path, n_expected=self.symbols.shape[0])

        seg_meta_path = symbols_dir / SEGMENT_META_JSONL
        if not seg_meta_path.exists():
            raise FileNotFoundError(f"Missing: {seg_meta_path}")
        self.segment_meta = load_segment_meta_jsonl(seg_meta_path, n_expected=self.symbols.shape[0])

        spl = np.load(symbols_dir / "splits.npz")
        self.test_idx = np.asarray(spl["test_idx"], dtype=np.int64)
        if self.test_idx.size == 0:
            raise RuntimeError("test_idx is empty")

    def __len__(self):
        return int(self.test_idx.shape[0])

    def __getitem__(self, i):
        ridx     = int(self.test_idx[i])
        src      = torch.from_numpy(self.symbols[ridx].astype(np.int64, copy=False))
        tgt      = torch.from_numpy(self.targets[ridx].astype(np.int64, copy=False))
        ctx      = str(self.context_text[ridx])
        seg_meta = self.segment_meta[ridx]
        return src, ctx, tgt, ridx, seg_meta


# ─────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────
def make_collate_middle(tokenizer, pad_sym: int, t5_pad: int):
    pad_id        = int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else 0
    outpref_ids   = tokenizer(OUTPUT_PREFIX, add_special_tokens=False).input_ids
    outpref_ids_t = torch.tensor(outpref_ids, dtype=torch.long)

    def collate(batch):
        srcs, ctxs, tgts, ridxs, metas = zip(*batch)

        inputs: List[torch.Tensor] = []
        ctx_texts: List[str] = []

        for s, ctx in zip(srcs, ctxs):
            s = s[s != pad_sym]
            if s.numel() == 0:
                s = torch.tensor([EMPTY_SYMBOL_FALLBACK_ID], dtype=torch.long)
            ctx = "" if ctx is None else str(ctx)
            prompt    = f"{PROMPT_PREFIX}{CONTEXT_PREFIX}{ctx}{SYMBOLS_PREFIX}"
            prompt_ids_t = torch.tensor(
                tokenizer(prompt, add_special_tokens=False).input_ids, dtype=torch.long
            )
            ctx_texts.append(ctx)
            inputs.append(torch.cat([prompt_ids_t, s, outpref_ids_t], dim=0))

        max_len   = max(x.numel() for x in inputs)
        input_ids = torch.full((len(inputs), max_len), pad_id, dtype=torch.long)
        attn      = torch.zeros_like(input_ids)
        for i, x in enumerate(inputs):
            L = x.numel()
            input_ids[i, :L] = x
            attn[i, :L]      = 1

        tgt      = torch.stack(tgts, dim=0)
        ridxs_t  = torch.tensor(ridxs, dtype=torch.int64)
        return input_ids, attn, ctx_texts, tgt, ridxs_t, list(metas)

    return collate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate middle-segment predictions (no scoring — scoring is in print_prefix_table.py)."
    )
    parser.add_argument("--machine-type",   required=True, choices=["laptop", "desktop"])
    parser.add_argument("--model",          required=True, choices=["phi", "llama"])
    parser.add_argument("--framework",      required=True, choices=["llamacpp", "huggingface"])
    parser.add_argument("--no-jsonl",       action="store_true",
                        help="Do not write per-segment JSONL (disables all downstream scoring)")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    global MACHINE_TYPE, PROCESS_TYPE, K, TARGET_MODEL, FRAMEWORK, CHECKPOINT_TAG

    args           = parse_args()
    MACHINE_TYPE   = args.machine_type
    TARGET_MODEL   = args.model
    FRAMEWORK      = args.framework
    write_jsonl    = not args.no_jsonl

    SYMBOLS_DIR = build_symbols_dir(MACHINE_TYPE, PROCESS_TYPE, FRAMEWORK, TARGET_MODEL, K)
    RUN_DIR     = build_run_dir(MACHINE_TYPE, PROCESS_TYPE, FRAMEWORK, TARGET_MODEL, K)

    seed_all(SEED)

    eval_tag     = build_eval_tag(MACHINE_TYPE, PROCESS_TYPE, FRAMEWORK, TARGET_MODEL)
    out_jsonl    = EVAL_DIR / f"{eval_tag}_{CHECKPOINT_TAG}.jsonl"
    out_summary  = EVAL_DIR / f"{eval_tag}_{CHECKPOINT_TAG}_summary.json"

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    print("Device:", device, "bf16:", use_bf16)
    print("Eval tag:", eval_tag)

    meta       = json.loads((SYMBOLS_DIR / "meta.json").read_text())
    base_vocab = int(meta["base_vocab"])
    K          = int(meta["K"])
    pad_sym    = int(meta.get("symbols_pad", 0))
    t5_pad     = int(meta.get("t5_pad", 0))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if int(tokenizer.vocab_size) != base_vocab:
        raise RuntimeError(f"Tokenizer vocab {tokenizer.vocab_size} != base_vocab {base_vocab}")

    if not RUN_DIR.exists():
        raise FileNotFoundError(f"model not found: {RUN_DIR}")

    print("Loading model:", RUN_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        RUN_DIR,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        low_cpu_mem_usage=True,
    )
    model.resize_token_embeddings(base_vocab + K)
    model.to(device)
    model.eval()

    ds = HeldoutMiddleTestDataset(SYMBOLS_DIR)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=make_collate_middle(tokenizer=tokenizer, pad_sym=pad_sym, t5_pad=t5_pad),
    )

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    f_seg   = open(out_jsonl, "w", encoding="utf-8") if write_jsonl else None
    n_total = 0

    # paragraph size diagnostics (no scoring needed)
    para_max_seg: Dict[str, int] = defaultdict(int)

    pbar = tqdm(dl, desc=f"Generating ({CHECKPOINT_TAG})", dynamic_ncols=True)
    for input_ids, attn, ctx_texts, tgt_raw, ridxs, metas in pbar:
        input_ids = input_ids.to(device, non_blocking=True)
        attn      = attn.to(device, non_blocking=True)

        with torch.no_grad():
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                gen = model.generate(input_ids=input_ids, attention_mask=attn, **GEN_CFG)

        preds = tokenizer.batch_decode(gen, skip_special_tokens=True)

        refs: List[str] = []
        for t in tgt_raw:
            ids = t[t != t5_pad].tolist()
            refs.append(tokenizer.decode(ids, skip_special_tokens=True))

        for ridx, p, r, ct, meta_row in zip(ridxs.tolist(), preds, refs, ctx_texts, metas):
            n_total += 1
            pid  = str(meta_row["paragraph_id"])
            sidx = int(meta_row["segment_idx"])
            para_max_seg[pid] = max(para_max_seg[pid], sidx)

            row = {
                "row_idx":      int(ridx),
                "paragraph_id": pid,
                "segment_idx":  sidx,
                "context_prev": ct,
                "pred":         p,
                "ref":          r,
            }

            if f_seg is not None:
                f_seg.write(json.dumps(row, ensure_ascii=False) + "\n")

        pbar.set_postfix(samples=n_total)

    if f_seg is not None:
        f_seg.close()

    # diagnostics
    n_paras = len(para_max_seg)
    if n_paras:
        sizes = [v + 1 for v in para_max_seg.values()]   # +1 because segment_idx is 0-based
        print(f"\nParagraph diagnostics (middle segments only):")
        print(f"  Total paragraphs: {n_paras}")
        print(f"  Segments per paragraph — "
              f"min={min(sizes)}, max={max(sizes)}, "
              f"mean={sum(sizes)/len(sizes):.1f}, "
              f"median={sorted(sizes)[len(sizes)//2]}")

    summary = {
        "eval_tag":      eval_tag,
        "path":    str(RUN_DIR),
        "symbols_dir":   str(SYMBOLS_DIR),
        "test_samples":  int(n_total),
        "num_paragraphs": int(n_paras),
        "gen_cfg":       GEN_CFG,
        "seed":          SEED,
        "machine_type":  MACHINE_TYPE,
        "process_type":  PROCESS_TYPE,
        "dataset":       DATASET_NAME,
        "scope":         SENTENCE_SCOPE,
        "framework":     FRAMEWORK,
        "target_model":  TARGET_MODEL,
        "jsonl_written": bool(write_jsonl),
        "jsonl_path":    str(out_jsonl) if write_jsonl else None,
        "note": (
            "Scoring removed from this script. "
            "Run print_prefix_table.py to compute metrics — it prepends the "
            "first-sentence prediction to each middle prefix before scoring."
        ),
    }

    Path(out_summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nDONE")
    print("path:", str(RUN_DIR))
    print("Test samples:", n_total)
    print("Saved summary:", str(out_summary))
    if write_jsonl:
        print("Saved segment JSONL:", str(out_jsonl))
    print("→ Run print_prefix_table.py to score and print the prefix table.")


if __name__ == "__main__":
    main()