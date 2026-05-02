#!/usr/bin/env python3
# eval.py
#
# Evaluate one (method, q) checkpoint on FIRST segments, test split only.
#
# Important for the cost-effective pipeline:
#   - inputs come from symbols_train.npy
#   - test examples are selected by test_idx from splits.npz
#
# Run example:
#   python eval.py \
#       --machine-type laptop \
#       --method smart \
#       --q 250 \
#       --checkpoint-tag final

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import json
import random
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import argparse


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
SEED = 42
MODEL_NAME = "google/flan-t5-xl"

MACHINE_TYPE = "laptop"
PROCESS_TYPE = "singleprocess"
METHOD = "uniform"
Q = 250
K = 64

DATASET_NAME = "ultrachat"
SENTENCE_SCOPE = "firstsentences"
FRAMEWORK = "llamacpp"
TARGET_MODEL = "phi"

SYMBOLS_DIR = Path(
    f"/data/llamacpp/"
    f"ultrachat_cluster_cost_effective/{MACHINE_TYPE}_{PROCESS_TYPE}/K_{K}/{METHOD}/q_{Q}/first"
)

RUN_DIR = Path(
    f"/data/llamacpp/"
    f"ultrachat_cluster_cost_effective/{MACHINE_TYPE}_{PROCESS_TYPE}/"
    f"flan_t5_xl_ultrachat_K{K}_runs/{METHOD}/q_{Q}/run_first_sentences"
)
CKPT_DIR = Path(f"{RUN_DIR}/checkpoints/")

CHECKPOINT_TAG = "final"

BATCH_SIZE = 128
NUM_WORKERS = 0

MAX_NEW_TOKENS = 38
NUM_BEAMS = 4
DO_SAMPLE = False

GEN_CFG = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "num_beams": NUM_BEAMS,
    "do_sample": DO_SAMPLE,
}

COS_THRESH = 0.5
ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

EVAL_ROOT = Path("./eval_test_synthetic")
WRITE_JSONL = True

EMPTY_SYMBOL_FALLBACK_ID = 1


# ─────────────────────────────────────────────────────────────
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_eval_tag(machine_type: str, process_type: str, method: str, q: int) -> str:
    return (
        f"dataset-{DATASET_NAME}_"
        f"{process_type}_"
        f"{machine_type}_"
        f"{SENTENCE_SCOPE}_"
        f"framework-{FRAMEWORK}_"
        f"target-{TARGET_MODEL}_"
        f"strategy-{method}_"
        f"q-{q}"
    )


def safe_stats(x: List[float]) -> Dict[str, float]:
    if not x:
        return {"mean": 0.0, "median": 0.0, "p05": 0.0, "p95": 0.0}
    arr = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p95": float(np.quantile(arr, 0.95)),
    }


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────
def rouge1_f1(pred: str, ref: str) -> float:
    p_toks = pred.lower().split()
    r_toks = ref.lower().split()
    if not p_toks and not r_toks:
        return 1.0
    if not p_toks or not r_toks:
        return 0.0

    from collections import Counter
    pc = Counter(p_toks)
    rc = Counter(r_toks)
    overlap = 0
    for k, v in pc.items():
        overlap += min(v, rc.get(k, 0))

    prec = overlap / max(len(p_toks), 1)
    rec = overlap / max(len(r_toks), 1)
    if prec + rec == 0:
        return 0.0
    return (2.0 * prec * rec) / (prec + rec)


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    if len(b) > len(a):
        a, b = b, a

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def levenshtein_norm_distance(a: str, b: str) -> float:
    d = levenshtein_distance(a, b)
    denom = max(len(a), len(b), 1)
    return d / denom


# ─────────────────────────────────────────────────────────────
# Dataset: test_idx only, using symbols_train.npy
# ─────────────────────────────────────────────────────────────
class TestDataset(Dataset):
    def __init__(self, symbols_dir: Path):
        meta = json.loads((symbols_dir / "meta.json").read_text())
        self.pad_sym = int(meta.get("symbols_pad", 0))
        self.t5_pad = int(meta.get("t5_pad", 0))

        self.s_max = int(meta["s_max"])
        self.t_max = int(meta["t_max"])

        self.symbols = np.lib.format.open_memmap(symbols_dir / "symbols_train.npy", mode="r")
        self.targets = np.lib.format.open_memmap(symbols_dir / "t5_labels.npy", mode="r")

        if self.symbols.shape[0] != self.targets.shape[0]:
            raise RuntimeError("symbols_train.npy and t5_labels.npy row count mismatch")

        spl = np.load(symbols_dir / "splits.npz")
        self.test_idx = np.asarray(spl["test_idx"], dtype=np.int64)
        if self.test_idx.size == 0:
            raise RuntimeError("test_idx is empty")

    def __len__(self):
        return int(self.test_idx.shape[0])

    def __getitem__(self, i):
        ridx = int(self.test_idx[i])
        src = torch.from_numpy(self.symbols[ridx].astype(np.int64, copy=False))
        tgt = torch.from_numpy(self.targets[ridx].astype(np.int64, copy=False))
        return src, tgt


# ─────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────
def make_collate(tokenizer, prompt_text: str, output_prefix: str,
                 add_sentinel: bool, sentinel_text: str,
                 pad_sym: int, t5_pad: int):

    full_prompt = prompt_text + (sentinel_text if add_sentinel else "")
    prompt_ids = tokenizer(full_prompt, add_special_tokens=False).input_ids
    outpref_ids = tokenizer(output_prefix, add_special_tokens=False).input_ids

    prompt_ids_t = torch.tensor(prompt_ids, dtype=torch.long)
    outpref_ids_t = torch.tensor(outpref_ids, dtype=torch.long)

    pad_id = int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else 0

    def collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
        srcs, tgts = zip(*batch)

        inputs = []
        for s in srcs:
            s = s[s != pad_sym]
            if s.numel() == 0:
                s = torch.tensor([EMPTY_SYMBOL_FALLBACK_ID], dtype=torch.long)
            inp = torch.cat([prompt_ids_t, s, outpref_ids_t], dim=0)
            inputs.append(inp)

        max_len = max(x.numel() for x in inputs)
        input_ids = torch.full((len(inputs), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros_like(input_ids)

        for i, x in enumerate(inputs):
            L = x.numel()
            input_ids[i, :L] = x
            attn[i, :L] = 1

        tgt = torch.stack(tgts, dim=0)
        return input_ids, attn, tgt

    return collate


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one cost-effective first-segment run")
    parser.add_argument("--machine-type", required=True, choices=["laptop", "desktop"])
    parser.add_argument("--process-type", default="singleprocess", choices=["singleprocess", "multiprocess"])
    parser.add_argument("--method", required=True, choices=["uniform", "threshold", "oracle"])
    parser.add_argument("--q", required=True, type=int)
    parser.add_argument("--checkpoint-tag", default=CHECKPOINT_TAG)
    parser.add_argument("--no-jsonl", action="store_true")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    global MACHINE_TYPE, PROCESS_TYPE, METHOD, Q, CHECKPOINT_TAG, WRITE_JSONL
    global SYMBOLS_DIR, RUN_DIR, CKPT_DIR

    args = parse_args()
    MACHINE_TYPE = args.machine_type
    PROCESS_TYPE = args.process_type
    METHOD = args.method
    Q = args.q
    CHECKPOINT_TAG = args.checkpoint_tag
    WRITE_JSONL = not args.no_jsonl

    SYMBOLS_DIR = Path(
        f"/data/llamacpp/"
        f"ultrachat_cluster_cost_effective/{MACHINE_TYPE}_{PROCESS_TYPE}/K_{K}/{METHOD}/q_{Q}/first"
    )
    RUN_DIR = Path(
        f"/data/llamacpp/"
        f"ultrachat_cluster_cost_effective/{MACHINE_TYPE}_{PROCESS_TYPE}/"
        f"flan_t5_xl_ultrachat_K{K}_runs/{METHOD}/q_{Q}/run_first_sentences"
    )
    CKPT_DIR = Path(f"{RUN_DIR}/checkpoints/")


    seed_all(SEED)

    eval_tag = build_eval_tag(MACHINE_TYPE, PROCESS_TYPE, METHOD, Q)
    eval_dir = EVAL_ROOT / f"{MACHINE_TYPE}_{PROCESS_TYPE}" / METHOD / f"q_{Q}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = eval_dir / f"{eval_tag}_{CHECKPOINT_TAG}.jsonl"
    out_summary = eval_dir / f"{eval_tag}_{CHECKPOINT_TAG}_summary.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = (device.type == "cuda" and torch.cuda.is_bf16_supported())
    print("Device:", device, "bf16:", use_bf16)
    print("Eval tag:", eval_tag)

    meta = json.loads((SYMBOLS_DIR / "meta.json").read_text())
    base_vocab = int(meta["base_vocab"])
    k_in_meta = int(meta["K"])
    pad_sym = int(meta.get("symbols_pad", 0))
    t5_pad = int(meta.get("t5_pad", 0))

    prompt_text = str(meta.get("prompt_text"))
    output_prefix = str(meta.get("output_prefix"))
    add_sentinel = bool(meta.get("add_sentinel_token", True))
    sentinel_text = str(meta.get("sentinel_text", " <extra_id_0>"))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if int(tokenizer.vocab_size) != base_vocab:
        raise RuntimeError(f"Tokenizer vocab_size {tokenizer.vocab_size} != base_vocab {base_vocab}")

    ckpt_path = CKPT_DIR / CHECKPOINT_TAG
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print("Loading checkpoint:", ckpt_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_path,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        low_cpu_mem_usage=True,
    )
    model.resize_token_embeddings(base_vocab + k_in_meta)
    model.to(device)
    model.eval()

    ds = TestDataset(SYMBOLS_DIR)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=make_collate(
            tokenizer,
            prompt_text,
            output_prefix,
            add_sentinel,
            sentinel_text,
            pad_sym,
            t5_pad
        ),
    )

    st = SentenceTransformer(ST_MODEL, device=device)

    f = None
    if WRITE_JSONL:
        f = open(out_jsonl, "w", encoding="utf-8")

    cos_vals: List[float] = []
    rouge1_vals: List[float] = []
    ed_vals: List[float] = []

    pass_cnt = 0
    n_total = 0

    pbar = tqdm(dl, desc=f"Eval test ({METHOD}, q={Q}, {CHECKPOINT_TAG})", dynamic_ncols=True)
    for input_ids, attn, tgt_raw in pbar:
        input_ids = input_ids.to(device, non_blocking=True)
        attn = attn.to(device, non_blocking=True)

        with torch.no_grad():
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                gen = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    **GEN_CFG,
                )

        preds = tokenizer.batch_decode(gen, skip_special_tokens=True)

        refs: List[str] = []
        for t in tgt_raw:
            ids = t[t != t5_pad].tolist()
            refs.append(tokenizer.decode(ids, skip_special_tokens=True))

        emb = st.encode(
            preds + refs,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        B = len(preds)
        pred_emb = emb[:B]
        ref_emb = emb[B:]
        cos = (pred_emb * ref_emb).sum(dim=1).detach().cpu().numpy()

        for p, r, c in zip(preds, refs, cos.tolist()):
            n_total += 1

            c = float(c)
            r1 = float(rouge1_f1(p, r))
            ed = float(levenshtein_norm_distance(p, r))

            cos_vals.append(c)
            rouge1_vals.append(r1)
            ed_vals.append(ed)

            if c >= COS_THRESH:
                pass_cnt += 1

            if f is not None:
                f.write(json.dumps(
                    {
                        "pred": p,
                        "ref": r,
                        "cosine": c,
                        "rouge1_f1": r1,
                        "edit_distance": ed,
                    },
                    ensure_ascii=False
                ) + "\n")

        asr = pass_cnt / max(n_total, 1)
        pbar.set_postfix(
            mean_cos=float(np.mean(cos_vals)) if cos_vals else 0.0,
            asr_at_05=float(asr),
            mean_r1=float(np.mean(rouge1_vals)) if rouge1_vals else 0.0,
            mean_ed=float(np.mean(ed_vals)) if ed_vals else 0.0,
        )

    if f is not None:
        f.close()

    asr = pass_cnt / max(n_total, 1)

    summary = {
        "eval_tag": eval_tag,
        "checkpoint": str(ckpt_path),
        "test_samples": int(n_total),
        "cosine": {
            **safe_stats(cos_vals),
            "asr_at_0p5": float(asr),
        },
        "rouge1_f1": safe_stats(rouge1_vals),
        "edit_distance": safe_stats(ed_vals),
        "gen_cfg": GEN_CFG,
        "st_model": ST_MODEL,
        "seed": SEED,
        "machine_type": MACHINE_TYPE,
        "process_type": PROCESS_TYPE,
        "dataset": DATASET_NAME,
        "scope": SENTENCE_SCOPE,
        "framework": FRAMEWORK,
        "target_model": TARGET_MODEL,
        "method": METHOD,
        "q": int(Q),
    }

    out_summary.write_text(json.dumps(summary, indent=2))

    print("\nDONE")
    print("Checkpoint:", str(ckpt_path))
    print("Test samples:", n_total)
    print("ASR@0.5:", summary["cosine"]["asr_at_0p5"])
    print("Saved summary:", str(out_summary))
    if WRITE_JSONL:
        print("Saved JSONL:", str(out_jsonl))


if __name__ == "__main__":
    main()