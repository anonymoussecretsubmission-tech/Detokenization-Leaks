#!/usr/bin/env python3
# eval_t5_first.py
#
# Evaluates Flan-T5-XL on HELDOUT trace symbols, ONLY on test split (test_idx from splits.npz).
# Computes:
#   - cosine similarity (sentence-transformers)
#   - ROUGE-1 F1 (unigram overlap)
#   - edit distance (Levenshtein) on character level
#
# Output filenames include:
#   dataset-ultrachat, laptop/desktop, firstsentences, framework-llamacpp, target-phi

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any

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

DATASET_NAME = "ultrachat"
SENTENCE_SCOPE = "firstsentences"
FRAMEWORK = "llamacpp"
TARGET_MODEL = "phi"
K = 64

CHECKPOINT_TAG = "final"

BATCH_SIZE = 256
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

EVAL_DIR = Path("./eval")
WRITE_JSONL = True

EMPTY_SYMBOL_FALLBACK_ID = 1
SEGMENT_META_JSONL = "segment_meta.jsonl"


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
        f"{machine_type}_{process_type}_{target_model}/K_{k}/first"
    )


def build_run_dir(machine_type: str, process_type: str, framework: str, target_model: str, k: int) -> Path:
    return Path(
        f"/outpath/{framework}/ultrachat_cluster/"
        f"{machine_type}_{process_type}_{target_model}/flan_t5_xl_ultrachat_K{k}_runs/run_first_sentences"
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
# segment_meta loader  (mirrors eval_t5_middle.py)
# ─────────────────────────────────────────────────────────────
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
                raise RuntimeError(
                    f"{path.name} line {i+1}: missing paragraph_id/group_id"
                )

            if "segment_idx" not in obj:
                raise RuntimeError(f"{path.name} line {i+1}: missing segment_idx")

            out = dict(obj)
            out["paragraph_id"] = paragraph_id
            out["segment_idx"] = int(obj["segment_idx"])
            rows.append(out)

    if len(rows) != n_expected:
        raise RuntimeError(
            f"{path.name}: {len(rows)} rows != n_expected {n_expected}"
        )
    return rows


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
    overlap = sum(min(v, rc.get(k, 0)) for k, v in pc.items())

    prec = overlap / max(len(p_toks), 1)
    rec  = overlap / max(len(r_toks), 1)
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
            cur.append(min(cur[j-1]+1, prev[j]+1, prev[j-1]+(0 if ca==cb else 1)))
        prev = cur
    return prev[-1]


def levenshtein_norm_distance(a: str, b: str) -> float:
    d = levenshtein_distance(a, b)
    return d / max(len(a), len(b), 1)


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────
class HeldoutTestDataset(Dataset):
    def __init__(self, symbols_dir: Path):
        meta = json.loads((symbols_dir / "meta.json").read_text())
        self.pad_sym = int(meta.get("symbols_pad", 0))
        self.t5_pad  = int(meta.get("t5_pad", 0))
        self.s_max   = int(meta["s_max"])
        self.t_max   = int(meta["t_max"])

        self.symbols = np.lib.format.open_memmap(symbols_dir / "symbols_heldout.npy", mode="r")
        self.targets = np.lib.format.open_memmap(symbols_dir / "t5_labels.npy",       mode="r")

        if self.symbols.shape[0] != self.targets.shape[0]:
            raise RuntimeError("symbols_heldout.npy and t5_labels.npy row count mismatch")

        # Load segment_meta so we can attach paragraph_id to each output record.
        seg_meta_path = symbols_dir / SEGMENT_META_JSONL
        if seg_meta_path.exists():
            self.segment_meta = load_segment_meta_jsonl(
                seg_meta_path, n_expected=self.symbols.shape[0]
            )
        else:
            # Fallback: no segment_meta — paragraph_id will be the row index.
            # This means first + middle cannot be merged.  Warn loudly.
            import warnings
            warnings.warn(
                f"segment_meta.jsonl not found in {symbols_dir}. "
                "paragraph_id will be a fallback index. "
                "Re-run prepare_flan_symbols_k48.py to generate it.",
                stacklevel=2,
            )
            self.segment_meta = [
                {"paragraph_id": f"fallback_{i}", "segment_idx": 0}
                for i in range(self.symbols.shape[0])
            ]

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
        seg_meta = self.segment_meta[ridx]
        return src, tgt, seg_meta


# ─────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────
def make_collate(tokenizer, prompt_text: str, output_prefix: str,
                 add_sentinel: bool, sentinel_text: str,
                 pad_sym: int, t5_pad: int):

    full_prompt   = prompt_text + (sentinel_text if add_sentinel else "")
    prompt_ids    = tokenizer(full_prompt,   add_special_tokens=False).input_ids
    outpref_ids   = tokenizer(output_prefix, add_special_tokens=False).input_ids

    prompt_ids_t  = torch.tensor(prompt_ids,  dtype=torch.long)
    outpref_ids_t = torch.tensor(outpref_ids, dtype=torch.long)
    pad_id        = int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else 0

    def collate(batch: List[Tuple[torch.Tensor, torch.Tensor, Dict]]):
        srcs, tgts, metas = zip(*batch)

        inputs = []
        for s in srcs:
            s = s[s != pad_sym]
            if s.numel() == 0:
                s = torch.tensor([EMPTY_SYMBOL_FALLBACK_ID], dtype=torch.long)
            inputs.append(torch.cat([prompt_ids_t, s, outpref_ids_t], dim=0))

        max_len    = max(x.numel() for x in inputs)
        input_ids  = torch.full((len(inputs), max_len), pad_id, dtype=torch.long)
        attn       = torch.zeros_like(input_ids)
        for i, x in enumerate(inputs):
            L = x.numel()
            input_ids[i, :L] = x
            attn[i, :L]      = 1

        tgt = torch.stack(tgts, dim=0)
        return input_ids, attn, tgt, list(metas)

    return collate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine-type", required=True, choices=["laptop", "desktop"])
    parser.add_argument("--model",        required=True, choices=["phi", "llama"])
    parser.add_argument("--framework",    required=True, choices=["llamacpp", "huggingface"])
    parser.add_argument("--checkpoint-tag", default=CHECKPOINT_TAG)
    parser.add_argument("--no-jsonl",     action="store_true")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    global MACHINE_TYPE, PROCESS_TYPE, K, TARGET_MODEL, FRAMEWORK
    global CHECKPOINT_TAG, WRITE_JSONL

    args          = parse_args()
    MACHINE_TYPE  = args.machine_type
    TARGET_MODEL  = args.model
    FRAMEWORK     = args.framework
    CHECKPOINT_TAG = args.checkpoint_tag
    WRITE_JSONL   = not args.no_jsonl

    SYMBOLS_DIR = build_symbols_dir(MACHINE_TYPE, PROCESS_TYPE, FRAMEWORK, TARGET_MODEL, K)
    RUN_DIR     = build_run_dir(MACHINE_TYPE, PROCESS_TYPE, FRAMEWORK, TARGET_MODEL, K)

    seed_all(SEED)

    eval_tag    = build_eval_tag(MACHINE_TYPE, PROCESS_TYPE, FRAMEWORK, TARGET_MODEL)
    OUT_JSONL   = EVAL_DIR / f"{eval_tag}_{CHECKPOINT_TAG}.jsonl"
    OUT_SUMMARY = EVAL_DIR / f"{eval_tag}_{CHECKPOINT_TAG}_summary.json"

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = (device.type == "cuda" and torch.cuda.is_bf16_supported())
    print("Device:", device, "bf16:", use_bf16)
    print("Eval tag:", eval_tag)

    meta        = json.loads((SYMBOLS_DIR / "meta.json").read_text())
    base_vocab  = int(meta["base_vocab"])
    K           = int(meta["K"])
    pad_sym     = int(meta.get("symbols_pad", 0))
    t5_pad      = int(meta.get("t5_pad", 0))
    prompt_text    = str(meta.get("prompt_text"))
    output_prefix  = str(meta.get("output_prefix"))
    add_sentinel   = bool(meta.get("add_sentinel_token", True))
    sentinel_text  = str(meta.get("sentinel_text", " <extra_id_0>"))

    if not prompt_text or not output_prefix:
        raise RuntimeError("meta.json missing prompt_text / output_prefix")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if int(tokenizer.vocab_size) != base_vocab:
        raise RuntimeError(f"Tokenizer vocab_size {tokenizer.vocab_size} != base_vocab {base_vocab}")

    if not RUN_DIR.exists():
        raise FileNotFoundError(f"Checkpoint not found: {RUN_DIR}")

    print("Loading model:", RUN_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        RUN_DIR,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        low_cpu_mem_usage=True,
    )
    model.resize_token_embeddings(base_vocab + K)
    model.to(device)
    model.eval()

    ds = HeldoutTestDataset(SYMBOLS_DIR)
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=make_collate(
            tokenizer, prompt_text, output_prefix,
            add_sentinel, sentinel_text, pad_sym, t5_pad,
        ),
    )

    st = SentenceTransformer(ST_MODEL, device=device)

    f = None
    if WRITE_JSONL:
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        f = open(OUT_JSONL, "w", encoding="utf-8")

    cos_vals:    List[float] = []
    rouge1_vals: List[float] = []
    ed_vals:     List[float] = []
    pass_cnt = 0
    n_total  = 0

    pbar = tqdm(dl, desc=f"Eval test heldout ({CHECKPOINT_TAG})", dynamic_ncols=True)
    for input_ids, attn, tgt_raw, metas in pbar:
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

        emb    = st.encode(preds + refs, convert_to_tensor=True, normalize_embeddings=True)
        B      = len(preds)
        cos    = (emb[:B] * emb[B:]).sum(dim=1).detach().cpu().numpy()

        for p, r, c, meta_row in zip(preds, refs, cos.tolist(), metas):
            n_total += 1
            c  = float(c)
            r1 = float(rouge1_f1(p, r))
            ed = levenshtein_norm_distance(p, r)

            cos_vals.append(c)
            rouge1_vals.append(r1)
            ed_vals.append(ed)

            if c >= COS_THRESH:
                pass_cnt += 1

            if f is not None:
                f.write(json.dumps(
                    {
                        "paragraph_id": str(meta_row["paragraph_id"]),
                        "segment_idx":  0,
                        "pred":         p,
                        "ref":          r,
                        "cosine":       c,
                        "rouge1_f1":    r1,
                        "edit_distance": ed,
                    },
                    ensure_ascii=False,
                ) + "\n")

        asr = pass_cnt / max(n_total, 1)
        pbar.set_postfix(
            mean_cos=float(np.mean(cos_vals)) if cos_vals else 0.0,
            asr=float(asr),
        )

    if f is not None:
        f.close()

    asr     = pass_cnt / max(n_total, 1)
    summary = {
        "eval_tag":    eval_tag,
        "path":  str(RUN_DIR),
        "test_samples": int(n_total),
        "cosine": {**safe_stats(cos_vals), "asr_at_0p5": float(asr)},
        "rouge1_f1": safe_stats(rouge1_vals),
        "edit_distance": safe_stats(ed_vals),
        "gen_cfg":     GEN_CFG,
        "st_model":    ST_MODEL,
        "seed":        SEED,
        "machine_type": MACHINE_TYPE,
        "process_type": PROCESS_TYPE,
        "dataset":     DATASET_NAME,
        "scope":       SENTENCE_SCOPE,
        "framework":   FRAMEWORK,
        "target_model": TARGET_MODEL,
        "paragraph_id_source": "segment_meta.jsonl (group_id)",
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    Path(OUT_SUMMARY).write_text(json.dumps(summary, indent=2))

    print("\nDONE")
    print("path:", str(RUN_DIR))
    print("Test samples:", n_total)
    print("ASR@0.5:", summary["cosine"]["asr_at_0p5"])
    print("Saved summary:", str(OUT_SUMMARY))
    if WRITE_JSONL:
        print("Saved JSONL:", str(OUT_JSONL))


if __name__ == "__main__":
    main()