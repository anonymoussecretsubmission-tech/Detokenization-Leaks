#!/usr/bin/env python3
# eval_noise_no_train.py
#
# Evaluates Flan-T5-XL on HELDOUT trace symbols for the ultrachat first-segment model,
# after injecting synthetic noise into the clustering output.
#
# Noise model:
#   For each original symbol, with probability p apply exactly one corruption
#   chosen uniformly at random from:
#       - deletion
#       - insertion
#       - substitution
#
# Computes:
#   - cosine similarity (sentence-transformers)
#   - ROUGE-1 F1
#   - normalized edit distance
#
# Writes:
#   - one JSONL file per noise level
#   - one summary JSON per noise level

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import json
import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
SEED = 42
MODEL_NAME = "google/flan-t5-xl"

MACHINE_TYPE = "laptop"
PROCESS_TYPE = "singleprocess"
K = 64

P_LIST = [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3]

DATASET_NAME = "ultrachat"
SENTENCE_SCOPE = "firstsentences"
FRAMEWORK = "llamacpp"
TARGET_MODEL = "phi"

SYMBOLS_DIR = Path(
    f"/data/{FRAMEWORK}/ultrachat_cluster/{MACHINE_TYPE}_{PROCESS_TYPE}_{TARGET_MODEL}/K_{K}/first"
)
RUN_DIR = Path(
    f"/data/{FRAMEWORK}/ultrachat_cluster/{MACHINE_TYPE}_{PROCESS_TYPE}_{TARGET_MODEL}/flan_t5_xl_ultrachat_K{K}_runs/run_first_sentences"
)
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

EVAL_DIR = Path("./eval_noise_no_train")
WRITE_JSONL = True

EMPTY_SYMBOL_FALLBACK_ID = 1


# ─────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_eval_tag(machine_type: str, process_type: str) -> str:
    return (
        f"dataset-{DATASET_NAME}_"
        f"{process_type}_"
        f"{machine_type}_"
        f"{SENTENCE_SCOPE}_"
        f"framework-{FRAMEWORK}_"
        f"target-{TARGET_MODEL}"
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


def fmt_p(x: float) -> str:
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


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
# Dataset
# ─────────────────────────────────────────────────────────────
class HeldoutTestDataset(Dataset):
    def __init__(self, symbols_dir: Path):
        meta = json.loads((symbols_dir / "meta.json").read_text())
        self.pad_sym = int(meta.get("symbols_pad", 0))
        self.t5_pad = int(meta.get("t5_pad", 0))

        self.s_max = int(meta["s_max"])
        self.t_max = int(meta["t_max"])

        self.symbols = np.lib.format.open_memmap(symbols_dir / "symbols_heldout.npy", mode="r")
        self.targets = np.lib.format.open_memmap(symbols_dir / "t5_labels.npy", mode="r")

        if self.symbols.shape[0] != self.targets.shape[0]:
            raise RuntimeError("symbols_heldout.npy and t5_labels.npy row count mismatch")

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
        return src, tgt, ridx


# ─────────────────────────────────────────────────────────────
# Noise
# ─────────────────────────────────────────────────────────────
def corrupt_symbol_sequence(
    seq: torch.Tensor,
    pad_sym: int,
    valid_symbol_ids: List[int],
    p_noise: float,
    rng: random.Random,
) -> Tuple[torch.Tensor, int, int, int]:
    """
    For each original token, with probability p_noise, choose uniformly from:
      - delete the token
      - insert one random token after it
      - substitute it with a random token
    """
    clean = [int(x) for x in seq.tolist() if int(x) != pad_sym]

    out: List[int] = []
    n_del = 0
    n_ins = 0
    n_sub = 0

    for tok in clean:
        if rng.random() < p_noise:
            action = rng.choice(("del", "ins", "sub"))

            if action == "del":
                n_del += 1
                continue

            if action == "ins":
                out.append(tok)
                out.append(rng.choice(valid_symbol_ids))
                n_ins += 1
                continue

            tok = rng.choice(valid_symbol_ids)
            out.append(tok)
            n_sub += 1
            continue

        out.append(tok)

    if len(out) == 0:
        out = [EMPTY_SYMBOL_FALLBACK_ID]

    return torch.tensor(out, dtype=torch.long), n_del, n_ins, n_sub


# ─────────────────────────────────────────────────────────────
# Collate: prompt + noisy symbols(no-pad) + output_prefix
# ─────────────────────────────────────────────────────────────
def make_noisy_collate(
    tokenizer,
    prompt_text: str,
    output_prefix: str,
    add_sentinel: bool,
    sentinel_text: str,
    pad_sym: int,
    p_noise: float,
    valid_symbol_ids: List[int],
    seed: int,
):
    full_prompt = prompt_text + (sentinel_text if add_sentinel else "")
    prompt_ids = tokenizer(full_prompt, add_special_tokens=False).input_ids
    outpref_ids = tokenizer(output_prefix, add_special_tokens=False).input_ids

    prompt_ids_t = torch.tensor(prompt_ids, dtype=torch.long)
    outpref_ids_t = torch.tensor(outpref_ids, dtype=torch.long)

    pad_id = int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else 0
    rng = random.Random(seed)

    def collate(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]):
        srcs, tgts, ridxs = zip(*batch)

        inputs = []
        noisy_lens = []
        del_counts = []
        ins_counts = []
        sub_counts = []

        for s in srcs:
            noisy_s, n_del, n_ins, n_sub = corrupt_symbol_sequence(
                seq=s,
                pad_sym=pad_sym,
                valid_symbol_ids=valid_symbol_ids,
                p_noise=p_noise,
                rng=rng,
            )
            noisy_lens.append(int(noisy_s.numel()))
            del_counts.append(n_del)
            ins_counts.append(n_ins)
            sub_counts.append(n_sub)

            inp = torch.cat([prompt_ids_t, noisy_s, outpref_ids_t], dim=0)
            inputs.append(inp)

        max_len = max(x.numel() for x in inputs)
        input_ids = torch.full((len(inputs), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros_like(input_ids)

        for i, x in enumerate(inputs):
            L = x.numel()
            input_ids[i, :L] = x
            attn[i, :L] = 1

        tgt = torch.stack(tgts, dim=0)
        ridx_t = torch.tensor(ridxs, dtype=torch.long)
        noisy_lens_t = torch.tensor(noisy_lens, dtype=torch.long)
        del_counts_t = torch.tensor(del_counts, dtype=torch.long)
        ins_counts_t = torch.tensor(ins_counts, dtype=torch.long)
        sub_counts_t = torch.tensor(sub_counts, dtype=torch.long)

        return input_ids, attn, tgt, ridx_t, noisy_lens_t, del_counts_t, ins_counts_t, sub_counts_t

    return collate


# ─────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Eval ultrachat first-segment heldout test set with added symbol noise"
    )
    parser.add_argument(
        "--machine-type",
        required=True,
        choices=["laptop", "desktop"],
        help="Type of machine: laptop or desktop"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    global MACHINE_TYPE, PROCESS_TYPE, K, P_LIST
    global SYMBOLS_DIR, RUN_DIR
    global CHECKPOINT_TAG, WRITE_JSONL, BATCH_SIZE

    args = parse_args()
    MACHINE_TYPE = args.machine_type

    SYMBOLS_DIR = Path(
        f"/data/{FRAMEWORK}/ultrachat_cluster/{MACHINE_TYPE}_{PROCESS_TYPE}_{TARGET_MODEL}/K_{K}/first"
    )
    RUN_DIR = Path(
        f"/data/{FRAMEWORK}/ultrachat_cluster/{MACHINE_TYPE}_{PROCESS_TYPE}_{TARGET_MODEL}/flan_t5_xl_ultrachat_K{K}_runs/run_first_sentences"
    )

    seed_all(SEED)

    eval_tag = build_eval_tag(MACHINE_TYPE, PROCESS_TYPE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = (device.type == "cuda" and torch.cuda.is_bf16_supported())
    print("Device:", device, "bf16:", use_bf16)
    print("Eval tag:", eval_tag)

    meta = json.loads((SYMBOLS_DIR / "meta.json").read_text())
    base_vocab = int(meta["base_vocab"])
    K = int(meta["K"])
    pad_sym = int(meta.get("symbols_pad", 0))
    t5_pad = int(meta.get("t5_pad", 0))

    prompt_text = str(meta.get("prompt_text"))
    output_prefix = str(meta.get("output_prefix"))
    add_sentinel = bool(meta.get("add_sentinel_token", True))
    sentinel_text = str(meta.get("sentinel_text", " <extra_id_0>"))

    if not prompt_text or not output_prefix:
        raise RuntimeError("meta.json is missing prompt_text / output_prefix")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if int(tokenizer.vocab_size) != base_vocab:
        raise RuntimeError(f"Tokenizer vocab_size {tokenizer.vocab_size} != base_vocab {base_vocab}")

    if not RUN_DIR.exists():
        raise FileNotFoundError(f"Checkpoint not found: {RUN_DIR}")

    print("Loading checkpoint:", RUN_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        RUN_DIR,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        low_cpu_mem_usage=True,
    )
    model.resize_token_embeddings(base_vocab + K)
    model.to(device)
    model.eval()

    ds = HeldoutTestDataset(SYMBOLS_DIR)
    st = SentenceTransformer(ST_MODEL, device=device)

    valid_symbol_ids = list(range(base_vocab, base_vocab + K))

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    for p_noise in P_LIST:
        noise_tag = fmt_p(p_noise)
        out_jsonl = EVAL_DIR / f"{eval_tag}_{CHECKPOINT_TAG}_noise-{noise_tag}.jsonl"
        out_summary = EVAL_DIR / f"{eval_tag}_{CHECKPOINT_TAG}_noise-{noise_tag}_summary.json"

        dl = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=make_noisy_collate(
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                output_prefix=output_prefix,
                add_sentinel=add_sentinel,
                sentinel_text=sentinel_text,
                pad_sym=pad_sym,
                p_noise=p_noise,
                valid_symbol_ids=valid_symbol_ids,
                seed=SEED + int(round(p_noise * 100000)),
            ),
        )

        f = None
        if WRITE_JSONL:
            f = open(out_jsonl, "w", encoding="utf-8")

        cos_vals: List[float] = []
        rouge1_vals: List[float] = []
        ed_vals: List[float] = []

        total_del = 0
        total_ins = 0
        total_sub = 0
        total_noisy_len = 0

        pass_cnt = 0
        n_total = 0

        print(f"\nRunning noise p = {p_noise}")
        pbar = tqdm(dl, desc=f"Eval noisy heldout p={p_noise}", dynamic_ncols=True)

        for input_ids, attn, tgt_raw, ridx, noisy_lens, del_counts, ins_counts, sub_counts in pbar:
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

            for i, (pred, ref, c) in enumerate(zip(preds, refs, cos.tolist())):
                c = float(c)
                r1 = float(rouge1_f1(pred, ref))
                ed = float(levenshtein_norm_distance(pred, ref))

                ndel = int(del_counts[i].item())
                nins = int(ins_counts[i].item())
                nsub = int(sub_counts[i].item())
                nlen = int(noisy_lens[i].item())
                row_idx = int(ridx[i].item())

                n_total += 1
                pass_cnt += int(c >= COS_THRESH)
                total_del += ndel
                total_ins += nins
                total_sub += nsub
                total_noisy_len += nlen

                cos_vals.append(c)
                rouge1_vals.append(r1)
                ed_vals.append(ed)

                if f is not None:
                    f.write(json.dumps(
                        {
                            "row_idx": row_idx,
                            "noise_p": float(p_noise),
                            "num_deletions": ndel,
                            "num_insertions": nins,
                            "num_substitutions": nsub,
                            "noisy_symbol_len": nlen,
                            "pred": pred,
                            "ref": ref,
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
            "checkpoint": str(RUN_DIR),
            "test_samples": int(n_total),
            "noise": {
                "base_p": float(p_noise),
                "noise_model": "for each original symbol, with probability p choose uniformly from deletion/insertion/substitution",
                "action_probs_given_noise": {
                    "deletion": 1.0 / 3.0,
                    "insertion": 1.0 / 3.0,
                    "substitution": 1.0 / 3.0,
                },
            },
            "cosine": {
                **safe_stats(cos_vals),
                "asr_at_0p5": float(asr),
            },
            "rouge1_f1": safe_stats(rouge1_vals),
            "edit_distance": {
                **safe_stats(ed_vals),
                "mean_int": float(np.mean(ed_vals)) if ed_vals else 0.0,
                "median_int": float(np.median(ed_vals)) if ed_vals else 0.0,
            },
            "avg_noisy_symbol_len": float(total_noisy_len / max(n_total, 1)),
            "avg_deletions_per_sample": float(total_del / max(n_total, 1)),
            "avg_insertions_per_sample": float(total_ins / max(n_total, 1)),
            "avg_substitutions_per_sample": float(total_sub / max(n_total, 1)),
            "total_deletions": int(total_del),
            "total_insertions": int(total_ins),
            "total_substitutions": int(total_sub),
            "gen_cfg": GEN_CFG,
            "st_model": ST_MODEL,
            "seed": SEED,
            "machine_type": MACHINE_TYPE,
            "process_type": PROCESS_TYPE,
            "dataset": DATASET_NAME,
            "scope": SENTENCE_SCOPE,
            "framework": FRAMEWORK,
            "target_model": TARGET_MODEL,
        }

        Path(out_summary).write_text(json.dumps(summary, indent=2))

        print("\nDONE for noise p =", p_noise)
        print("Checkpoint:", str(RUN_DIR))
        print("Test samples:", n_total)
        print("Cosine mean/median:", summary["cosine"]["mean"], summary["cosine"]["median"])
        print("ASR@0.5:", summary["cosine"]["asr_at_0p5"])
        print("ROUGE-1 F1 mean/median:", summary["rouge1_f1"]["mean"], summary["rouge1_f1"]["median"])
        print("Edit distance mean/median:", summary["edit_distance"]["mean_int"], summary["edit_distance"]["median_int"])
        print("Saved summary:", str(out_summary))
        if WRITE_JSONL:
            print("Saved JSONL:", str(out_jsonl))


if __name__ == "__main__":
    main()