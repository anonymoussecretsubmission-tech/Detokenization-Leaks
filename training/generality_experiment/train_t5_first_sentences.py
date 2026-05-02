#!/usr/bin/env python3
# train_t5_first_sentences.py

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import json
import random
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
SEED          = 42
MODEL_NAME    = "google/flan-t5-xl"
K             = 64
EPOCHS        = 5
BATCH_SIZE    = 32
LR            = 2e-4
WEIGHT_DECAY  = 1e-2
GRAD_CLIP     = 1.0
NUM_WORKERS   = 4

DECODE_SAMPLES      = 2
GEN_MAX_NEW_TOKENS  = 64
GEN_NUM_BEAMS       = 1
GEN_DO_SAMPLE       = False
GEN_EARLY_STOPPING  = False

EMPTY_SYMBOL_FALLBACK_ID = 1


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int):
    s = torch.initial_seed() % (2**32)
    np.random.seed(s + worker_id)
    random.seed(s + worker_id)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class SymbolToTextMemmapDataset(Dataset):
    def __init__(self, symbols_dir: Path, row_idx: np.ndarray):
        symbols_dir = Path(symbols_dir)
        meta = json.loads((symbols_dir / "meta.json").read_text())

        self.n_all  = int(meta["n_samples"])
        self.s_max  = int(meta["s_max"])
        self.t_max  = int(meta["t_max"])
        self.pad_sym = int(meta.get("symbols_pad", 0))
        self.t5_pad  = int(meta.get("t5_pad", 0))

        self.prompt_text   = str(meta.get("prompt_text", "Translate the following trace symbols into the beginning of the original sentence.\nTrace Symbols:"))
        self.output_prefix = str(meta.get("output_prefix", "\nOutput:"))
        self.add_sentinel  = bool(meta.get("add_sentinel_token", True))
        self.sentinel_text = str(meta.get("sentinel_text", " <extra_id_0>"))

        sym_path = symbols_dir / "symbols_train.npy"
        t5_path  = symbols_dir / "t5_labels.npy"
        if not sym_path.exists() or not t5_path.exists():
            raise FileNotFoundError(f"Missing memmaps under {symbols_dir}")

        self.symbols   = np.lib.format.open_memmap(sym_path, mode="r")
        self.t5_labels = np.lib.format.open_memmap(t5_path,  mode="r")

        if self.symbols.shape   != (self.n_all, self.s_max):
            raise RuntimeError(f"symbols shape {self.symbols.shape} != {(self.n_all, self.s_max)}")
        if self.t5_labels.shape != (self.n_all, self.t_max):
            raise RuntimeError(f"t5_labels shape {self.t5_labels.shape} != {(self.n_all, self.t_max)}")

        row_idx = np.asarray(row_idx, dtype=np.int64)
        if row_idx.ndim != 1 or row_idx.size == 0:
            raise ValueError("row_idx must be a non-empty 1D array")
        if row_idx.min() < 0 or row_idx.max() >= self.n_all:
            raise ValueError("row_idx out of range")

        self.row_idx = row_idx

    def __len__(self):
        return int(self.row_idx.shape[0])

    def __getitem__(self, i):
        ridx = int(self.row_idx[i])
        src = np.array(self.symbols[ridx],   copy=False).astype(np.int64, copy=False)
        tgt = np.array(self.t5_labels[ridx], copy=False).astype(np.int64, copy=False)
        return torch.from_numpy(src), torch.from_numpy(tgt)


# ─────────────────────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────────────────────
def make_collate(tokenizer, prompt_text: str, output_prefix: str,
                 add_sentinel: bool, sentinel_text: str,
                 pad_sym: int, t5_pad: int):

    full_prompt  = prompt_text + (sentinel_text if add_sentinel else "")
    prompt_ids_t = torch.tensor(tokenizer(full_prompt,    add_special_tokens=False).input_ids, dtype=torch.long)
    outpref_ids_t = torch.tensor(tokenizer(output_prefix, add_special_tokens=False).input_ids, dtype=torch.long)
    pad_id = int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else 0

    def _collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
        sym_list, tgt_list = zip(*batch)

        inputs: List[torch.Tensor] = []
        for sym in sym_list:
            sym_np = sym.numpy()
            sym_np = sym_np[sym_np != pad_sym]
            sym_clean = torch.from_numpy(sym_np.astype(np.int64, copy=False)) if sym_np.size > 0 \
                        else torch.tensor([EMPTY_SYMBOL_FALLBACK_ID], dtype=torch.long)
            inputs.append(torch.cat([prompt_ids_t, sym_clean, outpref_ids_t], dim=0))

        max_in = max(x.numel() for x in inputs)
        input_ids      = torch.full((len(inputs), max_in), fill_value=pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(inputs), max_in), dtype=torch.long)
        for i, inp in enumerate(inputs):
            L = inp.numel()
            input_ids[i, :L]      = inp
            attention_mask[i, :L] = 1

        tgt    = torch.stack(tgt_list, dim=0)
        labels = tgt.clone()
        labels[labels == t5_pad] = -100

        return input_ids, attention_mask, labels, tgt

    return _collate


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine-type", required=True, choices=["laptop", "desktop"])
    parser.add_argument("--model",        required=True, choices=["phi", "llama"])
    parser.add_argument("--framework",    required=True, choices=["llamacpp", "huggingface"])
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    machine_type     = args.machine_type
    target_model     = args.model
    target_framework = args.framework
    process_type     = "singleprocess"

    base = f"/data/{target_framework}/ultrachat_cluster/{machine_type}_{process_type}_{target_model}"
    symbols_dir = Path(f"{base}/K_{K}/first")
    output_dir  = Path(f"{base}/flan_t5_xl_ultrachat_K{K}_runs/run_first_sentences")

    seed_all(SEED)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    print(f"Device: {device}  bf16: {use_bf16}")

    meta       = json.loads((symbols_dir / "meta.json").read_text())
    base_vocab = int(meta["base_vocab"])
    pad_sym    = int(meta.get("symbols_pad", 0))
    t5_pad     = int(meta.get("t5_pad", 0))

    prompt_text   = str(meta.get("prompt_text"))
    output_prefix = str(meta.get("output_prefix"))
    add_sentinel  = bool(meta.get("add_sentinel_token", True))
    sentinel_text = str(meta.get("sentinel_text", " <extra_id_0>"))

    spl       = np.load(symbols_dir / "splits.npz")
    train_idx = spl["train_idx"]
    val_idx   = spl["val_idx"]
    print(f"Split sizes — train: {len(train_idx)}  val: {len(val_idx)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.vocab_size != base_vocab:
        raise RuntimeError(f"Tokenizer vocab_size {tokenizer.vocab_size} != meta base_vocab {base_vocab}")

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(base_vocab + K)
    model.to(device)

    collate_fn = make_collate(tokenizer, prompt_text, output_prefix,
                               add_sentinel, sentinel_text, pad_sym, t5_pad)

    dl_kwargs = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                     pin_memory=True, worker_init_fn=worker_init_fn, collate_fn=collate_fn)
    train_dl = DataLoader(SymbolToTextMemmapDataset(symbols_dir, train_idx), shuffle=True,  **dl_kwargs)
    val_dl   = DataLoader(SymbolToTextMemmapDataset(symbols_dir, val_idx),   shuffle=False, **dl_kwargs)

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, EPOCHS + 1):
        # ── Train
        model.train()
        running = 0.0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS}", dynamic_ncols=True, mininterval=0.5)
        for i, (input_ids, attn, labels, _) in enumerate(pbar, 1):
            input_ids = input_ids.to(device, non_blocking=True)
            attn      = attn.to(device,      non_blocking=True)
            labels    = labels.to(device,    non_blocking=True)

            ctx = autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16)
            with ctx:
                loss = model(input_ids=input_ids, attention_mask=attn, labels=labels).loss

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered during training.")

            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optim.step()

            running += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{running / i:.4f}")

        # ── Validate
        model.eval()
        vlosses, decoded = [], 0
        with torch.no_grad():
            for input_ids, attn, labels, tgt_raw in tqdm(val_dl, desc="Validating", dynamic_ncols=True, leave=False):
                input_ids = input_ids.to(device, non_blocking=True)
                attn      = attn.to(device,      non_blocking=True)
                labels    = labels.to(device,    non_blocking=True)

                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                if torch.isfinite(out.loss):
                    vlosses.append(float(out.loss.item()))

                if decoded < DECODE_SAMPLES:
                    gen  = model.generate(input_ids=input_ids[:1], attention_mask=attn[:1],
                                          max_new_tokens=GEN_MAX_NEW_TOKENS, num_beams=GEN_NUM_BEAMS,
                                          do_sample=GEN_DO_SAMPLE, early_stopping=GEN_EARLY_STOPPING)
                    pred = tokenizer.decode(gen[0].tolist(), skip_special_tokens=True).strip()
                    ref_ids = [t for t in tgt_raw[0].tolist() if t != t5_pad]
                    ref  = tokenizer.decode(ref_ids, skip_special_tokens=True).strip()
                    print(f"\n[DECODE epoch {epoch}]")
                    print("PRED:", pred[:500])
                    print("REF :", ref[:500])
                    decoded += 1

        if vlosses:
            print(f"Epoch {epoch} — val loss: {float(np.mean(vlosses)):.4f}")

    # ── Save final model
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving final model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done.")


if __name__ == "__main__":
    main()