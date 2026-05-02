#!/usr/bin/env python3
# train_t5_middle_sentences.py
#
# Train Flan-T5-XL to map trace-symbol sequences + context -> next-text prefix.
# Checkpoints saved at even epochs only, plus a final save.
#
# Inputs under SYMBOLS_DIR:
#   - symbols_train.npy    int32 [N, S_MAX]
#   - t5_labels.npy        int32 [N, T_MAX]
#   - context_text.npy     object/str array [N]  (preferred)
#   - context_text.jsonl   fallback ({"context_text": "..."} per line)
#   - meta.json            includes base_vocab, K, pads, kind="middle"
#   - splits.npz           train_idx/val_idx/test_idx

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
SEED         = 42
MODEL_NAME   = "google/flan-t5-xl"
K            = 64
EPOCHS       = 5
BATCH_SIZE   = 12
LR           = 2e-4
WEIGHT_DECAY = 1e-2
GRAD_CLIP    = 1.0
NUM_WORKERS  = 4

RESUME_FROM = None  # e.g. "epoch_4" or "final" or None

DECODE_SAMPLES     = 2
GEN_MAX_NEW_TOKENS = 64
GEN_NUM_BEAMS      = 1
GEN_DO_SAMPLE      = False
GEN_EARLY_STOPPING = False

EMPTY_SYMBOL_FALLBACK_ID = 1

# Prompt template
CONTEXT_PREFIX = "Context:\n"
SYMBOLS_PREFIX = "\nTrace Symbols:"
OUTPUT_PREFIX  = "\nOutput:"


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
# Context loading
# ─────────────────────────────────────────────────────────────────────────────
def load_context_array(symbols_dir: Path, n_expected: int) -> np.ndarray:
    npy_path   = symbols_dir / "context_text.npy"
    jsonl_path = symbols_dir / "context_text.jsonl"

    if npy_path.exists():
        arr = np.load(npy_path, allow_pickle=True)
        if arr.shape[0] != n_expected:
            raise RuntimeError(f"context_text.npy len {arr.shape[0]} != n_samples {n_expected}")
        return arr

    if jsonl_path.exists():
        ctx = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ctx.append(str(json.loads(line).get("context_text", "")))
        if len(ctx) != n_expected:
            raise RuntimeError(f"context_text.jsonl lines {len(ctx)} != n_samples {n_expected}")
        return np.asarray(ctx, dtype=object)

    raise FileNotFoundError(
        f"Missing context source. Provide either:\n  {npy_path}\nor\n  {jsonl_path}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class SymbolToTextWithContextMemmapDataset(Dataset):
    def __init__(self, symbols_dir: Path, row_idx: np.ndarray):
        symbols_dir = Path(symbols_dir)
        meta = json.loads((symbols_dir / "meta.json").read_text())

        kind = str(meta.get("kind", "")).lower()
        if kind and kind != "middle":
            raise RuntimeError(f"meta.kind={kind!r} but expected 'middle'")

        self.n_all   = int(meta["n_samples"])
        self.s_max   = int(meta["s_max"])
        self.t_max   = int(meta["t_max"])
        self.pad_sym = int(meta.get("symbols_pad", 0))
        self.t5_pad  = int(meta.get("t5_pad", 0))

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

        self.context = load_context_array(symbols_dir, self.n_all)

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
        ctx = str(self.context[ridx] if self.context[ridx] is not None else "")
        return src, ctx, tgt


# ─────────────────────────────────────────────────────────────────────────────
# Collate: "Context:\n{ctx}\nTrace Symbols:" + symbols + "\nOutput:"
# ─────────────────────────────────────────────────────────────────────────────
def make_collate(tokenizer, pad_sym: int, t5_pad: int):
    pad_id        = int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else 0
    outpref_ids_t = torch.tensor(tokenizer(OUTPUT_PREFIX, add_special_tokens=False).input_ids, dtype=torch.long)

    def _collate(batch: List[Tuple[np.ndarray, str, np.ndarray]]):
        sym_list, ctx_list, tgt_list = zip(*batch)

        inputs: List[torch.Tensor] = []
        for sym, ctx in zip(sym_list, ctx_list):
            sym_np = np.asarray(sym)
            sym_np = sym_np[sym_np != pad_sym]
            sym_clean = torch.from_numpy(sym_np.astype(np.int64, copy=False)) if sym_np.size > 0 \
                        else torch.tensor([EMPTY_SYMBOL_FALLBACK_ID], dtype=torch.long)

            prompt = f"Translate the following trace symbols using the following context.\n{CONTEXT_PREFIX}{ctx}{SYMBOLS_PREFIX}"
            prompt_ids_t = torch.tensor(tokenizer(prompt, add_special_tokens=False).input_ids, dtype=torch.long)

            inputs.append(torch.cat([prompt_ids_t, sym_clean, outpref_ids_t], dim=0))

        max_in         = max(x.numel() for x in inputs)
        input_ids      = torch.full((len(inputs), max_in), fill_value=pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(inputs), max_in), dtype=torch.long)
        for i, inp in enumerate(inputs):
            L = inp.numel()
            input_ids[i, :L]      = inp
            attention_mask[i, :L] = 1

        tgt    = torch.stack([torch.from_numpy(np.asarray(t)) for t in tgt_list], dim=0).long()
        labels = tgt.clone()
        labels[labels == t5_pad] = -100

        return input_ids, attention_mask, labels, tgt

    return _collate


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(tag: str, step: int, ckpt_dir: Path, model, tokenizer, optim):
    hf_dir = ckpt_dir / tag
    hf_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ckpt] Saving to {hf_dir}")
    model.save_pretrained(hf_dir)
    tokenizer.save_pretrained(hf_dir)
    torch.save({"optim": optim.state_dict(), "step": step}, hf_dir / "trainer_state.pt")


def try_resume(model, optim, ckpt_dir: Path, tag: str, device: torch.device) -> int:
    hf_dir = ckpt_dir / tag
    if not hf_dir.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {hf_dir}")
    print(f"Resuming from {hf_dir}")
    model.load_state_dict(AutoModelForSeq2SeqLM.from_pretrained(hf_dir).state_dict())
    model.to(device)
    state_path = hf_dir / "trainer_state.pt"
    if state_path.exists():
        cp = torch.load(state_path, map_location="cpu")
        optim.load_state_dict(cp["optim"])
        return int(cp.get("step", 0))
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine-type", required=True, choices=["laptop", "desktop"])
    parser.add_argument("--framework",    required=True, choices=["llamacpp", "huggingface"])
    parser.add_argument("--model",        required=True, choices=["llama", "phi"])
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args             = parse_args()
    machine_type     = args.machine_type
    target_framework = args.framework
    target_model     = args.model
    process_type     = "singleprocess"

    base        = f"/data/{target_framework}/ultrachat_cluster/{machine_type}_{process_type}_{target_model}"
    symbols_dir = Path(f"{base}/K_{K}/middle")
    output_dir  = Path(f"{base}/flan_t5_xl_ultrachat_K{K}_runs/run_middle_sentences")
    ckpt_dir    = output_dir / "checkpoints"

    seed_all(SEED)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    print(f"Device: {device}  bf16: {use_bf16}")

    meta       = json.loads((symbols_dir / "meta.json").read_text())
    base_vocab = int(meta["base_vocab"])
    pad_sym    = int(meta.get("symbols_pad", 0))
    t5_pad     = int(meta.get("t5_pad", 0))

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

    collate_fn = make_collate(tokenizer, pad_sym, t5_pad)

    dl_kwargs = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                     pin_memory=True, worker_init_fn=worker_init_fn, collate_fn=collate_fn)
    train_dl = DataLoader(SymbolToTextWithContextMemmapDataset(symbols_dir, train_idx), shuffle=True,  **dl_kwargs)
    val_dl   = DataLoader(SymbolToTextWithContextMemmapDataset(symbols_dir, val_idx),   shuffle=False, **dl_kwargs)

    optim       = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    global_step = 0

    if RESUME_FROM is not None:
        global_step = try_resume(model, optim, ckpt_dir, RESUME_FROM, device)

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

            global_step += 1
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

        save_checkpoint(f"epoch_{epoch}", global_step, ckpt_dir, model, tokenizer, optim)

    # ── Final save
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving final model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete.")


if __name__ == "__main__":
    main()