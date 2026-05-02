#!/usr/bin/env python3
# train.py
#
# Train Flan-T5-XL on FIRST segments for one (method, q) configuration
# from the cost-effective symbol datasets.
#
# Inputs under SYMBOLS_DIR:
#   - symbols_train.npy
#   - t5_labels.npy
#   - meta.json
#   - splits.npz
#
# Uses:
#   - train_idx for training
#   - val_idx for validation
#
# Run example:
#   python train.py \
#       --machine-type laptop \
#       --method smart \
#       --q 250

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
MODEL_NAME = "google/flan-t5-xl"

MACHINE_TYPE = "laptop"
PROCESS_TYPE = "singleprocess"
METHOD = "uniform"
Q = 250
K = 64

SYMBOLS_DIR = Path(
    f"/data/llamacpp/"
    f"ultrachat_cluster_cost_effective/{MACHINE_TYPE}_{PROCESS_TYPE}/K_{K}/{METHOD}/q_{Q}/first"
)

RUN_DIR = Path(
    f"/data/llamacpp/"
    f"ultrachat_cluster_cost_effective/{MACHINE_TYPE}_{PROCESS_TYPE}/"
    f"flan_t5_xl_ultrachat_K{K}_runs/{METHOD}/q_{Q}/run_first_sentences"
)
CKPT_DIR = Path(f"/scratch/cost_effective/{MACHINE_TYPE}_{PROCESS_TYPE}/K_{K}/{METHOD}/q_{Q}/checkpoints")

EPOCHS = 3
BATCH_SIZE = 32
LR = 2e-4
WEIGHT_DECAY = 1e-2
GRAD_CLIP = 1.0
NUM_WORKERS = 4

SAVE_EVERY_EPOCH = True
RESUME_FROM = None   # "epoch_3", "final", or None

DECODE_SAMPLES = 2
GEN_MAX_NEW_TOKENS = 64
GEN_NUM_BEAMS = 1
GEN_DO_SAMPLE = False
GEN_EARLY_STOPPING = False

EMPTY_SYMBOL_FALLBACK_ID = 1


# ─────────────────────────────────────────────────────────────────────────────
# Repro
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

        self.n_all = int(meta["n_samples"])
        self.s_max = int(meta["s_max"])
        self.t_max = int(meta["t_max"])

        self.base_vocab = int(meta["base_vocab"])
        self.K = int(meta["K"])
        self.pad_sym = int(meta.get("symbols_pad", 0))
        self.t5_pad = int(meta.get("t5_pad", 0))

        self.prompt_text = str(
            meta.get(
                "prompt_text",
                "Translate the following trace symbols into the beginning of the original sentence.\nTrace Symbols:"
            )
        )
        self.output_prefix = str(meta.get("output_prefix", "\nOutput:"))
        self.add_sentinel = bool(meta.get("add_sentinel_token", True))
        self.sentinel_text = str(meta.get("sentinel_text", " <extra_id_0>"))

        sym_path = symbols_dir / "symbols_train.npy"
        t5_path = symbols_dir / "t5_labels.npy"
        if not sym_path.exists() or not t5_path.exists():
            raise FileNotFoundError(f"Missing memmaps under {symbols_dir}: symbols_train.npy / t5_labels.npy")

        self.symbols = np.lib.format.open_memmap(sym_path, mode="r")
        self.t5_labels = np.lib.format.open_memmap(t5_path, mode="r")

        if self.symbols.shape != (self.n_all, self.s_max):
            raise RuntimeError(f"symbols shape {self.symbols.shape} != {(self.n_all, self.s_max)}")
        if self.t5_labels.shape != (self.n_all, self.t_max):
            raise RuntimeError(f"t5_labels shape {self.t5_labels.shape} != {(self.n_all, self.t_max)}")

        row_idx = np.asarray(row_idx, dtype=np.int64)
        if row_idx.ndim != 1:
            raise ValueError("row_idx must be 1D")
        if row_idx.size == 0:
            raise ValueError("row_idx is empty")
        if row_idx.min() < 0 or row_idx.max() >= self.n_all:
            raise ValueError("row_idx out of range")

        self.row_idx = row_idx

    def __len__(self):
        return int(self.row_idx.shape[0])

    def __getitem__(self, i):
        ridx = int(self.row_idx[i])
        src = np.array(self.symbols[ridx], copy=False).astype(np.int64, copy=False)
        tgt = np.array(self.t5_labels[ridx], copy=False).astype(np.int64, copy=False)
        return torch.from_numpy(src), torch.from_numpy(tgt)


# ─────────────────────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────────────────────
def make_collate(tokenizer: AutoTokenizer, prompt_text: str, output_prefix: str,
                 add_sentinel: bool, sentinel_text: str,
                 pad_sym: int, t5_pad: int):

    full_prompt = prompt_text + (sentinel_text if add_sentinel else "")
    prompt_ids: List[int] = tokenizer(full_prompt, add_special_tokens=False).input_ids
    outpref_ids: List[int] = tokenizer(output_prefix, add_special_tokens=False).input_ids

    prompt_ids_t = torch.tensor(prompt_ids, dtype=torch.long)
    outpref_ids_t = torch.tensor(outpref_ids, dtype=torch.long)

    pad_id = int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else 0

    def _collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
        sym_list, tgt_list = zip(*batch)

        inputs: List[torch.Tensor] = []
        for sym in sym_list:
            sym_np = sym.numpy()
            sym_np = sym_np[sym_np != pad_sym]

            if sym_np.size == 0:
                sym_clean = torch.tensor([EMPTY_SYMBOL_FALLBACK_ID], dtype=torch.long)
            else:
                sym_clean = torch.from_numpy(sym_np.astype(np.int64, copy=False))

            inp = torch.cat([prompt_ids_t, sym_clean, outpref_ids_t], dim=0)
            inputs.append(inp)

        max_in = max(x.numel() for x in inputs)
        input_ids = torch.full((len(inputs), max_in), fill_value=pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(inputs), max_in), dtype=torch.long)
        for i, inp in enumerate(inputs):
            L = inp.numel()
            input_ids[i, :L] = inp
            attention_mask[i, :L] = 1

        tgt = torch.stack(tgt_list, dim=0)
        labels = tgt.clone()
        labels[labels == t5_pad] = -100

        return input_ids, attention_mask, labels, tgt

    return _collate


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────────────────
def save_checkpoint(tag: str, step: int, epoch: int, ckpt_dir: Path, model, tokenizer, optim):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    hf_dir = ckpt_dir / tag
    hf_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(hf_dir)
    tokenizer.save_pretrained(hf_dir)

    torch.save(
        {
            "optim": optim.state_dict(),
            "step": step,
            "epoch": epoch,
        },
        hf_dir / "trainer_state.pt"
    )
    print(f"[HF] Checkpoint saved: {hf_dir}")


def try_resume(model, optim, ckpt_dir: Path, tag: str, device: torch.device):
    hf_dir = ckpt_dir / tag
    if not hf_dir.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {hf_dir}")

    print(f"Resuming from: {hf_dir}")
    model2 = AutoModelForSeq2SeqLM.from_pretrained(hf_dir)
    model.load_state_dict(model2.state_dict())

    state_path = hf_dir / "trainer_state.pt"
    step = 0
    start_epoch = 1
    if state_path.exists():
        cp = torch.load(state_path, map_location="cpu")
        optim.load_state_dict(cp["optim"])
        step = int(cp.get("step", 0))
        start_epoch = int(cp.get("epoch", 0)) + 1

    model.to(device)
    return step, start_epoch


def parse_args():
    parser = argparse.ArgumentParser(description="Train Flan-T5-XL on cost-effective first-segment symbols")
    parser.add_argument("--machine-type", required=True, choices=["laptop", "desktop"])
    parser.add_argument("--process-type", default="singleprocess", choices=["singleprocess", "multiprocess"])
    parser.add_argument("--method", required=True, choices=["uniform", "threshold", "oracle"])
    parser.add_argument("--q", required=True, type=int)
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global MACHINE_TYPE, PROCESS_TYPE, METHOD, Q, SYMBOLS_DIR, RUN_DIR, CKPT_DIR

    args = parse_args()
    MACHINE_TYPE = args.machine_type
    PROCESS_TYPE = args.process_type
    METHOD = args.method
    Q = args.q

    SYMBOLS_DIR = Path(
        f"/data/llamacpp/"
        f"ultrachat_cluster_cost_effective/{MACHINE_TYPE}_{PROCESS_TYPE}/K_{K}/{METHOD}/q_{Q}/first"
    )
    RUN_DIR = Path(
        f"/data/llamacpp/"
        f"ultrachat_cluster_cost_effective/{MACHINE_TYPE}_{PROCESS_TYPE}/"
        f"flan_t5_xl_ultrachat_K{K}_runs/{METHOD}/q_{Q}/run_first_sentences"
    )
    CKPT_DIR = Path(f"/scratch/cost_effective/{MACHINE_TYPE}_{PROCESS_TYPE}/K_{K}/{METHOD}/q_{Q}/checkpoints")

    if not SYMBOLS_DIR.exists():
        raise FileNotFoundError(f"SYMBOLS_DIR not found: {SYMBOLS_DIR}")

    seed_all(SEED)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    # CKPT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = (device.type == "cuda")
    use_bf16 = bool(use_cuda and torch.cuda.is_bf16_supported())
    print("Device:", device, "bf16:", use_bf16)
    print("SYMBOLS_DIR:", SYMBOLS_DIR)
    print("RUN_DIR:", RUN_DIR)

    meta = json.loads((SYMBOLS_DIR / "meta.json").read_text())
    base_vocab = int(meta["base_vocab"])
    k_in_meta = int(meta["K"])
    pad_sym = int(meta.get("symbols_pad", 0))
    t5_pad = int(meta.get("t5_pad", 0))

    prompt_text = str(meta.get("prompt_text"))
    output_prefix = str(meta.get("output_prefix"))
    add_sentinel = bool(meta.get("add_sentinel_token", True))
    sentinel_text = str(meta.get("sentinel_text", " <extra_id_0>"))

    spl = np.load(SYMBOLS_DIR / "splits.npz")
    train_idx = spl["train_idx"]
    val_idx = spl["val_idx"]

    print("Split sizes:", int(train_idx.shape[0]), int(val_idx.shape[0]))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if int(tokenizer.vocab_size) != base_vocab:
        raise RuntimeError(f"Tokenizer vocab_size {tokenizer.vocab_size} != base_vocab {base_vocab}")

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(base_vocab + k_in_meta)
    model.to(device)

    train_ds = SymbolToTextMemmapDataset(SYMBOLS_DIR, train_idx)
    val_ds = SymbolToTextMemmapDataset(SYMBOLS_DIR, val_idx)

    collate_fn = make_collate(
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        output_prefix=output_prefix,
        add_sentinel=add_sentinel,
        sentinel_text=sentinel_text,
        pad_sym=pad_sym,
        t5_pad=t5_pad,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn,
    )

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    global_step = 0
    start_epoch = 1

    if RESUME_FROM is not None:
        global_step, start_epoch = try_resume(model, optim, CKPT_DIR, RESUME_FROM, device)

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        running = 0.0
        optim.zero_grad(set_to_none=True)

        pbar = tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS}", dynamic_ncols=True, mininterval=0.5)
        for i, (input_ids, attn, labels, _tgt_raw) in enumerate(pbar, 1):
            input_ids = input_ids.to(device, non_blocking=True)
            attn = attn.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_bf16:
                ctx = autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
            else:
                ctx = autocast(device_type="cuda", enabled=False)

            with ctx:
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                loss = out.loss

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered during training.")

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optim.step()
            optim.zero_grad(set_to_none=True)
            global_step += 1

            running += float(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{running / i:.4f}")

        model.eval()
        vlosses = []
        decoded = 0

        with torch.no_grad():
            vpbar = tqdm(val_dl, desc="Validating", dynamic_ncols=True, leave=False, mininterval=0.5)
            for input_ids, attn, labels, tgt_raw in vpbar:
                input_ids = input_ids.to(device, non_blocking=True)
                attn = attn.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                if torch.isfinite(out.loss):
                    vlosses.append(float(out.loss.item()))
                    vpbar.set_postfix(loss=f"{out.loss.item():.4f}")

                if decoded < DECODE_SAMPLES:
                    gen = model.generate(
                        input_ids=input_ids[:1],
                        attention_mask=attn[:1],
                        max_new_tokens=GEN_MAX_NEW_TOKENS,
                        num_beams=GEN_NUM_BEAMS,
                        do_sample=GEN_DO_SAMPLE,
                        early_stopping=GEN_EARLY_STOPPING,
                    )
                    pred = tokenizer.decode(gen[0].tolist(), skip_special_tokens=True).strip()

                    ref_ids = tgt_raw[0].tolist()
                    ref_ids = [t for t in ref_ids if t != t5_pad]
                    ref = tokenizer.decode(ref_ids, skip_special_tokens=True).strip()

                    print("\n[DECODE]")
                    print("PRED:", pred[:500])
                    print("REF :", ref[:500])
                    decoded += 1

        val_loss_mean = float(np.mean(vlosses)) if vlosses else None
        if val_loss_mean is not None:
            print(f"Val loss={val_loss_mean:.4f}")

        epoch_summary = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss_mean": float(running / max(len(train_dl), 1)),
            "val_loss_mean": val_loss_mean,
            "method": METHOD,
            "q": int(Q),
            "machine_type": MACHINE_TYPE,
            "process_type": PROCESS_TYPE,
        }
        (RUN_DIR / f"epoch_{epoch}_summary.json").write_text(json.dumps(epoch_summary, indent=2))

        # if SAVE_EVERY_EPOCH:
            # save_checkpoint(f"epoch_{epoch}", global_step, epoch, CKPT_DIR, model, tokenizer, optim)

    FINAL_DIR = Path(
        f"/data/llamacpp/"
        f"ultrachat_cluster_cost_effective/laptop_singleprocess/"
        f"flan_t5_xl_ultrachat_K{K}_runs/{METHOD}/q_{Q}/run_first_sentences/checkpoints"
    )
    save_checkpoint("final", global_step, EPOCHS, FINAL_DIR, model, tokenizer, optim)

    final_summary = {
        "status": "done",
        "method": METHOD,
        "q": int(Q),
        "machine_type": MACHINE_TYPE,
        "process_type": PROCESS_TYPE,
        "symbols_dir": str(SYMBOLS_DIR),
        "run_dir": str(RUN_DIR),
        "final_model_dir": str(FINAL_DIR),   # ← add this line
        "epochs": int(EPOCHS),
        "batch_size": int(BATCH_SIZE),
        "lr": float(LR),
        "weight_decay": float(WEIGHT_DECAY),
        "seed": int(SEED),
    }
    (RUN_DIR / "train_done.json").write_text(json.dumps(final_summary, indent=2))
    tokenizer.save_pretrained(RUN_DIR / "tokenizer")

    print("Training complete.")


if __name__ == "__main__":
    main()