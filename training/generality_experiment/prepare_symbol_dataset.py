#!/usr/bin/env python3
# prepare_flan_symbols_ultrachat.py
#
# UltraChat: traces -> symbol memmaps + T5 label memmaps
#
# For kind="first":  one output row per parquet row, segment_meta.jsonl written.
# For kind="middle": each parquet row expanded into ceil(len(phi_ids)/S_MAX) windows,
#                    context_text.jsonl + segment_meta.jsonl written.
#
# group_id format: "{split}|row_idx|{source_row_idx}"
# This matches the ChatDoctor format so eval / print_prefix_table scripts can join.
#
# Usage:
#   python prepare_flan_symbols_ultrachat.py \
#       --machine-type laptop \
#       --framework llamacpp \
#       --model phi

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import json
import random
import argparse
import math
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from llama_cpp import Llama


# ─────────────────────────────────────────────────────────────────────────────
# Config (non-path constants — paths are derived in main() from args)
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42

K                  = 64
HELDOUT_PER_TOKEN  = 20
PER_TRACE_CENTER   = True

EPOCHS           = 10
STEPS_PER_EPOCH  = 500
P                = 512
M                = 4
EMB_DIM          = 128
HIDDEN           = 512
LR               = 2e-3
WEIGHT_DECAY     = 1e-4
GRAD_CLIP        = 1.0
TEMPERATURE      = 0.07
NUM_WORKERS      = 4

S_MAX                  = 32
TARGET_PHI_PREFIX_LEN  = 32
T_MAX                  = 128
CONTEXT_PHI_LEN        = 32

REBUILD_CLUSTER_MEMMAPS  = True
REBUILD_ENCODER          = True
REBUILD_CLUSTERING       = True
REBUILD_SYMBOLS          = True
REBUILD_PARQUET_INSPECT  = False

MODEL_PATHS = {
    "phi":   Path("/data/models/phi3-mini/Phi-3-mini-4k-instruct-q4.gguf"),
    "llama": Path("/data/models/llama-3.1/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
}

FLAN_MODEL_NAME    = "google/flan-t5-xl"
PROMPT_TEXT        = "Translate the following trace symbols into the beginning of the original sentence.\nTrace Symbols:"
OUTPUT_PREFIX      = "\nOutput:"
ADD_SENTINEL_TOKEN = True
SENTINEL_TEXT      = " <extra_id_0>"

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
# CSV -> memmaps
# ─────────────────────────────────────────────────────────────────────────────
def build_cluster_memmaps(csv_path: Path, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)

    meta_path  = outdir / "meta.json"
    x_path     = outdir / "X.memmap"
    t_path     = outdir / "token_id.memmap"
    stats_path = outdir / "norm_stats.json"

    if (not REBUILD_CLUSTER_MEMMAPS
            and meta_path.exists() and x_path.exists()
            and t_path.exists() and stats_path.exists()):
        return json.loads(meta_path.read_text())

    sample    = pd.read_csv(csv_path, nrows=5)
    feat_cols = [c for c in sample.columns if c.startswith("set_")]
    if len(feat_cols) != 64:
        raise RuntimeError(f"Expected 64 cols set_*, got {len(feat_cols)}")

    n_rows = sum(1 for _ in open(csv_path)) - 1
    if n_rows <= 0:
        raise RuntimeError("CSV empty")

    X = np.memmap(x_path, dtype=np.float32, mode="w+", shape=(n_rows, 64))
    T = np.memmap(t_path, dtype=np.int32,   mode="w+", shape=(n_rows,))

    offset = 0
    pbar = tqdm(desc="CSV -> memmap", total=n_rows, unit="rows")
    for chunk in pd.read_csv(csv_path, chunksize=200_000):
        feats = chunk[feat_cols].to_numpy(dtype=np.float32, copy=False)
        tok   = chunk["token_id"].to_numpy(dtype=np.int32, copy=False)
        n     = feats.shape[0]
        X[offset:offset + n] = feats
        T[offset:offset + n] = tok
        offset += n
        pbar.update(n)
    pbar.close()
    X.flush(); T.flush()

    mean  = np.zeros((64,), dtype=np.float64)
    m2    = np.zeros((64,), dtype=np.float64)
    count = 0

    for start in tqdm(range(0, n_rows, 200_000), desc="Mean/std"):
        end   = min(n_rows, start + 200_000)
        x     = np.asarray(X[start:end], dtype=np.float32)
        if PER_TRACE_CENTER:
            x = x - x.mean(axis=1, keepdims=True)
        x64         = x.astype(np.float64)
        batch_count = x64.shape[0]
        batch_mean  = x64.mean(axis=0)
        batch_var   = x64.var(axis=0)
        if count == 0:
            mean, m2, count = batch_mean, batch_var * batch_count, batch_count
        else:
            delta     = batch_mean - mean
            new_count = count + batch_count
            mean      = mean + delta * (batch_count / new_count)
            m2        = m2 + batch_var * batch_count + delta**2 * (count * batch_count / new_count)
            count     = new_count

    std  = np.sqrt(m2 / max(count, 1) + 1e-8).astype(np.float32)
    mean = mean.astype(np.float32)

    for start in tqdm(range(0, n_rows, 200_000), desc="Normalize"):
        end = min(n_rows, start + 200_000)
        x   = np.asarray(X[start:end], dtype=np.float32)
        if PER_TRACE_CENTER:
            x = x - x.mean(axis=1, keepdims=True)
        X[start:end] = (x - mean[None, :]) / std[None, :]
    X.flush()

    stats_path.write_text(json.dumps(
        {"mean": mean.tolist(), "std": std.tolist(), "per_trace_center": bool(PER_TRACE_CENTER)}, indent=2
    ))
    meta = {"n_rows": int(n_rows), "x_path": str(x_path), "t_path": str(t_path), "feature_cols": feat_cols}
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Index + heldout split
# ─────────────────────────────────────────────────────────────────────────────
class TokenTraceIndex:
    def __init__(self, token_ids: np.memmap, allowed_mask: Optional[np.ndarray] = None):
        toks = np.asarray(token_ids[:], dtype=np.int32)
        if allowed_mask is not None:
            idx_all = np.flatnonzero(allowed_mask)
            toks    = toks[allowed_mask]
        else:
            idx_all = np.arange(toks.shape[0], dtype=np.int64)

        order       = np.argsort(toks, kind="mergesort")
        toks_sorted = toks[order]
        idx_sorted  = idx_all[order]

        bounds = np.flatnonzero(np.diff(toks_sorted)) + 1
        splits = np.split(idx_sorted, bounds)
        keys   = np.split(toks_sorted, bounds)

        self.tokens = np.array([int(k[0]) for k in keys], dtype=np.int32)
        self.map    = {int(k[0]): s.astype(np.int64, copy=False) for k, s in zip(keys, splits)}

    def sample1(self, token_id: int, rng: np.random.RandomState) -> int:
        idxs = self.map.get(int(token_id))
        if idxs is None or idxs.size == 0:
            return -1
        return int(idxs[0]) if idxs.size == 1 else int(rng.choice(idxs, size=1, replace=False)[0])

    def sample(self, token_id: int, m: int, rng: np.random.RandomState) -> np.ndarray:
        idxs = self.map[token_id]
        return rng.choice(idxs, size=m, replace=(idxs.size < m))


def build_train_mask(
    token_ids: np.memmap, heldout_per_token: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    toks       = np.asarray(token_ids[:], dtype=np.int32)
    train_mask = np.ones(toks.shape[0],  dtype=bool)
    held_mask  = np.zeros(toks.shape[0], dtype=bool)
    if heldout_per_token <= 0:
        return train_mask, held_mask

    rng = np.random.RandomState(seed)
    for t in tqdm(np.unique(toks), desc="heldout_per_token"):
        idxs = np.flatnonzero(toks == t)
        if idxs.size == 0:
            continue
        held = rng.choice(idxs, size=min(heldout_per_token, idxs.size), replace=False)
        train_mask[held] = False
        held_mask[held]  = True
    return train_mask, held_mask


# ─────────────────────────────────────────────────────────────────────────────
# Encoder + SupCon
# ─────────────────────────────────────────────────────────────────────────────
class MLPEncoder(nn.Module):
    def __init__(self, in_dim=64, emb_dim=128, hidden=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def supcon_loss(z, y, temperature=0.07):
    B           = z.shape[0]
    sim         = (z @ z.T) / temperature - (z @ z.T).max(dim=1, keepdim=True).values.detach() / temperature
    mask        = ~torch.eye(B, device=z.device, dtype=torch.bool)
    y           = y.view(-1, 1)
    pos_mask    = (y == y.T) & mask
    exp_sim     = torch.exp(sim) * mask
    pos_sum     = (exp_sim * pos_mask).sum(dim=1)
    pos_cnt     = pos_mask.sum(dim=1).clamp_min(1)
    log_prob    = torch.log(pos_sum / pos_cnt / (exp_sim.sum(dim=1) + 1e-12) + 1e-12)
    return -log_prob.mean()


def make_collate(X_memmap: np.memmap, index: TokenTraceIndex, seed: int):
    rng = np.random.RandomState(seed)

    def _collate(_batch):
        tok_ids = rng.choice(index.tokens, size=P, replace=False).astype(np.int32)
        xs, ys  = [], []
        for t in tok_ids:
            chosen = index.sample(int(t), M, rng)
            xs.append(np.asarray(X_memmap[chosen], dtype=np.float32))
            ys.append(np.full((M,), int(t), dtype=np.int32))
        return torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(ys))

    return _collate


def infinite_loader(dl):
    while True:
        for batch in dl:
            yield batch


@torch.no_grad()
def compute_token_centroids_train_only(
    encoder: nn.Module, X: np.memmap, Tm: np.memmap, train_mask: np.ndarray, device: str
) -> Tuple[np.ndarray, np.ndarray]:
    encoder.eval()
    toks_all   = np.asarray(Tm[:], dtype=np.int32)
    token_ids  = np.unique(toks_all[train_mask]).astype(np.int32)
    tid_to_row = {int(t): i for i, t in enumerate(token_ids)}
    sum_z      = np.zeros((token_ids.shape[0], EMB_DIM), dtype=np.float64)
    cnt        = np.zeros((token_ids.shape[0],), dtype=np.int64)

    for start in tqdm(range(0, toks_all.shape[0], 200_000), desc="Centroids"):
        end  = min(toks_all.shape[0], start + 200_000)
        mask = train_mask[start:end]
        if not mask.any():
            continue
        z = encoder(torch.from_numpy(np.asarray(X[start:end][mask], dtype=np.float32)).to(device))
        z = z.detach().cpu().numpy().astype(np.float64)
        for i, t in enumerate(toks_all[start:end][mask]):
            r = tid_to_row[int(t)]
            sum_z[r] += z[i]
            cnt[r]   += 1

    centroids  = (sum_z / np.maximum(cnt[:, None], 1)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
    return token_ids, centroids


@torch.no_grad()
def torch_kmeans_cosine(X_np: np.ndarray, K: int, seed: int, niter: int = 50, device: str = "cuda") -> np.ndarray:
    device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    g      = torch.Generator(device=device); g.manual_seed(seed)
    X      = F.normalize(torch.from_numpy(X_np).to(device), dim=1)
    N, _   = X.shape
    C      = X[torch.randperm(N, generator=g, device=device)[:K]].clone()

    for _ in tqdm(range(niter), desc="k-means"):
        lab    = (X @ C.T).argmax(dim=1)
        C_new  = torch.zeros_like(C)
        counts = torch.zeros((K,), device=device, dtype=torch.int64)
        C_new.index_add_(0, lab, X)
        counts.index_add_(0, lab, torch.ones(N, device=device, dtype=torch.int64))
        empty = counts == 0
        if empty.any():
            C_new[empty]  = X[torch.randperm(N, generator=g, device=device)[:int(empty.sum())]]
            counts[empty] = 1
        C = F.normalize(C_new / counts.unsqueeze(1), dim=1)

    return lab.cpu().numpy().astype(np.int32)


def compute_cluster_centers_from_token_centroids(
    token_centroids: np.ndarray, token_cluster_labels: np.ndarray, K: int
) -> np.ndarray:
    D       = token_centroids.shape[1]
    centers = np.zeros((K, D), dtype=np.float64)
    counts  = np.zeros((K,),   dtype=np.int64)
    for i in range(token_centroids.shape[0]):
        k           = int(token_cluster_labels[i])
        centers[k] += token_centroids[i].astype(np.float64)
        counts[k]  += 1
    for k in range(K):
        if counts[k]:
            centers[k] /= float(counts[k])
    centers  = centers.astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12
    return centers


# ─────────────────────────────────────────────────────────────────────────────
# Symbolize traces
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def traces_to_symbol_ids(
    encoder: nn.Module, X_memmap: np.memmap, trace_row_ids: np.ndarray,
    cluster_centers: torch.Tensor, base_vocab: int, sym_pad: int, device: str,
) -> np.ndarray:
    out        = np.full(trace_row_ids.shape[0], sym_pad, dtype=np.int32)
    valid_mask = trace_row_ids >= 0
    if not valid_mask.any():
        return out
    idx  = trace_row_ids[valid_mask].astype(np.int64)
    z    = encoder(torch.from_numpy(np.asarray(X_memmap[idx], dtype=np.float32)).to(device)).float()
    lab  = (z @ cluster_centers.T).argmax(dim=1).cpu().numpy().astype(np.int32)
    out[valid_mask] = (base_vocab + lab).astype(np.int32)
    return out


def phi_ids_to_text(phi: Llama, phi_ids: List[int]) -> str:
    b = phi.detokenize([int(x) for x in phi_ids])
    return b.decode("utf-8", errors="ignore") if isinstance(b, (bytes, bytearray)) else str(b)


def normalize_uc_split(sp: str) -> str:
    sp = str(sp).strip().lower()
    if sp in ("train", "train_sft"):
        return "train_sft"
    if sp in ("test", "test_sft"):
        return "test_sft"
    return "train_sft"


# ─────────────────────────────────────────────────────────────────────────────
# Count expanded rows for middle (pass 1)
# ─────────────────────────────────────────────────────────────────────────────
def count_middle_expanded(phi_parquet: Path) -> int:
    total = 0
    for batch in ds.dataset(phi_parquet, format="parquet").scanner(columns=["phi_ids"], batch_size=4096).to_batches():
        for ids in batch.column("phi_ids").to_pylist():
            total += max(1, math.ceil((len(ids) if ids else 0) / S_MAX))
    return total


# ─────────────────────────────────────────────────────────────────────────────
# build_symbol_dataset
# ─────────────────────────────────────────────────────────────────────────────
def build_symbol_dataset(
    *,
    phi_parquet: Path,
    out_dir: Path,
    kind: str,
    trace_csv: Path,
    X: np.memmap,
    encoder: nn.Module,
    train_index: TokenTraceIndex,
    held_index: TokenTraceIndex,
    cluster_centers: np.ndarray,
    flan_tok: AutoTokenizer,
    base_vocab: int,
    phi: Llama,
) -> None:
    assert kind in ("first", "middle"), f"Unknown kind: {kind}"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path      = out_dir / "meta.json"
    splits_path    = out_dir / "splits.npz"
    sym_train_path = out_dir / "symbols_train.npy"
    sym_held_path  = out_dir / "symbols_heldout.npy"
    sym_path       = out_dir / "symbols.npy"
    t5_path        = out_dir / "t5_labels.npy"

    if (not REBUILD_SYMBOLS
            and meta_path.exists() and splits_path.exists()
            and sym_train_path.exists() and sym_held_path.exists() and t5_path.exists()):
        print(f"[{kind}] Symbols already exist; skipping.")
        return

    ds_par       = ds.dataset(phi_parquet, format="parquet")
    schema_names = set(ds_par.schema.names)

    required = ["phi_ids", "split", "text", "row_idx"] + (["context_text"] if kind == "middle" else [])
    for c in required:
        if c not in schema_names:
            raise RuntimeError(f"[{kind}] Missing column '{c}' in {phi_parquet}")

    # ── Pass 1: count output rows
    if kind == "middle":
        total = count_middle_expanded(phi_parquet)
        print(f"[{kind}] Expanded middle: {total} windows")
    else:
        total = sum(b.num_rows for b in ds_par.scanner(columns=["split"], batch_size=4096).to_batches())

    if total <= 0:
        raise RuntimeError(f"[{kind}] No rows found in {phi_parquet}")

    # ── Allocate memmaps
    sym_train_mm = np.lib.format.open_memmap(sym_train_path, mode="w+", dtype="int32", shape=(total, S_MAX))
    sym_held_mm  = np.lib.format.open_memmap(sym_held_path,  mode="w+", dtype="int32", shape=(total, S_MAX))
    sym_mm       = np.lib.format.open_memmap(sym_path,       mode="w+", dtype="int32", shape=(total, S_MAX))
    t5_mm        = np.lib.format.open_memmap(t5_path,        mode="w+", dtype="int32", shape=(total, T_MAX))

    # ── Open side-files
    segmeta_path = out_dir / "segment_meta.jsonl"
    if segmeta_path.exists():
        segmeta_path.unlink()
    segmeta_f = segmeta_path.open("w", encoding="utf-8")

    ctx_path, ctx_f = None, None
    if kind == "middle":
        ctx_path = out_dir / "context_text.jsonl"
        if ctx_path.exists():
            ctx_path.unlink()
        ctx_f = ctx_path.open("w", encoding="utf-8")

    writer = None
    if REBUILD_PARQUET_INSPECT:
        inspect_path = out_dir / "symbol_dataset.parquet"
        if inspect_path.exists():
            inspect_path.unlink()
        writer = pq.ParquetWriter(str(inspect_path), compression="zstd", schema=pa.schema([
            ("split",           pa.string()),
            ("symbols_train",   pa.list_(pa.int32())),
            ("symbols_heldout", pa.list_(pa.int32())),
            ("t5_input_ids",    pa.list_(pa.int32())),
        ]))

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    encoder.eval()
    centers_t = torch.from_numpy(cluster_centers).to(device)
    flan_pad  = int(flan_tok.pad_token_id) if flan_tok.pad_token_id is not None else 0
    sym_pad   = 0

    train_idx: List[int] = []
    test_idx:  List[int] = []
    rng = np.random.RandomState(SEED)

    scanner_cols = ["phi_ids", "split", "text", "row_idx"]
    if kind == "middle":
        scanner_cols = ["context_text"] + scanner_cols

    ptr = 0

    for batch in tqdm(
        ds_par.scanner(columns=scanner_cols, batch_size=256).to_batches(),
        desc=f"[{kind}] Symbolizing", unit="batch",
    ):
        phi_ids_list = batch.column("phi_ids").to_pylist()
        split_list   = [normalize_uc_split(s) for s in batch.column("split").to_pylist()]
        text_list    = batch.column("text").to_pylist()
        row_idx_list = batch.column("row_idx").to_pylist()

        def _sample_traces(ids_trunc):
            """Given a list of token ids (already clipped to S_MAX), return train/held trace id arrays."""
            tok_mat = np.full((1, S_MAX), -1, dtype=np.int32)
            tok_mat[0, :len(ids_trunc)] = np.asarray(ids_trunc, dtype=np.int32)
            flat = tok_mat.reshape(-1)
            tr   = np.full_like(flat, -1, dtype=np.int64)
            he   = np.full_like(flat, -1, dtype=np.int64)
            for j in range(flat.shape[0]):
                tid = int(flat[j])
                if tid < 0:
                    continue
                tr[j] = train_index.sample1(tid, rng)
                he[j] = held_index.sample1(tid, rng)
            return tr, he

        def _encode_t5(text: str) -> List[int]:
            enc     = flan_tok(text, max_length=T_MAX, truncation=True,
                               padding="max_length", return_attention_mask=False)
            out_ids = [int(v) for v in enc["input_ids"]]
            if len(out_ids) < T_MAX:
                out_ids = out_ids + [flan_pad] * (T_MAX - len(out_ids))
            return out_ids[:T_MAX]

        # ── FIRST ────────────────────────────────────────────────────────────
        if kind == "first":
            B = len(phi_ids_list)

            tok_mat = np.full((B, S_MAX), -1, dtype=np.int32)
            for i, ids in enumerate(phi_ids_list):
                ids_trunc = [int(v) for v in ids[:S_MAX]]
                if ids_trunc:
                    tok_mat[i, :len(ids_trunc)] = np.asarray(ids_trunc, dtype=np.int32)

            flat            = tok_mat.reshape(-1)
            train_trace_ids = np.full_like(flat, -1, dtype=np.int64)
            held_trace_ids  = np.full_like(flat, -1, dtype=np.int64)
            for j in range(flat.shape[0]):
                tid = int(flat[j])
                if tid < 0:
                    continue
                train_trace_ids[j] = train_index.sample1(tid, rng)
                held_trace_ids[j]  = held_index.sample1(tid, rng)

            sym_train = traces_to_symbol_ids(
                encoder=encoder, X_memmap=X, trace_row_ids=train_trace_ids,
                cluster_centers=centers_t, base_vocab=base_vocab, sym_pad=sym_pad, device=device,
            ).reshape(B, S_MAX)
            sym_held = traces_to_symbol_ids(
                encoder=encoder, X_memmap=X, trace_row_ids=held_trace_ids,
                cluster_centers=centers_t, base_vocab=base_vocab, sym_pad=sym_pad, device=device,
            ).reshape(B, S_MAX)

            t5_batch = [_encode_t5(phi_ids_to_text(phi, [int(x) for x in ids[:TARGET_PHI_PREFIX_LEN]]))
                        for ids in phi_ids_list]

            sym_train_mm[ptr:ptr + B, :] = sym_train.astype(np.int32)
            sym_held_mm [ptr:ptr + B, :] = sym_held.astype(np.int32)
            sym_mm      [ptr:ptr + B, :] = sym_train.astype(np.int32)
            t5_mm       [ptr:ptr + B, :] = np.asarray(t5_batch, dtype=np.int32)

            for i, sp in enumerate(split_list):
                ridx           = ptr + i
                source_row_idx = int(row_idx_list[i])
                group_id       = f"{sp}|row_idx|{source_row_idx}"

                (test_idx if sp == "test_sft" else train_idx).append(ridx)

                segmeta_f.write(json.dumps({
                    "row_idx": ridx, "group_id": group_id, "split": sp,
                    "source_row_idx": source_row_idx, "segment_idx": 0, "n_windows": 1,
                    "context_text": "", "ref_segment_text": str(text_list[i] or ""),
                }, ensure_ascii=False) + "\n")

            ptr += B

        # ── MIDDLE ───────────────────────────────────────────────────────────
        else:
            context_list = batch.column("context_text").to_pylist()

            for i in range(len(phi_ids_list)):
                all_ids        = [int(v) for v in phi_ids_list[i]]
                sp             = split_list[i]
                orig_ctx       = str(context_list[i] or "")
                source_row_idx = int(row_idx_list[i])
                group_id       = f"{sp}|row_idx|{source_row_idx}"
                n_ids          = len(all_ids)
                n_windows      = max(1, math.ceil(n_ids / S_MAX))

                for w in range(n_windows):
                    window_ids = all_ids[w * S_MAX: min((w + 1) * S_MAX, n_ids)]
                    if not window_ids:
                        break

                    ctx_text = orig_ctx if w == 0 else phi_ids_to_text(phi, all_ids[(w - 1) * S_MAX: w * S_MAX])
                    tgt_text = phi_ids_to_text(phi, window_ids)

                    tr, he = _sample_traces(window_ids)

                    sym_train_w = traces_to_symbol_ids(
                        encoder=encoder, X_memmap=X, trace_row_ids=tr,
                        cluster_centers=centers_t, base_vocab=base_vocab, sym_pad=sym_pad, device=device,
                    ).reshape(1, S_MAX)
                    sym_held_w = traces_to_symbol_ids(
                        encoder=encoder, X_memmap=X, trace_row_ids=he,
                        cluster_centers=centers_t, base_vocab=base_vocab, sym_pad=sym_pad, device=device,
                    ).reshape(1, S_MAX)

                    sym_train_mm[ptr, :] = sym_train_w[0]
                    sym_held_mm [ptr, :] = sym_held_w[0]
                    sym_mm      [ptr, :] = sym_train_w[0]
                    t5_mm       [ptr, :] = np.asarray(_encode_t5(tgt_text), dtype=np.int32)

                    (test_idx if sp == "test_sft" else train_idx).append(ptr)

                    ctx_s = ctx_text.replace("\r\n", "\n").replace("\r", "\n")
                    ctx_f.write(json.dumps({"context_text": ctx_s}, ensure_ascii=False) + "\n")
                    segmeta_f.write(json.dumps({
                        "row_idx": ptr, "group_id": group_id, "split": sp,
                        "source_row_idx": source_row_idx, "segment_idx": w, "n_windows": n_windows,
                        "context_text": ctx_s, "ref_segment_text": tgt_text,
                    }, ensure_ascii=False) + "\n")

                    ptr += 1

    # ── Flush and close
    for mm in (sym_train_mm, sym_held_mm, sym_mm, t5_mm):
        mm.flush()
    segmeta_f.close()
    if ctx_f is not None:
        ctx_f.close()
    if writer is not None:
        writer.close()

    # ── Train / val / test splits
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx  = np.asarray(test_idx,  dtype=np.int64)

    n_val = int(test_idx.shape[0])
    if n_val <= 0:
        raise RuntimeError(f"[{kind}] test_idx is empty")
    if train_idx.shape[0] <= n_val:
        raise RuntimeError(f"[{kind}] Not enough train rows for val. train={train_idx.shape[0]}, test={n_val}")

    perm      = rng.permutation(train_idx.shape[0])
    val_idx   = train_idx[perm[:n_val]]
    train_idx = train_idx[perm[n_val:]]
    for arr in (train_idx, val_idx, test_idx):
        arr.sort()

    np.savez_compressed(splits_path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    # ── meta.json
    meta = {
        "kind": kind, "n_samples": int(total),
        "s_max": int(S_MAX), "t_max": int(T_MAX),
        "K": int(K), "base_vocab": int(base_vocab),
        "symbols_pad": int(sym_pad), "t5_pad": int(flan_pad),
        "trace_csv": str(trace_csv),
        "ultrachat_parquet": str(phi_parquet),
        "heldout_per_token": int(HELDOUT_PER_TOKEN),
        "target_is_phi_prefix": True,
        "target_phi_prefix_len": int(TARGET_PHI_PREFIX_LEN),
        "context_phi_len": int(CONTEXT_PHI_LEN) if kind == "middle" else 0,
        "prompt_text": PROMPT_TEXT, "output_prefix": OUTPUT_PREFIX,
        "add_sentinel_token": ADD_SENTINEL_TOKEN, "sentinel_text": SENTINEL_TEXT,
        "outputs": {
            "symbols_train_memmap":   str(sym_train_path),
            "symbols_heldout_memmap": str(sym_held_path),
            "symbols_memmap":         str(sym_path),
            "t5_labels_memmap":       str(t5_path),
            "splits_npz":             str(splits_path),
            "segment_meta_jsonl":     str(segmeta_path),
        },
        "note": (
            "segment_meta.jsonl written for both first and middle. "
            "group_id format: '{split}|row_idx|{source_row_idx}'."
        ),
    }
    if kind == "middle":
        meta["outputs"]["context_jsonl"] = str(ctx_path)
        meta["segment_grouping"] = {
            "mode": "row_idx_multiwindow",
            "group_key_fields": ["split", "source_row_idx"],
            "segment_order_field": "segment_idx",
            "window_size_tokens": int(S_MAX),
        }

    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[{kind}] Wrote: {meta_path}")
    print(f"[{kind}] Wrote: {splits_path}")
    print(f"[{kind}] Total output rows: {ptr}")


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine-type", required=True, choices=["laptop", "desktop"])
    parser.add_argument("--process-type", default="singleprocess", choices=["singleprocess", "multiprocess"])
    parser.add_argument("--framework",    required=True, choices=["llamacpp", "huggingface"])
    parser.add_argument("--model",        required=True, choices=["phi", "llama"])
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args             = parse_args()
    machine_type     = args.machine_type
    process_type     = args.process_type
    target_framework = args.framework
    target_model     = args.model

    model_path = MODEL_PATHS[target_model]

    trace_csv = Path(
        f"/data/{target_framework}_{target_model}/dataset_top50_per_token_{machine_type}_{process_type}.csv"
    )
    cluster_workdir = Path(
        f"./cluster_workdir/{target_framework}/K_{K}_{machine_type}_{process_type}"
    )
    symbols_dir = Path(
        f"/outpath/{target_framework}/ultrachat_cluster/{machine_type}_{process_type}_{target_model}/K_{K}"
    )
    parquet_first = Path(
        f"/data/{target_framework}_{target_model}/ultrachat_{target_model}_first.parquet"
    )
    parquet_middle = Path(
        f"/data/{target_framework}_{target_model}/ultrachat_{target_model}_middle.parquet"
    )

    seed_all(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  framework: {target_framework}  |  model: {target_model}")

    cluster_workdir.mkdir(parents=True, exist_ok=True)
    symbols_dir.mkdir(parents=True, exist_ok=True)

    meta   = build_cluster_memmaps(trace_csv, cluster_workdir)
    n_rows = int(meta["n_rows"])
    X      = np.memmap(meta["x_path"], dtype=np.float32, mode="r", shape=(n_rows, 64))
    Tm     = np.memmap(meta["t_path"], dtype=np.int32,   mode="r", shape=(n_rows,))

    train_mask, held_mask = build_train_mask(Tm, HELDOUT_PER_TOKEN, SEED)

    encoder_ckpt = cluster_workdir / "encoder.pt"
    encoder      = MLPEncoder(in_dim=64, emb_dim=EMB_DIM, hidden=HIDDEN, dropout=0.1).to(device)

    if encoder_ckpt.exists() and not REBUILD_ENCODER:
        encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device)["encoder"])
        print("Loaded encoder:", encoder_ckpt)
    else:
        train_index_sc = TokenTraceIndex(Tm, allowed_mask=train_mask)
        dl  = DataLoader(list(range(1024)), batch_size=1, shuffle=True,
                         num_workers=NUM_WORKERS, pin_memory=True,
                         worker_init_fn=worker_init_fn,
                         collate_fn=make_collate(X, train_index_sc, SEED))
        opt = torch.optim.AdamW(encoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        encoder.train()
        it  = infinite_loader(dl)

        for epoch in range(EPOCHS):
            pbar = tqdm(range(STEPS_PER_EPOCH), desc=f"SupCon epoch {epoch+1}/{EPOCHS}", dynamic_ncols=True)
            for _ in pbar:
                xb, yb = next(it)
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                loss   = supcon_loss(encoder(xb), yb, temperature=TEMPERATURE)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(encoder.parameters(), GRAD_CLIP)
                opt.step()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        torch.save({"encoder": encoder.state_dict()}, encoder_ckpt)
        print("Saved encoder:", encoder_ckpt)

    token_ids, centroids = compute_token_centroids_train_only(encoder, X, Tm, train_mask, device=device)

    clustering_path = cluster_workdir / "clustering.json"
    if clustering_path.exists() and not REBUILD_CLUSTERING:
        clustering    = json.loads(clustering_path.read_text())
        token_cluster = np.asarray(clustering["token_cluster"], dtype=np.int32)
        print("Loaded clustering:", clustering_path)
    else:
        token_cluster = torch_kmeans_cosine(centroids, K=K, seed=SEED, niter=50, device=device)
        clustering_path.write_text(json.dumps(
            {"token_ids": token_ids.tolist(), "token_cluster": token_cluster.tolist()}, indent=2
        ))
        print("Wrote clustering:", clustering_path)

    cluster_centers = compute_cluster_centers_from_token_centroids(centroids, token_cluster, K=K)
    train_index     = TokenTraceIndex(Tm, allowed_mask=train_mask)
    held_index      = TokenTraceIndex(Tm, allowed_mask=held_mask)

    flan_tok   = AutoTokenizer.from_pretrained(FLAN_MODEL_NAME, use_fast=True)
    base_vocab = int(flan_tok.vocab_size)

    print(f"Loading tokenizer (vocab_only) from: {model_path}")
    phi = Llama(model_path=str(model_path), vocab_only=True, verbose=False)

    shared = dict(
        trace_csv=trace_csv, X=X, encoder=encoder,
        train_index=train_index, held_index=held_index,
        cluster_centers=cluster_centers, flan_tok=flan_tok,
        base_vocab=base_vocab, phi=phi,
    )

    build_symbol_dataset(phi_parquet=parquet_first,  out_dir=symbols_dir / "first",  kind="first",  **shared)
    build_symbol_dataset(phi_parquet=parquet_middle, out_dir=symbols_dir / "middle", kind="middle", **shared)

    print("Done. Outputs at:", symbols_dir)


if __name__ == "__main__":
    main()