# 2_prepare_symbols.py


import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('HF_HUB_CACHE', os.path.join(os.environ['HF_HOME'], 'hub'))
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

import json, random, argparse, copy
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from llama_cpp import Llama

SEED = 42
K = 64
PER_TRACE_CENTER = True
EPOCHS = 10
STEPS_PER_EPOCH = 500
P = 512
M = 4
EMB_DIM = 128
HIDDEN = 512
LR = 2e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
TEMPERATURE = 0.07
NUM_WORKERS = 4
DROPOUT = 0.1
S_MAX = 32
TARGET_PHI_PREFIX_LEN = 32
T_MAX = 128
REBUILD_CLUSTER_MEMMAPS = False
REBUILD_ENCODER = False
REBUILD_CLUSTERING = False

MACHINE_TYPE = 'laptop'
PROCESS_TYPE = 'singleprocess'
FULL_TRACE_CSV = Path(
    f"/data/measurments/llamacpp_phi/"
    f"dataset_top50_per_token_{MACHINE_TYPE}_{PROCESS_TYPE}.csv"
)

METHODS = ['uniform', 'threshold', 'oracle']
Q_LIST = [25, 50, 75, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000]

# METHODS = ['uniform']
# Q_LIST = [500]
PROCESSED_ROOT = Path('/data/processed/phi')
PHI_PARQUET_FIRST = PROCESSED_ROOT / 'ultrachat_phi_first.parquet'
PHI_MODEL_PATH = Path('/data/models/phi3-mini/Phi-3-mini-4k-instruct-q4.gguf')
FLAN_MODEL_NAME = 'google/flan-t5-xl'
PROMPT_TEXT = 'Translate the following trace symbols into the beginning of the original sentence.\nTrace Symbols:'
OUTPUT_PREFIX = '\nOutput:'
ADD_SENTINEL_TOKEN = True
SENTINEL_TEXT = ' <extra_id_0>'

VAL_FRAC = 0.1
MIN_VAL_PER_TOKEN = 1
EVAL_EVERY = 200
EARLY_STOP_PATIENCE = 2
MIN_IMPROVEMENT = 1e-4


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int):
    s = torch.initial_seed() % (2**32)
    np.random.seed(s + worker_id)
    random.seed(s + worker_id)


def build_cluster_memmaps_csv(csv_path: Path, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    meta_path = outdir / 'meta.json'
    x_path = outdir / 'X.memmap'
    t_path = outdir / 'token_id.memmap'
    stats_path = outdir / 'norm_stats.json'
    if (not REBUILD_CLUSTER_MEMMAPS) and meta_path.exists() and x_path.exists() and t_path.exists() and stats_path.exists():
        return json.loads(meta_path.read_text())
    sample = pd.read_csv(csv_path, nrows=5)
    feat_cols = [c for c in sample.columns if c.startswith('set_')]
    n_rows = sum(1 for _ in open(csv_path)) - 1
    X = np.memmap(x_path, dtype=np.float32, mode='w+', shape=(n_rows, 64))
    T = np.memmap(t_path, dtype=np.int32, mode='w+', shape=(n_rows,))
    off = 0
    for chunk in pd.read_csv(csv_path, chunksize=200_000):
        feats = chunk[feat_cols].to_numpy(dtype=np.float32, copy=False)
        toks = chunk['token_id'].to_numpy(dtype=np.int32, copy=False)
        n = len(chunk)
        X[off:off+n] = feats
        T[off:off+n] = toks
        off += n
    mean = np.zeros(64, dtype=np.float64)
    m2 = np.zeros(64, dtype=np.float64)
    count = 0
    for start in range(0, n_rows, 200_000):
        end = min(n_rows, start+200_000)
        x = np.asarray(X[start:end], dtype=np.float32)
        if PER_TRACE_CENTER:
            x = x - x.mean(axis=1, keepdims=True)
        x64 = x.astype(np.float64)
        bc = x64.shape[0]
        bm = x64.mean(axis=0)
        bv = x64.var(axis=0)
        if count == 0:
            mean = bm
            m2 = bv * bc
            count = bc
        else:
            delta = bm - mean
            new_count = count + bc
            mean = mean + delta * (bc / new_count)
            m2 = m2 + bv * bc + (delta * delta) * (count * bc / new_count)
            count = new_count
    std = np.sqrt(m2 / max(count, 1) + 1e-8).astype(np.float32)
    mean = mean.astype(np.float32)
    for start in range(0, n_rows, 200_000):
        end = min(n_rows, start+200_000)
        x = np.asarray(X[start:end], dtype=np.float32)
        if PER_TRACE_CENTER:
            x = x - x.mean(axis=1, keepdims=True)
        X[start:end] = (x - mean[None, :]) / std[None, :]
    X.flush()
    T.flush()
    stats_path.write_text(json.dumps({'mean': mean.tolist(), 'std': std.tolist(), 'per_trace_center': bool(PER_TRACE_CENTER)}, indent=2))
    meta = {'n_rows': int(n_rows), 'x_path': str(x_path), 't_path': str(t_path), 'feature_cols': feat_cols, 'trace_csv': str(csv_path)}
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def build_cluster_memmaps_parquet(parquet_path: Path, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    meta_path = outdir / 'meta.json'
    x_path = outdir / 'X.memmap'
    t_path = outdir / 'token_id.memmap'
    stats_path = outdir / 'norm_stats.json'
    if (not REBUILD_CLUSTER_MEMMAPS) and meta_path.exists() and x_path.exists() and t_path.exists() and stats_path.exists():
        return json.loads(meta_path.read_text())
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    feat_cols = sorted([c for c in df.columns if c.startswith('set_')], key=lambda c: int(c.split('_', 1)[1]))
    n_rows = len(df)
    X = np.memmap(x_path, dtype=np.float32, mode='w+', shape=(n_rows, 64))
    T = np.memmap(t_path, dtype=np.int32, mode='w+', shape=(n_rows,))
    X[:] = df[feat_cols].to_numpy(dtype=np.float32, copy=False)
    T[:] = df['token_id'].to_numpy(dtype=np.int32, copy=False)
    mean = np.zeros(64, dtype=np.float64)
    m2 = np.zeros(64, dtype=np.float64)
    count = 0
    for start in range(0, n_rows, 200_000):
        end = min(n_rows, start+200_000)
        x = np.asarray(X[start:end], dtype=np.float32)
        if PER_TRACE_CENTER:
            x = x - x.mean(axis=1, keepdims=True)
        x64 = x.astype(np.float64)
        bc = x64.shape[0]
        bm = x64.mean(axis=0)
        bv = x64.var(axis=0)
        if count == 0:
            mean = bm
            m2 = bv * bc
            count = bc
        else:
            delta = bm - mean
            new_count = count + bc
            mean = mean + delta * (bc / new_count)
            m2 = m2 + bv * bc + (delta * delta) * (count * bc / new_count)
            count = new_count
    std = np.sqrt(m2 / max(count, 1) + 1e-8).astype(np.float32)
    mean = mean.astype(np.float32)
    for start in range(0, n_rows, 200_000):
        end = min(n_rows, start+200_000)
        x = np.asarray(X[start:end], dtype=np.float32)
        if PER_TRACE_CENTER:
            x = x - x.mean(axis=1, keepdims=True)
        X[start:end] = (x - mean[None, :]) / std[None, :]
    X.flush()
    T.flush()
    stats_path.write_text(json.dumps({'mean': mean.tolist(), 'std': std.tolist(), 'per_trace_center': bool(PER_TRACE_CENTER)}, indent=2))
    meta = {'n_rows': int(n_rows), 'x_path': str(x_path), 't_path': str(t_path), 'feature_cols': feat_cols, 'trace_parquet': str(parquet_path)}
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


class TokenTraceIndex:
    def __init__(self, token_ids: np.memmap):
        toks = np.asarray(token_ids[:], dtype=np.int32)
        order = np.argsort(toks, kind='mergesort')
        toks_sorted = toks[order]
        idx_sorted = np.arange(toks.shape[0], dtype=np.int64)[order]
        bounds = np.flatnonzero(np.diff(toks_sorted)) + 1
        splits = np.split(idx_sorted, bounds)
        keys = np.split(toks_sorted, bounds)
        self.tokens = np.array([int(k[0]) for k in keys], dtype=np.int32)
        self.map = {int(k[0]): s.astype(np.int64, copy=False) for k, s in zip(keys, splits)}

    def sample(self, token_id: int, m: int, rng: np.random.RandomState) -> np.ndarray:
        idxs = self.map[token_id]
        if idxs.size < m:
            return rng.choice(idxs, size=m, replace=True)
        return rng.choice(idxs, size=m, replace=False)

    def sample1(self, token_id: int, rng: np.random.RandomState) -> int:
        idxs = self.map.get(int(token_id))
        if idxs is None or idxs.size == 0:
            return -1
        if idxs.size == 1:
            return int(idxs[0])
        return int(rng.choice(idxs, size=1, replace=False)[0])


class TokenTraceIndexSubset:
    def __init__(self, token_ids: np.memmap, allowed_rows: np.ndarray):
        allowed_rows = np.asarray(allowed_rows, dtype=np.int64)
        toks = np.asarray(token_ids[allowed_rows], dtype=np.int32)

        if toks.size == 0:
            self.tokens = np.zeros((0,), dtype=np.int32)
            self.map = {}
            return

        order = np.argsort(toks, kind='mergesort')
        toks_sorted = toks[order]
        rows_sorted = allowed_rows[order]
        bounds = np.flatnonzero(np.diff(toks_sorted)) + 1
        row_splits = np.split(rows_sorted, bounds)
        tok_splits = np.split(toks_sorted, bounds)

        self.tokens = np.array([int(k[0]) for k in tok_splits], dtype=np.int32)
        self.map = {int(k[0]): r.astype(np.int64, copy=False) for k, r in zip(tok_splits, row_splits)}

    def sample(self, token_id: int, m: int, rng: np.random.RandomState) -> np.ndarray:
        idxs = self.map[token_id]
        if idxs.size < m:
            return rng.choice(idxs, size=m, replace=True)
        return rng.choice(idxs, size=m, replace=False)

    def sample1(self, token_id: int, rng: np.random.RandomState) -> int:
        idxs = self.map.get(int(token_id))
        if idxs is None or idxs.size == 0:
            return -1
        if idxs.size == 1:
            return int(idxs[0])
        return int(rng.choice(idxs, size=1, replace=False)[0])


class MLPEncoder(nn.Module):
    def __init__(self, in_dim=64, emb_dim=128, hidden=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def supcon_loss(z, y, temperature=0.07):
    B = z.shape[0]
    sim = (z @ z.T) / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    logits_mask = torch.ones((B, B), device=z.device, dtype=torch.bool)
    logits_mask.fill_diagonal_(False)
    y = y.view(-1, 1)
    pos_mask = (y == y.T) & logits_mask
    exp_sim = torch.exp(sim) * logits_mask
    denom = exp_sim.sum(dim=1, keepdim=True) + 1e-12
    pos_sum = (exp_sim * pos_mask).sum(dim=1)
    pos_cnt = pos_mask.sum(dim=1).clamp_min(1)
    return -torch.log((pos_sum / pos_cnt) / denom.squeeze(1) + 1e-12).mean()


def make_collate(X_memmap, index, seed):
    rng = np.random.RandomState(seed)

    def _collate(_batch):
        n_classes = min(P, len(index.tokens))
        tok_ids = rng.choice(index.tokens, size=n_classes, replace=(len(index.tokens) < P)).astype(np.int32)
        xs, ys = [], []
        for t in tok_ids:
            chosen = index.sample(int(t), M, rng)
            xs.append(np.asarray(X_memmap[chosen], dtype=np.float32))
            ys.append(np.full((M,), int(t), dtype=np.int32))
        return torch.from_numpy(np.concatenate(xs, axis=0)), torch.from_numpy(np.concatenate(ys, axis=0))
    return _collate


def infinite_loader(dl):
    while True:
        for batch in dl:
            yield batch


def split_profile_train_val_indices(token_ids_memmap: np.memmap, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    toks = np.asarray(token_ids_memmap[:], dtype=np.int32)
    rng = np.random.RandomState(seed)

    if toks.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)

    train_parts = []
    val_parts = []

    order = np.argsort(toks, kind='mergesort')
    toks_sorted = toks[order]
    idx_sorted = np.arange(toks.shape[0], dtype=np.int64)[order]
    bounds = np.flatnonzero(np.diff(toks_sorted)) + 1
    idx_groups = np.split(idx_sorted, bounds)

    for idxs in idx_groups:
        n = idxs.shape[0]
        if n <= 1:
            train_parts.append(idxs)
            continue

        n_val = max(MIN_VAL_PER_TOKEN, int(np.ceil(VAL_FRAC * n)))
        n_val = min(n_val, n - 1)

        perm = rng.permutation(n)
        val_local = perm[:n_val]
        train_local = perm[n_val:]

        val_parts.append(idxs[val_local])
        train_parts.append(idxs[train_local])

    train_idx = np.concatenate(train_parts).astype(np.int64, copy=False) if train_parts else np.zeros((0,), dtype=np.int64)
    val_idx = np.concatenate(val_parts).astype(np.int64, copy=False) if val_parts else np.zeros((0,), dtype=np.int64)

    train_idx.sort()
    val_idx.sort()
    return train_idx, val_idx


@torch.no_grad()
def compute_token_centroids_from_subset(encoder, X, index, device):
    encoder.eval()
    token_ids = index.tokens.copy()
    tid_to_row = {int(t): i for i, t in enumerate(token_ids)}
    sum_z = np.zeros((token_ids.shape[0], EMB_DIM), dtype=np.float64)
    cnt = np.zeros((token_ids.shape[0],), dtype=np.int64)

    for tok in token_ids:
        rows = index.map[int(tok)]
        for start in range(0, rows.shape[0], 200_000):
            batch_rows = rows[start:start + 200_000]
            x_np = np.asarray(X[batch_rows], dtype=np.float32)
            z = encoder(torch.from_numpy(x_np).to(device)).detach().cpu().numpy().astype(np.float64, copy=False)
            r = tid_to_row[int(tok)]
            sum_z[r] += z.sum(axis=0)
            cnt[r] += z.shape[0]

    centroids = (sum_z / np.maximum(cnt[:, None], 1)).astype(np.float32)
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    return token_ids, centroids


@torch.no_grad()
def retrieval_top1_accuracy(encoder, X_memmap, train_index, val_index, device) -> float:
    if len(train_index.tokens) == 0 or len(val_index.tokens) == 0:
        return 0.0

    train_token_ids, train_centroids = compute_token_centroids_from_subset(encoder, X_memmap, train_index, device)
    tid_to_row = {int(t): i for i, t in enumerate(train_token_ids)}
    C = torch.from_numpy(train_centroids).to(device)

    correct = 0
    total = 0

    encoder.eval()
    for tok in val_index.tokens:
        rows = val_index.map.get(int(tok))
        if rows is None or rows.size == 0:
            continue
        gold = tid_to_row.get(int(tok))
        if gold is None:
            continue

        for start in range(0, rows.shape[0], 200_000):
            batch_rows = rows[start:start + 200_000]
            x_np = np.asarray(X_memmap[batch_rows], dtype=np.float32)
            z = encoder(torch.from_numpy(x_np).to(device)).float()
            pred = (z @ C.T).argmax(dim=1).detach().cpu().numpy()
            correct += int((pred == gold).sum())
            total += int(pred.shape[0])

    return float(correct / max(total, 1))


def train_encoder_with_early_stopping(encoder, Xp, Tp, device, workdir: Path):
    train_idx_rows, val_idx_rows = split_profile_train_val_indices(Tp, seed=SEED)
    train_index = TokenTraceIndexSubset(Tp, train_idx_rows)
    val_index = TokenTraceIndexSubset(Tp, val_idx_rows)

    if len(train_index.tokens) == 0:
        raise RuntimeError('No training tokens available for encoder training.')

    dl = DataLoader(
        list(range(1024)),
        batch_size=1,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        collate_fn=make_collate(Xp, train_index, SEED),
    )

    opt = torch.optim.AdamW(encoder.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    encoder.train()
    it = infinite_loader(dl)

    best_metric = -1.0
    best_state = None
    patience = 0
    global_step = 0
    history = []
    stop_training = False

    for epoch in range(EPOCHS):
        for _ in tqdm(range(STEPS_PER_EPOCH), desc=f'SupCon {epoch+1}/{EPOCHS}'):
            xb, yb = next(it)
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            loss = supcon_loss(encoder(xb), yb, temperature=TEMPERATURE)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), GRAD_CLIP)
            opt.step()

            global_step += 1

            if len(val_index.tokens) > 0 and (global_step % EVAL_EVERY == 0):
                val_acc = retrieval_top1_accuracy(encoder, Xp, train_index, val_index, device)
                history.append({
                    'epoch': int(epoch + 1),
                    'global_step': int(global_step),
                    'val_retrieval_top1': float(val_acc),
                })

                if val_acc > best_metric + MIN_IMPROVEMENT:
                    best_metric = val_acc
                    best_state = copy.deepcopy(encoder.state_dict())
                    patience = 0
                else:
                    patience += 1

                encoder.train()

                if patience >= EARLY_STOP_PATIENCE:
                    stop_training = True
                    break

        if stop_training:
            break

    if best_state is not None:
        encoder.load_state_dict(best_state)

    train_summary = {
        'n_train_rows': int(len(train_idx_rows)),
        'n_val_rows': int(len(val_idx_rows)),
        'n_train_tokens': int(len(train_index.tokens)),
        'n_val_tokens': int(len(val_index.tokens)),
        'eval_every': int(EVAL_EVERY),
        'early_stop_patience': int(EARLY_STOP_PATIENCE),
        'min_improvement': float(MIN_IMPROVEMENT),
        'best_val_retrieval_top1': float(best_metric if best_metric >= 0 else 0.0),
        'stopped_early': bool(stop_training),
        'history': history,
    }
    (workdir / 'encoder_train_summary.json').write_text(json.dumps(train_summary, indent=2))
    return encoder


@torch.no_grad()
def compute_token_centroids(encoder, X, Tm, device):
    encoder.eval()
    toks_all = np.asarray(Tm[:], dtype=np.int32)
    token_ids = np.unique(toks_all).astype(np.int32)
    tid_to_row = {int(t): i for i, t in enumerate(token_ids)}
    sum_z = np.zeros((token_ids.shape[0], EMB_DIM), dtype=np.float64)
    cnt = np.zeros((token_ids.shape[0],), dtype=np.int64)
    for start in range(0, toks_all.shape[0], 200_000):
        end = min(toks_all.shape[0], start + 200_000)
        x_np = np.asarray(X[start:end], dtype=np.float32)
        t_np = toks_all[start:end]
        z = encoder(torch.from_numpy(x_np).to(device)).detach().cpu().numpy().astype(np.float64, copy=False)
        for i in range(t_np.shape[0]):
            r = tid_to_row[int(t_np[i])]
            sum_z[r] += z[i]
            cnt[r] += 1
    centroids = (sum_z / np.maximum(cnt[:, None], 1)).astype(np.float32)
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    return token_ids, centroids


@torch.no_grad()
def torch_kmeans_cosine(X_np, K, seed, niter=50, device='cuda'):
    device = device if (device == 'cuda' and torch.cuda.is_available()) else 'cpu'
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    X = F.normalize(torch.from_numpy(X_np).to(device), dim=1)
    N, _ = X.shape
    if N < K:
        raise RuntimeError(f'Cannot run KMeans with K={K} when N={N}')
    C = X[torch.randperm(N, generator=g, device=device)[:K]].clone()
    for _ in range(niter):
        lab = (X @ C.T).argmax(dim=1)
        C_new = torch.zeros_like(C)
        counts = torch.zeros((K,), device=device, dtype=torch.int64)
        C_new.index_add_(0, lab, X)
        counts.index_add_(0, lab, torch.ones((N,), device=device, dtype=torch.int64))
        empty = counts == 0
        if empty.any():
            refill = torch.randperm(N, generator=g, device=device)[:int(empty.sum())]
            C_new[empty] = X[refill]
            counts[empty] = 1
        C = F.normalize(C_new / counts.unsqueeze(1), dim=1)
    return lab.cpu().numpy().astype(np.int32)


def compute_cluster_centers_from_token_centroids(token_centroids, token_cluster_labels, K):
    D = token_centroids.shape[1]
    centers = np.zeros((K, D), dtype=np.float64)
    counts = np.zeros((K,), dtype=np.int64)
    for i in range(token_centroids.shape[0]):
        k = int(token_cluster_labels[i])
        centers[k] += token_centroids[i].astype(np.float64)
        counts[k] += 1
    for k in range(K):
        if counts[k] > 0:
            centers[k] /= float(counts[k])
    centers = centers.astype(np.float32)
    centers /= (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)
    return centers


@torch.no_grad()
def traces_to_symbol_ids(encoder, X_memmap, trace_row_ids, cluster_centers, base_vocab, sym_pad, device):
    out = np.full((trace_row_ids.shape[0],), sym_pad, dtype=np.int32)
    valid_mask = trace_row_ids >= 0
    if not np.any(valid_mask):
        return out
    idx = trace_row_ids[valid_mask].astype(np.int64, copy=False)
    x_np = np.asarray(X_memmap[idx], dtype=np.float32)
    z = encoder(torch.from_numpy(x_np).to(device)).float()
    lab = (z @ cluster_centers.T).argmax(dim=1).detach().cpu().numpy().astype(np.int32)
    out[valid_mask] = (base_vocab + lab).astype(np.int32, copy=False)
    return out


def phi_ids_to_text_prefix(phi: Llama, phi_ids: List[int]) -> str:
    b = phi.detokenize([int(x) for x in phi_ids])
    return b.decode('utf-8', errors='ignore') if isinstance(b, (bytes, bytearray)) else str(b)


def _build_first_response_uid(df: pd.DataFrame) -> pd.Series:
    prompt_col = df['prompt_id'] if 'prompt_id' in df.columns else pd.Series([None] * len(df))
    dialogue_col = df['dialogue_idx'] if 'dialogue_idx' in df.columns else pd.Series([None] * len(df))
    message_col = df['message_idx'] if 'message_idx' in df.columns else pd.Series([None] * len(df))
    return pd.Series(
        [
            f"first_row{i}_p{prompt_col.iloc[i]}_d{dialogue_col.iloc[i]}_m{message_col.iloc[i]}"
            for i in range(len(df))
        ],
        index=df.index,
        dtype='object',
    )


def load_test_rows() -> pd.DataFrame:
    need_cols = ['phi_ids', 'split', 'text', 'prompt_id', 'dialogue_idx', 'message_idx']
    dset = ds.dataset(PHI_PARQUET_FIRST, format='parquet')
    table = dset.to_table(columns=[c for c in need_cols if c in dset.schema.names])
    df = table.to_pandas()
    # parquet uses 'test' but may also contain 'test_sft' — accept both
    df = df[df['split'].astype(str).isin({'test_sft', 'test'})].copy().reset_index(drop=True)
    return df


def load_selected_rows(selected_parquet: Path) -> pd.DataFrame:
    df = pq.read_table(selected_parquet).to_pandas()
    df = df[df['segment_type'].astype(str) == 'first'].copy().reset_index(drop=True)
    return df


def load_train_rows() -> pd.DataFrame:
    need_cols = ['phi_ids', 'split', 'text', 'prompt_id', 'dialogue_idx', 'message_idx']
    dset = ds.dataset(PHI_PARQUET_FIRST, format='parquet')
    table = dset.to_table(columns=[c for c in need_cols if c in dset.schema.names])
    df = table.to_pandas()
    # parquet uses 'train' but may also contain 'train_sft' — accept both
    df = df[df['split'].astype(str).isin({'train_sft', 'train'})].copy().reset_index(drop=True)
    return df


def load_source_row_ids_from_profile_dir(profile_dir: Path) -> np.ndarray:
    npy_path = profile_dir / 'profiling_source_row_ids.npy'
    parquet_path = profile_dir / 'profile_trace_assignments.parquet'
    csv_path = profile_dir / 'profile_traces.csv'

    if npy_path.exists():
        return np.load(npy_path).astype(np.int64, copy=False)

    if parquet_path.exists():
        df = pq.read_table(parquet_path, columns=['source_row_id']).to_pandas()
        return np.asarray(sorted(set(int(x) for x in df['source_row_id'].tolist())), dtype=np.int64)

    if csv_path.exists():
        df = pd.read_csv(csv_path, usecols=['source_row_id'])
        return np.asarray(sorted(set(int(x) for x in df['source_row_id'].tolist())), dtype=np.int64)

    raise FileNotFoundError(f'Could not find profiling source-row IDs under {profile_dir}')


def load_source_row_ids_from_parquet(parquet_path: Path) -> np.ndarray:
    df = pq.read_table(parquet_path, columns=['source_row_id']).to_pandas()
    return np.asarray(sorted(set(int(x) for x in df['source_row_id'].tolist())), dtype=np.int64)


def build_llm_memmaps_from_full_csv(csv_path: Path, excluded_source_row_ids: np.ndarray, outdir: Path) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    meta_path = outdir / 'meta.json'
    x_path = outdir / 'X.memmap'
    t_path = outdir / 'token_id.memmap'
    s_path = outdir / 'source_row_id.memmap'
    stats_path = outdir / 'norm_stats.json'

    if (not REBUILD_CLUSTER_MEMMAPS) and meta_path.exists() and x_path.exists() and t_path.exists() and s_path.exists() and stats_path.exists():
        return json.loads(meta_path.read_text())

    sample = pd.read_csv(csv_path, nrows=5)
    feat_cols = sorted([c for c in sample.columns if c.startswith('set_')], key=lambda c: int(c.split('_', 1)[1]))
    if len(feat_cols) != 64:
        raise RuntimeError(f'Expected 64 set_* columns in {csv_path}, got {len(feat_cols)}')

    excluded = set(int(x) for x in np.asarray(excluded_source_row_ids, dtype=np.int64).tolist())

    n_keep = 0
    row_base = 0
    for chunk in pd.read_csv(csv_path, chunksize=200_000, usecols=['token_id'] + feat_cols):
        n = len(chunk)
        row_ids = np.arange(row_base, row_base + n, dtype=np.int64)
        keep_mask = np.fromiter((int(r) not in excluded for r in row_ids), count=n, dtype=np.bool_)
        n_keep += int(keep_mask.sum())
        row_base += n

    X = np.memmap(x_path, dtype=np.float32, mode='w+', shape=(n_keep, 64))
    T = np.memmap(t_path, dtype=np.int32, mode='w+', shape=(n_keep,))
    S = np.memmap(s_path, dtype=np.int64, mode='w+', shape=(n_keep,))

    off = 0
    row_base = 0
    for chunk in pd.read_csv(csv_path, chunksize=200_000, usecols=['token_id'] + feat_cols):
        n = len(chunk)
        row_ids = np.arange(row_base, row_base + n, dtype=np.int64)
        keep_mask = np.fromiter((int(r) not in excluded for r in row_ids), count=n, dtype=np.bool_)
        if keep_mask.any():
            kept = chunk.loc[keep_mask]
            kept_row_ids = row_ids[keep_mask]
            nn = len(kept)
            X[off:off+nn] = kept[feat_cols].to_numpy(dtype=np.float32, copy=False)
            T[off:off+nn] = kept['token_id'].to_numpy(dtype=np.int32, copy=False)
            S[off:off+nn] = kept_row_ids
            off += nn
        row_base += n

    mean = np.zeros(64, dtype=np.float64)
    m2 = np.zeros(64, dtype=np.float64)
    count = 0
    for start in range(0, n_keep, 200_000):
        end = min(n_keep, start + 200_000)
        x = np.asarray(X[start:end], dtype=np.float32)
        if PER_TRACE_CENTER:
            x = x - x.mean(axis=1, keepdims=True)
        x64 = x.astype(np.float64)
        bc = x64.shape[0]
        bm = x64.mean(axis=0)
        bv = x64.var(axis=0)
        if count == 0:
            mean = bm
            m2 = bv * bc
            count = bc
        else:
            delta = bm - mean
            new_count = count + bc
            mean = mean + delta * (bc / new_count)
            m2 = m2 + bv * bc + (delta * delta) * (count * bc / new_count)
            count = new_count

    std = np.sqrt(m2 / max(count, 1) + 1e-8).astype(np.float32)
    mean = mean.astype(np.float32)

    for start in range(0, n_keep, 200_000):
        end = min(n_keep, start + 200_000)
        x = np.asarray(X[start:end], dtype=np.float32)
        if PER_TRACE_CENTER:
            x = x - x.mean(axis=1, keepdims=True)
        X[start:end] = (x - mean[None, :]) / std[None, :]

    X.flush()
    T.flush()
    S.flush()

    stats_path.write_text(json.dumps({'mean': mean.tolist(), 'std': std.tolist(), 'per_trace_center': bool(PER_TRACE_CENTER)}, indent=2))
    meta = {
        'n_rows': int(n_keep),
        'x_path': str(x_path),
        't_path': str(t_path),
        'source_row_id_path': str(s_path),
        'feature_cols': feat_cols,
        'trace_csv': str(csv_path),
        'n_excluded_source_rows': int(len(excluded)),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def encode_rows_to_dataset(rows_df: pd.DataFrame, X, encoder, trace_index, cluster_centers, flan_tok, base_vocab, phi, device, sample_mode: str):
    sym_pad = 0
    flan_pad = int(flan_tok.pad_token_id) if flan_tok.pad_token_id is not None else 0
    rng = np.random.RandomState(SEED)
    B = len(rows_df)
    symbols = np.full((B, S_MAX), sym_pad, dtype=np.int32)
    labels = np.full((B, T_MAX), flan_pad, dtype=np.int32)
    for i, row in enumerate(tqdm(rows_df.itertuples(index=False), total=B, desc=f'Encoding first-{sample_mode}')):
        ids = [int(x) for x in getattr(row, 'phi_ids')[:S_MAX]]
        tok_mat = np.full((S_MAX,), -1, dtype=np.int32)
        if ids:
            tok_mat[:len(ids)] = np.asarray(ids, dtype=np.int32)
        trace_ids = np.full((S_MAX,), -1, dtype=np.int64)
        for j, tid in enumerate(tok_mat):
            if tid >= 0:
                trace_ids[j] = trace_index.sample1(int(tid), rng)
        symbols[i] = traces_to_symbol_ids(encoder, X, trace_ids, cluster_centers, base_vocab, sym_pad, device)
        wanted = [int(x) for x in getattr(row, 'phi_ids')[:TARGET_PHI_PREFIX_LEN]]
        tgt_text = phi_ids_to_text_prefix(phi, wanted)
        out_ids = [int(v) for v in flan_tok(tgt_text, max_length=T_MAX, truncation=True, padding='max_length', return_attention_mask=False)['input_ids']]
        labels[i] = np.asarray(out_ids[:T_MAX] + [flan_pad] * max(0, T_MAX - len(out_ids)), dtype=np.int32)
    return symbols, labels


def build_unified_symbol_dataset(out_dir: Path, X_llm, llm_index, X_held, heldout_index, encoder, cluster_centers_np, flan_tok, base_vocab, phi, llm_trace_path: Path, method: str, q: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = load_train_rows()
    test_rows = load_test_rows()

    if len(train_rows) == 0:
        (out_dir / 'build_status.json').write_text(json.dumps({'status': 'skipped_no_train_rows', 'kind': 'first', 'method': method, 'q': int(q)}, indent=2))
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    centers_t = torch.from_numpy(cluster_centers_np).to(device)

    trainval_symbols, trainval_labels = encode_rows_to_dataset(train_rows, X_llm, encoder, llm_index, centers_t, flan_tok, base_vocab, phi, device, 'llm_train')
    test_symbols, test_labels = encode_rows_to_dataset(test_rows, X_held, encoder, heldout_index, centers_t, flan_tok, base_vocab, phi, device, 'heldout')

    n_trainval, n_test = len(train_rows), len(test_rows)
    n_all = n_trainval + n_test

    sym_train_path = out_dir / 'symbols_train.npy'
    sym_path = out_dir / 'symbols.npy'
    t5_path = out_dir / 't5_labels.npy'
    sym_mm = np.lib.format.open_memmap(sym_train_path, mode='w+', dtype='int32', shape=(n_all, S_MAX))
    sym2_mm = np.lib.format.open_memmap(sym_path, mode='w+', dtype='int32', shape=(n_all, S_MAX))
    t5_mm = np.lib.format.open_memmap(t5_path, mode='w+', dtype='int32', shape=(n_all, T_MAX))

    sym_mm[:n_trainval] = trainval_symbols
    sym_mm[n_trainval:] = test_symbols
    sym2_mm[:] = sym_mm[:]
    t5_mm[:n_trainval] = trainval_labels
    t5_mm[n_trainval:] = test_labels
    sym_mm.flush()
    sym2_mm.flush()
    t5_mm.flush()

    rng = np.random.RandomState(SEED)
    all_trainval = np.arange(n_trainval, dtype=np.int64)
    n_val = min(n_test, max(1, int(round(0.2 * n_trainval)))) if n_trainval > 1 else 0
    perm = rng.permutation(n_trainval)
    val_idx = np.sort(all_trainval[perm[:n_val]]) if n_val > 0 else np.zeros((0,), dtype=np.int64)
    train_idx = np.sort(all_trainval[perm[n_val:]]) if n_val > 0 else all_trainval
    test_idx = np.arange(n_trainval, n_all, dtype=np.int64)
    np.savez_compressed(out_dir / 'splits.npz', train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    meta = {
        'kind': 'first',
        'n_samples': int(n_all),
        'n_trainval': int(n_trainval),
        'n_test': int(n_test),
        's_max': int(S_MAX),
        't_max': int(T_MAX),
        'K': int(K),
        'base_vocab': int(base_vocab),
        'symbols_pad': 0,
        't5_pad': int(flan_tok.pad_token_id) if flan_tok.pad_token_id is not None else 0,
        'trace_csv': str(llm_trace_path),
        'prompt_text': PROMPT_TEXT,
        'output_prefix': OUTPUT_PREFIX,
        'add_sentinel_token': bool(ADD_SENTINEL_TOKEN),
        'sentinel_text': SENTINEL_TEXT,
        'method': method,
        'q': int(q),
        'train_source': str(PHI_PARQUET_FIRST),
        'test_source': str(PHI_PARQUET_FIRST),
        'note': 'Train/val rows come from train_sft and are encoded with non-profiling, non-heldout traces from the full trace CSV. Test rows come from test_sft and are encoded with global heldout traces.'
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2))
    (out_dir / 'build_status.json').write_text(json.dumps({'status': 'done', 'kind': 'first', 'method': method, 'q': int(q), 'n_trainval': int(n_trainval), 'n_test': int(n_test)}, indent=2))


def run_one_budget(cost_root: Path, cluster_root: Path, symbols_root: Path, method: str, q: int, device: str):
    budget_dir = cost_root / method / f'q_{q}'
    trace_csv = budget_dir / 'profile_traces.csv'
    heldout_parquet = cost_root / 'global_heldout_traces.parquet'
    workdir = cluster_root / method / f'q_{q}'
    symbols_dir = symbols_root / method / f'q_{q}'
    workdir.mkdir(parents=True, exist_ok=True)
    symbols_dir.mkdir(parents=True, exist_ok=True)
    summary_path = workdir / 'run_summary.json'

    if not trace_csv.exists() or not heldout_parquet.exists():
        summary_path.write_text(json.dumps({
            'status': 'skipped_missing_input',
            'method': method,
            'q': int(q),
            'trace_csv_exists': trace_csv.exists(),
            'heldout_parquet_exists': heldout_parquet.exists()
        }, indent=2))
        return

    meta_profile = build_cluster_memmaps_csv(trace_csv, workdir / 'profile_memmaps')
    n_rows = int(meta_profile['n_rows'])
    Xp = np.memmap(meta_profile['x_path'], dtype=np.float32, mode='r', shape=(n_rows, 64))
    Tp = np.memmap(meta_profile['t_path'], dtype=np.int32, mode='r', shape=(n_rows,))
    uniq = np.unique(np.asarray(Tp[:], dtype=np.int32))

    if uniq.shape[0] < K:
        summary_path.write_text(json.dumps({
            'status': 'skipped_low_unique_tokens',
            'method': method,
            'q': int(q),
            'n_unique_tokens': int(uniq.shape[0]),
            'required_k': int(K)
        }, indent=2))
        return

    encoder_ckpt = workdir / 'encoder.pt'
    encoder = MLPEncoder(in_dim=64, emb_dim=EMB_DIM, hidden=HIDDEN, dropout=DROPOUT).to(device)

    if encoder_ckpt.exists() and (not REBUILD_ENCODER):
        encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device)['encoder'])
    else:
        encoder = train_encoder_with_early_stopping(encoder, Xp, Tp, device, workdir)
        torch.save({'encoder': encoder.state_dict()}, encoder_ckpt)

    token_ids, centroids = compute_token_centroids(encoder, Xp, Tp, device)

    clustering_path = workdir / 'clustering.json'
    if clustering_path.exists() and (not REBUILD_CLUSTERING):
        token_cluster = np.asarray(json.loads(clustering_path.read_text())['token_cluster'], dtype=np.int32)
    else:
        token_cluster = torch_kmeans_cosine(centroids, K=K, seed=SEED, niter=50, device=device)
        clustering_path.write_text(json.dumps({'token_ids': token_ids.tolist(), 'token_cluster': token_cluster.tolist()}, indent=2))

    cluster_centers = compute_cluster_centers_from_token_centroids(centroids, token_cluster, K=K)

    profiling_source_row_ids = load_source_row_ids_from_profile_dir(budget_dir)
    heldout_source_row_ids = load_source_row_ids_from_parquet(heldout_parquet)
    excluded_source_row_ids = np.asarray(
        sorted(set(int(x) for x in profiling_source_row_ids.tolist()) | set(int(x) for x in heldout_source_row_ids.tolist())),
        dtype=np.int64,
    )

    meta_llm = build_llm_memmaps_from_full_csv(FULL_TRACE_CSV, excluded_source_row_ids, workdir / 'llm_memmaps')
    n_llm = int(meta_llm['n_rows'])
    Xl = np.memmap(meta_llm['x_path'], dtype=np.float32, mode='r', shape=(n_llm, 64))
    Tl = np.memmap(meta_llm['t_path'], dtype=np.int32, mode='r', shape=(n_llm,))
    llm_index = TokenTraceIndex(Tl)

    meta_held = build_cluster_memmaps_parquet(heldout_parquet, workdir / 'heldout_memmaps')
    n_held = int(meta_held['n_rows'])
    Xh = np.memmap(meta_held['x_path'], dtype=np.float32, mode='r', shape=(n_held, 64))
    Th = np.memmap(meta_held['t_path'], dtype=np.int32, mode='r', shape=(n_held,))
    heldout_index = TokenTraceIndex(Th)

    flan_tok = AutoTokenizer.from_pretrained(FLAN_MODEL_NAME, use_fast=True)
    base_vocab = int(flan_tok.vocab_size)
    phi = Llama(model_path=str(PHI_MODEL_PATH), vocab_only=True, verbose=False)

    build_unified_symbol_dataset(
        symbols_dir / 'first',
        Xl,
        llm_index,
        Xh,
        heldout_index,
        encoder,
        cluster_centers,
        flan_tok,
        base_vocab,
        phi,
        FULL_TRACE_CSV,
        method,
        q,
    )

    summary = {
        'status': 'done',
        'method': method,
        'q': int(q),
        'n_profile_trace_rows': int(n_rows),
        'n_profile_unique_tokens': int(uniq.shape[0]),
        'n_profile_source_rows': int(len(profiling_source_row_ids)),
        'n_heldout_source_rows': int(len(heldout_source_row_ids)),
        'n_excluded_source_rows_for_llm': int(len(excluded_source_row_ids)),
        'n_llm_trace_rows': int(n_llm),
        'workdir': str(workdir),
        'symbols_dir': str(symbols_dir / 'first'),
        'profile_trace_csv': str(trace_csv),
        'llm_trace_csv': str(FULL_TRACE_CSV),
    }
    train_summary_path = workdir / 'encoder_train_summary.json'
    if train_summary_path.exists():
        try:
            train_summary = json.loads(train_summary_path.read_text())
            summary['encoder_best_val_retrieval_top1'] = float(train_summary.get('best_val_retrieval_top1', 0.0))
            summary['encoder_stopped_early'] = bool(train_summary.get('stopped_early', False))
        except Exception:
            pass

    summary_path.write_text(json.dumps(summary, indent=2))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--machine-type', required=True, choices=['laptop', 'desktop'])
    p.add_argument('--process-type', default='singleprocess', choices=['singleprocess', 'multiprocess'])
    p.add_argument('--method', default=None, choices=METHODS,
                   help='Single strategy to run (default: all)')
    p.add_argument('--q', default=None, type=int,
                   help='Single Q value to run (default: all in Q_LIST)')
    return p.parse_args()


def main():
    global MACHINE_TYPE, PROCESS_TYPE, FULL_TRACE_CSV
    args = parse_args()
    MACHINE_TYPE = args.machine_type
    PROCESS_TYPE = args.process_type
    FULL_TRACE_CSV = Path(
        f"/data/measurments/llamacpp_phi/"
        f"dataset_top50_per_token_{MACHINE_TYPE}_{PROCESS_TYPE}.csv"
    )

    cost_root = Path(f'/data/llamacpp/ultrachat_cluster_cost_effective/{MACHINE_TYPE}_{PROCESS_TYPE}')
    cluster_root = Path(f'./cluster_workdir_cost_effective/{MACHINE_TYPE}_{PROCESS_TYPE}/K_{K}')
    symbols_root = Path(f'/data/llamacpp/ultrachat_cluster_cost_effective/{MACHINE_TYPE}_{PROCESS_TYPE}/K_{K}')

    seed_all(SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    methods_to_run = [args.method] if args.method is not None else METHODS
    qs_to_run = [args.q] if args.q is not None else Q_LIST

    for method in methods_to_run:
        for q in qs_to_run:
            run_one_budget(cost_root, cluster_root, symbols_root, method, q, device)

    print('Done.')


if __name__ == '__main__':
    main()