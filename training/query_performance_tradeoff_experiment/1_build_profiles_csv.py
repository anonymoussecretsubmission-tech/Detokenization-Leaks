#!/usr/bin/env python3
# 1_build_profiles_csv.py
#
#  Profiling budget builder — Q-budget framing.
#
# The primary axis is Q = number of virtual queries the attacker is allowed
# to make. Every strategy spends exactly Q queries (or fewer if the vocabulary
# is exhausted before the budget runs out).
#
# A virtual query has two fixed physical parameters:
#
#   X = TOKENS_PER_QUERY = 10   distinct tokens packed into one query
#   Y = REPS_PER_QUERY   =  5   times the sequence is repeated
#
# These were chosen so that a 7B–13B quantized local LLM (llama.cpp) reliably
# follows the repeat instruction. One query generates X*Y = 50 tokens and
# yields Y = 5 trace samples for each of the X = 10 tokens it covers.
#
# ─────────────────────────────────────────────────────────────────────────────
# Strategies
# ─────────────────────────────────────────────────────────────────────────────
#
#   uniform   — maximum breadth. Spread Q queries across the top Q*X most
#               frequent domain tokens, each covered once (Y=5 traces).
#               Baseline attacker with no domain knowledge beyond frequency.
#
#   threshold — maximum depth. Profile only the top Q*X*COVERAGE_FRAC tokens
#               (25% of uniform's vocabulary), but spend all Q queries on that
#               smaller set so each token gets 4x more traces.
#               Tests whether trace quality beats vocabulary breadth.
#
#   oracle    — targeted attacker with perfect test-set knowledge. Profiles
#               only tokens that actually appear in the test set, ordered by
#               domain frequency. Upper bound on domain-targeted profiling.
#               Saturates once all reachable test tokens are covered.
#
# All strategies output the same profile_traces.csv format so that
# 2_prepare_symbols.py and all downstream scripts are completely unchanged.
#
# ─────────────────────────────────────────────────────────────────────────────
# Output layout
# ─────────────────────────────────────────────────────────────────────────────
#
#   OUT_ROOT / <strategy> / q_<Q> /
#       profile_traces.csv      ← identical format to original pipeline
#       virtual_queries.json    ← human-readable query plan
#       stats.json
#
# ─────────────────────────────────────────────────────────────────────────────

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import math
import random
import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# Global constants
# ─────────────────────────────────────────────────────────────────────────────

SEED = 42

MACHINE_TYPE = "laptop"
PROCESS_TYPE = "singleprocess"

TRACE_CSV = Path(
    f"/data/measurments/llamacpp_phi/"
    f"dataset_top50_per_token_{MACHINE_TYPE}_{PROCESS_TYPE}.csv"
)
PROCESSED_ROOT = Path(
    "/data/processed/phi"
)
FIRST_PARQUET = PROCESSED_ROOT / "ultrachat_phi_first.parquet"

OUT_ROOT = Path(
    f"/data/llamacpp/"
    f"ultrachat_cluster_cost_effective/{MACHINE_TYPE}_{PROCESS_TYPE}"
)

# ── Virtual query physical parameters ────────────────────────────────────────

# X: distinct tokens packed into one query.
#    10 is the sweet spot for reliable instruction-following on 7B-13B local
#    LLMs.  Empirically, sequences of 10 tokens are repeated faithfully;
#    at 15+ the model starts reordering or dropping items.
TOKENS_PER_QUERY = 10

# Y (baseline): repetitions per query = traces per token per query pass.
#    5 is above the observed N=3 sufficiency threshold and keeps the total
#    generated sequence length at X*Y = 50 tokens, well within what any
#    quantized 7B-13B model handles reliably.
REPS_PER_QUERY = 5

# ── Budget axis ───────────────────────────────────────────────────────────────
Q_LIST = [25, 50, 75, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000]
TEST_SPLITS = {"test"}

STRATEGIES = ["uniform", "threshold", "oracle"]

# ── Trace pool settings ───────────────────────────────────────────────────────
HELDOUT_PER_TOKEN    = 3
TOKEN_ID_COL         = "token_id"
TRACE_FEATURE_PREFIX = "set_"
TRAIN_SPLITS         = {"train_sft", "train"}  # parquet uses "train", kept both for safety
MIN_RESPONSE_TOKENS  = 8


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def json_dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def _chunk(lst: List, size: int) -> List[List]:
    return [lst[i : i + size] for i in range(0, len(lst), size)]


# ─────────────────────────────────────────────────────────────────────────────
# Trace CSV I/O  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def read_trace_csv(path: Path) -> pd.DataFrame:
    sample = pd.read_csv(path, nrows=5)
    feat_cols = sorted(
        [c for c in sample.columns if c.startswith(TRACE_FEATURE_PREFIX)],
        key=lambda c: int(c.split("_", 1)[1]),
    )
    if len(feat_cols) != 64:
        raise RuntimeError(f"Expected 64 set_* columns, got {len(feat_cols)}")
    df = pd.read_csv(path, usecols=[TOKEN_ID_COL] + feat_cols)
    df["source_row_id"] = np.arange(len(df), dtype=np.int64)
    return df


def build_trace_pools(
    df_traces: pd.DataFrame,
    heldout_per_token: int,
    seed: int,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    rng = np.random.RandomState(seed)
    profiling_pool: Dict[int, List[int]] = {}
    heldout_pool:   Dict[int, List[int]] = {}

    by_token = df_traces.groupby(TOKEN_ID_COL).indices
    for tok, idxs in tqdm(by_token.items(), desc="Sampling heldout traces"):
        idxs = np.asarray(idxs, dtype=np.int64)
        if idxs.size < heldout_per_token:
            raise RuntimeError(
                f"Token {tok} has only {idxs.size} traces, need {heldout_per_token}"
            )
        perm = rng.permutation(idxs.size)
        held = idxs[perm[:heldout_per_token]].tolist()
        prof = idxs[perm[heldout_per_token:]].tolist()
        rng.shuffle(prof)
        profiling_pool[int(tok)] = prof
        heldout_pool[int(tok)]   = held

    return profiling_pool, heldout_pool


def save_global_heldout_traces(
    df_traces: pd.DataFrame,
    heldout_pool: Dict[int, List[int]],
    out_path: Path,
):
    rows = []
    for _, idxs in heldout_pool.items():
        rows.extend(idxs)
    held = (
        df_traces
        .loc[np.asarray(rows, dtype=np.int64)]
        .copy()
        .reset_index(drop=True)
    )
    pq.write_table(
        pa.Table.from_pandas(held, preserve_index=False),
        str(out_path),
        compression="zstd",
    )


def save_source_row_ids(idxs: List[int], out_path: Path):
    np.save(out_path, np.asarray(sorted(int(x) for x in idxs), dtype=np.int64))


# ─────────────────────────────────────────────────────────────────────────────
# Domain vocabulary
# ─────────────────────────────────────────────────────────────────────────────

def _safe_ids(x):
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return x.astype(np.int64, copy=False).tolist()
    return [int(v) for v in x]


def load_test_token_set(parquet_path: Path) -> set:
    """
    Collect the set of token ids that appear in the test split.
    Used by the oracle strategy to profile only tokens it will actually see.
    """
    dataset = ds.dataset(parquet_path, format="parquet")
    all_cols = dataset.schema.names
    id_col_candidates = ["phi_ids", "token_ids", "input_ids", "token_list"]
    id_col = next((c for c in id_col_candidates if c in all_cols), None)
    if id_col is None:
        raise RuntimeError(f"Cannot find token-id column. Columns: {all_cols}")

    load_cols = [c for c in ["split", id_col] if c in all_cols]
    df = dataset.to_table(columns=load_cols).to_pandas()

    if "split" in df.columns:
        df["split"] = df["split"].astype(str)
        df = df[df["split"].isin(TEST_SPLITS)].copy()

    test_tokens: set = set()
    for row in tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc="Collecting test token set",
    ):
        ids = _safe_ids(getattr(row, id_col))
        if len(ids) >= MIN_RESPONSE_TOKENS:
            test_tokens.update(int(x) for x in ids)

    print(f"  Test token set size: {len(test_tokens)} unique tokens")
    return test_tokens


def load_domain_token_frequencies(parquet_path: Path) -> Counter:
    """
    Count per-token occurrences across UltraChat training responses.
    Used only to rank the vocabulary by importance; not used for response
    selection (that was the original approach which gave poor coverage).
    """
    dataset = ds.dataset(parquet_path, format="parquet")
    all_cols = dataset.schema.names
    print(f"  Parquet columns: {all_cols}")

    # Find the token-id list column (original script uses phi_ids)
    id_col_candidates = ["phi_ids", "token_ids", "input_ids", "token_list"]
    id_col = next((c for c in id_col_candidates if c in all_cols), None)
    if id_col is None:
        raise RuntimeError(
            f"Cannot find a token-id column in parquet. "
            f"Columns present: {all_cols}. "
            f"Expected one of: {id_col_candidates}"
        )
    print(f"  Using token-id column: '{id_col}'")

    load_cols = [c for c in ["split", id_col] if c in all_cols]
    df = dataset.to_table(columns=load_cols).to_pandas()

    # Filter to training splits if a split column exists
    if "split" in df.columns:
        df["split"] = df["split"].astype(str)
        present_splits = df["split"].unique().tolist()
        print(f"  Splits present in parquet: {present_splits}")

        # Use TRAIN_SPLITS if there's overlap, otherwise warn and use all rows
        overlap = set(present_splits) & TRAIN_SPLITS
        if overlap:
            df = df[df["split"].isin(overlap)].copy()
            print(f"  Filtering to splits: {overlap}  ({len(df)} rows kept)")
        else:
            print(
                f"  WARNING: TRAIN_SPLITS={TRAIN_SPLITS} not found in parquet. "
                f"Using all {len(df)} rows."
            )
    else:
        print(f"  No 'split' column found — using all {len(df)} rows.")

    freq: Counter = Counter()
    for row in tqdm(
        df.itertuples(index=False),
        total=len(df),
        desc="Computing domain frequencies",
    ):
        ids = _safe_ids(getattr(row, id_col))
        if len(ids) >= MIN_RESPONSE_TOKENS:
            freq.update(int(x) for x in ids)

    print(f"  Domain frequency computed: {len(freq)} unique tokens")
    return freq


# ─────────────────────────────────────────────────────────────────────────────
# Query plan builders
#
# Each planner receives:
#   tokens_by_freq : List[(token_id, domain_freq)] sorted DESC by freq
#   Q              : virtual query budget
#
# Returns a list of query dicts:
#   {
#     "query_id":  int,
#     "token_ids": List[int],   # distinct tokens in this query (len <= X)
#     "reps":      int,         # repetitions = REPS_PER_QUERY for all strategies
#   }
# ─────────────────────────────────────────────────────────────────────────────

def plan_uniform(
    tokens_by_freq: List[Tuple[int, int]],
    Q: int,
) -> List[Dict]:
    """
    Maximise vocabulary coverage: spread Q queries across as many distinct
    tokens as possible, each token appearing in exactly one query pass.

    Tokens are selected most-frequent-first so the Q*X most useful tokens
    are always covered.  Every covered token gets exactly REPS_PER_QUERY
    traces.  Tail tokens (beyond Q*X) get zero traces.

    This is the baseline attacker: maximum breadth, minimum depth.
    """
    max_tokens = Q * TOKENS_PER_QUERY
    tokens = [tok for tok, _ in tokens_by_freq[:max_tokens]]

    queries = []
    for qid, chunk in enumerate(_chunk(tokens, TOKENS_PER_QUERY)):
        queries.append({
            "query_id":  qid,
            "token_ids": chunk,
            "reps":      REPS_PER_QUERY,
        })
    return queries[:Q]


def plan_threshold(
    tokens_by_freq: List[Tuple[int, int]],
    Q: int,
) -> List[Dict]:
    """
    Maximise trace depth on a small, high-value vocabulary subset.

    Profiles only the top COVERAGE_FRAC fraction of what uniform would cover,
    but spends all Q queries on that smaller set so each token gets more passes.

    With k = Q * X * COVERAGE_FRAC tokens and n_chunks = ceil(k / X) chunks:
        passes_per_chunk = Q // n_chunks          (base)
        leftover_queries = Q % n_chunks           (distributed round-robin)

    So the full query budget Q is always consumed — no wasted slots.

    At COVERAGE_FRAC=0.25 and Q=100:
        k=250 tokens, 25 chunks, 4 passes each → 100 queries, 20 traces/token
    Compare to uniform at Q=100: 1000 tokens, 1 pass, 5 traces/token.
    """
    COVERAGE_FRAC = 0.25

    max_tokens_uniform = Q * TOKENS_PER_QUERY
    n_tokens = max(TOKENS_PER_QUERY, int(round(max_tokens_uniform * COVERAGE_FRAC)))

    top_tokens = [tok for tok, _ in tokens_by_freq[:n_tokens]]
    chunks = _chunk(top_tokens, TOKENS_PER_QUERY)
    n_chunks = len(chunks)

    base_passes    = Q // n_chunks      # every chunk gets at least this many passes
    leftover       = Q % n_chunks       # first `leftover` chunks get one extra pass

    queries = []
    qid = 0
    for i, chunk in enumerate(chunks):
        passes = base_passes + (1 if i < leftover else 0)
        for _ in range(passes):
            queries.append({
                "query_id":  qid,
                "token_ids": chunk,
                "reps":      REPS_PER_QUERY,
            })
            qid += 1

    return queries


def plan_oracle(
    tokens_by_freq: List[Tuple[int, int]],
    Q: int,
    test_token_set: set,
) -> List[Dict]:
    """
    Oracle / targeted attacker: profiles exactly the tokens that appear in
    the test set, nothing more.

    This is the upper bound on domain-targeted profiling — the attacker has
    perfect knowledge of which tokens they'll encounter.  Within the Q budget
    the oracle uses the same uniform allocation as plan_uniform, but restricted
    to test-set tokens ordered by domain frequency.

    Comparing oracle vs uniform at the same Q shows how much ASR is left on
    the table by profiling tokens that never appear in the target text.
    """
    # Filter to tokens that actually appear in the test set, keep freq order
    test_tokens = [tok for tok, _ in tokens_by_freq if tok in test_token_set]

    # Allocate uniformly within budget (same logic as plan_uniform)
    max_tokens = Q * TOKENS_PER_QUERY
    tokens = test_tokens[:max_tokens]

    queries = []
    for qid, chunk in enumerate(_chunk(tokens, TOKENS_PER_QUERY)):
        queries.append({
            "query_id":  qid,
            "token_ids": chunk,
            "reps":      REPS_PER_QUERY,
        })
    return queries[:Q]


# ─────────────────────────────────────────────────────────────────────────────
# Trace assignment
# ─────────────────────────────────────────────────────────────────────────────

def assign_traces(
    queries:               List[Dict],
    profiling_pool_master: Dict[int, List[int]],
    df_traces:             pd.DataFrame,
    feat_cols:             List[str],
) -> Tuple[pd.DataFrame, Dict]:
    """
    Walk the query plan and draw traces from the profiling pool.

    For each (query, repetition, token) triple we pop one trace index from
    a local copy of the pool so the master is never mutated.

    Returns a DataFrame with columns:
        token_id, virtual_query_id, rep_in_query, source_row_id, set_0…set_63
    which is the same format as profile_traces.csv in the original pipeline.
    """
    pool = {tok: lst.copy() for tok, lst in profiling_pool_master.items()}

    token_ids_col  = []
    query_ids_col  = []
    rep_col        = []
    trace_idxs_col = []
    skipped        = 0

    for vq in queries:
        qid       = vq["query_id"]
        tok_list  = vq["token_ids"]
        reps      = vq["reps"]

        for rep in range(reps):
            for tok in tok_list:
                tok_pool = pool.get(tok)
                if not tok_pool:
                    skipped += 1
                    continue
                token_ids_col.append(tok)
                query_ids_col.append(qid)
                rep_col.append(rep)
                trace_idxs_col.append(tok_pool.pop())

    if not token_ids_col:
        return pd.DataFrame(), {"error": "no traces assigned"}

    trace_indices = np.asarray(trace_idxs_col, dtype=np.int64)
    feat_part = (
        df_traces
        .loc[trace_indices, [TOKEN_ID_COL, "source_row_id"] + feat_cols]
        .copy()
        .reset_index(drop=True)
    )
    meta_part = pd.DataFrame({
        TOKEN_ID_COL:       token_ids_col,
        "virtual_query_id": query_ids_col,
        "rep_in_query":     rep_col,
    })

    result = pd.concat(
        [
            meta_part.reset_index(drop=True),
            feat_part[["source_row_id"] + feat_cols].reset_index(drop=True),
        ],
        axis=1,
    )

    n_requested = sum(len(vq["token_ids"]) * vq["reps"] for vq in queries)
    stats = {
        "n_virtual_queries":            len(queries),
        "total_tokens_generated":       len(queries) * TOKENS_PER_QUERY * REPS_PER_QUERY,
        "traces_requested":             int(n_requested),
        "traces_assigned":              int(len(result)),
        "traces_skipped_pool_empty":    int(skipped),
        "n_tokens_profiled":            int(result[TOKEN_ID_COL].nunique()),
    }
    return result, stats


# ─────────────────────────────────────────────────────────────────────────────
# Per (strategy, Q) run
# ─────────────────────────────────────────────────────────────────────────────

def run_one(
    strategy:              str,
    Q:                     int,
    tokens_by_freq:        List[Tuple[int, int]],
    profiling_pool_master: Dict[int, List[int]],
    df_traces:             pd.DataFrame,
    feat_cols:             List[str],
    domain_freq:           Counter,
    out_root:              Path,
    test_token_set:        set,
):
    outdir = out_root / strategy / f"q_{Q}"
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path     = outdir / "profile_traces.csv"
    stats_path   = outdir / "stats.json"
    queries_path = outdir / "virtual_queries.json"

    if csv_path.exists() and stats_path.exists():
        print(f"  [SKIP] {strategy}/q_{Q} — already done")
        return

    # Build query plan
    if strategy == "uniform":
        queries = plan_uniform(tokens_by_freq, Q)
    elif strategy == "threshold":
        queries = plan_threshold(tokens_by_freq, Q)
    elif strategy == "oracle":
        queries = plan_oracle(tokens_by_freq, Q, test_token_set)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    if not queries:
        json_dump({"status": "skipped_empty_plan", "strategy": strategy, "q": Q}, stats_path)
        return

    # Assign real traces from pool
    prof_df, stats = assign_traces(queries, profiling_pool_master, df_traces, feat_cols)

    if prof_df.empty:
        json_dump({"status": "skipped_no_traces", "strategy": strategy, "q": Q, **stats}, stats_path)
        return

    # ── Write profile_traces.csv in the format expected by 2_prepare_symbols.py
    keep_cols = [TOKEN_ID_COL] + feat_cols + [
        c for c in ["source_row_id", "virtual_query_id", "rep_in_query"]
        if c in prof_df.columns
    ]
    prof_df[keep_cols].to_csv(csv_path, index=False)

    # ── Write human-readable query plan
    json_dump(queries, queries_path)

    # ── Write stats
    n_domain = len(domain_freq)
    stats.update({
        "strategy":          strategy,
        "q":                 Q,
        "tokens_per_query":  TOKENS_PER_QUERY,
        "reps_per_query":    REPS_PER_QUERY,
        "coverage_frac":     round(stats["n_tokens_profiled"] / max(n_domain, 1), 4),
        "n_domain_tokens":   n_domain,
    })
    json_dump(stats, stats_path)

    print(
        f"  [{strategy}/q_{Q:4d}]  "
        f"queries={len(queries):4d}  "
        f"tokens_profiled={stats['n_tokens_profiled']:5d}  "
        f"traces={stats['traces_assigned']:6d}  "
        f"generated_tokens={stats['total_tokens_generated']:7d}  "
        f"coverage={stats['coverage_frac']:.1%}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Profiling budget builder (Q-budget framing)"
    )
    p.add_argument("--machine-type", required=True, choices=["laptop", "desktop"])
    p.add_argument(
        "--process-type",
        default="singleprocess",
        choices=["singleprocess", "multiprocess"],
    )
    p.add_argument(
        "--strategies",
        nargs="+",
        default=STRATEGIES,
        choices=STRATEGIES,
        help="Which strategies to run (default: all three)",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global MACHINE_TYPE, PROCESS_TYPE, TRACE_CSV, OUT_ROOT, FIRST_PARQUET

    args = parse_args()
    MACHINE_TYPE = args.machine_type
    PROCESS_TYPE = args.process_type

    TRACE_CSV = Path(
        f"/data/measurments/llamacpp_phi/"
        f"dataset_top50_per_token_{MACHINE_TYPE}_{PROCESS_TYPE}.csv"
    )
    OUT_ROOT = Path(
        f"/data/llamacpp/"
        f"ultrachat_cluster_cost_effective/{MACHINE_TYPE}_{PROCESS_TYPE}"
    )
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    seed_all(SEED)

    # ── Load traces ──────────────────────────────────────────────────────────
    print("Loading trace CSV …")
    df_traces = read_trace_csv(TRACE_CSV)
    feat_cols = sorted(
        [c for c in df_traces.columns if c.startswith(TRACE_FEATURE_PREFIX)],
        key=lambda c: int(c.split("_", 1)[1]),
    )

    # ── Heldout split (identical to original pipeline) ───────────────────────
    print("Building heldout split …")
    profiling_pool_master, heldout_pool = build_trace_pools(
        df_traces, HELDOUT_PER_TOKEN, SEED
    )
    save_global_heldout_traces(
        df_traces, heldout_pool, OUT_ROOT / "global_heldout_traces.parquet"
    )
    json_dump(
        {
            "heldout_per_token":       HELDOUT_PER_TOKEN,
            "n_tokens_in_trace_csv":   len(profiling_pool_master),
            "n_global_profiling_rows": int(sum(len(v) for v in profiling_pool_master.values())),
            "n_global_heldout_rows":   int(sum(len(v) for v in heldout_pool.values())),
        },
        OUT_ROOT / "heldout_stats.json",
    )

    # ── Domain vocabulary ────────────────────────────────────────────────────
    print("Computing domain token frequencies from UltraChat …")
    domain_freq = load_domain_token_frequencies(FIRST_PARQUET)

    # Restrict to tokens that exist in the trace CSV
    trace_token_set = set(profiling_pool_master.keys())
    valid_tokens = sorted(
        trace_token_set & set(domain_freq.keys()),
        key=lambda t: -domain_freq[t],  # most frequent first
    )
    tokens_by_freq: List[Tuple[int, int]] = [
        (tok, domain_freq[tok]) for tok in valid_tokens
    ]

    print(f"Tokens in trace CSV : {len(trace_token_set)}")
    print(f"Tokens in domain    : {len(domain_freq)}")
    print(f"Valid (intersection): {len(valid_tokens)}")
    print(
        f"\nVirtual query parameters:"
        f"\n  X = TOKENS_PER_QUERY = {TOKENS_PER_QUERY}"
        f"\n  Y = REPS_PER_QUERY   = {REPS_PER_QUERY}"
        f"\n  → {TOKENS_PER_QUERY * REPS_PER_QUERY} generated tokens per query"
        f"\n  → At Q=100: up to {100 * TOKENS_PER_QUERY} tokens profiled, "
        f"{100 * TOKENS_PER_QUERY * REPS_PER_QUERY} total generated tokens"
    )

    json_dump(
        {
            "tokens_per_query": TOKENS_PER_QUERY,
            "reps_per_query":   REPS_PER_QUERY,
            "n_valid_tokens":   len(valid_tokens),
            "n_trace_tokens":   len(trace_token_set),
            "n_domain_tokens":  len(domain_freq),
            "q_list":           Q_LIST,
            "strategies":       STRATEGIES,
        },
        OUT_ROOT / "experiment_config.json",
    )

    # ── Test token set (oracle strategy) ────────────────────────────────────
    print("Collecting test token set for oracle strategy …")
    test_token_set = load_test_token_set(FIRST_PARQUET)

    # ── Run all (strategy, Q) combinations ──────────────────────────────────
    for strategy in args.strategies:
        print(f"\n{'='*70}\nStrategy: {strategy}\n{'='*70}")
        for Q in Q_LIST:
            run_one(
                strategy              = strategy,
                Q                     = Q,
                tokens_by_freq        = tokens_by_freq,
                profiling_pool_master = profiling_pool_master,
                df_traces             = df_traces,
                feat_cols             = feat_cols,
                domain_freq           = domain_freq,
                out_root              = OUT_ROOT,
                test_token_set        = test_token_set,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()