#!/usr/bin/env python3
"""
UltraChat (first + middle segments) -> token-id parquets (two files)

Goal:
  Produce the minimal parquet outputs needed by prepare_symbol_dataset later, WITHOUT
  inventing a new train/val/test split. We keep UltraChat's original split labels
  (typically: train / test, set by 3_clean_sentences.py).

Inputs:
  - Cleaned first segments CSV: ultrachat_first_sentences_clean_balanced.csv
  - Middle segments CSV:        ultrachat_middle_sentences_clean_balanced.csv

Outputs:
  - ultrachat_{model}_first.parquet   columns: [text, phi_ids, phi_len, split, row_idx, prompt_id, dialogue_idx, message_idx]
  - ultrachat_{model}_middle.parquet  columns: [context_text, text, phi_ids, phi_len, split, row_idx, prompt_id, dialogue_idx, message_idx]
  - ultrachat_{model}_meta.json

row_idx is the ORIGINAL row_idx from the cleaned CSVs (assigned once in
2_extract_sentences.py and preserved through 3_clean_sentences.py).
It is the stable join key between first and middle parquets:
  first.row_idx == middle.row_idx  ⟺  same assistant message

It is used by prepare_symbol_dataset.py to build
  group_id = '{split}|row_idx|{row_idx}'
for segment_meta.jsonl, matching the ChatDoctor format so eval /
print_prefix_table scripts can join first + middle by paragraph_id.

Usage:
  python 4_trace_mapping.py --model phi
  python 4_trace_mapping.py --model llama
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq

from llama_cpp import Llama


# ==================== CONFIG ====================

DATA_ROOT = Path("../data")

CLEANED_FIRST_CSV = DATA_ROOT / "ultrachat_first_sentences_clean_balanced.csv"
RAW_FIRST_CSV     = DATA_ROOT / "ultrachat_first_sentences.csv"
MIDDLE_CSV        = DATA_ROOT / "ultrachat_middle_sentences_clean_balanced.csv"

MODEL_PATHS = {
    "phi":   Path("../data/models/phi3-mini/Phi-3-mini-4k-instruct-q4.gguf"),
    "llama": Path("../data/models/llama-3.1/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
}

SPLIT_COL          = "split"
ROW_IDX_COL        = "row_idx"          # original stable join key from step 2/3
ID_COLS            = ["prompt_id", "dialogue_idx", "message_idx"]
FIRST_TEXT_COL     = "first_sentence"
MIDDLE_CONTEXT_COL = "context_sentence"
MIDDLE_TEXT_COL    = "middle_sentence"

MAX_PHI_TOKENS_PER_ROW = 4096
CHUNK_SIZE             = 5000

# =================================================


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["phi", "llama"],
                        help="Target model for tokenization")
    return parser.parse_args()


def clean_border_quotes(sent: str) -> str:
    s = (sent or "").strip()
    s = re.sub(r'"{2,}', '"', s)
    s = re.sub(r"'{2,}", "'", s)
    return s.strip('"').strip("'").strip()


def _ensure_int_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(int)
    return df


def load_first_with_split() -> pd.DataFrame:
    if not CLEANED_FIRST_CSV.exists():
        raise FileNotFoundError(f"Missing cleaned first CSV: {CLEANED_FIRST_CSV}")

    df = pd.read_csv(CLEANED_FIRST_CSV)
    if FIRST_TEXT_COL not in df.columns:
        raise KeyError(f"Expected '{FIRST_TEXT_COL}' in {CLEANED_FIRST_CSV}. Found: {list(df.columns)}")
    if ROW_IDX_COL not in df.columns:
        raise KeyError(
            f"Expected '{ROW_IDX_COL}' in {CLEANED_FIRST_CSV}. Found: {list(df.columns)}\n"
            f"Re-run 2_extract_sentences.py and 3_clean_sentences.py to regenerate the CSVs."
        )

    df[FIRST_TEXT_COL] = df[FIRST_TEXT_COL].astype(str).map(clean_border_quotes)
    df = _ensure_int_cols(df, ID_COLS + [ROW_IDX_COL])

    if SPLIT_COL in df.columns:
        return df

    # Reconstruct split by joining against RAW_FIRST_CSV
    if not RAW_FIRST_CSV.exists():
        raise RuntimeError(
            f"Cleaned first CSV is missing '{SPLIT_COL}', and cannot reconstruct because "
            f"RAW_FIRST_CSV is missing: {RAW_FIRST_CSV}"
        )

    df_raw = pd.read_csv(RAW_FIRST_CSV)
    missing = [c for c in ([SPLIT_COL] + ID_COLS + [FIRST_TEXT_COL]) if c not in df_raw.columns]
    if missing:
        raise RuntimeError(
            f"Cannot reconstruct split: RAW_FIRST_CSV missing columns {missing}. "
            f"Found: {list(df_raw.columns)}"
        )

    df_raw = _ensure_int_cols(df_raw, ID_COLS)
    df_raw[FIRST_TEXT_COL] = df_raw[FIRST_TEXT_COL].astype(str).map(clean_border_quotes)
    df_raw = df_raw[[SPLIT_COL] + ID_COLS + [FIRST_TEXT_COL]].drop_duplicates()

    df2 = df.merge(df_raw, on=ID_COLS + [FIRST_TEXT_COL], how="left")
    df2[SPLIT_COL] = df2[SPLIT_COL].fillna("train_sft")
    return df2


def load_middle() -> pd.DataFrame:
    if not MIDDLE_CSV.exists():
        raise FileNotFoundError(f"Missing middle CSV: {MIDDLE_CSV}")

    df = pd.read_csv(MIDDLE_CSV)
    missing = [c for c in ([SPLIT_COL, ROW_IDX_COL] + ID_COLS + [MIDDLE_CONTEXT_COL, MIDDLE_TEXT_COL])
               if c not in df.columns]
    if missing:
        raise KeyError(f"Middle CSV missing columns {missing}. Found: {list(df.columns)}")

    df = _ensure_int_cols(df, ID_COLS + [ROW_IDX_COL])
    df[MIDDLE_CONTEXT_COL] = df[MIDDLE_CONTEXT_COL].astype(str).map(clean_border_quotes)
    df[MIDDLE_TEXT_COL]    = df[MIDDLE_TEXT_COL].astype(str).map(clean_border_quotes)
    return df


def flush_rows(rows: List[Dict[str, Any]], out_path: Path, writer_holder: Dict[str, Any]) -> None:
    if not rows:
        return
    table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
    if writer_holder.get("writer") is None:
        writer_holder["writer"] = pq.ParquetWriter(str(out_path), table.schema, compression="zstd")
    writer_holder["writer"].write_table(table)
    rows.clear()


def close_writer(writer_holder: Dict[str, Any]) -> None:
    w = writer_holder.get("writer")
    if w is not None:
        w.close()
        writer_holder["writer"] = None


def tokenize(model: Llama, text: str) -> List[int]:
    return [int(x) for x in model.tokenize(text.encode("utf-8"), add_bos=False, special=True)]


def main() -> None:
    args         = parse_args()
    target_model = args.model
    model_path   = MODEL_PATHS[target_model]

    out_dir            = DATA_ROOT / "processed" / target_model
    out_first_parquet  = out_dir / f"ultrachat_{target_model}_first.parquet"
    out_middle_parquet = out_dir / f"ultrachat_{target_model}_middle.parquet"
    out_meta           = out_dir / f"ultrachat_{target_model}_meta.json"

    out_dir.mkdir(parents=True, exist_ok=True)
    for p in (out_first_parquet, out_middle_parquet):
        if p.exists():
            p.unlink()

    df_first = load_first_with_split()
    df_mid   = load_middle()

    for c in ID_COLS:
        if c not in df_first.columns:
            df_first[c] = -1

    # ── Sanity: row_idx must be unique in first CSV (it's a per-message key)
    n_first_dupes = df_first[ROW_IDX_COL].duplicated().sum()
    if n_first_dupes > 0:
        raise RuntimeError(
            f"{n_first_dupes} duplicate row_idx values in first CSV. "
            "Re-run 2_extract_sentences.py and 3_clean_sentences.py."
        )

    # ── Sanity: every middle row_idx must exist in first
    first_idx_set = set(df_first[ROW_IDX_COL].tolist())
    mid_orphans   = (~df_mid[ROW_IDX_COL].isin(first_idx_set)).sum()
    if mid_orphans > 0:
        raise RuntimeError(
            f"{mid_orphans} middle rows have row_idx not present in first CSV. "
            "Re-run 3_clean_sentences.py."
        )

    print(f"Loading tokenizer (vocab_only) from: {model_path}")
    llm = Llama(model_path=str(model_path), vocab_only=True, verbose=False)

    writer_first = {"writer": None}
    writer_mid   = {"writer": None}
    rows_first:  List[Dict[str, Any]] = []
    rows_mid:    List[Dict[str, Any]] = []

    stats = {
        "first":  {"input_rows": int(len(df_first)), "kept": 0, "dropped_empty": 0,
                   "dropped_tokenize_fail": 0, "dropped_zero_len": 0, "truncated": 0},
        "middle": {"input_rows": int(len(df_mid)),   "kept": 0, "dropped_empty": 0,
                   "dropped_tokenize_fail": 0, "dropped_zero_len": 0, "truncated": 0},
    }
    split_counts_first:  Dict[str, int] = {}
    split_counts_middle: Dict[str, int] = {}

    print(f"First rows:  {len(df_first)}  |  Middle rows: {len(df_mid)}")
    print(f"Output dir:  {out_dir}")

    # ── FIRST PARQUET ─────────────────────────────────────────────────────────
    # row_idx is read directly from the CSV — it is the original stable join key
    # assigned in 2_extract_sentences.py and preserved by 3_clean_sentences.py.
    # We MUST NOT replace it with a sequential counter here.

    iter_cols_first = [SPLIT_COL, ROW_IDX_COL] + ID_COLS + [FIRST_TEXT_COL]

    for (split_name, orig_row_idx, prompt_id, dialogue_idx, message_idx, text) in tqdm(
        df_first[iter_cols_first].itertuples(index=False, name=None),
        total=len(df_first), desc="Tokenizing FIRST"
    ):
        text = clean_border_quotes(str(text))
        if not text:
            stats["first"]["dropped_empty"] += 1
            continue

        try:
            ids = tokenize(llm, text)
        except Exception:
            stats["first"]["dropped_tokenize_fail"] += 1
            continue

        if not ids:
            stats["first"]["dropped_zero_len"] += 1
            continue

        if len(ids) > MAX_PHI_TOKENS_PER_ROW:
            ids = ids[:MAX_PHI_TOKENS_PER_ROW]
            stats["first"]["truncated"] += 1

        sp = str(split_name)
        split_counts_first[sp] = split_counts_first.get(sp, 0) + 1

        rows_first.append({
            "text":         text,
            "phi_ids":      ids,
            "phi_len":      len(ids),
            "split":        sp,
            "row_idx":      int(orig_row_idx),   # ← original CSV row_idx, NOT a new counter
            "prompt_id":    int(prompt_id)    if prompt_id    is not None else -1,
            "dialogue_idx": int(dialogue_idx) if dialogue_idx is not None else -1,
            "message_idx":  int(message_idx)  if message_idx  is not None else -1,
        })
        stats["first"]["kept"] += 1

        if len(rows_first) >= CHUNK_SIZE:
            flush_rows(rows_first, out_first_parquet, writer_first)

    # ── MIDDLE PARQUET ────────────────────────────────────────────────────────
    # Same fix: pass orig_row_idx through unchanged so it matches first parquet.

    iter_cols_mid = [SPLIT_COL, ROW_IDX_COL] + ID_COLS + [MIDDLE_CONTEXT_COL, MIDDLE_TEXT_COL]

    for (split_name, orig_row_idx, prompt_id, dialogue_idx, message_idx, context_text, middle_text) in tqdm(
        df_mid[iter_cols_mid].itertuples(index=False, name=None),
        total=len(df_mid), desc="Tokenizing MIDDLE"
    ):
        context_text = clean_border_quotes(str(context_text))
        middle_text  = clean_border_quotes(str(middle_text))

        if not middle_text:
            stats["middle"]["dropped_empty"] += 1
            continue

        try:
            ids = tokenize(llm, middle_text)
        except Exception:
            stats["middle"]["dropped_tokenize_fail"] += 1
            continue

        if not ids:
            stats["middle"]["dropped_zero_len"] += 1
            continue

        if len(ids) > MAX_PHI_TOKENS_PER_ROW:
            ids = ids[:MAX_PHI_TOKENS_PER_ROW]
            stats["middle"]["truncated"] += 1

        sp = str(split_name)
        split_counts_middle[sp] = split_counts_middle.get(sp, 0) + 1

        rows_mid.append({
            "context_text": context_text,
            "text":         middle_text,
            "phi_ids":      ids,
            "phi_len":      len(ids),
            "split":        sp,
            "row_idx":      int(orig_row_idx),   # ← original CSV row_idx, NOT a new counter
            "prompt_id":    int(prompt_id)    if prompt_id    is not None else -1,
            "dialogue_idx": int(dialogue_idx) if dialogue_idx is not None else -1,
            "message_idx":  int(message_idx)  if message_idx  is not None else -1,
        })
        stats["middle"]["kept"] += 1

        if len(rows_mid) >= CHUNK_SIZE:
            flush_rows(rows_mid, out_middle_parquet, writer_mid)

    flush_rows(rows_first, out_first_parquet, writer_first)
    flush_rows(rows_mid,   out_middle_parquet, writer_mid)
    close_writer(writer_first)
    close_writer(writer_mid)

    meta = {
        "target_model": target_model,
        "inputs": {
            "cleaned_first_csv":                      str(CLEANED_FIRST_CSV),
            "raw_first_csv_for_split_reconstruction": str(RAW_FIRST_CSV),
            "middle_csv":                             str(MIDDLE_CSV),
        },
        "outputs": {
            "first_parquet":  str(out_first_parquet),
            "middle_parquet": str(out_middle_parquet),
            "meta_json":      str(out_meta),
        },
        "model_path":         str(model_path),
        "max_tokens_per_row": MAX_PHI_TOKENS_PER_ROW,
        "row_idx_note": (
            "row_idx is the ORIGINAL value from the cleaned CSVs, assigned once in "
            "2_extract_sentences.py (one integer per assistant message, globally unique "
            "across all splits). It is preserved unchanged through 3_clean_sentences.py "
            "and now through this script. "
            "first.row_idx == middle.row_idx iff they come from the same assistant message. "
            "prepare_symbol_dataset.py uses it to build "
            "group_id = '{split}|row_idx|{row_idx}' for segment_meta.jsonl."
        ),
        "split_counts": {
            "first":  {k: int(v) for k, v in split_counts_first.items()},
            "middle": {k: int(v) for k, v in split_counts_middle.items()},
        },
        "stats": stats,
    }
    out_meta.write_text(json.dumps(meta, indent=2))

    print("\nDone.")
    print("  first parquet: ", out_first_parquet)
    print("  middle parquet:", out_middle_parquet)
    print("  meta:          ", out_meta)
    print("  first rows written: ", stats["first"]["kept"])
    print("  middle rows written:", stats["middle"]["kept"])


if __name__ == "__main__":
    main()