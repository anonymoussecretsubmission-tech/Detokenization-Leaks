#!/usr/bin/env python3
"""
3_clean_sentences_ultrachat.py

Cleaning pipeline (vital steps only):
  1. Drop empty target text
  2. Token-length filter (>= MIN_FIRST_TOKENS / MIN_MIDDLE_TOKENS)
  3. Target-text dedup (lowercased)
  4. Remove AI refusal responses (^as an ai ... model)
  5. Prompt 8-gram dedup
  6. Train/test split  (FIRST only)

Middle inherits row_idx -> split mapping from first.
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

# ---------- CONFIG ----------
DATA_ROOT = Path("../data")

IN_FIRST_CSV  = DATA_ROOT / "ultrachat_first_sentences.csv"
IN_MIDDLE_CSV = DATA_ROOT / "ultrachat_middle_sentences.csv"

OUT_FIRST_CSV  = DATA_ROOT / "ultrachat_first_sentences_clean_balanced.csv"
OUT_MIDDLE_CSV = DATA_ROOT / "ultrachat_middle_sentences_clean_balanced.csv"

PROMPT_COL      = "prompt"
FIRST_TEXT_COL  = "first_sentence"
MIDDLE_TEXT_COL = "middle_sentence"
CONTEXT_COL     = "context_sentence"

PHI_TOKENIZER_NAME = os.environ.get(
    "PHI_TOKENIZER_NAME", "microsoft/Phi-3-mini-4k-instruct"
)

MIN_FIRST_TOKENS  = 20
MIN_MIDDLE_TOKENS = 5

NORM_COL = "_norm_ws"

SPLIT_COL   = "split"
TRAIN_LABEL = "train"
TEST_LABEL  = "test"
TEST_FRACTION = 0.10
SPLIT_SEED    = 42

MIN_ALPHA_RATIO = 0.6
# -----------------------------


# ── text helpers ──────────────────────────────────────────────────────────────

def normalize_ws_only(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower().strip())


_AI_REFUSAL_PATTERNS = [
    r"^as an ai (language )?model\b",
    r"^as an ai assistant\b",
    r"^i('m| am) (just |only )?an ai\b",
    r"^i('m| am) an artificial intelligence\b",
    r"^i cannot and will not\b",
    r"^i('m| am) not able to (provide|help|assist|generate|create)\b",
    r"^i do not have (the ability|personal opinions|feelings|emotions)\b",
    r"^i don't have (the ability|personal opinions|feelings|emotions)\b",
]
_AI_REFUSAL_RE = re.compile("|".join(_AI_REFUSAL_PATTERNS), re.IGNORECASE)


def is_ai_refusal(text: str) -> bool:
    return bool(_AI_REFUSAL_RE.match((text or "").strip()))

def alpha_ratio(text: str) -> float:
    s = (text or "").strip()
    if not s:
        return 0.0
    alpha_chars = sum(1 for c in s if c.isalpha())
    return alpha_chars / len(s)



# ── shared utils ──────────────────────────────────────────────────────────────

def token_len_batch(tokenizer, texts: list, batch_size: int = 1024) -> np.ndarray:
    n = len(texts)
    out = np.empty(n, dtype=np.int32)
    for start in tqdm(range(0, n, batch_size), desc="Tokenizing", unit="rows"):
        end = min(start + batch_size, n)
        batch = [(t or "").strip() for t in texts[start:end]]
        enc = tokenizer(
            batch,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            padding=False,
            truncation=False,
        )
        out[start:end] = np.fromiter(
            (len(ids) for ids in enc["input_ids"]),
            dtype=np.int32,
            count=(end - start),
        )
    return out


def get_8grams(text: str) -> set:
    tokens = normalize_ws_only(text).split()
    if len(tokens) < 8:
        return set()
    return {" ".join(tokens[i:i + 8]) for i in range(len(tokens) - 7)}


def get_first_8_words(text: str) -> str:
    return " ".join(normalize_ws_only(text).split()[:8])


def prompt_8gram_dedup(df: pd.DataFrame, prompt_col: str) -> pd.DataFrame:
    if prompt_col not in df.columns:
        raise KeyError(f"Expected column '{prompt_col}', found: {list(df.columns)}")

    seen_8grams: set = set()
    seen_first_8_words: set = set()
    keep_mask = []

    prompts = df[prompt_col].fillna("").astype(str).tolist()

    for prompt in tqdm(prompts, desc="8-gram prompt dedup", unit="rows"):
        first_8 = get_first_8_words(prompt)
        current_8grams = get_8grams(prompt)

        if first_8 and first_8 in seen_first_8_words:
            keep_mask.append(False)
            continue

        if current_8grams and current_8grams & seen_8grams:
            keep_mask.append(False)
            continue

        keep_mask.append(True)
        if first_8:
            seen_first_8_words.add(first_8)
        seen_8grams.update(current_8grams)

    before = len(df)
    df = df.loc[keep_mask].reset_index(drop=True)
    print(f"  After prompt 8-gram dedup: {len(df)} (removed {before - len(df)})")
    return df


def split_and_append_test_to_end(
    df: pd.DataFrame,
    test_fraction: float,
    seed: int,
    split_col: str,
    train_label: str,
    test_label: str,
) -> pd.DataFrame:
    if not (0.0 < test_fraction < 1.0):
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")

    n = len(df)
    if n == 0:
        df = df.copy()
        df[split_col] = pd.Series(dtype="object")
        return df

    n_test = int(round(n * test_fraction))
    n_test = max(1, n_test) if n >= 2 else 0

    if n_test == 0:
        out = df.copy()
        out[split_col] = train_label
    else:
        df_shuf = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        df_test  = df_shuf.iloc[:n_test].copy()
        df_train = df_shuf.iloc[n_test:].copy()
        df_train[split_col] = train_label
        df_test[split_col]  = test_label
        out = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    cols = list(out.columns)
    if split_col in cols:
        cols = [split_col] + [c for c in cols if c != split_col]
        out = out[cols]
    return out


# ── FIRST: full cleaning pipeline ────────────────────────────────────────────

def clean_first(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    required = [PROMPT_COL, FIRST_TEXT_COL, "row_idx"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"First CSV missing columns: {missing}. Found: {list(df.columns)}")

    n0 = len(df)
    print(f"  Loaded {n0} rows")

    df[PROMPT_COL]     = df[PROMPT_COL].fillna("").astype(str)
    df[FIRST_TEXT_COL] = df[FIRST_TEXT_COL].fillna("").astype(str)

    # 1) Drop empty target
    df = df[df[FIRST_TEXT_COL].str.strip().ne("")].copy()
    print(f"  After dropping empty: {len(df)} (removed {n0 - len(df)})")

    # 2) Token-length filter
    lens = token_len_batch(tokenizer, df[FIRST_TEXT_COL].tolist())
    df = df.loc[lens >= MIN_FIRST_TOKENS].copy()
    print(f"  After token filter (>={MIN_FIRST_TOKENS}): {len(df)}")

    # 3) Target-text dedup (lowercased for chat)
    df[NORM_COL] = [
        normalize_text(x)
        for x in tqdm(df[FIRST_TEXT_COL].tolist(), desc="Normalizing", unit="rows")
    ]
    df = df.drop_duplicates(subset=[NORM_COL]).reset_index(drop=True)
    df = df.drop(columns=[NORM_COL])
    print(f"  After text dedup: {len(df)}")

    # 4) Remove AI refusal responses
    mask = df[FIRST_TEXT_COL].apply(is_ai_refusal)
    df = df[~mask].copy().reset_index(drop=True)
    print(f"  After removing AI refusals: {len(df)}")

    # 5) Prompt 8-gram dedup
    df = prompt_8gram_dedup(df, prompt_col=PROMPT_COL)

    # 6) Train/test split
    df = split_and_append_test_to_end(
        df=df,
        test_fraction=TEST_FRACTION,
        seed=SPLIT_SEED,
        split_col=SPLIT_COL,
        train_label=TRAIN_LABEL,
        test_label=TEST_LABEL,
    )

    # 7) Remove responses that are mostly non-alphabetic (code/URL/number dumps)
    ar = df[FIRST_TEXT_COL].apply(alpha_ratio)
    before = len(df)
    df = df.loc[ar >= MIN_ALPHA_RATIO].copy().reset_index(drop=True)
    print(f"  After alpha ratio filter (>={MIN_ALPHA_RATIO}): {len(df)} (removed {before - len(df)})")


    n_test  = int((df[SPLIT_COL] == TEST_LABEL).sum())
    n_train = int((df[SPLIT_COL] == TRAIN_LABEL).sum())
    print(f"  Split: train={n_train}, test={n_test}")
    return df


# ── MIDDLE: filter-only, inherit splits from first ───────────────────────────

def clean_middle(
    df_mid: pd.DataFrame,
    row_idx_to_split: dict,
    tokenizer,
) -> pd.DataFrame:
    required = [CONTEXT_COL, MIDDLE_TEXT_COL, "row_idx"]
    missing = [c for c in required if c not in df_mid.columns]
    if missing:
        raise KeyError(f"Middle CSV missing columns: {missing}. Found: {list(df_mid.columns)}")

    n0 = len(df_mid)
    print(f"  Loaded {n0} rows")

    df_mid[CONTEXT_COL]     = df_mid[CONTEXT_COL].fillna("").astype(str)
    df_mid[MIDDLE_TEXT_COL] = df_mid[MIDDLE_TEXT_COL].fillna("").astype(str)

    # 1) Keep only row_idx present in cleaned first
    mask = df_mid["row_idx"].isin(row_idx_to_split)
    df_mid = df_mid[mask].copy()
    print(f"  After filtering to first's row_idx: {len(df_mid)} "
          f"(removed {n0 - len(df_mid)})")

    # 2) Drop empty middle text
    df_mid = df_mid[df_mid[MIDDLE_TEXT_COL].str.strip().ne("")].copy()
    print(f"  After dropping empty middle text: {len(df_mid)}")

    # 3) Token-length filter on middle text
    lens = token_len_batch(tokenizer, df_mid[MIDDLE_TEXT_COL].tolist())
    df_mid = df_mid.loc[lens >= MIN_MIDDLE_TOKENS].copy()
    print(f"  After token filter (>={MIN_MIDDLE_TOKENS}): {len(df_mid)}")

    # 4) Inherit split from first
    df_mid[SPLIT_COL] = df_mid["row_idx"].map(row_idx_to_split)

    cols = [SPLIT_COL] + [c for c in df_mid.columns if c != SPLIT_COL]
    df_mid = df_mid[cols].reset_index(drop=True)

    n_test  = int((df_mid[SPLIT_COL] == TEST_LABEL).sum())
    n_train = int((df_mid[SPLIT_COL] == TRAIN_LABEL).sum())
    print(f"  Split (inherited from first): train={n_train}, test={n_test}")
    return df_mid


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(f"PHI_TOKENIZER_NAME = {PHI_TOKENIZER_NAME}")
    print(f"DATA_ROOT          = {DATA_ROOT}")
    print("-" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        PHI_TOKENIZER_NAME,
        use_fast=True,
        trust_remote_code=True,
    )

    # ── FIRST ────────────────────────────────────────────────────────────────
    if not IN_FIRST_CSV.exists():
        raise FileNotFoundError(f"Not found: {IN_FIRST_CSV}")

    print("\n=== FIRST FILE ===")
    df_first = pd.read_csv(IN_FIRST_CSV)
    df_first = clean_first(df_first, tokenizer)

    OUT_FIRST_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_first.to_csv(OUT_FIRST_CSV, index=False)
    print(f"  Saved: {OUT_FIRST_CSV}  ({len(df_first)} rows)")

    if "row_idx" not in df_first.columns:
        raise KeyError("'row_idx' not found in first CSV after cleaning.")

    row_idx_to_split = dict(
        zip(df_first["row_idx"].tolist(), df_first[SPLIT_COL].tolist())
    )
    n_test_idx  = sum(1 for v in row_idx_to_split.values() if v == TEST_LABEL)
    n_train_idx = sum(1 for v in row_idx_to_split.values() if v == TRAIN_LABEL)
    print(f"\n  row_idx->split mapping: {len(row_idx_to_split)} entries "
          f"({n_train_idx} train, {n_test_idx} test)")

    # ── MIDDLE ───────────────────────────────────────────────────────────────
    if not IN_MIDDLE_CSV.exists():
        raise FileNotFoundError(f"Not found: {IN_MIDDLE_CSV}")

    print("\n=== MIDDLE FILE ===")
    df_mid = pd.read_csv(IN_MIDDLE_CSV)
    df_mid = clean_middle(df_mid, row_idx_to_split, tokenizer)

    OUT_MIDDLE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_mid.to_csv(OUT_MIDDLE_CSV, index=False)
    print(f"  Saved: {OUT_MIDDLE_CSV}  ({len(df_mid)} rows)")

    # ── sanity check ─────────────────────────────────────────────────────────
    print("\n=== SANITY CHECK ===")
    first_test_idx  = set(df_first.loc[df_first[SPLIT_COL] == TEST_LABEL,  "row_idx"])
    first_train_idx = set(df_first.loc[df_first[SPLIT_COL] == TRAIN_LABEL, "row_idx"])
    mid_test_idx    = set(df_mid.loc[df_mid[SPLIT_COL]   == TEST_LABEL,  "row_idx"])
    mid_train_idx   = set(df_mid.loc[df_mid[SPLIT_COL]   == TRAIN_LABEL, "row_idx"])

    print(f"  First  test  rows : {len(first_test_idx)}")
    print(f"  Middle test  rows : {len(mid_test_idx)}")
    print(f"  First  train rows : {len(first_train_idx)}")
    print(f"  Middle train rows : {len(mid_train_idx)}")

    for label, mid_set, first_set in [
        ("test",  mid_test_idx,  first_test_idx),
        ("train", mid_train_idx, first_train_idx),
    ]:
        leakage = mid_set - first_set
        if leakage:
            print(f"  [ERROR] {len(leakage)} middle {label} row_idx not in first {label} "
                  f"— sample: {list(leakage)[:5]}")
        else:
            print(f"  [OK] All middle {label} row_idx present in first {label}.")
        if len(mid_set) > len(first_set):
            print(f"  [ERROR] middle {label} ({len(mid_set)}) > first {label} ({len(first_set)})")

    print("\nDone.")


if __name__ == "__main__":
    main()