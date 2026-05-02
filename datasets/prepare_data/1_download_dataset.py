#!/usr/bin/env python3
import os

# ── Force HF cache dirs BEFORE importing datasets ─────────────────────────────
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
os.environ["HF_HUB_DISABLE_XET"]      = "1"
# ──────────────────────────────────────────────────────────────────────────────

from pathlib import Path
from datasets import load_dataset

# --------- CONFIG ---------
DATA_ROOT          = Path("../data/raw")
DATASET_NAME       = "HuggingFaceTB/ultrachat_questions_about_world"
SPLITS_TO_DOWNLOAD = ["train"]
# --------------------------

def first_assistant_msg(ex):
    """Extract first assistant message regardless of schema format."""
    # Try messages field first, then data
    msgs = ex.get("messages") or ex.get("data") or []
    if not msgs:
        return None

    m0 = msgs[0]

    if isinstance(m0, dict):
        # {role, content} format (like ultrachat_200k)
        for m in msgs:
            if "assistant" in str(m.get("role", "")).lower():
                return m.get("content") or m.get("text") or ""
    else:
        # Plain string format: alternating [user, assistant, user, assistant, ...]
        # assistant is at odd indices (1, 3, 5, ...)
        if len(msgs) > 1:
            return str(msgs[1])

    return None


def main():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"HF_HOME           = {os.environ['HF_HOME']}")
    print(f"HF_DATASETS_CACHE = {os.environ['HF_DATASETS_CACHE']}")
    print(f"DATA_ROOT         = {DATA_ROOT}")
    print(f"Dataset           = {DATASET_NAME}")
    print(f"Splits            = {SPLITS_TO_DOWNLOAD}")
    print("-" * 60)

    for split in SPLITS_TO_DOWNLOAD:
        print(f"\n=== Downloading split: {split} ===")
        ds = load_dataset(DATASET_NAME, split=split)
        print(f"  -> {len(ds)} examples")
        print(f"  -> columns: {ds.column_names}")

        # Print raw first example to understand schema
        ex = ds[0]
        msgs = ex.get("messages") or ex.get("data") or []
        print(f"  -> msgs type: {type(msgs[0]) if msgs else 'empty'}")
        print(f"  -> num turns in ex[0]: {len(msgs)}")

        content = first_assistant_msg(ex)
        if content:
            print(f"  -> first assistant msg preview:\n     {content[:300]}")
        else:
            print(f"  -> [could not extract assistant msg — raw ex[0]]: {str(ex)[:300]}")

        out_dir = DATA_ROOT / split
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  -> saving to: {out_dir}")
        ds.save_to_disk(str(out_dir))

    print("\nDone.")


if __name__ == "__main__":
    main()