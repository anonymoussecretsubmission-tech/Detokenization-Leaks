#!/usr/bin/env python3
"""
2_extract_sentences_ultrachat.py
"""

import os
import csv
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer

# ---------------- CONFIG ----------------
DATA_ROOT = Path("../data")

LOCAL_SPLITS = {
    "train_sft": DATA_ROOT / "raw" / "train"
}

PHI_TOKENIZER_NAME = os.environ.get(
    "PHI_TOKENIZER_NAME", "microsoft/Phi-3-mini-4k-instruct"
)

SPLIT_AFTER_TOKENS = 32

OUT_FIRST_CSV  = DATA_ROOT / "ultrachat_first_sentences.csv"
OUT_MIDDLE_CSV = DATA_ROOT / "ultrachat_middle_sentences.csv"

MIN_LEN_CHARS = 10
# ----------------------------------------


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def split_by_tokens_phi(tokenizer, text: str, split_after: int):
    text = normalize_whitespace((text or "").strip())
    if not text:
        return "", ""

    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    ids = enc["input_ids"]
    if len(ids) <= split_after:
        return text, ""

    first_part  = tokenizer.decode(ids[:split_after], skip_special_tokens=True).strip()
    middle_part = tokenizer.decode(ids[split_after:],  skip_special_tokens=True).strip()

    return normalize_whitespace(first_part), normalize_whitespace(middle_part)


def process_split(
    split_name: str,
    dataset_path: Path,
    w_first,
    w_mid,
    stats: dict,
    tokenizer,
    row_idx_counter: list,   # single-element list used as a mutable int
):
    print(f"\n=== Processing split: {split_name} ===")
    print(f"  -> loading from disk: {dataset_path}")

    ds = load_from_disk(str(dataset_path))
    print(f"  -> loaded {len(ds)} dialogues")

    # ── Detect schema on first example ──────────────────────────────────────
    # HuggingFaceH4/ultrachat_200k  → field "messages", list of {role, content}
    # HuggingFaceTB/ultrachat_*     → field "data",     list of plain strings
    #                                  even indices = user, odd indices = assistant
    ex0       = ds[0]
    use_data_field = "data" in ex0 and "messages" not in ex0
    if use_data_field:
        print("  -> schema: plain-string turns in 'data' field")
    else:
        print("  -> schema: {role,content} dicts in 'messages' field")
    # ────────────────────────────────────────────────────────────────────────

    num_dialogues      = 0
    num_assistant_msgs = 0
    num_first_kept     = 0
    num_middle_kept    = 0

    for dialogue_idx, ex in enumerate(ds):
        num_dialogues += 1

        # ── Extract turns depending on schema ────────────────────────────────
        if use_data_field:
            # plain strings: [user0, asst0, user1, asst1, ...]
            turns = ex.get("data", [])
            # prompt_id not present in this dataset — use empty string
            prompt_id   = ex.get("id", ex.get("prompt_id", ""))
            # user turn is index 0, assistant turn is index 1
            user_prompt     = normalize_whitespace(turns[0]) if len(turns) > 0 else ""
            assistant_content = turns[1]                     if len(turns) > 1 else ""
            msg_idx = 1   # assistant is always at index 1 (first assistant turn)
        else:
            # {role, content} dicts
            messages  = ex.get("messages", [])
            prompt_id = ex.get("prompt_id", "")
            user_prompt       = ""
            assistant_content = ""
            msg_idx           = -1
            for i, msg in enumerate(messages):
                role = msg.get("role", "")
                if role == "user":
                    user_prompt = normalize_whitespace(msg.get("content", "") or "")
                elif role == "assistant":
                    assistant_content = msg.get("content", "") or ""
                    msg_idx = i
                    break
        # ────────────────────────────────────────────────────────────────────

        if not assistant_content:
            continue

        num_assistant_msgs += 1

        first_part, middle_part = split_by_tokens_phi(
            tokenizer=tokenizer,
            text=assistant_content,
            split_after=SPLIT_AFTER_TOKENS,
        )

        # Assign and increment the global row_idx for this assistant message
        row_idx = row_idx_counter[0]
        row_idx_counter[0] += 1

        # Write first row
        if len(first_part) >= MIN_LEN_CHARS:
            w_first.writerow([
                split_name,
                row_idx,
                prompt_id,
                dialogue_idx,
                msg_idx,
                user_prompt,
                first_part,
            ])
            num_first_kept += 1

        # Write middle row (only if there is a middle part)
        if middle_part and len(middle_part) >= MIN_LEN_CHARS:
            w_mid.writerow([
                split_name,
                row_idx,
                prompt_id,
                dialogue_idx,
                msg_idx,
                user_prompt,
                first_part,    # context_sentence
                middle_part,
            ])
            num_middle_kept += 1

    stats[split_name] = {
        "dialogues":       num_dialogues,
        "assistant_msgs":  num_assistant_msgs,
        "kept_first_rows": num_first_kept,
        "kept_middle_rows": num_middle_kept,
    }
    print(
        f"  -> dialogues: {num_dialogues}, "
        f"assistant msgs: {num_assistant_msgs}, "
        f"kept first rows: {num_first_kept}, "
        f"kept middle rows: {num_middle_kept}"
    )

def main():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"DATA_ROOT           = {DATA_ROOT}")
    print(f"PHI_TOKENIZER_NAME  = {PHI_TOKENIZER_NAME}")
    print(f"SPLIT_AFTER_TOKENS  = {SPLIT_AFTER_TOKENS}")
    print(f"Output FIRST CSV    = {OUT_FIRST_CSV}")
    print(f"Output MIDDLE CSV   = {OUT_MIDDLE_CSV}")
    print("-" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        PHI_TOKENIZER_NAME,
        use_fast=True,
        trust_remote_code=True,
    )

    stats = {}
    row_idx_counter = [0]   # mutable counter shared across all splits

    with OUT_FIRST_CSV.open("w", encoding="utf-8", newline="") as f_first, \
         OUT_MIDDLE_CSV.open("w", encoding="utf-8", newline="") as f_mid:

        w_first = csv.writer(f_first, quoting=csv.QUOTE_ALL, escapechar='\\')
        w_mid   = csv.writer(f_mid,   quoting=csv.QUOTE_ALL, escapechar='\\')
        # row_idx is second column, right after split — matches chatdoctor convention
        w_first.writerow([
            "split", "row_idx", "prompt_id", "dialogue_idx", "message_idx",
            "prompt", "first_sentence",
        ])
        w_mid.writerow([
            "split", "row_idx", "prompt_id", "dialogue_idx", "message_idx",
            "prompt", "context_sentence", "middle_sentence",
        ])

        for split_name, path in LOCAL_SPLITS.items():
            if not path.exists():
                print(f"WARNING: path for split '{split_name}' does not exist: {path}")
                continue
            process_split(
                split_name, path, w_first, w_mid, stats, tokenizer, row_idx_counter
            )

    total_assistant = sum(s["assistant_msgs"]   for s in stats.values())
    total_first     = sum(s["kept_first_rows"]  for s in stats.values())
    total_middle    = sum(s["kept_middle_rows"] for s in stats.values())

    print("\nDone. Summary:")
    for split_name, s in stats.items():
        print(
            f"  {split_name}: dialogues={s['dialogues']}, "
            f"assistant_msgs={s['assistant_msgs']}, "
            f"kept_first={s['kept_first_rows']}, "
            f"kept_middle={s['kept_middle_rows']}"
        )
    print(f"\n  Total assistant messages : {total_assistant}")
    print(f"  Total row_idx assigned   : {row_idx_counter[0]}  "
          f"(== total assistant messages)")
    print(f"  Total first rows written : {total_first}")
    print(f"  Total middle rows written: {total_middle}")
    print(f"  Sanity: middle <= first  : {total_middle <= total_first}")
    print(f"\nFirst CSV  -> {OUT_FIRST_CSV}")
    print(f"Middle CSV -> {OUT_MIDDLE_CSV}")


if __name__ == "__main__":
    main()
