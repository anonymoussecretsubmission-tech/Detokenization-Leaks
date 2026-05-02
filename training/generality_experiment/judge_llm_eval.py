#!/usr/bin/env python3
# judge_llm_eval.py
#
# Cost-efficient ultrachat reconstruction judging with the OpenAI Batch API.
#
# Stages:
#   1) prepare : build tasks and write batch input JSONL
#   2) submit  : upload JSONL (purpose="batch") and create batch on /v1/responses
#   3) collect : poll/retrieve batch, download output, convert to judged JSONL
#
# Why this is cheaper:
#   - Batch API pricing is lower than synchronous requests
#   - prompt caching applies automatically for repeated prompt prefixes
#   - user prompt is shortened
#   - text is compacted before sending
#
# Examples:
#   python judge_llm_eval.py prepare --machine-type laptop --framework llamacpp --model llama --scope full
#   python judge_llm_eval.py submit  --machine-type laptop --framework llamacpp --model llama --scope full
#   python judge_llm_eval.py collect --machine-type laptop --framework llamacpp --model llama --scope full
#   python judge_llm_eval.py submit-all
#   python judge_llm_eval.py submit-all --dry-run
#   python judge_llm_eval.py collect-all
#   python judge_llm_eval.py collect-all --dry-run
#
# With --limit N, the synchronous API is used (no batch job) and output filenames
# include "_limitN" so they never collide with full-run outputs:
#   python judge_llm_eval.py prepare --machine-type laptop --framework llamacpp --model llama --scope full --limit 20
#   python judge_llm_eval.py submit  --machine-type laptop --framework llamacpp --model llama --scope full --limit 20
#   python judge_llm_eval.py submit-all --limit 20
#
# Optional:
#   python judge_llm_eval.py status  --batch-id batch_123
#
# Notes:
#   - Requires OPENAI_API_KEY in the environment.
#   - Uses /v1/responses as the batch endpoint, as documented for Batch API input files.
#   - The batch input file is uploaded with purpose="batch".
#

import os
import re
import json
import time
import argparse
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

from tqdm import tqdm


SYSTEM_PROMPT = (
    "You are a privacy-leakage evaluator for reconstructed responses obtained from side-channel observations of a reference language model. "
    "The goal of the reconstruction is not exact correctness, but to reveal nontrivial and accurate information about the original reference response. "
    "Do not judge whether the predicted response is well-written, stylistically polished, or fully factually perfect. "
    "Instead, judge whether it reveals meaningful and accurate information about the reference response. "
    "Answer Yes if the prediction captures the same topic, intent, key ideas, structure, reasoning, or distinctive content of the reference, "
    "even if it is partial, noisy, disorganized, or contains unrelated text. "
    "Generic or vague responses alone are not sufficient; the prediction must reflect specific and identifiable content from the reference. "
    "A partially corrupted reconstruction still counts as leakage if a human reader could infer nontrivial and correct information about the original response from it. "
    "Answer No only if the prediction is mostly unrelated or does not reveal meaningful and correct information about the reference. "
    "Output only Yes or No."
)

JUDGE_MODEL = "gpt-4.1-mini"

DATASET = "ultrachat"
PROCESS = "singleprocess"

FIRST_SCOPE = "firstsentences"
MIDDLE_SCOPE = "middlesentences"

BATCH_DIR = Path("./eval/batch_jobs")

ALL_MACHINE_TYPES = ["laptop", "desktop"]
ALL_FRAMEWORKS    = ["llamacpp", "huggingface"]
ALL_MODELS        = ["phi", "llama"]
ALL_SCOPES        = ["first", "full"]


# ── file helpers ──────────────────────────────────────────────────────────────

def eval_tag(machine_type: str, scope: str, framework: str, target: str) -> str:
    return (
        f"dataset-{DATASET}_{PROCESS}_{machine_type}_{scope}"
        f"_framework-{framework}_target-{target}"
    )

def _limit_suffix(limit: Optional[int]) -> str:
    """Return '_limitN' when a limit is active, empty string otherwise."""
    return f"_limit{limit}" if limit else ""

def first_jsonl_path(machine_type: str, checkpoint_tag: str, framework: str, target: str) -> Path:
    tag = eval_tag(machine_type, FIRST_SCOPE, framework, target)
    return Path("./eval") / f"{tag}_{checkpoint_tag}.jsonl"

def middle_jsonl_path(machine_type: str, checkpoint_tag: str, framework: str, target: str) -> Path:
    tag = eval_tag(machine_type, MIDDLE_SCOPE, framework, target)
    return Path("./eval") / f"{tag}_{checkpoint_tag}.jsonl"

def judged_output_jsonl_path(
    machine_type: str, checkpoint_tag: str, framework: str, target: str,
    scope: str, limit: Optional[int] = None,
) -> Path:
    tag = eval_tag(machine_type, scope, framework, target)
    return Path("./eval") / f"{tag}_{checkpoint_tag}_{JUDGE_MODEL}_judged{_limit_suffix(limit)}.jsonl"

def batch_job_prefix(
    machine_type: str, checkpoint_tag: str, framework: str, target: str,
    scope: str, limit: Optional[int] = None,
) -> str:
    tag = eval_tag(machine_type, scope, framework, target)
    return f"{tag}_{checkpoint_tag}_{JUDGE_MODEL}{_limit_suffix(limit)}"

def batch_input_path(
    machine_type: str, checkpoint_tag: str, framework: str, target: str,
    scope: str, limit: Optional[int] = None,
) -> Path:
    return BATCH_DIR / f"{batch_job_prefix(machine_type, checkpoint_tag, framework, target, scope, limit)}.requests.jsonl"

def batch_meta_path(
    machine_type: str, checkpoint_tag: str, framework: str, target: str,
    scope: str, limit: Optional[int] = None,
) -> Path:
    return BATCH_DIR / f"{batch_job_prefix(machine_type, checkpoint_tag, framework, target, scope, limit)}.meta.json"

def batch_output_raw_path(
    machine_type: str, checkpoint_tag: str, framework: str, target: str,
    scope: str, limit: Optional[int] = None,
) -> Path:
    return BATCH_DIR / f"{batch_job_prefix(machine_type, checkpoint_tag, framework, target, scope, limit)}.output.jsonl"

def batch_error_raw_path(
    machine_type: str, checkpoint_tag: str, framework: str, target: str,
    scope: str, limit: Optional[int] = None,
) -> Path:
    return BATCH_DIR / f"{batch_job_prefix(machine_type, checkpoint_tag, framework, target, scope, limit)}.errors.jsonl"


# ── loaders ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[warn] {path.name} line {i}: {e}")
    return records


# ── text helpers ──────────────────────────────────────────────────────────────

def norm(s: Any) -> str:
    return "" if s is None else str(s).strip()

def join_segs(*parts: str) -> str:
    return "\n".join(p for p in (norm(x) for x in parts) if p)

def compact_code(s: str) -> str:
    # Cheaper input while preserving semantic leakage evidence.
    s = norm(s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def build_user_prompt(reference: str, prediction: str) -> str:
    # Keep this short and stable for cost and prompt-caching benefits.
    reference = compact_code(reference)
    prediction = compact_code(prediction)
    return f"Reference:\n{reference}\n\nPrediction:\n{prediction}"

def normalize_yes_no(text: str) -> Optional[str]:
    t = norm(text).lower()
    if t.startswith("yes"):
        return "Yes"
    if t.startswith("no"):
        return "No"
    return None

def cheap_local_verdict(pred: str, ref: str) -> Optional[str]:
    # Free fast-path for trivial cases.
    p = compact_code(pred)
    r = compact_code(ref)
    if not p or not r:
        return "No"
    if p == r:
        return "Yes"
    return None


# ── task builders ─────────────────────────────────────────────────────────────

def build_first_tasks(first_records: List[dict]) -> List[Tuple[int, str, str, str]]:
    return [
        (i, norm(r.get("pred", "")), norm(r.get("ref", "")), str(r.get("paragraph_id", "")))
        for i, r in enumerate(first_records)
    ]

def _is_real_pid(pid: str) -> bool:
    return bool(pid) and not pid.startswith("fallback_")

def build_full_tasks(
    first_records: List[dict],
    middle_records: List[dict],
) -> Tuple[List[Tuple[int, str, str, str]], str]:
    n_real = sum(1 for r in first_records if _is_real_pid(str(r.get("paragraph_id", ""))))
    use_pid_join = n_real > len(first_records) * 0.5

    if use_pid_join:
        return _join_by_paragraph_id(first_records, middle_records), "by_paragraph_id"
    return _join_by_position(first_records, middle_records), "by_position"

def _join_by_paragraph_id(
    first_records: List[dict],
    middle_records: List[dict],
) -> List[Tuple[int, str, str, str]]:
    first_by_pid: Dict[str, dict] = {
        str(r.get("paragraph_id", "")): r
        for r in first_records
        if _is_real_pid(str(r.get("paragraph_id", "")))
    }

    middle_by_pid: Dict[str, Dict[int, dict]] = defaultdict(dict)
    for r in middle_records:
        pid = str(r.get("paragraph_id", ""))
        sidx = int(r.get("segment_idx", 0))
        if pid:
            middle_by_pid[pid][sidx] = r

    tasks = []
    for idx, pid in enumerate(sorted(first_by_pid.keys())):
        first = first_by_pid[pid]
        first_pred = norm(first.get("pred", ""))
        first_ref = norm(first.get("ref", ""))

        middle = middle_by_pid.get(pid, {})
        if middle:
            sorted_sidxs = sorted(middle.keys())
            full_pred = join_segs(first_pred, *[norm(middle[s].get("pred", "")) for s in sorted_sidxs])
            full_ref = join_segs(first_ref, *[norm(middle[s].get("ref", "")) for s in sorted_sidxs])
        else:
            full_pred = first_pred
            full_ref = first_ref

        tasks.append((idx, full_pred, full_ref, pid))
    return tasks

def _join_by_position(
    first_records: List[dict],
    middle_records: List[dict],
) -> List[Tuple[int, str, str, str]]:
    middle_groups: Dict[str, Dict[int, dict]] = OrderedDict()
    for r in middle_records:
        pid = str(r.get("paragraph_id", ""))
        sidx = int(r.get("segment_idx", 0))
        if pid not in middle_groups:
            middle_groups[pid] = {}
        middle_groups[pid][sidx] = r

    middle_para_list = list(middle_groups.values())

    n_first = len(first_records)
    n_middle = len(middle_para_list)
    if n_middle != n_first:
        print(
            f"[warn] Positional join: {n_first} first records but {n_middle} middle paragraphs. "
            f"Paragraphs beyond min({n_first}, {n_middle}) will be first-only or dropped."
        )

    tasks = []
    for idx, first in enumerate(first_records):
        first_pred = norm(first.get("pred", ""))
        first_ref = norm(first.get("ref", ""))
        pid = str(first.get("paragraph_id", f"pos_{idx}"))

        if idx < n_middle:
            middle = middle_para_list[idx]
            sorted_sidxs = sorted(middle.keys())
            full_pred = join_segs(first_pred, *[norm(middle[s].get("pred", "")) for s in sorted_sidxs])
            full_ref = join_segs(first_ref, *[norm(middle[s].get("ref", "")) for s in sorted_sidxs])
        else:
            full_pred = first_pred
            full_ref = first_ref

        tasks.append((idx, full_pred, full_ref, pid))
    return tasks


# ── OpenAI helpers ────────────────────────────────────────────────────────────

def get_client():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    from openai import OpenAI
    return OpenAI()

def extract_output_text(resp_body: dict) -> str:
    # Responses API output is structured. Prefer output_text if present.
    text = resp_body.get("output_text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    parts: List[str] = []
    for item in resp_body.get("output", []) or []:
        for content in item.get("content", []) or []:
            if content.get("type") == "output_text":
                t = content.get("text", "")
                if t:
                    parts.append(t)
    return "\n".join(parts).strip()


# ── Synchronous judging (used when --limit is set) ────────────────────────────

def judge_tasks_sync(
    tasks: List[Tuple[int, str, str, str]],
    limit: int,
) -> List[Dict[str, Any]]:
    """
    Call the OpenAI API synchronously for each task (no batch job).
    Returns rows in the same format as cmd_collect would produce.
    Only called when --limit is active.
    """
    client = get_client()
    rows = []

    for idx, pred, ref, pid in tqdm(tasks, desc="Judging (sync)"):
        local = cheap_local_verdict(pred, ref)
        if local is not None:
            rows.append({
                "idx": idx,
                "paragraph_id": pid,
                "pred": pred,
                "ref": ref,
                f"{JUDGE_MODEL}_judge": local,
                "raw_judge_text": f"[local:{local}]",
            })
            continue

        try:
            response = client.responses.create(
                model=JUDGE_MODEL,
                instructions=SYSTEM_PROMPT,
                input=build_user_prompt(ref, pred),
                max_output_tokens=16,
            )
            # responses.create returns a Response object; extract text the same way
            output_text = ""
            if hasattr(response, "output_text") and response.output_text:
                output_text = response.output_text.strip()
            elif hasattr(response, "output"):
                for item in response.output or []:
                    for content in getattr(item, "content", []) or []:
                        if getattr(content, "type", None) == "output_text":
                            output_text = getattr(content, "text", "").strip()
            verdict = normalize_yes_no(output_text)
        except Exception as e:
            print(f"  [warn] sync judge failed for idx={idx}: {e}")
            output_text = None
            verdict = None

        rows.append({
            "idx": idx,
            "paragraph_id": pid,
            "pred": pred,
            "ref": ref,
            f"{JUDGE_MODEL}_judge": verdict,
            "raw_judge_text": output_text,
        })

    return rows


def run_sync_and_save(
    tasks: List[Tuple[int, str, str, str]],
    limit: int,
    machine_type: str,
    checkpoint_tag: str,
    framework: str,
    model: str,
    scope: str,
) -> Path:
    """
    Judge `tasks` synchronously and write the judged JSONL.
    The output filename includes '_limitN' to distinguish it from a full batch run.
    Returns the path of the written file.
    """
    print(f"\nLimit mode — using synchronous API for {len(tasks)} tasks (no batch job).")
    rows = judge_tasks_sync(tasks, limit)

    judged_out = judged_output_jsonl_path(machine_type, checkpoint_tag, framework, model, scope, limit)
    judged_out.parent.mkdir(parents=True, exist_ok=True)

    with judged_out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(rows)
    yes     = sum(1 for r in rows if r.get(f"{JUDGE_MODEL}_judge") == "Yes")
    no      = sum(1 for r in rows if r.get(f"{JUDGE_MODEL}_judge") == "No")
    errors  = sum(1 for r in rows if r.get("error"))
    unclear = total - yes - no - errors

    print(f"\nSYNC JUDGE DONE")
    print(f"Judged output: {judged_out}")
    print(f"Total        : {total}")
    print(f"Yes          : {yes} ({yes / total * 100:.2f}%)" if total else "Yes          : 0")
    print(f"No           : {no} ({no / total * 100:.2f}%)" if total else "No           : 0")
    print(f"Unclear      : {unclear}")
    print(f"Errors       : {errors}")

    return judged_out


# ── Batch JSONL creation ──────────────────────────────────────────────────────

def build_tasks_from_args(args) -> List[Tuple[int, str, str, str]]:
    fp = first_jsonl_path(args.machine_type, args.checkpoint_tag, args.framework, args.model)
    if not fp.exists():
        raise FileNotFoundError(f"First-sentence JSONL not found: {fp}")
    first_records = load_jsonl(fp)
    print(f"Loaded first segments : {fp.name} ({len(first_records)} records)")

    if args.scope == "full":
        mp = middle_jsonl_path(args.machine_type, args.checkpoint_tag, args.framework, args.model)
        if not mp.exists():
            raise FileNotFoundError(
                f"Middle segment JSONL not found: {mp}\n"
                f"Run eval_t5_middle.py first, or use --scope first."
            )
        middle_records = load_jsonl(mp)
        print(f"Loaded middle segments: {mp.name} ({len(middle_records)} records)")

        tasks, join_strategy = build_full_tasks(first_records, middle_records)
        print(f"Join strategy: {join_strategy}")
        avg_pred_len = sum(len(t[1]) for t in tasks) / max(len(tasks), 1)
        avg_ref_len = sum(len(t[2]) for t in tasks) / max(len(tasks), 1)
        print(f"Avg pred length: {avg_pred_len:.0f} chars | Avg ref length: {avg_ref_len:.0f} chars")
        print(f"Built {len(tasks)} full-paragraph tasks")
    else:
        tasks = build_first_tasks(first_records)
        print(f"Built {len(tasks)} first-segment tasks")

    if args.limit:
        tasks = tasks[:args.limit]
        print(f"Limited to {len(tasks)} tasks (--limit {args.limit})")
    return tasks

def write_batch_input_jsonl(tasks: List[Tuple[int, str, str, str]], path: Path) -> Tuple[int, int]:
    path.parent.mkdir(parents=True, exist_ok=True)

    n_local = 0
    n_remote = 0

    with path.open("w", encoding="utf-8") as f:
        for idx, pred, ref, pid in tqdm(tasks, desc="Preparing batch input"):
            local = cheap_local_verdict(pred, ref)
            if local is not None:
                n_local += 1
                continue

            req = {
                "custom_id": f"judge-{idx}",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": JUDGE_MODEL,
                    "instructions": SYSTEM_PROMPT,
                    "input": build_user_prompt(ref, pred),
                    "max_output_tokens": 16,
                },
            }
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
            n_remote += 1

    return n_local, n_remote

def write_meta(
    tasks: List[Tuple[int, str, str, str]],
    meta_path: Path,
    n_local: int,
    n_remote: int,
    limit: Optional[int] = None,
) -> None:
    local_map = {}
    for idx, pred, ref, pid in tasks:
        local = cheap_local_verdict(pred, ref)
        if local is not None:
            local_map[str(idx)] = {
                "paragraph_id": pid,
                "pred": pred,
                "ref": ref,
                "judge": local,
                "raw_judge_text": f"[local:{local}]",
            }

    meta = {
        "dataset": DATASET,
        "process": PROCESS,
        "judge_model": JUDGE_MODEL,
        "limit": limit,
        "n_tasks_total": len(tasks),
        "n_local": n_local,
        "n_remote": n_remote,
        "tasks": {
            str(idx): {
                "paragraph_id": pid,
                "pred": pred,
                "ref": ref,
            }
            for idx, pred, ref, pid in tasks
        },
        "local_results": local_map,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Batch submit / status / collect ───────────────────────────────────────────

def cmd_prepare(args):
    tasks = build_tasks_from_args(args)

    input_path = batch_input_path(args.machine_type, args.checkpoint_tag, args.framework, args.model, args.scope)
    meta_path  = batch_meta_path(args.machine_type, args.checkpoint_tag, args.framework, args.model, args.scope)

    n_local, n_remote = write_batch_input_jsonl(tasks, input_path)
    write_meta(tasks, meta_path, n_local, n_remote)

    print("\nPREPARE DONE")
    print(f"Batch input : {input_path}")
    print(f"Meta        : {meta_path}")
    print(f"Total tasks : {len(tasks)}")
    print(f"Local fast  : {n_local}")
    print(f"Remote batch: {n_remote}")

def cmd_submit(args):
    input_path = batch_input_path(args.machine_type, args.checkpoint_tag, args.framework, args.model, args.scope)
    meta_path  = batch_meta_path(args.machine_type, args.checkpoint_tag, args.framework, args.model, args.scope)

    if not input_path.exists():
        raise FileNotFoundError(f"Batch input JSONL not found: {input_path}\nRun prepare first.")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta JSON not found: {meta_path}\nRun prepare first.")

    client = get_client()

    with input_path.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={
            "job_name": batch_job_prefix(args.machine_type, args.checkpoint_tag, args.framework, args.model, args.scope),
            "scope": args.scope,
            "framework": args.framework,
            "target_model": args.model,
            "machine_type": args.machine_type,
        },
    )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["uploaded_input_file_id"] = uploaded.id
    meta["batch_id"] = batch.id
    meta["batch_status"] = batch.status
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nSUBMIT DONE")
    print(f"Uploaded file id: {uploaded.id}")
    print(f"Batch id        : {batch.id}")
    print(f"Initial status  : {batch.status}")
    print(f"Meta updated    : {meta_path}")

def cmd_status(args):
    client = get_client()
    batch = client.batches.retrieve(args.batch_id)

    print(f"Batch id           : {batch.id}")
    print(f"Status             : {batch.status}")
    print(f"Endpoint           : {batch.endpoint}")
    print(f"Input file id      : {batch.input_file_id}")
    print(f"Output file id     : {getattr(batch, 'output_file_id', None)}")
    print(f"Error file id      : {getattr(batch, 'error_file_id', None)}")
    print(f"Created at         : {getattr(batch, 'created_at', None)}")
    print(f"Completed at       : {getattr(batch, 'completed_at', None)}")
    print(f"Completion window  : {batch.completion_window}")

def _download_file_content(client, file_id: str) -> bytes:
    content = client.files.content(file_id)
    data = content.read()
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        return data.encode("utf-8")
    return bytes(data)

def cmd_collect(args):
    meta_path = batch_meta_path(args.machine_type, args.checkpoint_tag, args.framework, args.model, args.scope)
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta JSON not found: {meta_path}\nRun prepare and submit first.")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    batch_id = meta.get("batch_id")
    if not batch_id:
        raise RuntimeError(f"No batch_id in {meta_path}\nRun submit first.")

    client = get_client()
    batch = client.batches.retrieve(batch_id)

    print(f"Batch {batch.id} status: {batch.status}")

    if batch.status not in {"completed", "failed", "expired", "cancelled"}:
        print("Batch is not terminal yet. Re-run collect later.")
        return

    output_path = batch_output_raw_path(args.machine_type, args.checkpoint_tag, args.framework, args.model, args.scope)
    error_path  = batch_error_raw_path(args.machine_type, args.checkpoint_tag, args.framework, args.model, args.scope)

    if getattr(batch, "output_file_id", None):
        raw = _download_file_content(client, batch.output_file_id)
        output_path.write_bytes(raw)
        print(f"Saved raw output JSONL: {output_path}")

    if getattr(batch, "error_file_id", None):
        raw = _download_file_content(client, batch.error_file_id)
        error_path.write_bytes(raw)
        print(f"Saved raw error JSONL : {error_path}")

    tasks_meta    = meta["tasks"]
    local_results = meta.get("local_results", {})

    final_rows: Dict[int, Dict[str, Any]] = {}

    # Insert local fast-path decisions first
    for idx_str, row in local_results.items():
        idx = int(idx_str)
        final_rows[idx] = {
            "idx": idx,
            "paragraph_id": row["paragraph_id"],
            "pred": row["pred"],
            "ref": row["ref"],
            f"{JUDGE_MODEL}_judge": row["judge"],
            "raw_judge_text": row["raw_judge_text"],
        }

    # Parse batch output if present
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[warn] output parse line {line_no}: {e}")
                    continue

                custom_id = rec.get("custom_id", "")
                try:
                    idx = int(custom_id.replace("judge-", ""))
                except Exception:
                    print(f"[warn] unexpected custom_id: {custom_id}")
                    continue

                body = (((rec.get("response") or {}).get("body")) or {})
                output_text = extract_output_text(body)
                verdict = normalize_yes_no(output_text)

                task = tasks_meta[str(idx)]
                final_rows[idx] = {
                    "idx": idx,
                    "paragraph_id": task["paragraph_id"],
                    "pred": task["pred"],
                    "ref": task["ref"],
                    f"{JUDGE_MODEL}_judge": verdict,
                    "raw_judge_text": output_text,
                }

    # Mark remote failures from the error file if present
    if error_path.exists():
        with error_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[warn] error parse line {line_no}: {e}")
                    continue

                custom_id = rec.get("custom_id", "")
                try:
                    idx = int(custom_id.replace("judge-", ""))
                except Exception:
                    continue

                if idx in final_rows:
                    continue

                task = tasks_meta[str(idx)]
                err  = rec.get("error") or {}
                final_rows[idx] = {
                    "idx": idx,
                    "paragraph_id": task["paragraph_id"],
                    "pred": task["pred"],
                    "ref": task["ref"],
                    f"{JUDGE_MODEL}_judge": None,
                    "raw_judge_text": None,
                    "error": json.dumps(err, ensure_ascii=False),
                }

    # Fill any missing rows defensively
    for idx_str, task in tasks_meta.items():
        idx = int(idx_str)
        if idx not in final_rows:
            final_rows[idx] = {
                "idx": idx,
                "paragraph_id": task["paragraph_id"],
                "pred": task["pred"],
                "ref": task["ref"],
                f"{JUDGE_MODEL}_judge": None,
                "raw_judge_text": None,
                "error": "Missing from batch output and local results",
            }

    judged_out = judged_output_jsonl_path(
        args.machine_type, args.checkpoint_tag, args.framework, args.model, args.scope
    )
    judged_out.parent.mkdir(parents=True, exist_ok=True)

    ordered = [final_rows[i] for i in sorted(final_rows.keys())]
    with judged_out.open("w", encoding="utf-8") as f:
        for row in ordered:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total   = len(ordered)
    yes     = sum(1 for r in ordered if r.get(f"{JUDGE_MODEL}_judge") == "Yes")
    no      = sum(1 for r in ordered if r.get(f"{JUDGE_MODEL}_judge") == "No")
    errors  = sum(1 for r in ordered if r.get("error"))
    unclear = total - yes - no - errors

    meta["batch_status"] = batch.status
    meta["output_file_id"] = getattr(batch, "output_file_id", None)
    meta["error_file_id"]  = getattr(batch, "error_file_id", None)
    meta["judged_output_jsonl"] = str(judged_out)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nCOLLECT DONE")
    print(f"Judged output: {judged_out}")
    print(f"Total        : {total}")
    print(f"Yes          : {yes} ({yes / total * 100:.2f}%)" if total else "Yes          : 0")
    print(f"No           : {no} ({no / total * 100:.2f}%)" if total else "No           : 0")
    print(f"Unclear      : {unclear}")
    print(f"Errors       : {errors}")


# ── submit-all ────────────────────────────────────────────────────────────────

def _config_label(machine_type, framework, model, scope, checkpoint_tag) -> str:
    return f"{machine_type}/{framework}/{model}/{scope} (ckpt={checkpoint_tag})"


def discover_pending_configs(checkpoint_tag: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """
    Scan all combinations of machine_type × framework × model × scope.

    A configuration is considered *ready to judge* when:
      - The first-sentence eval JSONL exists (required for both scopes).
      - For scope='full': the middle-sentence eval JSONL also exists.

    A configuration is skipped when:
      - The judged output JSONL already exists (judging already done or in flight).
      - With no limit: the batch meta file already exists and contains a batch_id
        (already submitted via batch API).
    """
    pending = []

    for machine_type in ALL_MACHINE_TYPES:
        for framework in ALL_FRAMEWORKS:
            for model in ALL_MODELS:
                for scope in ALL_SCOPES:
                    label = _config_label(machine_type, framework, model, scope, checkpoint_tag)

                    fp = first_jsonl_path(machine_type, checkpoint_tag, framework, model)
                    if not fp.exists():
                        print(f"  [skip] {label} — first-segment eval file missing: {fp.name}")
                        continue

                    if scope == "full":
                        mp = middle_jsonl_path(machine_type, checkpoint_tag, framework, model)
                        if not mp.exists():
                            print(f"  [skip] {label} — middle-segment eval file missing: {mp.name}")
                            continue

                    judged = judged_output_jsonl_path(machine_type, checkpoint_tag, framework, model, scope, limit)
                    if judged.exists():
                        print(f"  [done] {label} — judged output already exists: {judged.name}")
                        continue

                    # For batch (no-limit) runs, skip configs already submitted
                    if not limit:
                        meta_p = batch_meta_path(machine_type, checkpoint_tag, framework, model, scope)
                        if meta_p.exists():
                            try:
                                meta = json.loads(meta_p.read_text(encoding="utf-8"))
                                if meta.get("batch_id"):
                                    print(
                                        f"  [inflight] {label} — batch already submitted "
                                        f"(id={meta['batch_id']}, status={meta.get('batch_status', '?')})"
                                    )
                                    continue
                            except Exception:
                                pass  # malformed meta — treat as not submitted

                    pending.append({
                        "machine_type": machine_type,
                        "framework": framework,
                        "model": model,
                        "scope": scope,
                        "checkpoint_tag": checkpoint_tag,
                        "label": label,
                    })

    return pending


class _Namespace:
    """Minimal stand-in for argparse.Namespace used when calling cmd_* directly."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def cmd_submit_all(args):
    """
    Discover every configuration that has eval files but no judged output yet,
    then run prepare → submit (batch) or judge synchronously (limit mode) for each.
    """
    checkpoint_tag = args.checkpoint_tag
    limit          = args.limit or None
    dry_run        = args.dry_run

    print("=" * 70)
    print(f"SUBMIT-ALL  checkpoint_tag={checkpoint_tag}  limit={limit}  dry_run={dry_run}")
    print("=" * 70)
    print("\nScanning configurations …\n")

    pending = discover_pending_configs(checkpoint_tag, limit)

    if not pending:
        print("\nNothing to do — no pending configurations found.")
        return

    mode_label = f"sync (limit={limit})" if limit else "batch"
    print(f"\nFound {len(pending)} configuration(s) to process ({mode_label}):\n")
    for cfg in pending:
        print(f"  • {cfg['label']}")

    if dry_run:
        print("\n[dry-run] Stopping here. Remove --dry-run to actually submit.")
        return

    succeeded = []
    failed    = []

    for i, cfg in enumerate(pending, 1):
        label = cfg["label"]
        print(f"\n{'─' * 70}")
        print(f"[{i}/{len(pending)}] {label}")
        print("─" * 70)

        ns = _Namespace(
            machine_type=cfg["machine_type"],
            framework=cfg["framework"],
            model=cfg["model"],
            scope=cfg["scope"],
            checkpoint_tag=cfg["checkpoint_tag"],
            limit=limit,
        )

        if limit:
            # ── sync path ─────────────────────────────────────────────────────
            try:
                tasks = build_tasks_from_args(ns)
                run_sync_and_save(
                    tasks, limit,
                    ns.machine_type, ns.checkpoint_tag, ns.framework, ns.model, ns.scope,
                )
                succeeded.append(label)
            except Exception as e:
                print(f"   [ERROR] sync judge failed: {e}")
                failed.append((label, str(e)))
        else:
            # ── batch path ────────────────────────────────────────────────────
            print("\n>> PREPARE")
            try:
                tasks      = build_tasks_from_args(ns)
                input_path = batch_input_path(ns.machine_type, ns.checkpoint_tag, ns.framework, ns.model, ns.scope)
                meta_path  = batch_meta_path(ns.machine_type, ns.checkpoint_tag, ns.framework, ns.model, ns.scope)
                n_local, n_remote = write_batch_input_jsonl(tasks, input_path)
                write_meta(tasks, meta_path, n_local, n_remote)
                print(f"   Total tasks : {len(tasks)}  (local={n_local}, remote={n_remote})")
            except Exception as e:
                print(f"   [ERROR] prepare failed: {e}")
                failed.append((label, f"prepare: {e}"))
                continue

            if n_remote == 0:
                print("   All tasks resolved locally — no batch job needed.")
                try:
                    _flush_local_only(ns, meta_path)
                    succeeded.append(label)
                except Exception as e:
                    print(f"   [ERROR] local flush failed: {e}")
                    failed.append((label, f"local flush: {e}"))
                continue

            print("\n>> SUBMIT")
            try:
                cmd_submit(ns)
                succeeded.append(label)
            except Exception as e:
                print(f"   [ERROR] submit failed: {e}")
                failed.append((label, f"submit: {e}"))

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SUBMIT-ALL COMPLETE")
    print(f"  Processed : {len(succeeded)}")
    print(f"  Failed    : {len(failed)}")
    if succeeded:
        print("\n  Processed configurations:")
        for s in succeeded:
            print(f"    ✓ {s}")
    if failed:
        print("\n  Failed configurations:")
        for label, reason in failed:
            print(f"    ✗ {label}  →  {reason}")
    print("=" * 70)


def _flush_local_only(ns, meta_path: Path) -> None:
    """Write a judged JSONL for configs where every task resolved locally (n_remote == 0)."""
    meta          = json.loads(meta_path.read_text(encoding="utf-8"))
    tasks_meta    = meta["tasks"]
    local_results = meta.get("local_results", {})
    limit         = meta.get("limit") or None

    final_rows: Dict[int, Dict[str, Any]] = {}
    for idx_str, row in local_results.items():
        idx = int(idx_str)
        final_rows[idx] = {
            "idx": idx,
            "paragraph_id": row["paragraph_id"],
            "pred": row["pred"],
            "ref": row["ref"],
            f"{JUDGE_MODEL}_judge": row["judge"],
            "raw_judge_text": row["raw_judge_text"],
        }
    for idx_str, task in tasks_meta.items():
        idx = int(idx_str)
        if idx not in final_rows:
            final_rows[idx] = {
                "idx": idx,
                "paragraph_id": task["paragraph_id"],
                "pred": task["pred"],
                "ref": task["ref"],
                f"{JUDGE_MODEL}_judge": None,
                "raw_judge_text": None,
                "error": "Missing from local results",
            }

    judged_out = judged_output_jsonl_path(
        ns.machine_type, ns.checkpoint_tag, ns.framework, ns.model, ns.scope, limit
    )
    judged_out.parent.mkdir(parents=True, exist_ok=True)
    ordered = [final_rows[i] for i in sorted(final_rows.keys())]
    with judged_out.open("w", encoding="utf-8") as f:
        for row in ordered:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"   Flushed {len(ordered)} local-only rows → {judged_out.name}")


# ── collect-all ───────────────────────────────────────────────────────────────

def discover_collectable_configs(checkpoint_tag: str) -> List[Dict[str, str]]:
    """
    Collect-all only makes sense for batch (no-limit) runs.
    Limit runs are judged synchronously in submit and produce output immediately.

    A configuration is collectable when:
      - The first-sentence eval JSONL exists.
      - For scope='full': the middle-sentence eval JSONL also exists.
      - The judged output JSONL (no-limit variant) does NOT exist yet.
      - The batch meta file exists (i.e. prepare was already run).
    """
    collectable = []

    for machine_type in ALL_MACHINE_TYPES:
        for framework in ALL_FRAMEWORKS:
            for model in ALL_MODELS:
                for scope in ALL_SCOPES:
                    label = _config_label(machine_type, framework, model, scope, checkpoint_tag)

                    fp = first_jsonl_path(machine_type, checkpoint_tag, framework, model)
                    if not fp.exists():
                        print(f"  [skip] {label} — first-segment eval file missing")
                        continue

                    if scope == "full":
                        mp = middle_jsonl_path(machine_type, checkpoint_tag, framework, model)
                        if not mp.exists():
                            print(f"  [skip] {label} — middle-segment eval file missing")
                            continue

                    judged = judged_output_jsonl_path(machine_type, checkpoint_tag, framework, model, scope)
                    if judged.exists():
                        print(f"  [done] {label} — judged output already exists: {judged.name}")
                        continue

                    meta_p = batch_meta_path(machine_type, checkpoint_tag, framework, model, scope)
                    if not meta_p.exists():
                        print(f"  [skip] {label} — no meta file (run prepare/submit first)")
                        continue

                    try:
                        meta = json.loads(meta_p.read_text(encoding="utf-8"))
                    except Exception as e:
                        print(f"  [skip] {label} — could not read meta: {e}")
                        continue

                    collectable.append({
                        "machine_type": machine_type,
                        "framework": framework,
                        "model": model,
                        "scope": scope,
                        "checkpoint_tag": checkpoint_tag,
                        "label": label,
                        "batch_id": meta.get("batch_id"),
                        "batch_status": meta.get("batch_status", "?"),
                        "n_remote": meta.get("n_remote", -1),
                    })

    return collectable


def cmd_collect_all(args):
    """
    Discover every batch (no-limit) configuration that has eval files and a meta
    file but no judged output yet, then run collect on each one sequentially.
    Configs whose batch is not terminal yet are reported and skipped.
    """
    checkpoint_tag = args.checkpoint_tag
    dry_run        = args.dry_run

    print("=" * 70)
    print(f"COLLECT-ALL  checkpoint_tag={checkpoint_tag}  dry_run={dry_run}")
    print("=" * 70)
    print("\nScanning configurations …\n")

    collectable = discover_collectable_configs(checkpoint_tag)

    if not collectable:
        print("\nNothing to do — no collectable configurations found.")
        return

    print(f"\nFound {len(collectable)} configuration(s) to collect:\n")
    for cfg in collectable:
        if cfg["batch_id"]:
            print(f"  • {cfg['label']}  [batch_id={cfg['batch_id']}, status={cfg['batch_status']}]")
        else:
            print(f"  • {cfg['label']}  [local-only]")

    if dry_run:
        print("\n[dry-run] Stopping here. Remove --dry-run to actually collect.")
        return

    succeeded     = []
    still_running = []
    failed        = []

    for i, cfg in enumerate(collectable, 1):
        label = cfg["label"]
        print(f"\n{'─' * 70}")
        print(f"[{i}/{len(collectable)}] {label}")
        print("─" * 70)

        ns = _Namespace(
            machine_type=cfg["machine_type"],
            framework=cfg["framework"],
            model=cfg["model"],
            scope=cfg["scope"],
            checkpoint_tag=cfg["checkpoint_tag"],
            limit=None,
        )
        meta_p = batch_meta_path(ns.machine_type, ns.checkpoint_tag, ns.framework, ns.model, ns.scope)

        # Local-only: no batch was ever submitted, just flush
        if not cfg["batch_id"]:
            print("   No batch job — flushing local-only results.")
            try:
                _flush_local_only(ns, meta_p)
                succeeded.append(label)
            except Exception as e:
                print(f"   [ERROR] local flush failed: {e}")
                failed.append((label, str(e)))
            continue

        # Remote batch: peek at live status before delegating to cmd_collect
        try:
            client = get_client()
            batch  = client.batches.retrieve(cfg["batch_id"])
            print(f"   Batch status: {batch.status}")

            if batch.status not in {"completed", "failed", "expired", "cancelled"}:
                print("   Batch is not terminal yet — skipping (re-run collect-all later).")
                still_running.append(label)
                continue

            cmd_collect(ns)
            succeeded.append(label)
        except Exception as e:
            print(f"   [ERROR] collect failed: {e}")
            failed.append((label, str(e)))

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("COLLECT-ALL COMPLETE")
    print(f"  Collected     : {len(succeeded)}")
    print(f"  Still running : {len(still_running)}")
    print(f"  Failed        : {len(failed)}")
    if succeeded:
        print("\n  Collected configurations:")
        for s in succeeded:
            print(f"    ✓ {s}")
    if still_running:
        print("\n  Still running (re-run collect-all later):")
        for s in still_running:
            print(f"    … {s}")
    if failed:
        print("\n  Failed configurations:")
        for label, reason in failed:
            print(f"    ✗ {label}  →  {reason}")
    print("=" * 70)


# ── CLI ───────────────────────────────────────────────────────────────────────

def add_batch_eval_args(p):
    """Args shared by prepare / submit / collect (batch flow, no --limit)."""
    p.add_argument("--machine-type", required=True, choices=["laptop", "desktop"])
    p.add_argument("--framework", required=True, choices=["llamacpp", "huggingface"])
    p.add_argument("--model", "--target-model", dest="model", required=True, choices=["phi", "llama"])
    p.add_argument("--checkpoint-tag", default="final")
    p.add_argument("--scope", required=True, choices=["first", "full"])


def cmd_judge(args):
    """
    One-shot synchronous judging for a single configuration with --limit.
    Loads tasks, judges via the live API, writes output immediately.
    Output filename includes '_limit{N}' so it never collides with batch outputs.
    """
    # Re-use build_tasks_from_args by setting limit on args
    tasks = build_tasks_from_args(args)
    run_sync_and_save(
        tasks, args.limit,
        args.machine_type, args.checkpoint_tag, args.framework, args.model, args.scope,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Batch judge for ultrachat reconstruction")
    sub = p.add_subparsers(dest="command", required=True)

    # ── judge (sync, limit mode) ──────────────────────────────────────────────
    p_judge = sub.add_parser(
        "judge",
        help="Judge a limited number of tasks immediately using the synchronous API (no batch job).",
    )
    p_judge.add_argument("--machine-type", required=True, choices=["laptop", "desktop"])
    p_judge.add_argument("--framework", required=True, choices=["llamacpp", "huggingface"])
    p_judge.add_argument("--model", "--target-model", dest="model", required=True, choices=["phi", "llama"])
    p_judge.add_argument("--checkpoint-tag", default="final")
    p_judge.add_argument("--scope", required=True, choices=["first", "full"])
    p_judge.add_argument(
        "--limit", type=int, required=True,
        help="Number of tasks to judge. Output filename will include '_limit{N}'.",
    )

    # ── batch flow ────────────────────────────────────────────────────────────
    p_prepare = sub.add_parser("prepare", help="Build tasks and write Batch API input JSONL.")
    add_batch_eval_args(p_prepare)

    p_submit = sub.add_parser("submit", help="Upload input JSONL and create the batch job.")
    add_batch_eval_args(p_submit)

    p_collect = sub.add_parser("collect", help="Retrieve completed batch and build judged JSONL.")
    add_batch_eval_args(p_collect)

    p_status = sub.add_parser("status", help="Check status of a batch by ID.")
    p_status.add_argument("--batch-id", required=True)

    # ── bulk batch flow ───────────────────────────────────────────────────────
    p_submit_all = sub.add_parser(
        "submit-all",
        help="Auto-discover all configurations with eval files but no judged output, then prepare and submit each.",
    )
    p_submit_all.add_argument("--checkpoint-tag", default="final")
    p_submit_all.add_argument("--dry-run", action="store_true",
                              help="List pending configurations without doing anything.")

    p_collect_all = sub.add_parser(
        "collect-all",
        help="Auto-discover all submitted batch configurations with no judged output yet, then collect each.",
    )
    p_collect_all.add_argument("--checkpoint-tag", default="final")
    p_collect_all.add_argument("--dry-run", action="store_true",
                               help="List collectable configurations without doing anything.")

    return p.parse_args()


def main():
    args = parse_args()

    # Batch commands have no --limit; set it to None so build_tasks_from_args works
    if not hasattr(args, "limit"):
        args.limit = None

    if args.command == "judge":
        cmd_judge(args)
    elif args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "submit":
        cmd_submit(args)
    elif args.command == "collect":
        cmd_collect(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "submit-all":
        cmd_submit_all(args)
    elif args.command == "collect-all":
        cmd_collect_all(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()