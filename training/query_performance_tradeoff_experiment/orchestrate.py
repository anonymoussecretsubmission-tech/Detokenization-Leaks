#!/usr/bin/env python3
# orchestrate.py
#
# Job grid: strategy × Q  (same Q_LIST as the original experiment)
#
# Usage:
#   python orchestrate.py \
#       --machine-type laptop \
#       --job-id 0
#
#   # Run only the pairs listed in DO_CONFIGS["rerun_failures"]:
#   python orchestrate.py \
#       --machine-type laptop \
#       --job-id 0 \
#       --do-config rerun_failures
#
# Workers are assigned jobs satisfying: global_job_index % MAX_NODES == job_id

import os
import json
import socket
import subprocess
from pathlib import Path
import argparse


# ─────────────────────────────────────────────────────────────────────────────
# Globals — must stay in sync with 1_build_profiles_csv_synthetic.py
# ─────────────────────────────────────────────────────────────────────────────

MAX_NODES = 4
K         = 64

STRATEGIES = ["uniform", "threshold", "oracle"]
Q_LIST = [25, 50, 75, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000]

# ─────────────────────────────────────────────────────────────────────────────
# Configs to skip due to known bugs or bad data.
# Each entry is a (strategy, q) pair. Jobs are NOT reshuffled — the assigned
# worker simply skips the entry and does one fewer task.
# ─────────────────────────────────────────────────────────────────────────────

SKIP_CONFIGS = [
    # ("uniform", 500),   # example: remove comment to activate
]

# ─────────────────────────────────────────────────────────────────────────────
# Explicit do-configs (activated via --do-config <name>).
# When a named config is selected it fully replaces the STRATEGIES × Q_LIST
# grid. Job assignment (idx % MAX_NODES == job_id) is computed over the
# do-config pairs in the order they are listed, so each worker gets a
# deterministic, non-overlapping subset — exactly as with the full grid.
# SKIP_CONFIGS still applies on top of whatever do-config is active.
# ─────────────────────────────────────────────────────────────────────────────

DO_CONFIGS: dict[str, list[tuple[str, int]]] = {
    "rerun_failures": [
        ("uniform",   500),
        ("uniform",   3000),
        ("threshold", 500),
        ("threshold",   3000),
        ("oracle", 500),
        ("oracle",   3000),
    ]
}

PYTHON_BIN     = "python3"
TRAIN_SCRIPT   = Path("./train_first_cost_effective.py")
EVAL_SCRIPT    = Path("./eval_first_cost_effective.py")
CHECKPOINT_TAG = "final"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Distributed orchestrator for synthetic profiling experiment"
    )
    p.add_argument("--machine-type", required=True, choices=["laptop", "desktop"])
    p.add_argument(
        "--process-type",
        default="singleprocess",
        choices=["singleprocess", "multiprocess"],
    )
    p.add_argument(
        "--job-id",
        required=True,
        type=int,
        help=f"Worker partition id in [0, {MAX_NODES})",
    )
    p.add_argument(
        "--do-config",
        default=None,
        choices=list(DO_CONFIGS.keys()) or None,
        metavar="CONFIG_NAME",
        help=(
            "If set, replace the full STRATEGIES × Q_LIST grid with the named "
            f"entry from DO_CONFIGS. Available: {list(DO_CONFIGS.keys())}"
        ),
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Job enumeration
# ─────────────────────────────────────────────────────────────────────────────

def build_jobs(pairs=None):
    """Build the job list from strategy×Q_LIST, or from explicit (strategy, q) pairs."""
    if pairs is not None:
        return [
            {"job_name": f"{strategy}__q_{q}", "strategy": strategy, "q": q}
            for strategy, q in pairs
        ]
    jobs = []
    for strategy in STRATEGIES:
        for q in Q_LIST:
            jobs.append({
                "job_name": f"{strategy}__q_{q}",
                "strategy": strategy,
                "q":        q,
            })
    return jobs


def get_jobs_for_worker(job_id: int, pairs=None):
    jobs = build_jobs(pairs)
    assigned = [job for idx, job in enumerate(jobs) if idx % MAX_NODES == job_id]
    return jobs, assigned


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_root(machine_type: str, process_type: str) -> Path:
    return Path(
        f"/data/llamacpp/"
        f"ultrachat_cluster_cost_effective/{machine_type}_{process_type}"
    )


def get_profile_csv(machine_type, process_type, strategy, q) -> Path:
    return _synthetic_root(machine_type, process_type) / strategy / f"q_{q}" / "profile_traces.csv"


def get_symbols_dir(machine_type, process_type, strategy, q) -> Path:
    return _synthetic_root(machine_type, process_type) / f"K_{K}" / strategy / f"q_{q}" / "first"


def get_run_dir(machine_type, process_type, strategy, q) -> Path:
    return (
        _synthetic_root(machine_type, process_type)
        / f"flan_t5_xl_ultrachat_K{K}_runs"
        / strategy
        / f"q_{q}"
        / "run_first_sentences"
    )


def get_eval_summary_path(machine_type, process_type, strategy, q) -> Path:
    eval_dir = (
        Path("./eval_test_synthetic")
        / f"{machine_type}_{process_type}"
        / strategy
        / f"q_{q}"
    )
    eval_tag = (
        f"dataset-ultrachat_"
        f"{process_type}_{machine_type}_"
        f"firstsentences_framework-llamacpp_target-phi_"
        f"strategy-{strategy}_q-{q}"
    )
    return eval_dir / f"{eval_tag}_{CHECKPOINT_TAG}_summary.json"


def get_final_checkpoint_dir(machine_type, process_type, strategy, q) -> Path:
    return get_run_dir(machine_type, process_type, strategy, q) / "checkpoints" / CHECKPOINT_TAG


def get_done_marker_path(machine_type, process_type, strategy, q) -> Path:
    return get_run_dir(machine_type, process_type, strategy, q) / "orchestrator_done.json"


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def run_cmd(cmd, env=None):
    print("\n[RUN]", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True, env=env)


def write_done_marker(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not (0 <= args.job_id < MAX_NODES):
        raise ValueError(f"--job-id must be in [0, {MAX_NODES}), got {args.job_id}")

    machine_type = args.machine_type
    process_type = args.process_type

    # Resolve the job grid: explicit do-config or full STRATEGIES × Q_LIST
    do_config_pairs = None
    if args.do_config is not None:
        if args.do_config not in DO_CONFIGS:
            raise ValueError(
                f"--do-config '{args.do_config}' not found in DO_CONFIGS. "
                f"Available: {list(DO_CONFIGS.keys())}"
            )
        do_config_pairs = DO_CONFIGS[args.do_config]
        print(f"[DO_CONFIG] '{args.do_config}' active — grid overridden with "
              f"{len(do_config_pairs)} explicit pair(s).")

    all_jobs, assigned_jobs = get_jobs_for_worker(args.job_id, pairs=do_config_pairs)

    print(f"MAX_NODES={MAX_NODES}  total_jobs={len(all_jobs)}  assigned={len(assigned_jobs)}")
    for j in assigned_jobs:
        print(f"  {j['job_name']}")

    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    host = socket.gethostname()

    for pos, job in enumerate(assigned_jobs, start=1):
        strategy = job["strategy"]
        q        = job["q"]
        job_name = job["job_name"]

        print(f"\n{'='*100}")
        print(f"[WORKER {args.job_id}] Job {pos}/{len(assigned_jobs)}: {job_name}")

        # ── 0. Skip explicitly broken configs ─────────────────────────────
        if (strategy, q) in SKIP_CONFIGS:
            print(f"[SKIP] {job_name} is in SKIP_CONFIGS — skipping.")
            continue

        # ── 1. Ensure profile_traces.csv exists (built by step 1) ─────────
        profile_csv = get_profile_csv(machine_type, process_type, strategy, q)
        if not profile_csv.exists():
            print(f"[SKIP] profile_traces.csv missing: {profile_csv}")
            print(f"       Run 1_build_profiles_csv_synthetic.py first.")
            continue

        # ── 2. Ensure symbols exist (built by step 2) ─────────────────────
        symbols_dir = get_symbols_dir(machine_type, process_type, strategy, q)
        if not symbols_dir.exists():
            print(f"[SKIP] symbols dir missing: {symbols_dir}")
            print(f"       Run 2_prepare_symbols.py for strategy={strategy} q={q}.")
            continue

        eval_summary = get_eval_summary_path(machine_type, process_type, strategy, q)
        done_marker  = get_done_marker_path(machine_type, process_type, strategy, q)

        if eval_summary.exists():
            print(f"[SKIP] eval summary already exists: {eval_summary}")
            if not done_marker.exists():
                write_done_marker(done_marker, {
                    "status":       "done",
                    "job_name":     job_name,
                    "strategy":     strategy,
                    "q":            q,
                    "host":         host,
                    "note":         "done marker written after detecting existing eval summary",
                })
            continue

        # ── 3. Train ──────────────────────────────────────────────────────
        final_ckpt = get_final_checkpoint_dir(machine_type, process_type, strategy, q)

        if not final_ckpt.exists():
            train_cmd = [
                PYTHON_BIN, str(TRAIN_SCRIPT),
                "--machine-type", machine_type,
                "--process-type", process_type,
                "--method",       strategy,   # train script still uses --method
                "--q",            str(q),
            ]
            run_cmd(train_cmd, env=env)
        else:
            print(f"[SKIP] final checkpoint exists: {final_ckpt}")

        # ── 4. Eval ────────────────────────────────────────────────────────
        eval_cmd = [
            PYTHON_BIN, str(EVAL_SCRIPT),
            "--machine-type",   machine_type,
            "--process-type",   process_type,
            "--method",         strategy,
            "--q",              str(q),
            "--checkpoint-tag", CHECKPOINT_TAG,
        ]
        run_cmd(eval_cmd, env=env)

        write_done_marker(done_marker, {
            "status":         "done",
            "job_name":       job_name,
            "machine_type":   machine_type,
            "process_type":   process_type,
            "strategy":       strategy,
            "q":              int(q),
            "checkpoint_tag": CHECKPOINT_TAG,
            "eval_summary":   str(eval_summary),
            "worker_job_id":  int(args.job_id),
            "host":           host,
        })

    print("\nAll assigned jobs completed.")


if __name__ == "__main__":
    main()