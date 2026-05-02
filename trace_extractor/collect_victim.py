# src/collect_victim.py
import signal
import sys
import os
import gc
import ctypes
from llama_cpp import Llama

# ================== CONFIG ==================

TIMING_FILE = "/dev/shm/victim_times.csv"
SHM_TOKEN_PATH = "/dev/shm/target_token"
VICTIM_CPU = 15  # must be HT-sibling of attacker CPU

# ============================================

# --- TIMING SETUP (CLOCK_MONOTONIC_RAW) ---

CLOCK_MONOTONIC_RAW = 4


class timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


# Load libc to access clock_gettime
libc = ctypes.CDLL("libc.so.6", use_errno=True)


def now_ns_raw() -> int:
    ts = timespec()
    if libc.clock_gettime(CLOCK_MONOTONIC_RAW, ctypes.byref(ts)) != 0:
        # On error just return 0 (should not happen in practice)
        return 0
    return ts.tv_sec * 1_000_000_000 + ts.tv_nsec


def read_cpu_mhz() -> float:
    """Approximate CPU MHz from /proc/cpuinfo (for cycles_est)."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.lower().startswith("cpu mhz"):
                    return float(line.split(":")[1].strip())
    except Exception:
        pass
    return 0.0


CPU_MHZ = read_cpu_mhz()


def log_timing(token: int, start_ns: int, end_ns: int) -> None:
    duration = end_ns - start_ns
    if CPU_MHZ > 0:
        cycles_est = int((duration * CPU_MHZ) / 1000.0)
    else:
        cycles_est = 0

    # Token,v_start_ns,v_end_ns,duration_ns,cycles_est
    with open(TIMING_FILE, "a") as f:
        f.write(f"{token},{start_ns},{end_ns},{duration},{cycles_est}\n")


# ------------------------------------------


def bind_cpu(cpu_id: int) -> None:
    try:
        os.sched_setaffinity(0, {cpu_id})
    except Exception:
        # Not fatal – just means we couldn't pin the process
        pass


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]

    # Pin to a specific CPU (HT sibling of attacker)
    bind_cpu(VICTIM_CPU)

    # Initialize timing file with header
    with open(TIMING_FILE, "w") as f:
        f.write("Token,v_start_ns,v_end_ns,duration_ns,cycles_est\n")

    # Load LLaMA tokenizer-only model
    llm = Llama(
        model_path=model_path,
        vocab_only=True,
        use_mmap=True,
        verbose=False,
        logits_all=False,
        Llama_disable_cache=True,  # keep behavior consistent & avoid KV reuse
    )
    print(f"[Victim] Model loaded from {model_path}")
    print(f"[Victim] Logging timings to {TIMING_FILE}")

    # Reduce noise from GC
    gc.disable()

    def handle_sigusr1(signum, frame):
        try:
            # Read current target token from shm
            with open(SHM_TOKEN_PATH, "r") as f:
                content = f.read().strip()

            if content == "STOP":
                print("[Victim] Received STOP. Exiting.")
                sys.exit(0)

            t_id = int(content)

            # --- critical section: measure detokenize ---
            t_start = now_ns_raw()
            llm.detokenize([t_id])
            t_end = now_ns_raw()
            # --------------------------------------------

            log_timing(t_id, t_start, t_end)

        except SystemExit:
            raise
        except Exception as e:
            # Swallow errors to avoid killing the victim on a bad token
            # (but print once so you can debug if needed)
            print(f"[Victim] Error in handler: {e}", file=sys.stderr)

    # Register signal handler
    signal.signal(signal.SIGUSR1, handle_sigusr1)

    # Optional: write a "ready" file for external orchestration/debugging
    try:
        with open("/dev/shm/victim_ready", "w") as f:
            f.write(str(os.getpid()))
    except Exception:
        pass

    print("[Victim] Ready. Waiting for SIGUSR1...")
    # Main loop: sleep until signals arrive
    while True:
        signal.pause()


if __name__ == "__main__":
    main()
