# LLM Token Recovery via Cache Side-Channel — Data Collection

This repository contains the data-collection artifact for our CCS submission on
recovering tokens from a locally running LLM via a cache side-channel attack on
`llama_detokenize`. It combines a **Flush+Reload** trigger on the detokenize
call with an **L1 Prime+Probe** measurement on the HT-sibling core to capture a
per-token cache footprint.

The artifact produces a labeled dataset
(`token_id`, `rep`, `set_0` … `set_63`) suitable for training a token
classifier, plus a separate timing log used to validate that the probe window
actually overlaps the victim's detokenize call.

## File layout

```
.
├── Makefile                 # Builds the attacker binary
├── collect_attacker.c       # Attacker: F+R trigger + L1 P+P measurement
├── collect_victim.py        # Victim: loads tokenizer, detokenizes on SIGUSR1
└── analyze_timings.py       # (Optional) Pairs attacker/victim timestamps
```

## Notice
Important thing to notice is that the paths might not be correct and ready to ran as is.
The main most important thing to take about this repository is how we extract the traces
from the LLM.

## Prerequisites

We assume the following are already installed and available on the system:

- **Mastik** (Flush+Reload primitives — `libmastik`)
- **CacheSC** (L1 Prime+Probe primitives — `libcachesc`, headers)
- `gcc`, `make`, `python3` (≥ 3.10)
- Python package: `llama-cpp-python` (must expose `libllama.so`)
- A GGUF model file. The paper uses **Phi-3-mini-4k-instruct (q4)**.

The attack relies on shared physical resources, so the attacker and victim
**must run on hyper-threaded sibling cores** of the same physical core. Confirm
the sibling pair on your machine with:

```bash
cat /sys/devices/system/cpu/cpu*/topology/thread_siblings_list
```

## Configuration (must edit before building/running)

A handful of values in the source are machine-, model-, or build-specific.
Replace each placeholder below with a value appropriate to your setup.

### `Makefile`

```makefile
CACHESC_DIR ?= <PATH_TO_CACHESC>
```

Override on the command line if you prefer not to edit the file:

```bash
make CACHESC_DIR=/opt/CacheSC
```

### `collect_attacker.c`

| Macro            | What to set it to                                                                       |
|------------------|-----------------------------------------------------------------------------------------|
| `LIBLLAMA_PATH`  | Absolute path to `libllama.so` inside your `llama-cpp-python` install.                  |
| `MON_OFFS`       | Offset of `llama_detokenize` inside that `libllama.so`.                                 |
| `CPU_ATTACKER`   | Logical CPU id to pin the attacker to (HT sibling of `VICTIM_CPU`).                     |
| `VICTIM_CPU`     | Logical CPU id to pin the victim to (HT sibling of `CPU_ATTACKER`).                     |
| `VOCAB_SIZE`     | Vocabulary size of the model under attack (Phi-3-mini = `32064`).                       |
| `REPS`           | Number of measurements to collect per token (default `50`).                             |
| `FR_THRESH`, `AFTER_FR_DELAY_NS`, `FR_MAX_WAIT_NS` | Empirically tuned per CPU — see paper §[X].           |


### `collect_victim.py`

| Constant      | What to set it to                                          |
|---------------|------------------------------------------------------------|
| `VICTIM_CPU`  | Must match `VICTIM_CPU` in `collect_attacker.c`.           |

The victim's model path is passed as a command-line argument (see below), so
no edit is needed for that.

## Build

```bash
make CACHESC_DIR=/path/to/CacheSC
```

Produces `../bin/collect_attacker` (the Makefile writes one directory up; adjust
`BIN_DIR` if you'd like it elsewhere).

## Running the collection

### 1. Start the victim

```bash
nohup python3 collect_victim.py <PATH_TO_MODEL_GGUF> \
    > logs/victim.log 2>&1 &
```

For the paper:

```bash
nohup python3 collect_victim.py \
    /path/to/models/phi3-mini/Phi-3-mini-4k-instruct-q4.gguf \
    > logs/victim.log 2>&1 &
```

The victim pins itself to `VICTIM_CPU`, loads the tokenizer in `vocab_only`
mode, and waits for `SIGUSR1`. When ready it writes its PID to
`/dev/shm/victim_ready`.

### 2. Start the attacker

```bash
victim_pid=$(cat /dev/shm/victim_ready)
nohup ./bin/collect_attacker "$victim_pid" > logs/attacker.log 2>&1 &
```

The attacker pins itself to `CPU_ATTACKER`, sets up F+R on `llama_detokenize`
and L1 P+P with CacheSC, then iterates over all token ids `[0, VOCAB_SIZE)`,
collecting `REPS` measurements per token.

### Outputs

| File                                      | Producer | Contents                                                |
|-------------------------------------------|----------|---------------------------------------------------------|
| `data/dataset_top50_per_token.csv`        | Attacker | `token_id, rep, set_0 … set_63` — the main dataset.     |
| `/dev/shm/attacker_times.csv`             | Attacker | Per-rep prime / FR-detect / probe-start / probe-end ns. |
| `/dev/shm/victim_times.csv`               | Victim   | Per-rep `detokenize` start/end ns + estimated cycles.   |
| `/dev/shm/target_token`                   | Both     | One-byte channel: current target token id, or `STOP`.   |

Expected runtime is roughly proportional to `VOCAB_SIZE × REPS`; for
Phi-3-mini-4k (`32064 × 50`) on our machines the full sweep takes several
hours.

## (Optional) Timing analysis

`analyze_timings.py` cross-references the attacker and victim timing CSVs to
verify that the probe window actually overlapped the victim's detokenize call.
It is not required to reproduce the dataset, but is useful for validation.

```bash
python3 analyze_timings.py
```

Reads `/dev/shm/attacker_times.csv` and `/dev/shm/victim_times.csv` and writes
`dataset_timing_analysis.csv`. Each row records the pairing decision
(`max_overlap`, `nearest_time`, or `unpaired`) and the overlap statistics. A
high `max_overlap` rate indicates a well-synchronised collection.

## Cleanup

```bash
# Stop the victim cleanly (the attacker does this on normal exit)
echo STOP > /dev/shm/target_token
kill -SIGUSR1 "$(cat /dev/shm/victim_ready)"

# Or, if the attacker exited abnormally:
pkill -f collect_victim.py
rm -f /dev/shm/target_token /dev/shm/victim_ready \
      /dev/shm/attacker_times.csv /dev/shm/victim_times.csv
```

## Troubleshooting

- **Many "Missed detection for token …" lines** — `FR_THRESH` or
  `FR_MAX_WAIT_NS` likely need tuning for your CPU; see the paper for the
  procedure we used.
- **Attacker and victim not on HT siblings** — re-check
  `thread_siblings_list`. Without HT co-location the L1 P+P signal will not
  carry the victim's footprint.
- **`/dev/shm/victim_ready` missing** — the victim hasn't finished loading the
  model yet. Wait a few seconds and re-read it.