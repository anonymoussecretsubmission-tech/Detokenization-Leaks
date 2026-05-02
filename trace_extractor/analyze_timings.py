#!/usr/bin/env python3
import csv
import sys
import os
from collections import defaultdict

# === CONFIGURATION ===
ATTACKER_CSV = "/dev/shm/attacker_times.csv"
VICTIM_CSV   = "/dev/shm/victim_times.csv"
OUTPUT_CSV   = "dataset_timing_analysis.csv"

# Time limit to consider a victim event "paired" with an attacker event (1 second)
MAX_PAIRING_DELTA_NS = 1_000_000_000 

def load_csv(filepath):
    data = []
    if not os.path.exists(filepath):
        print(f"[!] File not found: {filepath}")
        return []
    
    with open(filepath, 'r') as f:
        # Strip whitespace from header keys
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            clean_row = {}
            for k, v in row.items():
                if k is None: continue
                key = k.strip()
                try:
                    clean_row[key] = int(v)
                except ValueError:
                    clean_row[key] = v
            data.append(clean_row)
    return data

def analyze():
    print("--- Loading Data ---")
    attacker_data = load_csv(ATTACKER_CSV)
    victim_data = load_csv(VICTIM_CSV)

    print(f"Loaded {len(attacker_data)} attacker rows.")
    print(f"Loaded {len(victim_data)} victim rows.")

    # Group victim events by token
    victim_map = defaultdict(list)
    for v in victim_data:
        # Handle case sensitivity "token" vs "Token"
        t = v.get('token') if 'token' in v else v.get('Token')
        victim_map[t].append(v)

    # Sort victim events chronologically
    for t in victim_map:
        victim_map[t].sort(key=lambda x: x['v_start_ns'])

    merged_rows = []
    
    # Track used victim events to avoid double pairing (simple greedy approach)
    used_victim_indices = set()

    print("\n--- Pairing Events ---")
    
    for i, att in enumerate(attacker_data):
        token = att.get('token')
        
        # Extract Attacker Timestamps
        t_prime = att.get('prime_time_ns', 0)
        t_fr    = att.get('fr_detect_time_ns', 0)
        t_ps    = att.get('probe_start_time_ns', 0)
        t_pe    = att.get('probe_end_time_ns', 0)
        
        candidates = victim_map.get(token, [])
        
        best_vic = None
        pairing_decision = "no_victim"
        
        # 1. First pass: Look for overlapping victim events
        # Logic: Valid overlap implies causality
        best_overlap = -1
        best_cand_idx = -1
        
        for idx, vic in enumerate(candidates):
            # Unique ID for set tracking: token_index
            uid = (token, idx)
            if uid in used_victim_indices:
                continue

            v_start = vic['v_start_ns']
            v_end   = vic['v_end_ns']
            
            # Check overlap: max(start_a, start_b) < min(end_a, end_b)
            ov_start = max(t_ps, v_start)
            ov_end   = min(t_pe, v_end)
            ov = ov_end - ov_start
            
            if ov > 0 and ov > best_overlap:
                best_overlap = ov
                best_cand_idx = idx

        if best_cand_idx != -1:
            best_vic = candidates[best_cand_idx]
            used_victim_indices.add((token, best_cand_idx))
            pairing_decision = "max_overlap"
        
        # 2. Second pass: If no overlap, find nearest neighbour (failed sync/missed)
        else:
            min_dist = float('inf')
            best_cand_idx = -1
            
            for idx, vic in enumerate(candidates):
                uid = (token, idx)
                if uid in used_victim_indices:
                    continue
                
                v_start = vic['v_start_ns']
                dist = abs(t_fr - v_start)
                
                if dist < min_dist and dist < MAX_PAIRING_DELTA_NS:
                    min_dist = dist
                    best_cand_idx = idx
            
            if best_cand_idx != -1:
                best_vic = candidates[best_cand_idx]
                used_victim_indices.add((token, best_cand_idx))
                pairing_decision = "nearest_time"

        # --- Build Output Row ---
        row = {}
        
        # Attacker basics
        row["token"] = token
        row["prime_time_ns"] = t_prime
        row["fr_detect_time_ns"] = t_fr
        row["probe_start_time_ns"] = t_ps
        row["probe_end_time_ns"] = t_pe
        row["probe_window_duration_ns"] = t_pe - t_ps

        if best_vic:
            v_start = best_vic['v_start_ns']
            v_end   = best_vic['v_end_ns']
            v_dur   = best_vic.get('duration_ns', v_end - v_start)
            
            row["victim_detokenize_start_time_ns"] = v_start
            row["victim_detokenize_end_time_ns"] = v_end
            row["victim_detokenize_duration_ns"] = v_dur
            row["victim_detokenize_cycles_estimated"] = best_vic.get('cycles_est', 0)
            
            # Calculations
            ov_start = max(t_ps, v_start)
            ov_end   = min(t_pe, v_end)
            overlap_ns = max(0, ov_end - ov_start)
            
            row["overlap_between_probe_and_victim_ns"] = overlap_ns
            row["probe_coverage_of_victim_ratio"] = (overlap_ns / v_dur) if v_dur > 0 else 0
            
            row["victim_start_minus_probe_start_ns"] = v_start - t_ps
            row["victim_start_minus_fr_detect_ns"] = v_start - t_fr
            row["victim_end_minus_fr_detect_ns"] = v_end - t_fr
            row["pairing_decision"] = pairing_decision
        else:
            # Fill with empty/zero if unpaired
            row["victim_detokenize_start_time_ns"] = ""
            row["victim_detokenize_end_time_ns"] = ""
            row["victim_detokenize_duration_ns"] = ""
            row["victim_detokenize_cycles_estimated"] = ""
            row["overlap_between_probe_and_victim_ns"] = 0
            row["probe_coverage_of_victim_ratio"] = 0
            row["victim_start_minus_probe_start_ns"] = ""
            row["victim_start_minus_fr_detect_ns"] = ""
            row["victim_end_minus_fr_detect_ns"] = ""
            row["pairing_decision"] = "unpaired"

        merged_rows.append(row)

    # --- SAVE CSV ---
    fieldnames = [
        "token",
        "prime_time_ns",
        "fr_detect_time_ns",
        "probe_start_time_ns",
        "probe_end_time_ns",
        "victim_detokenize_start_time_ns",
        "victim_detokenize_end_time_ns",
        "victim_detokenize_duration_ns",
        "victim_detokenize_cycles_estimated",
        "probe_window_duration_ns",
        "overlap_between_probe_and_victim_ns",
        "probe_coverage_of_victim_ratio",
        "victim_start_minus_probe_start_ns",
        "victim_start_minus_fr_detect_ns",
        "victim_end_minus_fr_detect_ns",
        "pairing_decision",
    ]

    with open(OUTPUT_CSV, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)
        
    print(f"\n[+] Analysis Complete. Saved {len(merged_rows)} rows to: {OUTPUT_CSV}")
    
    # Quick Stat
    valid = sum(1 for r in merged_rows if r['pairing_decision'] == 'max_overlap')
    print(f"    Successful Overlap Pairing: {valid}/{len(merged_rows)} ({valid/len(merged_rows)*100:.1f}%)")

if __name__ == "__main__":
    analyze()