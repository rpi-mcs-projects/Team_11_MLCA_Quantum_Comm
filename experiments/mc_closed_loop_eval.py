#!/usr/bin/env python3
import subprocess
import re
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm  # pip install tqdm

# Configuration
NUM_TRIALS = 50
OUTPUT_FILE = "results/monte_carlo_results.csv"

# The base command (minus the seed)
BASE_CMD = [
    "uv", "run", "experiments/closed_loop_backtest.py",
    "--npz", "./data/amp_sweep_bandpower_bins2_40.npz",
    "--budget", "20",
    "--seed-points", "5",
    "--lambda-xtalk", "0.5",
    "--norm-mode", "oracle",
    "--explore-first-k-steps", "2",
    "--refine-last-k-steps", "2",
    "--proposals", "5",
    "--eval-grid", "2001",
    "--global-scan-grid", "401",
    "--device", "cpu"
]

def parse_output(output_str, seed):
    """
    Parses the 'Overall Summary' block from the backtest output.
    Returns a list of dictionaries (one per qubit).
    """
    # Regex to capture: q{k}, online_amp, full_amp, amp_err, util_full_online
    # Looks for lines like: 
    # q0: online_amp=0.123456 full_amp=0.123456 amp_err=0.000000 util_full(online)=+1.234 ...
    
    # We look for floating point numbers that might be negative or positive
    float_re = r"([-+]?\d*\.\d+|\d+)"
    
    pattern = re.compile(
        r"q(\d+): "
        r"online_amp=" + float_re + r"\s+"
        r"full_amp=" + float_re + r"\s+"
        r"amp_err=" + float_re + r"\s+"
        r"util_full\(online\)=" + float_re
    )

    results = []
    for line in output_str.splitlines():
        match = pattern.search(line)
        if match:
            q_idx, on_amp, full_amp, err, util = match.groups()
            results.append({
                "seed": seed,
                "qubit": int(q_idx),
                "online_amp": float(on_amp),
                "full_amp": float(full_amp),
                "amp_err": float(err),
                "final_utility": float(util)
            })
    return results

def main():
    all_data = []
    
    print(f"Starting Monte Carlo Simulation ({NUM_TRIALS} trials)...")
    print(f"Command: {' '.join(BASE_CMD)} --seed <seed>")

    # Run the loop with a progress bar
    for seed in tqdm(range(NUM_TRIALS)):
        cmd = BASE_CMD + ["--seed", str(seed)]
        
        try:
            # Run the subprocess and capture stdout
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse the output
            parsed = parse_output(result.stdout, seed)
            
            if not parsed:
                print(f"Warning: No results found for seed {seed}. Did the script crash?")
                # Optional: print(result.stderr)
            
            all_data.extend(parsed)

        except subprocess.CalledProcessError as e:
            print(f"\nError running seed {seed}:")
            print(e.stderr)
            continue
        except KeyboardInterrupt:
            print("\nStopping early...")
            break

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    if df.empty:
        print("No data collected.")
        return

    # --- Analysis & Output ---

    # Group by qubit to see spread per channel
    stats = df.groupby("qubit")["amp_err"].agg(
        Count='count',
        Mean='mean',
        Std='std',
        Min='min',
        Max='max',
        Median='median'
    )

    print("\n" + "="*60)
    print("MONTE CARLO RESULTS: AMPLITUDE ERROR SPREAD")
    print("="*60)
    print(stats.round(6))
    print("\n")

    # Overall aggregate (across all qubits and seeds)
    total_mean = df["amp_err"].mean()
    total_std = df["amp_err"].std()
    print(f"Global Average Error: {total_mean:.6f} ± {total_std:.6f}")

    # Save to CSV for plotting
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nFull dataset saved to: {OUTPUT_FILE}")
    print("Format: seed, qubit, online_amp, full_amp, amp_err, final_utility")

if __name__ == "__main__":
    main()
