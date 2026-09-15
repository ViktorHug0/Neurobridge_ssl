#!/usr/bin/env python3
import argparse
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def sanitize_tag(value):
    value = str(value).replace('.', 'p').replace('-', 'm')
    return value

def parse_config(config_str):
    """
    Extracts hyperparameters from the config string.
    Example: ...saw0p96_k3_tau0p1_steps14_pow1p0_iters14
    """
    patterns = {
        "saw_shrink": r"saw([\dp]+)",
        "k": r"k(\d+)",
        "tau": r"tau([\dp]+)",
        "steps": r"steps(\d+)",
        "pow": r"pow([\dp]+)",
        "iters": r"iters(\d+)"
    }
    params = {}
    for p_name, pattern in patterns.items():
        match = re.search(pattern, config_str)
        if match:
            val = match.group(1).replace('p', '.')
            params[p_name] = float(val) if '.' in val else int(val)
    return params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--metric", type=str, default="top1 acc")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    # Filter out rows without the metric
    df = df.dropna(subset=[args.metric])
    df[args.metric] = pd.to_numeric(df[args.metric])

    # Parse all configs
    parsed_params = df["config"].apply(parse_config).apply(pd.Series)
    df = pd.concat([df, parsed_params], axis=1)

    # Find the anchor (best config)
    best_idx = df[args.metric].idxmax()
    anchor = df.loc[best_idx]
    
    print(f"Anchor Configuration (Best {args.metric}):")
    for p in ["saw_shrink", "k", "tau", "steps", "pow", "iters"]:
        print(f"  {p}: {anchor[p]}")
    print(f"  {args.metric}: {anchor[args.metric]}%")

    os.makedirs(args.output_dir, exist_ok=True)

    params_to_sweep = ["saw_shrink", "k", "tau", "steps", "pow", "iters"]
    
    for p_sweep in params_to_sweep:
        # Fix all other parameters
        others = [p for p in params_to_sweep if p != p_sweep]
        mask = True
        for p_other in others:
            mask &= (df[p_other] == anchor[p_other])
        
        sweep_df = df[mask].sort_values(p_sweep)
        
        if sweep_df.empty:
            print(f"Warning: No data found for sweep of {p_sweep}")
            continue

        plt.figure(figsize=(8, 5))
        plt.plot(sweep_df[p_sweep], sweep_df[args.metric], marker='o', linestyle='-', linewidth=2, markersize=8)
        
        # Highlight the anchor point
        plt.plot(anchor[p_sweep], anchor[args.metric], marker='*', color='red', markersize=15, label='Best Config')
        
        plt.title(f"Sensitivity Analysis: {p_sweep}", fontsize=14)
        plt.xlabel(p_sweep, fontsize=12)
        plt.ylabel(f"{args.metric} (%)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plot
        out_name = f"sensitivity_{p_sweep}.png"
        plt.savefig(os.path.join(args.output_dir, out_name), dpi=300)
        plt.close()
        print(f"Saved sensitivity plot for {p_sweep} to {out_name}")

if __name__ == "__main__":
    main()
