#!/usr/bin/env python3
import argparse
import os
import subprocess
import glob
import pandas as pd
import numpy as np
from datetime import datetime
import itertools

def main():
    parser = argparse.ArgumentParser(description="Sweep TTA grid on existing intra-subject runs.")
    parser.add_argument("--repo_root", default="/nasbrain/p20fores/Neurobridge_SSL", type=str)
    parser.add_argument("--source_dir", default="/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/intra-subjects/TTA", type=str, help="Directory containing per-subject run folders")
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--subjects", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    args = parser.parse_args()

    if args.output_root is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_root = os.path.join(args.repo_root, "results", "things_eeg", "intra-subjects", f"tta_grid_sweep_{timestamp}")
    
    os.makedirs(args.output_root, exist_ok=True)
    
    python_exe = os.path.join(args.repo_root, ".venv", "bin", "python3")
    if not os.path.exists(python_exe):
        python_exe = "python3"

    # TTA Hyperparameters Grid
    tta_grid = {
        "saw_shrink": [0.95],
        "csls_k": [3],
        "sinkhorn_tau": [0.1],
        "sinkhorn_iters": [12],
        "soft_steps": [16],
        "soft_power": [1.2]
    }

    # Find relevant subject directories in source_dir
    subject_dirs = []
    for sub_id in args.subjects:
        sub_label = f"sub-{sub_id:02d}"
        matches = glob.glob(os.path.join(args.source_dir, f"*-{sub_label}"))
        if matches:
            # Take the most recent if multiple matches
            subject_dirs.append(max(matches, key=os.path.getmtime))
        else:
            print(f"Warning: No directory found for {sub_label} in {args.source_dir}")

    if not subject_dirs:
        print("No subject directories found. Exiting.")
        return

    # Generate grid combinations
    keys, values = zip(*tta_grid.items())
    grid_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"\n>>> Starting TTA Grid Sweep on {len(subject_dirs)} subjects...")
    print(f">>> {len(grid_combinations)} combinations per subject.")
    
    all_results = []

    for sub_path in subject_dirs:
        sub_id_str = sub_path.split("-sub-")[-1]
        sub_id = int(sub_id_str)
        sub_label = f"sub-{sub_id:02d}"
        
        print(f"\n>>> Processing TTA grid for {sub_label}...")

        # Read Baseline Performance (without TTA)
        baseline_csv = os.path.join(sub_path, "result.csv")
        baseline_top1 = np.nan
        baseline_top5 = np.nan
        if os.path.exists(baseline_csv):
            try:
                base_df = pd.read_csv(baseline_csv)
                baseline_top1 = float(base_df["top1 acc"].iloc[0])
                baseline_top5 = float(base_df["top5 acc"].iloc[0])
            except Exception as e:
                print(f"Error reading baseline result for {sub_label}: {e}")

        for combo in grid_combinations:
            combo_tag = "-".join([f"{k}_{str(v).replace('.', 'p')}" for k, v in combo.items()])
            output_name = f"tta-{sub_label}-{combo_tag}"
            
            # TTA Evaluation command
            cmd = [
                python_exe, os.path.join(args.repo_root, "evaluate.py"),
                "--checkpoint_dir", sub_path,
                "--output_dir", os.path.join(args.output_root, "tta_grid"),
                "--output_name", output_name,
                "--eval_mode", "saw_csls",
                "--test_subject_id", str(sub_id),
                "--batch_size", "1024",
                "--num_workers", "4",
                "--device", args.device,
                "--feature_dim", "512", # From evaluate_config.json
                "--sattc_saw_shrink", str(combo["saw_shrink"]),
                "--sattc_csls_k", str(combo["csls_k"]),
                "--sattc_sinkhorn",
                "--sattc_sinkhorn_tau", str(combo["sinkhorn_tau"]),
                "--sattc_sinkhorn_iters", str(combo["sinkhorn_iters"])
            ]
            
            if combo["soft_steps"] > 0:
                cmd.extend([
                    "--sattc_soft_procrustes",
                    "--sattc_soft_procrustes_steps", str(combo["soft_steps"]),
                    "--sattc_soft_procrustes_power", str(combo["soft_power"])
                ])

            try:
                # Use capture_output=True to keep logs clean, but check=True to catch errors
                subprocess.run(cmd, check=True, capture_output=True)
                
                # Find result
                res_dirs = glob.glob(os.path.join(args.output_root, "tta_grid", f"*-{output_name}"))
                if res_dirs:
                    res_dir = max(res_dirs, key=os.path.getmtime)
                    res_file = os.path.join(res_dir, "result.csv")
                    if os.path.exists(res_file):
                        res_df = pd.read_csv(res_file)
                        tta_top1 = float(res_df["top1 acc"].iloc[0])
                        tta_top5 = float(res_df["top5 acc"].iloc[0])
                        
                        res_entry = {
                            "subject": sub_label,
                            "baseline_top1": baseline_top1,
                            "baseline_top5": baseline_top5,
                            "tta_top1": tta_top1,
                            "tta_top5": tta_top5,
                            "improvement_top1": tta_top1 - baseline_top1
                        }
                        res_entry.update(combo)
                        all_results.append(res_entry)
            except subprocess.CalledProcessError as e:
                print(f"Error in TTA combo {combo_tag} for {sub_label}: {e.stderr.decode()}")
            except Exception as e:
                print(f"Unexpected error for {sub_label}: {e}")

    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = os.path.join(args.output_root, "tta_grid_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        # Report best TTA per subject
        best_per_sub = summary_df.sort_values("tta_top1", ascending=False).groupby("subject").head(1)
        print("\n=== Best TTA Hyperparameters per Subject ===")
        print(best_per_sub.to_string(index=False))
        
        # Report overall best hyperparameters
        avg_per_combo = summary_df.groupby(list(tta_grid.keys()))["tta_top1"].mean().reset_index().sort_values("tta_top1", ascending=False)
        print("\n=== Best TTA Hyperparameters (Average across subjects) ===")
        print(avg_per_combo.head(10).to_string(index=False))
        
        print(f"\nFull grid results saved to: {summary_path}")
    else:
        print("No results were generated.")

if __name__ == "__main__":
    main()
