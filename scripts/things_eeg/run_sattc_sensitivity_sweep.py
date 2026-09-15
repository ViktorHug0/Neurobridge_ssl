#!/usr/bin/env python3
import argparse
import os
import subprocess
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_run_dir", required=True, type=str, help="Path to seed 3300 source run (e.g. results/.../param_..._seed3300)")
    parser.add_argument("--repo_root", required=True, type=str)
    parser.add_argument("--output_root", required=True, type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    args = parser.parse_args()

    # Anchor (Best) Hyperparameters
    anchor = {
        "saw_shrink": 0.94,
        "csls_k": 3,
        "sinkhorn_tau": 0.1,
        "sinkhorn_iters": 12,
        "soft_steps": 16,
        "soft_power": 1.2
    }

    # Define sweep ranges for each parameter
    sweeps = {
        "saw_shrink": [0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0],
        "csls_k": [1, 3, 5, 10, 15, 20],
        "sinkhorn_tau": [0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16],
        "sinkhorn_iters": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        "soft_steps": [2, 4, 6,8, 10,12, 14, 16, 18, 20, 22],
        "soft_power": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    }

    os.makedirs(args.output_root, exist_ok=True)
    
    # Find all subject checkpoint directories in the source run
    # Format is usually: source_run_dir/timestamp-sub-XX
    sub_dirs = sorted(glob.glob(os.path.join(args.source_run_dir, "*-sub-*")))
    if not sub_dirs:
        print(f"No subject directories found in {args.source_run_dir}")
        return

    python_exe = os.path.join(args.repo_root, ".venv", "bin", "python3")
    if not os.path.exists(python_exe):
        python_exe = "python3"

    all_sweep_results = []

    for param_name, values in sweeps.items():
        print(f"\n>>> Sweeping {param_name}...")
        param_results = []
        
        for val in values:
            # Prepare current params (anchor with one modification)
            current_params = anchor.copy()
            current_params[param_name] = val
            
            tag = f"sweep_{param_name}_{str(val).replace('.', 'p')}"
            temp_output_dir = os.path.join(args.output_root, "temp_evals", tag)
            os.makedirs(temp_output_dir, exist_ok=True)
            
            subject_accs = []
            for sub_path in sub_dirs:
                sub_id_match = os.path.basename(sub_path).split("-sub-")[-1]
                sub_id = int(sub_id_match)
                sub_label = f"sub-{sub_id:02d}"
                done = glob.glob(os.path.join(temp_output_dir, f"*-{sub_label}", "result.csv"))
                if done:
                    subject_accs.append(float(pd.read_csv(done[0])["top1 acc"].iloc[0]))
                    continue
                # Handle checkpoint name (prefer test_best, fallback to last)
                best_pth = os.path.join(sub_path, "checkpoint_test_best.pth")
                last_pth = os.path.join(sub_path, "checkpoint_last.pth")
                
                # We use a symlink trick if evaluate.py is rigid about the name
                target_pth = best_pth
                created_symlink = False
                if not os.path.exists(best_pth) and os.path.exists(last_pth):
                    os.symlink(last_pth, best_pth)
                    created_symlink = True
                
                cmd = [
                    python_exe, os.path.join(args.repo_root, "evaluate.py"),
                    "--checkpoint_dir", sub_path,
                    "--output_dir", temp_output_dir,
                    "--output_name", sub_label,
                    "--eval_mode", "saw_csls",
                    "--test_subject_id", str(sub_id),
                    "--batch_size", "1024",
                    "--num_workers", "4",
                    "--device", args.device,
                    "--sattc_saw_shrink", str(current_params["saw_shrink"]),
                    "--sattc_csls_k", str(current_params["csls_k"]),
                    "--sattc_sinkhorn_tau", str(current_params["sinkhorn_tau"]),
                    "--sattc_sinkhorn_iters", str(current_params["sinkhorn_iters"]),
                    "--sattc_soft_procrustes_steps", str(current_params["soft_steps"]),
                    "--sattc_soft_procrustes_power", str(current_params["soft_power"])
                ]
                
                # Add flags if non-zero
                if current_params["sinkhorn_iters"] > 0:
                    cmd.append("--sattc_sinkhorn")
                if current_params["soft_steps"] > 0:
                    cmd.append("--sattc_soft_procrustes")

                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    # Read result
                    res_file = glob.glob(os.path.join(temp_output_dir, f"*-{sub_label}", "result.csv"))[0]
                    res_df = pd.read_csv(res_file)
                    subject_accs.append(float(res_df["top1 acc"].iloc[0]))
                except Exception as e:
                    print(f"Error evaluating {sub_label} for {param_name}={val}: {e}")
                finally:
                    if created_symlink:
                        os.unlink(best_pth)
            
            if subject_accs:
                avg_acc = np.mean(subject_accs)
                param_results.append({"value": val, "acc": avg_acc})
                print(f"  {param_name}={val} -> Avg Acc: {avg_acc:.2f}%")

        if param_results:
            pdf = pd.DataFrame(param_results)
            all_sweep_results.append({"param": param_name, "data": pdf})
            
            # Plot
            plt.figure(figsize=(8, 5))
            plt.plot(pdf["value"], pdf["acc"], marker='o', linewidth=2)
            # Mark anchor
            plt.plot(anchor[param_name if param_name != "csls_k" else "csls_k"], 
                     pdf[pdf["value"] == anchor[param_name if param_name != "csls_k" else "csls_k"]]["acc"].iloc[0], 
                     marker='*', color='red', markersize=12, label='Anchor')
            
            plt.title(f"Sensitivity: {param_name}")
            plt.xlabel(param_name)
            plt.ylabel("Top-1 Acc (%)")
            plt.ylim(60, 70)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(args.output_root, f"sensitivity_{param_name}.png"), dpi=300)
            plt.close()

    print(f"\nAll sensitivity plots saved to {args.output_root}")

if __name__ == "__main__":
    main()
