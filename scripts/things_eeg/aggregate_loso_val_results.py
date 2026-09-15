import os
import pandas as pd
import argparse
import glob
import subprocess
import re

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_root', required=True, type=str)
    parser.add_argument('--repo_root', required=True, type=str)
    parser.add_argument('--source_run_root', required=True, type=str)
    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--output_csv', default="loso_val_selected_summary.csv", type=str)
    args = parser.parse_args()

    source_dirs = [d for d in glob.glob(os.path.join(args.run_root, "*")) if os.path.isdir(d)]
    all_final_results = []

    for source_dir in source_dirs:
        source_label = os.path.basename(source_dir)
        tag_dirs = [d for d in glob.glob(os.path.join(source_dir, "*")) if os.path.isdir(d)]
        
        # subject_best[sub_id] = { "best_val_acc": -1, "best_tag": None, "params": {} }
        subject_best = {i: {"best_val_acc": -1.0, "best_tag": None} for i in range(1, 11)}

        for tag_dir in tag_dirs:
            tag_name = os.path.basename(tag_dir)
            
            for sub_id in range(1, 11):
                val_id = (sub_id % 10) + 1
                val_csv = glob.glob(os.path.join(tag_dir, f"*-val-sub-{val_id:02d}", "result.csv"))
                
                if not val_csv:
                    continue
                
                val_df = pd.read_csv(val_csv[0])
                val_acc = float(val_df["top1 acc"].iloc[0])
                
                if val_acc > subject_best[sub_id]["best_val_acc"]:
                    subject_best[sub_id]["best_val_acc"] = val_acc
                    subject_best[sub_id]["best_tag"] = tag_name

        # Now, for each subject, run the final TEST evaluation using its best tag
        source_results = []
        for sub_id, data in subject_best.items():
            if data["best_tag"] is None:
                continue
            
            tag_name = data["best_tag"]
            # Extract parameters from tag_name (e.g. ...saw0p85_k5_tau0p1_steps4_pow1p0_iters12)
            # We skip the source_label part which might contain similar patterns (like k30)
            tag_suffix = tag_name.replace(source_label + "_", "")
            params = {}
            patterns = {
                "saw_shrink": r"saw([\dp]+)",
                "csls_k": r"k(\d+)",
                "sinkhorn_tau": r"tau([\dp]+)",
                "soft_steps": r"steps(\d+)",
                "soft_power": r"pow([\dp]+)",
                "sinkhorn_iters": r"iters(\d+)"
            }
            for p_name, pattern in patterns.items():
                match = re.search(pattern, tag_suffix)
                if match:
                    val = match.group(1).replace('p', '.')
                    params[p_name] = val

            test_output_name = f"sub-{sub_id:02d}"
            checkpoint_dir_search = glob.glob(os.path.join(args.source_run_root, source_label, f"*-{test_output_name}"))
            if not checkpoint_dir_search:
                print(f"Warning: Could not find checkpoint directory for {test_output_name} in {os.path.join(args.source_run_root, source_label)}")
                continue
            checkpoint_dir = checkpoint_dir_search[0]

            final_test_dir = os.path.join(args.run_root, source_label, "final_val_selected_test")
            os.makedirs(final_test_dir, exist_ok=True)

            import sys
            python_exe = sys.executable
            # If we are not in a venv, try to find one in repo_root
            if not hasattr(sys, 'real_prefix') and not sys.base_prefix != sys.prefix:
                venv_python = os.path.join(args.repo_root, ".venv", "bin", "python3")
                if os.path.exists(venv_python):
                    python_exe = venv_python

            cmd = [
                python_exe, os.path.join(args.repo_root, "evaluate.py"),
                "--checkpoint_dir", checkpoint_dir,
                "--output_dir", final_test_dir,
                "--output_name", test_output_name,
                "--eval_mode", "saw_csls",
                "--test_subject_id", str(sub_id),
                "--batch_size", str(args.batch_size),
                "--num_workers", str(args.num_workers),
                "--device", args.device,
                "--sattc_saw_shrink", params["saw_shrink"],
                "--sattc_csls_k", params["csls_k"],
                "--sattc_sinkhorn",
                "--sattc_sinkhorn_tau", params["sinkhorn_tau"],
                "--sattc_sinkhorn_iters", params["sinkhorn_iters"],
                "--sattc_soft_procrustes",
                "--sattc_soft_procrustes_steps", params["soft_steps"],
                "--sattc_soft_procrustes_power", params["soft_power"]
            ]
            subprocess.run(cmd, check=True)

            # Load the result
            res_path = glob.glob(os.path.join(final_test_dir, f"*-{test_output_name}", "result.csv"))[0]
            res_df = pd.read_csv(res_path)
            row = res_df.iloc[0].to_dict()
            row["sub"] = f"sub-{sub_id:02d}"
            row["source_run"] = source_label
            row["val_best_tag"] = tag_name
            row["val_best_acc"] = data["best_val_acc"]
            source_results.append(row)

        if source_results:
            df = pd.DataFrame(source_results)
            numeric_cols = ["top1 acc", "top5 acc", "best top1 acc", "best top5 acc", "best test loss", "val_best_acc"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col])
            
            avg_row = df[numeric_cols].mean().to_dict()
            avg_row["sub"] = "Average"
            avg_row["source_run"] = source_label
            avg_row["val_best_tag"] = "N/A"
            avg_row["architecture"] = df["architecture"].iloc[0] if "architecture" in df.columns else ""
            avg_row["eval_mode"] = df["eval_mode"].iloc[0] if "eval_mode" in df.columns else ""
            
            df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
            all_final_results.append(df)

    if all_final_results:
        final_df = pd.concat(all_final_results, ignore_index=True)
        out_path = os.path.join(args.run_root, args.output_csv)
        final_df.to_csv(out_path, index=False)
        print(f"Final aggregated results saved to {out_path}")
        print(final_df[final_df["sub"] == "Average"])
    else:
        print("No results found to aggregate.")

if __name__ == "__main__":
    main()
