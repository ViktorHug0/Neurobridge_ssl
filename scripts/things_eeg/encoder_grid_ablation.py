import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse

def parse_config(config_name):
    # Example: EVA02-E-14_layer35_mean_ATM_seed3300_mixup (training flag name unchanged on disk)
    parts = config_name.split('_')
    archs = ["EEGConformer", "ATM", "EEGNet", "EEGProject", "TSConv"]
    encoders = ["EVA02-E-14", "InternViT-6B", "ViT-bigG-14", "ViT-H-14", "dinov2-giant", "ViT-B-16"]

    found_arch = next((a for a in archs if a in parts), "Unknown")
    found_enc = next((e for e in encoders if e in parts), "Unknown")

    # Special handling for dinov2-giant which might be split or named differently
    if found_enc == "Unknown":
        if "dinov2" in config_name and "giant" in config_name:
            found_enc = "dinov2-giant"
        elif "ViT-B-16" in config_name:
            found_enc = "ViT-B-16"

    return found_enc, found_arch

def load_and_pivot(csv_path, metric, source_col='config', ignore_conformer=False):
    df = pd.read_csv(csv_path)
    if 'sub' in df.columns:
        df = df[df['sub'] == 'Average']

    data = []
    for _, row in df.iterrows():
        enc, arch = parse_config(row[source_col])
        if ignore_conformer and arch == "EEGConformer":
            continue
        data.append({'Image Encoder': enc, 'EEG Encoder': arch, 'Value': row[metric]})

    df_plot = pd.DataFrame(data)
    pivot = df_plot.pivot(index='EEG Encoder', columns='Image Encoder', values='Value')

    image_encoders = [
        "EVA02-E-14",
        "InternViT-6B",
        "ViT-bigG-14",
        "ViT-H-14",
        "dinov2-giant",
        "ViT-B-16",
    ]
    eeg_encoders = ["ATM", "EEGNet", "EEGProject", "TSConv"]
    if not ignore_conformer and "EEGConformer" in pivot.index:
        eeg_encoders = eeg_encoders + ["EEGConformer"]
    pivot = pivot.reindex(index=eeg_encoders, columns=image_encoders)

    pivot['Avg'] = pivot.mean(axis=1)
    pivot.loc['Avg'] = pivot.mean(axis=0)

    return pivot

def plot_heatmap(pivot, filename, output_dir, cmap='inferno', fmt=".1f", annot_size=26):
    tick_label_fontsize = round(22 * 1.2)

    plt.figure(figsize=(16, 12))
    sns.set_context("talk", font_scale=2.0)
    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        annot_kws={"size": annot_size, "weight": "semibold"},
    )

    plt.xlabel("Image Encoder", fontsize=30, labelpad=18)
    plt.ylabel("EEG Encoder", fontsize=30, labelpad=18)

    plt.yticks(rotation=0, fontsize=tick_label_fontsize)
    plt.xticks(rotation=45, ha='right', fontsize=tick_label_fontsize)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', pad_inches=0.1)
    plt.close()
    sns.reset_orig()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nomixup_csv", required=True)
    parser.add_argument("--mixup_csv", required=True)
    parser.add_argument("--tta_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--ignore_conformer",
        action="store_true",
        help="Drop EEGConformer sweep rows before building heatmaps.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    metrics = ["best top1 acc", "best top5 acc"]

    for metric in metrics:
        metric_label = metric.replace(" ", "_")

        pivot_nosubjectmix = load_and_pivot(
            args.nomixup_csv, metric, ignore_conformer=args.ignore_conformer
        )
        plot_heatmap(
            pivot_nosubjectmix,
            f"{metric_label}_1_nosubjectmix.png",
            args.output_dir,
        )

        pivot_subjectmix = load_and_pivot(
            args.mixup_csv, metric, ignore_conformer=args.ignore_conformer
        )
        diff_subjectmix = pivot_subjectmix - pivot_nosubjectmix
        plot_heatmap(
            diff_subjectmix,
            f"{metric_label}_2_subjectmix_diff.png",
            args.output_dir,
            fmt="+.1f",
        )

        pivot_tta = load_and_pivot(
            args.tta_csv,
            metric,
            source_col='source_run',
            ignore_conformer=args.ignore_conformer,
        )
        diff_tta = pivot_tta - pivot_subjectmix
        plot_heatmap(
            diff_tta,
            f"{metric_label}_3_tta_diff.png",
            args.output_dir,
            fmt="+.1f",
            annot_size=24,
        )

        plot_heatmap(
            pivot_tta,
            f"{metric_label}_4_subjectmix_tta_final.png",
            args.output_dir,
        )

if __name__ == "__main__":
    main()
