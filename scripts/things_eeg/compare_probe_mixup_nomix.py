import os

import matplotlib.pyplot as plt
import pandas as pd


FIGSIZE_IN = (12, 6)
FS_LABEL = 16
FS_TICK = 14
FS_LEGEND = 16
LINE_WIDTH = 3
SAVE_DPI = 300
YLIM = (60, 103)


def compare_probe_accuracy(base_dir):
    mixup_dir = os.path.join(base_dir, "mixup_featdim512_seed3300")
    nomix_dir = os.path.join(base_dir, "nomix_featdim512_seed3300")

    representations = [
        "eeg_backbone_val_acc",
        "eeg_align_val_acc",
        "eeg_temporal_val_acc",
        "eeg_spatial_val_acc",
    ]

    mixup_file = os.path.join(mixup_dir, "probe_metrics_all_subjects.csv")
    nomix_file = os.path.join(nomix_dir, "probe_metrics_all_subjects.csv")

    if not os.path.exists(mixup_file) or not os.path.exists(nomix_file):
        print(f"Error: Could not find one of the required files:\n{mixup_file}\n{nomix_file}")
        return

    df_mixup = pd.read_csv(mixup_file)
    df_nomix = pd.read_csv(nomix_file)

    avg_mixup = df_mixup.groupby("epoch")[representations].mean().reset_index()
    avg_nomix = df_nomix.groupby("epoch")[representations].mean().reset_index()

    # Same paper style for all probe panels
    plot_jobs = [
        ("eeg_backbone_val_acc", None),
        ("eeg_align_val_acc", "probe_accuracy_alignment_space.png"),
        ("eeg_temporal_val_acc", None),
        ("eeg_spatial_val_acc", None),
    ]
    for rep_col, outfile_override in plot_jobs:
        plt.figure(figsize=FIGSIZE_IN)

        plt.plot(
            avg_mixup["epoch"],
            avg_mixup[rep_col],
            label="SubjectMix",
            color="C0",
            linestyle="-",
            linewidth=LINE_WIDTH,
        )
        plt.plot(
            avg_nomix["epoch"],
            avg_nomix[rep_col],
            label="No SubjectMix",
            color="C1",
            linestyle="-",
            linewidth=LINE_WIDTH,
        )

        plt.xlabel("Epoch", fontsize=FS_LABEL)
        plt.ylabel("Subject identification accuracy (%)", fontsize=FS_LABEL)
        plt.xticks(fontsize=FS_TICK)
        plt.yticks(fontsize=FS_TICK)
        plt.ylim(*YLIM)
        plt.legend(loc="lower right", fontsize=FS_LEGEND)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()

        if outfile_override is not None:
            out_name = outfile_override
        else:
            out_name = f"probe_accuracy_{rep_col.replace('_val_acc', '')}.png"

        output_plot = os.path.join(base_dir, out_name)
        plt.savefig(output_plot, dpi=SAVE_DPI)
        plt.close()
        print(f"Paper-ready plot saved to: {output_plot}")

    print("\nSummary Table (Mixup):")
    print(avg_mixup.to_string(index=False))
    print("\nSummary Table (No Mixup):")
    print(avg_nomix.to_string(index=False))


if __name__ == "__main__":
    base_path = (
        "/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/inter-subjects/"
        "holdout_probe_projector_mixup_20260501-170040"
    )
    compare_probe_accuracy(base_path)
