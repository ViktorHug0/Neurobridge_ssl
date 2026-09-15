import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Base 1.3× vs original matplotlib defaults; then +10% on all text sizes.
FONT_SCALE = 1.3 * 1.1
LEGEND_FONT_SCALE = 1.45  # larger legend than other text
FIG_W, FIG_H = 8 * 1.2, 5  # 20% wider, same height as previous (8, 5)


def plot_comparison(session_dir: str, nomix_only: bool = False) -> None:
    all_configs = [
        "nomix_featdim1024_seed3300",
        "nomix_featdim64_seed3300",
        "mixup_featdim1024_seed3300",
        "mixup_featdim64_seed3300",
    ]
    all_labels = {
        "nomix_featdim1024_seed3300": "Dim 1024",
        "nomix_featdim64_seed3300": "Dim 64",
        "mixup_featdim1024_seed3300": "SubjectMix, Dim 1024",
        "mixup_featdim64_seed3300": "SubjectMix, Dim 64",
    }

    if nomix_only:
        configs = [c for c in all_configs if c.startswith("nomix_")]
        out_suffix = "_nomix"
    else:
        configs = all_configs
        out_suffix = ""

    labels = {k: all_labels[k] for k in configs}

    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "axes.titlesize": 15 * FONT_SCALE,
            "axes.labelsize": 12 * FONT_SCALE,
            "xtick.labelsize": 11 * FONT_SCALE,
            "ytick.labelsize": 11 * FONT_SCALE,
            "legend.fontsize": 12 * FONT_SCALE * LEGEND_FONT_SCALE,
            "figure.titlesize": 16 * FONT_SCALE,
        }
    )

    def finalize_plot(fig, ax, title, out_path):
        pad = 15 * FONT_SCALE
        ax.set_title(title, pad=pad)
        ax.set_xlabel("Training Epochs")
        ax.set_ylabel("Subject Identification Accuracy (%)")
        ax.set_ylim(20, 100)
        ax.set_xlim(0, 51)
        ax.tick_params(axis="both", which="major", labelsize=11 * FONT_SCALE)

        ax.grid(True, linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.legend(
            loc="lower right",
            frameon=True,
            framealpha=0.9,
            edgecolor="0.8",
            prop={"size": 12 * FONT_SCALE * LEGEND_FONT_SCALE},
        )
        
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out_path}")

    fig_bb, ax_bb = plt.subplots(figsize=(FIG_W, FIG_H))
    fig_al, ax_al = plt.subplots(figsize=(FIG_W, FIG_H))

    for config in configs:
        config_path = os.path.join(session_dir, config)
        if not os.path.exists(config_path): continue

        subject_data = []
        for sub_id in range(1, 11):
            sub_dirs = glob.glob(os.path.join(config_path, f"*-sub-{sub_id:02d}"))
            if sub_dirs:
                probe_csv = os.path.join(sub_dirs[0], "probe_metrics.csv")
                if os.path.exists(probe_csv):
                    subject_data.append(pd.read_csv(probe_csv))
        
        if not subject_data: continue

        mean_df = pd.concat(subject_data).groupby("epoch").mean().reset_index()
        
        # Plotting only the lines (no markers)
        ax_bb.plot(mean_df["epoch"], mean_df["eeg_backbone_val_acc"], 
                   label=labels[config], linewidth=2.5, alpha=0.9)
        
        ax_al.plot(mean_df["epoch"], mean_df["eeg_align_val_acc"], 
                   label=labels[config], linewidth=2.5, alpha=0.9)

    finalize_plot(
        fig_bb,
        ax_bb,
        "",
        os.path.join(session_dir, f"probe_comparison_backbone{out_suffix}.png"),
    )
    finalize_plot(
        fig_al,
        ax_al,
        "",
        os.path.join(session_dir, f"probe_comparison_alignment{out_suffix}.png"),
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot holdout-probe backbone vs alignment curves for a session directory."
    )
    parser.add_argument(
        "--session-dir",
        default=os.path.join(
            REPO_ROOT,
            "results/things_eeg/inter-subjects/holdout_probe_projector_mixup_20260426-020407",
        ),
        help="Directory containing nomix_*/mixup_* config subfolders.",
    )
    parser.add_argument(
        "--nomix-only",
        action="store_true",
        help="Plot only nomix_featdim* configs; save as probe_comparison_*_nomix.png",
    )
    args = parser.parse_args()
    plot_comparison(os.path.abspath(args.session_dir), nomix_only=args.nomix_only)
