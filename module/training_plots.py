"""Save training / subject-probe metric figures (non-interactive backend)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_training_plot(history, out_path):
    epochs = history["epoch"]
    fig, ax_loss = plt.subplots(figsize=(9, 5))
    ax_acc = ax_loss.twinx()

    ax_loss.plot(epochs, history["train_loss"], label="Train Loss", color="blue", linewidth=2)
    ax_loss.plot(epochs, history["test_loss"], label="Test Loss", color="blue", linestyle="--", linewidth=2)
    ax_acc.plot(epochs, history["top1_acc"], label="Top-1 Acc", color="black", linestyle="--", linewidth=2)

    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_acc.set_ylabel("Top-1 Accuracy (%)")
    ax_acc.set_ylim(0, 60)
    ax_loss.grid(alpha=0.3)

    lines_1, labels_1 = ax_loss.get_legend_handles_labels()
    lines_2, labels_2 = ax_acc.get_legend_handles_labels()
    ax_loss.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_probe_plot(probe_history, out_path):
    epochs = probe_history["epoch"]
    bb = np.asarray(probe_history["eeg_backbone_val_acc"], dtype=float)
    clip = np.asarray(probe_history["eeg_align_val_acc"], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, bb, label="Backbone", color="tab:orange", linewidth=2)
    ax.plot(epochs, clip, label="CLIP alignment", color="tab:green", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Subject probe val accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
