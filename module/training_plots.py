"""Training diagnostics: training / subject-probe metric figures (non-interactive backend) and
L0 sparsity stats of post-ReLU alignment vectors."""

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


def empty_alignment_sparsity_accumulator():
    return {'l0_sum': 0.0, 'n_samples': 0, 'active_mask': None}


def update_alignment_sparsity_accumulator(acc, features):
    """Update running L0 stats without storing all samples (O(dim) memory)."""
    features = np.asarray(features, dtype=np.float32)
    if features.size == 0:
        return acc
    if acc['active_mask'] is None:
        acc['active_mask'] = np.zeros(features.shape[1], dtype=bool)
    acc['l0_sum'] += float((features > 0).sum(axis=1).sum())
    acc['n_samples'] += features.shape[0]
    acc['active_mask'] |= (features > 0).any(axis=0)
    return acc


def finalize_alignment_sparsity_accumulator(acc):
    if acc['n_samples'] == 0 or acc['active_mask'] is None:
        return {'l0_mean': 0.0, 'l0_frac': 0.0, 'active_feat_frac': 0.0}
    dim = acc['active_mask'].shape[0]
    l0_mean = acc['l0_sum'] / acc['n_samples']
    active_dims = int(acc['active_mask'].sum())
    return {
        'l0_mean': l0_mean,
        'l0_frac': l0_mean / dim,
        'active_feat_frac': active_dims / dim,
    }


def compute_alignment_sparsity_stats(features):
    """L0 stats on post-ReLU, pre-L2-norm alignment vectors."""
    acc = empty_alignment_sparsity_accumulator()
    update_alignment_sparsity_accumulator(acc, features)
    return finalize_alignment_sparsity_accumulator(acc)
