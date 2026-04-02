"""
plot_training_curves.py
-----------------------
Reads TensorBoard event files from a single run directory and produces a
two-panel figure:
    - Left:  Top-1 accuracy (%) on the test subset vs. epoch
    - Right: Loss on the test subset vs. epoch

Usage
-----
    python plot_training_curves.py --run_dir <path_to_run_dir> [--out <fig.png>]

The run_dir is expected to contain a TF events file (created by the TB
SummaryWriter in train.py).  The script walks subdirectories to find it.

Called automatically by run_three_stage_grid.sh after each sub-run.
"""

import argparse
import os
import glob
import re

import matplotlib
matplotlib.use('Agg')       # headless
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Optional: try to import tfevents reader; fall back to CSV if unavailable
# ---------------------------------------------------------------------------
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    _HAS_TB = True
except ImportError:
    _HAS_TB = False


def _read_tb(run_dir):
    """Return (epochs, losses, top1_accs) from TensorBoard event files."""
    acc  = EventAccumulator(run_dir)
    acc.Reload()

    # Prefer test-specific tags; fall back to generic names
    def _scalars(tag):
        if tag in acc.Tags()['scalars']:
            return [(s.step, s.value) for s in acc.Scalars(tag)]
        return []

    # Loss
    for loss_tag in ('Loss/test', 'Loss/test_at_best_val', 'Loss/train'):
        pts = _scalars(loss_tag)
        if pts:
            break

    # Accuracy
    for acc_tag in ('Acc/top1_test', 'Acc/top1_test_at_best_val'):
        apt = _scalars(acc_tag)
        if apt:
            break
    else:
        apt = []

    epochs_l = [p[0] for p in pts]
    losses   = [p[1] for p in pts]
    epochs_a = [p[0] for p in apt]
    accs     = [p[1] for p in apt]
    return epochs_l, losses, epochs_a, accs


def _read_log_txt(run_dir):
    """Fallback: parse train.log text file for loss and accuracy."""
    log_path = os.path.join(run_dir, 'train.log')
    if not os.path.exists(log_path):
        return [], [], [], []

    epoch_pat   = re.compile(r'epoch\s+(\d+)', re.I)
    loss_pat    = re.compile(r'Loss:\s*([\d.]+)', re.I)
    top1_pat    = re.compile(r'top1\s+acc\s+([\d.]+)', re.I)

    epochs_l, losses, epochs_a, accs = [], [], [], []
    with open(log_path) as f:
        for line in f:
            em = epoch_pat.search(line)
            lm = loss_pat.search(line)
            am = top1_pat.search(line)
            if em and lm:
                epochs_l.append(int(em.group(1)))
                losses.append(float(lm.group(1)))
            if em and am:
                epochs_a.append(int(em.group(1)))
                accs.append(float(am.group(1)))
    return epochs_l, losses, epochs_a, accs


def find_event_files(root):
    """Recursively find TF event files under root."""
    return glob.glob(os.path.join(root, '**', 'events.out.tfevents.*'), recursive=True)


def plot_run(run_dir, out_path=None):
    """Generate and save the training-curve figure for one sub-run."""
    # Try TensorBoard first
    event_files = find_event_files(run_dir)
    if _HAS_TB and event_files:
        # Use the directory containing the event file
        tb_dir = os.path.dirname(event_files[0])
        epochs_l, losses, epochs_a, accs = _read_tb(tb_dir)
    else:
        epochs_l, losses, epochs_a, accs = _read_log_txt(run_dir)

    if not epochs_l and not epochs_a:
        print(f"[plot_training_curves] No data found in {run_dir}, skipping plot.")
        return

    # ---------- derive a readable title from the directory name ----------
    dirname = os.path.basename(run_dir.rstrip('/'))
    title = dirname

    # ---------- plot ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(title, fontsize=10, y=1.01)

    # Left: Top-1 accuracy
    ax_acc = axes[0]
    if epochs_a and accs:
        ax_acc.plot(epochs_a, accs, color='steelblue', linewidth=1.8, marker='o',
                    markersize=3, label='Top-1 acc')
        ax_acc.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Top-1 Accuracy (%)')
    ax_acc.set_title('Test Top-1 Accuracy')
    ax_acc.grid(alpha=0.3)

    # Right: Loss
    ax_loss = axes[1]
    if epochs_l and losses:
        ax_loss.plot(epochs_l, losses, color='tomato', linewidth=1.8, marker='o',
                     markersize=3, label='Loss')
        ax_loss.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Test Loss')
    ax_loss.grid(alpha=0.3)

    plt.tight_layout()

    if out_path is None:
        out_path = os.path.join(run_dir, 'training_curves.png')
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot_training_curves] Saved → {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training curves from a run directory.')
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Path to the sub-run directory (contains tfevents or train.log).')
    parser.add_argument('--out', type=str, default=None,
                        help='Output PNG path (default: <run_dir>/training_curves.png).')
    args = parser.parse_args()
    plot_run(args.run_dir, args.out)
