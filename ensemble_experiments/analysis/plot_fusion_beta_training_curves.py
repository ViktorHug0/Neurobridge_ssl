"""Plot matched fusion-training trajectories for beta = 0, .10, and .30."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "things_eeg" / "decorrelated_models"
SUBJECTS = (1, 2, 4, 5)
ARMS = {
    0.00: RESULTS
    / "valcon_wave"
    / "rescue_valcon_atm_tsconv_g000_b0_bs512"
    / "seed3300_3301",
    0.10: RESULTS
    / "fusion_wave"
    / "fusion_valcon_atm_tsconv_b010_g000_bs512"
    / "seed3300_3301",
    0.30: RESULTS
    / "fusion_wave"
    / "fusion_valcon_atm_tsconv_b030_g000_bs512"
    / "seed3300_3301",
}
OUT = ROOT / "ensemble_experiments" / "analysis"


def completed_history(arm_root: Path, subject: int) -> pd.DataFrame:
    candidates = sorted(arm_root.glob(f"*-sub-{subject:02d}"))
    completed = [path for path in candidates if (path / "result.csv").exists()]
    if not completed:
        raise FileNotFoundError(f"No completed result for subject {subject}: {arm_root}")
    history = pd.read_csv(completed[-1] / "history.csv").sort_values("epoch")
    if len(history) != 50 or history["epoch"].tolist() != list(range(1, 51)):
        raise ValueError(f"Expected epochs 1--50 in {completed[-1] / 'history.csv'}")
    return history


def aggregate(beta: float, arm_root: Path) -> pd.DataFrame:
    histories = [completed_history(arm_root, subject) for subject in SUBJECTS]
    train = np.stack([frame["train_ensemble"].to_numpy() for frame in histories])
    validation = np.stack(
        [frame["selection_pair_loss"].to_numpy() for frame in histories]
    )
    test = np.stack([frame["pair_top1"].to_numpy() for frame in histories])
    peak_test = np.maximum.accumulate(test, axis=1)
    return pd.DataFrame(
        {
            "beta": beta,
            "epoch": histories[0]["epoch"].to_numpy(),
            "mean_train_ensemble_loss": train.mean(axis=0),
            "mean_val_pair_loss": validation.mean(axis=0),
            "mean_peak_so_far_test_top1": peak_test.mean(axis=0),
        }
    )


def main() -> None:
    frames = [aggregate(beta, root) for beta, root in ARMS.items()]
    summary = pd.concat(frames, ignore_index=True)
    summary.to_csv(OUT / "fusion_beta_training_curves.csv", index=False)

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 240,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.grid": True,
            "grid.alpha": 0.20,
            "legend.frameon": False,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8), sharex=True, sharey=True,
                             layout="constrained")
    loss_values = summary[
        ["mean_train_ensemble_loss", "mean_val_pair_loss"]
    ].to_numpy()
    accuracy_values = summary["mean_peak_so_far_test_top1"].to_numpy()
    loss_padding = 0.06 * (loss_values.max() - loss_values.min())
    accuracy_padding = 0.08 * (accuracy_values.max() - accuracy_values.min())

    for axis, (beta, frame) in zip(axes, summary.groupby("beta", sort=True)):
        accuracy_axis = axis.twinx()
        train_line = axis.plot(
            frame["epoch"],
            frame["mean_train_ensemble_loss"],
            color="#4C78A8",
            lw=2.2,
            label="Training ensemble loss",
        )[0]
        val_line = axis.plot(
            frame["epoch"],
            frame["mean_val_pair_loss"],
            color="#F58518",
            lw=2.2,
            label="ValCon pair loss",
        )[0]
        accuracy_line = accuracy_axis.plot(
            frame["epoch"],
            frame["mean_peak_so_far_test_top1"],
            color="#029E73",
            lw=2.2,
            label="Peak-so-far test top-1",
        )[0]
        axis.set_title(rf"$\beta={beta:g}$")
        axis.set_xlabel("Epoch")
        axis.set_ylim(loss_values.min() - loss_padding, loss_values.max() + loss_padding)
        accuracy_axis.set_ylim(
            accuracy_values.min() - accuracy_padding,
            accuracy_values.max() + accuracy_padding,
        )
        if axis is axes[0]:
            axis.set_ylabel("Loss")
        if axis is axes[-1]:
            accuracy_axis.set_ylabel("Mean peak-so-far test top-1 (%)")
        else:
            accuracy_axis.set_yticklabels([])
        axis.legend(
            [train_line, val_line, accuracy_line],
            [line.get_label() for line in (train_line, val_line, accuracy_line)],
            loc="best",
            fontsize=8,
        )

    fig.suptitle(
        "Fusion-aware training trajectories (subjects 1, 2, 4, and 5)", fontsize=16
    )
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"fusion_beta_training_curves.{extension}", bbox_inches="tight")
    plt.close(fig)
    print(OUT / "fusion_beta_training_curves.png")


if __name__ == "__main__":
    main()
