"""Plot ten-fold mean objective contributions for the ATM+TSConv lambda=.05 arm."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(
    "results/things_eeg/decorrelated_models/heterogeneous_wave/"
    "decor_atm_tsconv_l005_b0/seed3300_3301"
)
OUTPUT = Path(
    "ensemble_experiments/analysis/decor_atm_tsconv_l005_loss_components.png"
)
CSV_OUTPUT = OUTPUT.with_suffix(".csv")


def main() -> None:
    histories = sorted(ROOT.glob("*-sub-*/history.csv"))
    if len(histories) != 10:
        raise RuntimeError(f"expected 10 histories, found {len(histories)}")

    frames = []
    common_last_epoch = 50
    for history in histories:
        frame = pd.read_csv(history)
        expected = list(range(1, len(frame) + 1))
        if frame["epoch"].tolist() != expected:
            raise RuntimeError(f"incomplete or malformed history: {history}")
        common_last_epoch = min(common_last_epoch, len(frame))
        frame = frame.copy()
        frame["subject"] = int(history.parent.name.rsplit("sub-", 1)[1])
        frame["infonce_contribution"] = frame["train_a"] + frame["train_b"]
        frame["lambda_contribution"] = (
            frame["active_lambda"] * frame["train_diversity"]
        )
        frames.append(frame)

    frames = [frame[frame["epoch"] <= common_last_epoch] for frame in frames]

    mean = (
        pd.concat(frames, ignore_index=True)
        .groupby("epoch", as_index=False)[
            ["infonce_contribution", "lambda_contribution"]
        ]
        .mean()
    )
    mean.to_csv(CSV_OUTPUT, index=False)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.plot(
        mean["epoch"],
        mean["infonce_contribution"],
        color="#276FBF",
        linewidth=2.5,
        label=r"InfoNCE: $L_{ATM}+L_{TSConv}$",
    )
    ax.plot(
        mean["epoch"],
        mean["lambda_contribution"],
        color="#D1495B",
        linewidth=2.5,
        label=r"Weighted leash: $\lambda L_{div}$",
    )
    ax.axvspan(1, 10, color="#777777", alpha=0.10, linewidth=0)
    ax.text(5.5, 0.012, r"$\lambda=0$ warm-up", ha="center", va="bottom", fontsize=9)
    ax.set_yscale("symlog", linthresh=0.01, linscale=0.8)
    ax.set_xlim(1, common_last_epoch)
    ticks = [tick for tick in [1, 5, 10, 11, 20, 30, 40, 50] if tick <= common_last_epoch]
    if common_last_epoch not in ticks:
        ticks.append(common_last_epoch)
    ax.set_xticks(ticks)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean objective contribution (symlog scale)")
    ax.set_title(
        "ATM + TSConv loss components, λ=0.05 "
        f"(10-fold mean, epochs 1–{common_last_epoch})"
    )
    ax.legend(frameon=True, loc="center right")
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(OUTPUT)
    print(CSV_OUTPUT)


if __name__ == "__main__":
    main()
