"""
Publication-Quality Figure Generator for GAN-LMNN-FARCI+ Empirical Study.
Generates grouped metric bar charts (fig_metrics.pdf/.png) and 2x2 factorial
decomposition plots (fig_analysis.pdf/.png) sorted by Imbalance Ratio (IR).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def resolve_csv_path():
    """Locate the experimental results CSV from CLI argument or default paths."""
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        return sys.argv[1]
    for candidate in ["experiment_results_detail.csv", "hasil_eksperimen_detail.csv"]:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find 'experiment_results_detail.csv' or 'hasil_eksperimen_detail.csv'."
    )


def generate_figures():
    csv_path = resolve_csv_path()
    # Sort datasets in ascending order of Imbalance Ratio (IR)
    df = pd.read_csv(csv_path).sort_values("IR").reset_index(drop=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
        }
    )

    methods = ["FRBCS", "GAN-LMNN-FRBCS", "FARCI+", "GAN-LMNN-FARCI+"]
    labels = [
        "FRBCS",
        "GAN-LMNN-FRBCS",
        "FARCI+-style",
        "GAN-LMNN-FARCI+ (hybrid)",
    ]
    colors = ["#7f8c8d", "#e67e22", "#2980b9", "#16a085"]
    hatches = ["", "//", "", "\\\\"]

    x = np.arange(len(df))
    xt = [f"{d}\n(IR={ir:.2f})" for d, ir in zip(df["Dataset"], df["IR"])]
    # Threshold separating moderate imbalance (IR < 9) from high imbalance (IR >= 9)
    n_low = int((df["IR"] < 9).sum())
    split = n_low - 0.5

    # --------------------------------------------------------------------------
    # Figure 1: Per-dataset F1-score, Precision, and Recall
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.2), sharex=True)
    w = 0.20

    for ax, (col_suffix, y_label) in zip(
        axes, [("F1", "$F_1$-score"), ("P", "Precision"), ("R", "Recall")]
    ):
        for i, (m, lbl, col, h) in enumerate(zip(methods, labels, colors, hatches)):
            ax.bar(
                x + (i - 1.5) * w,
                df[f"{m}_{col_suffix}"],
                w,
                label=lbl,
                color=col,
                edgecolor="black",
                linewidth=0.5,
                hatch=h,
            )
        ax.axvline(split, color="gray", linestyle=":", linewidth=1.1)
        ax.set_ylabel(y_label, fontweight="bold")
        ax.set_ylim(0.0, 1.08)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.28),
        ncol=2,
        frameon=True,
        edgecolor="lightgray",
    )
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(xt)
    axes[-1].set_xlabel(
        "Datasets (sorted by IR; dotted line: IR $\\geq$ 9)",
        fontweight="bold",
        labelpad=6,
    )

    plt.tight_layout()
    plt.savefig("fig_metrics.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("fig_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()

    # --------------------------------------------------------------------------
    # Figure 2: 2x2 Factorial Decomposition & F1 Trajectory across IR
    # --------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.6))

    deltas = {
        "+GAN-LMNN (to FRBCS)": (
            df["GAN-LMNN-FRBCS_F1"] - df["FRBCS_F1"],
            "#e74c3c",
        ),
        "+GAN-LMNN (to FARCI+-style)": (
            df["GAN-LMNN-FARCI+_F1"] - df["FARCI+_F1"],
            "#f39c12",
        ),
        "+Choquet (raw data)": (
            df["FARCI+_F1"] - df["FRBCS_F1"],
            "#2980b9",
        ),
        "+Choquet (GAN-LMNN data)": (
            df["GAN-LMNN-FARCI+_F1"] - df["GAN-LMNN-FRBCS_F1"],
            "#16a085",
        ),
    }

    w2 = 0.20
    ax1.axhline(0.0, color="black", linewidth=0.8)
    for k, (lbl, (vals, col)) in enumerate(deltas.items()):
        ax1.bar(
            x + (k - 1.5) * w2,
            vals,
            w2,
            label=lbl,
            color=col,
            edgecolor="black",
            linewidth=0.4,
        )
    ax1.axvline(split, color="gray", linestyle=":", linewidth=1.1)
    ax1.set_title("(a) $2\\times2$ decomposition of $\\Delta F_1$", fontweight="bold")
    ax1.set_ylabel("$\\Delta F_1$", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Dataset"], rotation=35, ha="right")
    ax1.set_ylim(-0.55, 0.32)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.32),
        ncol=2,
        fontsize=6.5,
        frameon=True,
    )

    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)

    for m, lbl, col, mk in zip(methods, labels, colors, ["o", "s", "^", "D"]):
        ax2.plot(
            x,
            df[f"{m}_F1"],
            marker=mk,
            markersize=5,
            linestyle="none",
            label=lbl,
            color=col,
        )
    ax2.axvline(split, color="gray", linestyle=":", linewidth=1.1)
    ax2.text(split + 0.1, 0.80, "IR $\\geq$ 9", fontsize=7)
    ax2.set_title("(b) $F_1$ per dataset (sorted by IR)", fontweight="bold")
    ax2.set_ylabel("$F_1$-score", fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(xt, rotation=35, ha="right", fontsize=6.5)
    ax2.set_ylim(-0.04, 0.88)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.36),
        ncol=2,
        fontsize=6.5,
        frameon=True,
    )

    plt.tight_layout()
    plt.savefig("fig_analysis.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("fig_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(
        f"[DONE] Generated 'fig_metrics.pdf' and 'fig_analysis.pdf' "
        f"from '{csv_path}' (IR >= 9 split after {n_low} datasets)."
    )


if __name__ == "__main__":
    generate_figures()