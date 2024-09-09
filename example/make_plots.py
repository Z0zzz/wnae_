import sys

import pandas as pd
import matplotlib.pyplot as plt


def set_plot_style_mpl():
    """Define a scientific style for matplotlib canvas."""

    rc_params = {
        "mathtext.default": "regular",
        "font.size": 25,
        "axes.labelsize": "large",
        "axes.unicode_minus": False,
        "xtick.labelsize": "large",
        "ytick.labelsize": "large",
        "legend.fontsize": "large",
        "legend.handlelength": 1.5,
        "legend.borderpad": 0.5,
        "legend.frameon": False,
        "xtick.direction": "in",
        "xtick.major.size": 12,
        "xtick.minor.size": 6,
        "xtick.major.pad": 6,
        "xtick.top": True,
        "xtick.major.top": True,
        "xtick.major.bottom": True,
        "xtick.minor.top": True,
        "xtick.minor.bottom": True,
        "xtick.minor.visible": True,
        "ytick.direction": "in",
        "ytick.major.size": 12,
        "ytick.minor.size": 6.0,
        "ytick.right": True,
        "ytick.major.left": True,
        "ytick.major.right": True,
        "ytick.minor.left": True,
        "ytick.minor.right": True,
        "ytick.minor.visible": True,
        "grid.alpha": 0.8,
        "grid.linestyle": ":",
        "axes.linewidth": 2,
        "savefig.transparent": False,
        "figure.figsize": (15.0, 10.0),
        "legend.numpoints": 1,
        "lines.markersize": 8,
    }
    
    for k, v in rc_params.items():
        plt.rcParams[k] = v


def main(output_path):

    set_plot_style_mpl()
    metrics_file_name = f"{output_path}/training.csv"
    df = pd.read_csv(metrics_file_name)

    plt.figure()
    plt.plot(
        df["epoch"],
        df["training_loss"],
        color='red',
        linestyle='solid',
        linewidth=2,
        label="Training",
    )
    plt.plot(
        df["epoch"],
        df["validation_loss"],
        color='blue',
        linestyle='dashed',
        linewidth=2,
        label="Validation",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{output_path}/loss.pdf")
    plt.savefig("loss.pdf")
    plt.close()

    plt.figure()
    plt.plot(
        df["epoch"],
        df["auc"],
        color='red',
        linestyle='solid',
        linewidth=2,
    )
    plt.xlabel("Epoch")
    plt.ylabel("ROC AUC")
    plt.savefig(f"{output_path}/auc.pdf")
    plt.savefig("auc.pdf")
    plt.close()


if __name__ == "__main__":
    output_path = sys.argv[1]
    main(output_path)

