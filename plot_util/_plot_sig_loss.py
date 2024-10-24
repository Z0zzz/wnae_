import matplotlib.pyplot as plt
import numpy as np

def plot_sig_loss(sig_loss, bins, savepath=None):
    num_plots = len(sig_loss)
    fig, axes = plt.subplots(nrows=int(np.ceil(num_plots / 2)),
                             ncols=2, 
                             figsize=(20, 60)
                            )

    axes = axes.flatten()

    for ax in axes[num_plots:]:
        ax.axis('off')

    for i, key in enumerate(sig_loss):
        ax = axes[i]
        ax.hist(
            np.clip(sig_loss[key], bins[0], bins[-1]),
            bins=bins,
            label=key, 
            density=True)

        ax.set_xlabel('Total Loss')
        ax.set_ylabel('Density')
        ax.set_title(key, fontsize=20)

    plt.tight_layout()

    if savepath:
        plt.savefig(savepath)

    plt.show()