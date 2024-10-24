import numpy as np
import matplotlib.pyplot as plt
from ._roc import *
from sklearn.metrics import auc


def plot_roc(sig_loss, background_loss, groups, savepath=None):
	num_plots = len(groups)
	fig, axes = plt.subplots(nrows=int(np.ceil(num_plots / 2)),
		ncols=2, 
		figsize=(15, 30))
	axes = axes.flatten()

	for ax in axes[num_plots:]:
		ax.axis('off')
	
	for i, group in enumerate(groups):
		ax = axes[i]
		for key in group:
			loss_sig = sig_loss[key]
			loss_bkg = np.random.choice(background_loss.numpy(), loss_sig.shape[0], replace=True) #Random sample from background_loss to match shape
			print(loss_sig.shape)
			print(loss_bkg.shape)
			FP, TP = fast_roc(loss_sig, loss_bkg)
			roc_auc = auc(FP, TP)
			ax.plot(FP, TP, linestyle='--', label=f'{key} (AUC = %0.2f)' % roc_auc)
			ax.set_xlabel('False Positive Rate')
			ax.set_ylabel('True Positive Rate')
			ax.set_title('ROC Curve')
			ax.legend()
	plt.tight_layout()
	
	if savepath:
		plt.savefig(savepath)

	plt.show() 