import matplotlib.pyplot as plt
import numpy as np

def plot_loss(bkg_loss, bins, savepath=None):
	# bkg_data = background.reshape(-1, 19 * 3)
	# bkg_data = np.reshape(background,(background.shape[0],-1))
	# bkg_reco, z_mean, z_log_var = model.predict(bkg_data)
	# bkg_reco_loss = reco_loss.call(bkg_data, bkg_reco)
	# bkg_kl_loss = kl_loss(z_mean, z_log_var)
	# bkg_loss = bkg_reco_loss + bkg_kl_loss
	# background_loss = bkg_loss.numpy()

	plt.figure(figsize=(7, 6))

	plt.hist(
		np.clip(bkg_loss, bins[0], bins[-1]), #Handle overflow
		bins=bins, 
		density=True, 
		label = 'background'
		)

	plt.xlabel('Total Loss')
	plt.ylabel('Density')
	plt.title('Total Loss Distribution')
	plt.legend()
	plt.grid(True)

	if savepath:
		plt.savefig(savepath)

	plt.show()