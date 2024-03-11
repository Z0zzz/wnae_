# ------------------------------------------------------------------------------------------------
# This is the default config file for the auto-encoder. Please copy it in case you want to change
# some parameters.
# ------------------------------------------------------------------------------------------------

enable_data_seed = False

n_models = 1
best_model = 0


# ---------------------------------------------
# Model info
model_type = "NormalizedAutoEncoder"
train_on_signal = False


# ---------------------------------------------
# Output paths
__output_directory_name = __file__.replace(".py", "").split("/")[-1]
output_path = f"/work/fleble/SVJ/MLFramework/trainingResults/nae_tChannel/{__output_directory_name}/"


# ---------------------------------------------
# Build general training/evaluation settings dictionary

__features_labels = {
    "FatJet_softDropMassLog": "log(softdrop mass)",
    "FatJet_tau2": "#tau_{2}",
    "FatJet_tau3": "#tau_{3}",
    "FatJet_efp1d3": "EFP1",
    "FatJet_c2b0p5": "C_{2}^{#beta=0.5}",
    "FatJet_axisMajor": "Axis major",
    "FatJet_axisMinor": "Axis minor",
    "FatJet_ptD": "p_{T}D",
}


# ---------------------------------------------
# Training parameters
__checkpoint_interval = "1 if i_epoch < 1 else 100"


__high_stat_emd_interval = 1


__hyper_parameters = {
    "z_step": 10,
    "z_stepsize": 1,
    "z_noise_std": 1,
    "z_noise_anneal": None,
    "x_step": 10,
    "x_stepsize": None,
    "x_noise_std": 0.22,
    "x_noise_anneal": None,
    "x_bound": (-3, 3),
    "z_bound": None,
    "z_clip_langevin_grad": None,
    "x_clip_langevin_grad": None,
    "l1_regularization_coef_positive_energy": 0.001,
    "l1_regularization_coef_negative_energy": None,
    "l2_regularization_coef_positive_energy": None,
    "l2_regularization_coef_negative_energy": None,
    "l2_regularization_coef_decoder": None,
    "l2_regularization_coef_encoder": None,
    "l2_regularization_coef_latent_space": None,
    "coef_energy_difference": None,
    "spherical": False,
    "buffer_size": 10000,
    "replay_ratio": 0.95,
    "replay": True,
    "sampling": "pcd",
    "temperature": 0.063,
    "temperature_trainable": False,
    "initial_dist": "gaussian",
    "mh": False,
    "mh_z": False,
    "reject_boundary": False,
    "reject_boundary_z": False,
    "use_log_cosh": True,
}

training_params = {
    "batch_size": 256,
    "nae": {
        # Best model metric
        # Either a string or a tuple
        # If string, can use energy_difference, validation loss, training loss, a function of these,
        # If tuple, first element is a string of a tuple of arguments to a function, and second element is that function
        "best_model_metric": "i_epoch",
        # Rule: minimum, maximum, minimum_positive
        "best_model_rule": "maximum",

        "optimizer": "Adam", # default
        "learning_rate": 1e-5, # best
        "lr_scheduler": None,
        "es_patience": 5000,
        "n_epochs": 2,

        "checkpoint_saving_interval": __checkpoint_interval,

        "negative_samples_1d_histograms_saving_interval": __checkpoint_interval,
        "make_negative_samples_1d_histograms": True,

        "emd_computation_saving_interval": __high_stat_emd_interval,
        "compute_emd": True,

        "emd_one_batch_computation_saving_interval": 1,
        "compute_emd_one_batch": False,

        "restore_best_model_after_training": True,

        "hyper_parameters": __hyper_parameters,
    },
    "features_labels": __features_labels,
}

