training_params = {
    "batch_size": 2048,
    "es_patience": 5000,
    "n_epochs": 500,
    
    "optimizer": "AdamW", # default
    "learning_rate": 2e-4,
    "lr_scheduler": "ReduceLROnPlateau", # default
    "lr_scheduler_args": {
        "mode": "min",
        "factor": 0.8,
        "patience": 20,
        "threshold": 0.001,
        "cooldown": 10,
        "threshold_mode": "rel",
        "min_lr": 0,
    },

    "wnae_parameters": {
        "sampling": "pcd",
        "x_step": 10,
        "x_step_size": None,
        "x_noise_std": 0.22,
        "x_temperature": 0.063,
        "x_bound": (-3, 3),
        "x_clip_grad": None,
        "x_reject_boundary": False,
        "x_mh": False,
        "z_step": 10,
        "z_step_size": 1,
        "z_temperature": 0.063,
        "z_noise_std": 1,
        "z_bound": None,
        "z_clip_grad": None,
        "z_reject_boundary": False,
        "z_mh": False,
        "spherical": False,
        "initial_dist": "gaussian",
        "replay": True,
        "replay_ratio": 0.95,
        "buffer_size": 10000,
    },
}

