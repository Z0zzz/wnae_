# Loading the data from the processed files :)
import json
import numpy as np
from torch.utils import data
import h5py
import torch
from example.trainer import TrainerWassersteinNormalizedAutoEncoder
from example.loader import Loader
from example.architectures import Encoder, Decoder
from wnae._logger import log
from pathlib import Path
import os
import shutil

device = torch.device('cpu')
    
batch_size = 2048

# f = h5py.File("./data/newdata/Data.h5","r")
f = h5py.File("/pfvolcentral/notebooks/btagging/Data.h5","r")

x_train = f["Background_data"]["Train"]["DATA"][:]
x_test = f["Background_data"]["Test"]["DATA"][:]
x_sig = f["Signal_data"]["GluGluHToBB_M-125"]["DATA"][:]

scale = f["Normalisation"]["norm_scale"][:]
bias = f["Normalisation"]["norm_bias"][:]

data_config = json.loads(f.attrs["config"])
constituents = data_config["Read_configs"]["BACKGROUND"]["constituents"]

x_train = torch.tensor(np.reshape(x_train,(x_train.shape[0],-1))).to(torch.float32).to(device)
x_test = torch.tensor(np.reshape(x_test,(x_test.shape[0],-1))).to(torch.float32).to(device)
x_sig = torch.tensor(np.reshape(x_sig,(x_sig.shape[0],-1))).to(torch.float32).to(device)

train_loader = data.DataLoader(
            dataset=data.TensorDataset(x_train),
            batch_size=batch_size,
        )

val_loader = data.DataLoader(
            dataset=data.TensorDataset(x_test),
            batch_size=batch_size,
        )

val_loader_no_batch = data.DataLoader(
    dataset=data.TensorDataset(x_test),
    batch_size=len(x_train),
)

sig_loader = data.DataLoader(
            dataset=data.TensorDataset(x_sig),
            batch_size=batch_size,
        )

from importlib import import_module

class MyLoader():
    def __init__(self, train_loader, val_loader, val_loader_no_batch, ood_loader) -> None:
        self.training_loader = train_loader
        self.validation_loader = val_loader
        self.validation_loader_no_batch = val_loader_no_batch
        self.ood_loader = ood_loader
        
loaders = MyLoader(train_loader, val_loader, val_loader_no_batch, sig_loader)

config = import_module("example.config")
config.training_params["batch_size"] = batch_size

input_size = x_train.shape[-1]
intermediate_architecture_encoder = (28,15)
intermediate_architecture_decoder = (24, 32, 64, 128, 57)
bottleneck_size = 8
output_path = "/pfvolcentral/notebooks/btagging/wnae6"
# output_path = "~/Desktop"
if os.path.exists(output_path) and os.path.isdir(output_path):
    shutil.rmtree(output_path)
    print(f"Deleted directory: {output_path}")
else:
    print(f"Directory does not exist: {output_path}")
    
config_file = f"{output_path}/config.json"
Path(f"{output_path}/").mkdir(parents=True, exist_ok=True)
with open(config_file, "w") as file:
    json.dump(config.training_params, file, indent=4) 
    
print("Saving to ", output_path)
encoder = Encoder(
    input_size=input_size,
    intermediate_architecture=intermediate_architecture_encoder,
    bottleneck_size=bottleneck_size,
    drop_out=None,
)
decoder = Decoder(
    output_size=input_size,
    intermediate_architecture=intermediate_architecture_decoder,
    bottleneck_size=bottleneck_size,
    drop_out=None,
)

trainer = TrainerWassersteinNormalizedAutoEncoder(
    config=config,
    loader=loaders,
    encoder=encoder,
    decoder=decoder,
    device=device,
    output_path=output_path,
    loss_function="wnae",  # can change to "ae" or "nae"
)

trainer.train()
log.info("Saving...")
trainer.save_train_plot()
log.info("Done.")