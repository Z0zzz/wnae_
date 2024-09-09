from importlib import import_module
import torch

from trainer import TrainerWassersteinNormalizedAutoEncoder
from loader import Loader
from architectures import Encoder, Decoder

import sys

output_path = sys.argv[1]


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

config = import_module("config")

loader = Loader(
    device=device,
    batch_size=config.training_params["batch_size"],
)

input_size = loader.input_size
intermediate_architecture = (64, 32, 16, 16)
bottleneck_size = 4

encoder = Encoder(
    input_size=input_size,
    intermediate_architecture=intermediate_architecture,
    bottleneck_size=bottleneck_size,
    drop_out=None,
)
decoder = Decoder(
    output_size=input_size,
    intermediate_architecture=intermediate_architecture,
    bottleneck_size=bottleneck_size,
    drop_out=None,
)

trainer = TrainerWassersteinNormalizedAutoEncoder(
    config=config,
    loader=loader,
    encoder=encoder,
    decoder=decoder,
    device=device,
    output_path=output_path,
)

trainer.train()

