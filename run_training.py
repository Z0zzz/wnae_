from importlib import import_module
import torch

from SVJanalysis.MLFramework.module.DataProcessor import DataProcessor   # Black magic to avoid import bug
from trainer import TrainerWassersteinNormalizedAutoEncoder
from loader.Loader import Loader
from architectures.ae import Encoder, Decoder


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

config = "configs.example"
config = import_module(config)

loader = Loader(
    device=device,
    batch_size=config.training_params["batch_size"],
)

input_size = loader.input_size
intermediate_architecture = (10, 10)
bottleneck_size = 6

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
)

trainer.train()

