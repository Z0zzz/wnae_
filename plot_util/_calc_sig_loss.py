import numpy as np
from ._calc_loss import calc_loss
import torch

def calc_sig_loss(model, data, loss_func):
    sig_loss = {}
    for key in data.keys():
        sig = torch.tensor(data[key]['DATA'][:]).to(torch.float32)
        sig = sig.reshape(sig.shape[0], -1)
        loss = model(sig)
        sig_loss[key] = loss.detach().numpy()
    return sig_loss