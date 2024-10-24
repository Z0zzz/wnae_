import numpy as np
import torch

def calc_loss(model, data, loss_func):
    data = torch.reshape(data, (data.shape[0],-1))
    prediction_outputs = model.predict(data)
    
    return loss_func(data, prediction_outputs)