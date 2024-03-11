import random

import numpy as np
import torch
import torch.nn as nn


class DummyDistribution(nn.Module):
    """ Function-less class introduced for backward-compatibility of model checkpoint files. """
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer('sigma', torch.tensor(0., dtype=torch.float))

    def forward(self, x):
        return self.net(x)


class IsotropicGaussian(nn.Module):
    """Isotripic Gaussian density function paramerized by a neural net.
    standard deviation is a free scalar parameter"""
    def __init__(self, net, sigma=1., sigma_trainable=False, error_normalize=True, deterministic=False):
        super().__init__()
        self.net = net
        self.sigma_trainable = sigma_trainable
        self.error_normalize = error_normalize
        self.deterministic = deterministic
        if sigma_trainable:
            # self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float))
            self.register_parameter('sigma', nn.Parameter(torch.tensor(sigma, dtype=torch.float)))
        else:
            self.register_buffer('sigma', torch.tensor(sigma, dtype=torch.float))

    def log_likelihood(self, x, z):
        decoder_out = self.net(z)
        if self.deterministic:
            return - ((x - decoder_out)**2).view((x.shape[0], -1)).sum(dim=1) 
        else:
            D = torch.prod(torch.tensor(x.shape[1:]))
            # sig = torch.tensor(1, dtype=torch.float32)
            sig = self.sigma
            const = - D * 0.5 * torch.log(2 * torch.tensor(np.pi, dtype=torch.float32)) - D * torch.log(sig)
            loglik = const - 0.5 * ((x - decoder_out)**2).view((x.shape[0], -1)).sum(dim=1) / (sig ** 2)
            return loglik

    def error(self, x, x_hat):
        if not self.error_normalize:
            return (((x - x_hat) / self.sigma) ** 2).view(len(x), -1).sum(-1)
        else:
            return ((x - x_hat) ** 2).view(len(x), -1).mean(-1)

    def forward(self, z):
        """returns reconstruction"""
        return self.net(z)

    def sample(self, z):
        if self.deterministic:
            return self.mean(z)
        else:
            x_hat = self.net(z)
            return x_hat + torch.randn_like(x_hat) * self.sigma

    def mean(self, z):
        return self.net(z)

    def max_log_likelihood(self, x):
        if self.deterministic:
            return torch.tensor(0., dtype=torch.float, device=x.device)
        else:
            D = torch.prod(torch.tensor(x.shape[1:]))
            sig = self.sigma
            const = - D * 0.5 * torch.log(2 * torch.tensor(np.pi, dtype=torch.float32)) - D * torch.log(sig)
            return const


class IsotropicLaplace(nn.Module):
    """Isotropic Laplace density function -- equivalent to using L1 error """
    def __init__(self, net, sigma=0.1, sigma_trainable=False):
        super().__init__()
        self.net = net
        self.sigma_trainable = sigma_trainable
        if sigma_trainable:
            self.sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float))
        else:
            self.register_buffer('sigma', torch.tensor(sigma, dtype=torch.float))

    def log_likelihood(self, x, z):
        # decoder_out = self.net(z)
        # D = torch.prod(torch.tensor(x.shape[1:]))
        # sig = torch.tensor(1, dtype=torch.float32)
        # const = - D * 0.5 * torch.log(2 * torch.tensor(np.pi, dtype=torch.float32)) - D * torch.log(sig)
        # loglik = const - 0.5 * (torch.abs(x - decoder_out)).view((x.shape[0], -1)).sum(dim=1) / (sig ** 2)
        # return loglik
        raise NotImplementedError

    def error(self, x, x_hat):
        if self.sigma_trainable:
            return ((torch.abs(x - x_hat) / self.sigma)).view(len(x), -1).sum(-1)
        else:
            return (torch.abs(x - x_hat)).view(len(x), -1).mean(-1)

    def forward(self, z):
        """returns reconstruction"""
        return self.net(z)

    def sample(self, z):
        # x_hat = self.net(z) 
        # return x_hat + torch.randn_like(x_hat) * self.sigma
        raise NotImplementedError


class SampleBuffer:
    def __init__(self, max_samples=10000, replay_ratio=0.95):
        self.max_samples = max_samples
        self.buffer = []
        self.replay_ratio = replay_ratio

    def __len__(self):
        return len(self.buffer)

    def push(self, samples):
        samples = samples.detach().to('cpu')

        for sample in samples:
            self.buffer.append(sample)

            if len(self.buffer) > self.max_samples:
                self.buffer.pop(0)

    def get(self, n_samples):
        samples = random.choices(self.buffer, k=n_samples)
        samples = torch.stack(samples, 0)
        return samples