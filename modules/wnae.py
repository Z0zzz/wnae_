import numpy as np
import torch
import torch.nn as nn
import ot

from utils.mcmcUtils import sample_langevin
from modules.modules import SampleBuffer, DummyDistribution
from Logger import *


def log_cosh(x):
    """LogCosh loss.

    Uses the following trick: https://datascience.stackexchange.com/questions/96271/logcoshloss-on-pytorch
    """

    return x + torch.nn.functional.softplus(-2*x) - torch.ones_like(x) * np.log(2)


def mse(y_true, y_pred):
    """Mean Squared Error (MSE) with periodic variables.

    Args:
        y_true (torch.Tensor)
        y_pred (torch.Tensor)
    """

    n_dim = np.prod(y_true.shape[1:])

    return ((y_true - y_pred) ** 2).view((y_true.shape[0], -1)).sum(dim=1) / n_dim



class FFEBM(nn.Module):
    """feed-forward energy-based model"""

    def __init__(
        self,
        net,
        x_step=None,
        x_stepsize=None,
        x_noise_std=None,
        x_noise_anneal=None,
        x_bound=None,
        x_clip_langevin_grad=None,
        l2_norm_reg=None,
        buffer_size=10000,
        replay_ratio=0.95,
        replay=True,
        gamma=1,
        sampling="pcd",
        initial_dist="gaussian",
        temperature=1.0,
        temperature_trainable=False,
        mh=False,
        reject_boundary=False,
    ):
        super().__init__()
        self.net = net

        self.x_bound = x_bound
        self.l2_norm_reg = l2_norm_reg
        self.gamma = gamma
        self.sampling = sampling

        assert x_stepsize is not None or x_noise_std is not None
        assert x_stepsize is None or x_stepsize > 0

        if x_stepsize is None or x_noise_std is None:
            if x_stepsize is None:
                x_stepsize = x_noise_std**2 / 2.
            else:
                x_noise_std = np.sqrt(2 * x_stepsize)
        
        self.x_step = x_step
        self.x_stepsize = x_stepsize
        self.x_noise_std = x_noise_std
        self.x_noise_anneal = x_noise_anneal
        self.x_bound = x_bound
        self.x_clip_langevin_grad = x_clip_langevin_grad
        self.mh = mh
        self.reject_boundary = reject_boundary

        self.buffer_size = buffer_size
        self.replay_ratio = replay_ratio
        self.replay = replay

        self.buffer = SampleBuffer(max_samples=buffer_size, replay_ratio=replay_ratio)

        self.x_shape = None
        self.initial_dist = initial_dist
        temperature = np.log(temperature)
        self.temperature_trainable = temperature_trainable
        if temperature_trainable:
            self.register_parameter(
                "temperature_",
                torch.nn.Parameter(torch.tensor(temperature, dtype=torch.float)),
            )
        else:
            self.register_buffer(
                "temperature_", torch.tensor(temperature, dtype=torch.float)
            )

    @property
    def temperature(self):
        return torch.exp(self.temperature_)

    @property
    def sample_shape(self):
        return self.x_shape

    def forward(self, x):
        return self.net(x)

    def predict(self, x):
        return self.forward(x)

    def energy(self, x):
        return self.forward(x)

    def energy_T(self, x):
        return self.energy(x) / self.temperature

    def sample(self, x0=None, n_sample=None, device=None, replay=None):
        """Sampling factory function.
        
        Takes either x0 or n_sample and device.
        """

        if x0 is not None:
            n_sample = len(x0)
            device = x0.device
        if replay is None:
            replay = self.replay

        if self.sampling == "pcd":
            return self.sample_x(n_sample, device, replay=replay)
        elif self.sampling == "cd":
            return self.sample_x(n_sample, device, x0=x0, replay=False)
        elif self.sampling == "omi":
            return self.sample_omi(n_sample, device, replay=replay)

    def sample_x(self, n_sample=None, device=None, x0=None, replay=False):
        if x0 is None:
            x0 = self.initial_sample(n_sample, device=device)
        d_sample_result = sample_langevin(
            x0.detach(),
            self.energy,
            step_size=self.x_stepsize,
            n_steps=self.x_step,
            noise_scale=self.x_noise_std,
            clip_x=self.x_bound,
            noise_anneal=self.x_noise_anneal,
            clip_grad=self.x_clip_langevin_grad,
            spherical=False,
            mh=self.mh,
            temperature=self.temperature,
            reject_boundary=self.reject_boundary,
        )
        sample_result = d_sample_result["sample"]
        if replay:
            self.buffer.push(sample_result)
        d_sample_result["sample_x"] = sample_result
        d_sample_result["sample_x0"] = x0
        return d_sample_result

    def initial_sample(self, n_samples, device):
        l_sample = []
        if not self.replay or len(self.buffer) == 0:
            n_replay = 0
        else:
            n_replay = (np.random.rand(n_samples) < self.replay_ratio).sum()
            l_sample.append(self.buffer.get(n_replay))

        shape = (n_samples - n_replay,) + self.sample_shape
        if self.initial_dist == "gaussian":
            x0_new = torch.randn(shape, dtype=torch.float)
        elif self.initial_dist == "uniform":
            x0_new = torch.rand(shape, dtype=torch.float)
            if self.sampling != "omi" and self.x_bound is not None:
                x0_new = x0_new * (self.x_bound[1] - self.x_bound[0]) + self.x_bound[0]
            elif self.sampling == "omi" and self.z_bound is not None:
                x0_new = x0_new * (self.z_bound[1] - self.z_bound[0]) + self.z_bound[0]
        else:
            log.critical(f"Invalid initial distribution {self.initial_dist}")
            exit(1)

        l_sample.append(x0_new)
        return torch.cat(l_sample).to(device)

    def _set_x_shape(self, x):
        if self.x_shape is not None:
            return
        self.x_shape = x.shape[1:]

    def weight_norm(self, net):
        norm = 0
        for param in net.parameters():
            norm += (param**2).sum()
        return norm

    def train_step(self, x, opt):
        self._set_x_shape(x)

        # negative sample
        d_sample = self.sample(x)
        x_neg = d_sample["sample_x"]

        opt.zero_grad()
        negative_energy = self.energy(x_neg)

        # ae recon pass
        positive_energy = self.energy(x)

        loss = (positive_energy.mean() - negative_energy.mean()) / self.temperature

        if self.gamma is not None:
            loss += self.gamma * (positive_energy**2 + negative_energy**2).mean()

        # weight regularization
        l2_norm = self.weight_norm(self.net)
        if self.l2_norm_reg is not None:
            loss = loss + self.l2_norm_reg * l2_norm

        loss.backward()
        opt.step()

        d_result = {
            "positive_energy": positive_energy.mean().item(),
            "negative_energy": negative_energy.mean().item(),
            "x_neg": x_neg.detach().cpu(),
            "x_neg_0": d_sample["sample_x0"].detach().cpu(),
            "loss": loss.item(),
            "sample": x_neg.detach().cpu(),
            "l2_norm": l2_norm.item(),
        }
        return d_result

    def validation_step(self, x):
        positive_energy = self.energy(x)
        loss = positive_energy.mean().item()
        predict = positive_energy.detach().cpu().flatten()
        return {"loss": positive_energy, "predict": predict}



class WNAE(FFEBM):
    """Wasserstien Normalized Autoencoder"""

    def __init__(
        self,
        encoder,
        decoder,
        z_step=50,
        z_stepsize=0.2,
        z_noise_std=0.2,
        z_noise_anneal=None,
        x_step=50,
        x_stepsize=10,
        x_noise_std=0.05,
        x_noise_anneal=None,
        x_bound=(0, 1),
        z_bound=None,
        z_clip_langevin_grad=None,
        x_clip_langevin_grad=None,
        l1_regularization_coef_positive_energy=None,
        l1_regularization_coef_negative_energy=None,
        l2_regularization_coef_positive_energy=None,
        l2_regularization_coef_negative_energy=None,
        l2_regularization_coef_decoder=None,
        l2_regularization_coef_encoder=None,
        l2_regularization_coef_ae=None,
        l2_regularization_coef_latent_space=None,
        coef_energy_difference=None,
        regularization_coef_emd=None,
        spherical=False,
        buffer_size=10000,
        replay_ratio=0.95,
        replay=True,
        sampling="omi",
        temperature=1.0,
        temperature_trainable=True,
        initial_dist="gaussian",
        mh=False,
        mh_z=False,
        reject_boundary=False,
        reject_boundary_z=False,
        use_log_cosh=False,
    ):
        """
        encoder: An encoder network, an instance of nn.Module.
        decoder: A decoder network, an instance of nn.Module.

        **Sampling Parameters**
        sampling: Sampling methods.
                  'omi' - on-manifold initialization.
                  'cd' - Contrastive Divergence.
                  'pcd' - Persistent CD.

        z_step: The number of steps in latent chain.
        z_stepsize: The step size of latent chain
        z_noise_std: The standard deviation of noise in latent chain
        z_noise_anneal: Noise annealing parameter in latent chain. If None, no annealing.
        mh_z: If True, use Metropolis-Hastings rejection in latent chain.
        z_clip_langevin_grad: Clip the norm of gradient in latent chain.
        z_bound: [z_min, z_max]

        x_step: The number of steps in visible chain.
        x_stepsize: The step size of visible chain
        x_noise_std: The standard deviation of noise in visible chain
        x_noise_anneal: Noise annealing parameter in visible chain. If None, no annealing.
        mh: If True, use Metropolis-Hastings rejection in latent chain.
        x_clip_langevin_grad: Clip the norm of gradient in visible chain.
        x_bound: [x_min, x_bound].

        replay: Whether to use the replay buffer.
        buffer_size: The size of replay buffer.
        replay_ratio: The probability of applying persistent CD. A chain is re-initialized with the probability of
                      (1 - replay_ratio).
        initial_dist: The distribution from which initial samples are generated.
                      'gaussian' or 'uniform'



        **Regularization Parameters**
        l2_regularization_coef_positive_energy: The coefficient for regularizing the positive sample energy.
        l2_regularization_coef_negative_energy: The coefficient for regularizing the negative sample energy.
        l2_regularization_coef_decoder: The coefficient for L2 norm of decoder weights.
        l2_regularization_coef_encoder: The coefficient for L2 norm of encoder weights.
        l2_regularization_coef_latent_space: The coefficient for regularizing the L2 norm of Z vector.
        regularization_coef_emd: The coefficient for regularizing the Energy Mover's Distance between
            negative and positive samples.
        coef_energy_difference: The coefficient for regularizing the energy difference.

        use_log_cosh (bool): Take the log(cosh) of the loss instead of the usual loss
        """

        super().__init__(
            net=None,
            x_step=x_step,
            x_stepsize=x_stepsize,
            x_noise_std=x_noise_std,
            x_noise_anneal=x_noise_anneal,
            x_bound=x_bound,
            x_clip_langevin_grad=x_clip_langevin_grad,
            l2_norm_reg=l2_regularization_coef_decoder,
            buffer_size=buffer_size,
            replay_ratio=replay_ratio,
            replay=replay,
            gamma=None,
            sampling=sampling,
            initial_dist=initial_dist,
            temperature=temperature,
            temperature_trainable=temperature_trainable,
            mh=mh,
            reject_boundary=reject_boundary,
        )

        assert z_stepsize is not None or z_noise_std is not None
        assert z_stepsize is None or z_stepsize > 0

        if z_stepsize is None or z_noise_std is None:
            if z_stepsize is None:
                z_stepsize = z_noise_std**2 / 2.
            else:
                z_noise_std = np.sqrt(2 * z_stepsize)

        self.encoder = encoder
        self.decoder = DummyDistribution(decoder)

        self.z_step = z_step
        self.z_stepsize = z_stepsize
        self.z_noise_std = z_noise_std
        self.z_noise_anneal = z_noise_anneal
        self.z_clip_langevin_grad = z_clip_langevin_grad
        self.mh_z = mh_z
        self.reject_boundary_z = reject_boundary_z
        self.z_bound = z_bound

        self.l1_regularization_coef_positive_energy = l1_regularization_coef_positive_energy
        self.l1_regularization_coef_negative_energy = l1_regularization_coef_negative_energy
        self.l2_regularization_coef_positive_energy = l2_regularization_coef_positive_energy
        self.l2_regularization_coef_negative_energy = l2_regularization_coef_negative_energy
        self.l2_regularization_coef_decoder = l2_regularization_coef_decoder
        self.l2_regularization_coef_encoder = l2_regularization_coef_encoder
        self.l2_regularization_coef_ae = l2_regularization_coef_ae
        self.l2_regularization_coef_latent_space = l2_regularization_coef_latent_space
        self.regularization_coef_emd = regularization_coef_emd
        self.coef_energy_difference = 1. if coef_energy_difference is None else coef_energy_difference
        self.regularisation_coefficients_names = [
            "l1_regularization_coef_positive_energy",
            "l1_regularization_coef_positive_energy",
            "l2_regularization_coef_negative_energy",
            "l2_regularization_coef_negative_energy",
            "l2_regularization_coef_decoder",
            "l2_regularization_coef_encoder",
            "l2_regularization_coef_ae",
            "l2_regularization_coef_latent_space",
            "coef_energy_difference",
        ]

        self.spherical = spherical
        self.sampling = sampling

        self.z_shape = None
        self.x_shape = None

        self.use_log_cosh = use_log_cosh

    @property
    def sample_shape(self):
        if self.sampling == "omi":
            return self.z_shape
        else:
            return self.x_shape

    def error(self, x, recon):
        """MSE error"""
        return mse(x, recon)

    def forward(self, x):
        """Computes error per dimension"""
        z = self.encode(x)
        recon = self.decoder(z)
        return self.error(x, recon)

    def energy_with_z(self, x):
        z = self.encode(x)
        recon = self.decoder(z)
        return self.error(x, recon), z

    def normalize(self, z):
        """normalize to unit length"""
        if self.spherical:
            if len(z.shape) == 4:
                z = z / z.view(len(z), -1).norm(dim=-1)[:, None, None, None]
            else:
                z = z / z.view(len(z), -1).norm(dim=1, keepdim=True)
            return z
        else:
            return z

    def encode(self, x):
        if self.spherical:
            return self.normalize(self.encoder(x))
        else:
            return self.encoder(x)

    def sample_omi(self, n_sample, device, replay=False):
        """using on-manifold initialization"""
        # Step 1: On-manifold initialization: LMC on Z space
        z0 = self.initial_sample(n_sample, device)
        if self.spherical:
            z0 = self.normalize(z0)
        d_sample_z = self.sample_z(z0=z0, replay=replay)
        sample_z = d_sample_z["sample"]

        sample_x_1 = self.decoder(sample_z).detach()
        if self.x_bound is not None:
            sample_x_1.clamp_(self.x_bound[0], self.x_bound[1])

        # Step 2: LMC on X space
        d_sample_x = self.sample_x(x0=sample_x_1, replay=False)
        sample_x_2 = d_sample_x["sample_x"]
        return {
            "sample_x": sample_x_2,
            "sample_z": sample_z.detach(),
            "sample_x0": sample_x_1,
            "sample_z0": z0.detach(),
        }

    def sample_z(self, n_sample=None, device=None, replay=False, z0=None):
        if z0 is None:
            z0 = self.initial_sample(n_sample, device)
        energy = lambda z: self.energy(self.decoder(z))
        d_sample_result = sample_langevin(
            z0,
            energy,
            step_size=self.z_stepsize,
            n_steps=self.z_step,
            noise_scale=self.z_noise_std,
            clip_x=self.z_bound,
            clip_grad=self.z_clip_langevin_grad,
            spherical=self.spherical,
            mh=self.mh_z,
            temperature=self.temperature,
            reject_boundary=self.reject_boundary_z,
        )
        sample_z = d_sample_result["sample"]
        if replay:
            self.buffer.push(sample_z)
        return d_sample_result

    def _set_z_shape(self, x):
        if self.z_shape is not None:
            return
        # infer z_shape by computing forward
        with torch.no_grad():
            dummy_z = self.encode(x[[0]])
        z_shape = dummy_z.shape
        self.z_shape = z_shape[1:]

    def predict(self, x):
        z = self.encode(x)
        return self.decoder(z)

    def set_shapes(self, x):
        self._set_z_shape(x)
        self._set_x_shape(x)

    def get_energies_and_z(self, x, d_sample):
        x_neg = d_sample["sample_x"]
        negative_energy, neg_z = self.energy_with_z(x_neg)
        positive_energy, pos_z = self.energy_with_z(x)

        return positive_energy, negative_energy, pos_z, neg_z
    
    def get_norms(self, pos_z, neg_z):
        encoder_norm = self.weight_norm(self.encoder)
        decoder_norm = self.weight_norm(self.decoder)
        ae_norm = encoder_norm + decoder_norm
        positive_latent_space_norm = (pos_z ** 2).mean()
        if neg_z is not None:
            negative_latent_space_norm = (neg_z ** 2).mean()
            latent_space_norm = (len(pos_z) * positive_latent_space_norm + len(neg_z) * negative_latent_space_norm) / (len(pos_z) + len(neg_z))
        else:
            negative_latent_space_norm = None
            latent_space_norm = None
        return decoder_norm, encoder_norm, ae_norm, positive_latent_space_norm, negative_latent_space_norm, latent_space_norm

    def __compute_emd(self, positive_samples, negative_samples):
        if int(ot.__version__.split(".")[1]) < 9:
            log.warning(f"Your optimal transport ot version is {ot.__version__}")
            log.warning(f"EMD calculation not supported for gradient descent, will probably crash.")
        loss_matrix = ot.dist(positive_samples, negative_samples)
        n_examples = len(positive_samples)
        weights = torch.ones(n_examples) / n_examples
        emd = ot.emd2(
            weights,
            weights,
            loss_matrix,
            numItermax=1e6,
        )
    
        return emd

    def __regularization_name_to_human_readable(self, name):
        return name.replace("regularization_coef_", "").replace("coef_", "")

    def regularize_loss(
            self,
            loss_function_expression,
            loss,
            positive_energy,
            negative_energy,
            encoder_norm,
            decoder_norm,
            ae_norm,
            latent_space_norm,
            positive_samples,
            negative_samples,
        ):

        regularization_terms = {}
        
        # EMD calculation takes time, only calculating if used
        if self.regularization_coef_emd is not None:
            emd = self.__compute_emd(positive_samples, negative_samples)
        else:
            emd = 0.

        # Terms info dict decription:
        # Key: regularisation coef name
        # Value: [regularisation term name to build loss fct expression, regularisation term value]
        terms_info = {
            "l1_regularization_coef_negative_energy": [
                "E-",
                negative_energy.mean() if negative_energy is not None else None,
            ],
            "l1_regularization_coef_positive_energy": [
                "E+",
                positive_energy.mean(),
            ],
            "l2_regularization_coef_negative_energy": [
                "<E-**2>",
                ((negative_energy) ** 2).mean() if negative_energy is not None else None,
            ],
            "l2_regularization_coef_positive_energy": [
                "<E+**2>",
                ((positive_energy) ** 2).mean(),
            ],
            "l2_regularization_coef_decoder": [
                "||D||",
                decoder_norm,
            ],
            "l2_regularization_coef_encoder": [
                "||E||",
                encoder_norm,
            ],
            "l2_regularization_coef_ae": [
                "||AE||",
                ae_norm,
            ],
            "l2_regularization_coef_latent_space": [
                "||L||",
                latent_space_norm,
            ],
            "regularization_coef_emd": [
                "EMD",
                emd,
            ],
        }

        for regularization_term_name, expression_value in terms_info.items():
            term_expression, value = expression_value
            regularization_coef = getattr(self, regularization_term_name)
            human_readable_name = self.__regularization_name_to_human_readable(regularization_term_name)
            if regularization_coef is not None and value is not None:
                log.info(f"Adding {regularization_term_name} to loss.", repeat=False)
                regularization_term = regularization_coef * value
                loss += regularization_term
                loss_function_expression += f" + {regularization_coef} * {term_expression}"
                regularization_term = regularization_term.detach().item()
            else:
                regularization_term = 0.

            regularization_terms[human_readable_name] = regularization_term

        return loss_function_expression, loss, regularization_terms

    def __add_regularisation_coefficients_to_result_dict(self, d_result):
        for regularisation_coef_name in self.regularisation_coefficients_names:
            coef = getattr(self, regularisation_coef_name) or 0.
            d_result[regularisation_coef_name] = coef

    def __add_regularisation_terms_to_result_dict(self, d_result, regularization_terms):
        for regularisation_name, value in regularization_terms.items():
            regularisation_name += "_term"
            d_result[regularisation_name] = value

    def get_d_result_ae(
            self, positive_energy, loss, encoder_norm, decoder_norm, ae_norm,
            positive_latent_space_norm, pos_z,
        ):

        d_result = {
            "positive_energy": positive_energy.mean().item(),
            "loss": loss.item(),
            "loss_sample": positive_energy.detach().cpu(),
            "decoder_norm": decoder_norm.item(),
            "encoder_norm": encoder_norm.item(),
            "ae_norm": ae_norm.item(),
            "positive_latent_space_norm": positive_latent_space_norm.item(),
            "pos_z": pos_z.detach().cpu(),
        }
        self.__add_regularisation_coefficients_to_result_dict(d_result)

        return d_result

    def get_d_result_nae(
            self, d_sample, energy_difference_term, positive_energy, negative_energy,
            loss, encoder_norm, decoder_norm, ae_norm,
            positive_latent_space_norm, negative_latent_space_norm, latent_space_norm,
            pos_z, neg_z, regularization_terms,
        ):

        # for debugging
        x_neg = d_sample["sample_x"]
        x_neg_0 = d_sample["sample_x0"]
        negative_energy_x0 = self.energy(x_neg_0)  # energy of samples from latent chain
        recon_neg = self.predict(x_neg)

        positive_energy_scalar = positive_energy.mean()
        negative_energy_scalar = negative_energy.mean()
        energy_difference_scalar = positive_energy_scalar - negative_energy_scalar

        d_result = {
            "energy_difference_term": energy_difference_term,
            "positive_energy": positive_energy_scalar.item(),
            "negative_energy": negative_energy_scalar.item(),
            "energy_difference": energy_difference_scalar.item(),
            "recon_neg": recon_neg.detach().cpu(),
            "loss": loss.item(),
            "loss_sample": positive_energy.detach().cpu(),
            "negative_samples": x_neg.detach().cpu(),
            "initial_negative_samples": x_neg_0.detach().cpu(),
            "decoder_norm": decoder_norm.item(),
            "encoder_norm": encoder_norm.item(),
            "ae_norm": ae_norm.item(),
            "positive_latent_space_norm": positive_latent_space_norm.item(),
            "negative_latent_space_norm": negative_latent_space_norm.item(),
            "latent_space_norm": latent_space_norm.item(),
            "negative_energy_x0": negative_energy_x0.mean().item(),
            "temperature": self.temperature.item(),
            "pos_z": pos_z.detach().cpu(),
            "neg_z": neg_z.detach().cpu(),
        }

        self.__add_regularisation_coefficients_to_result_dict(d_result)
        self.__add_regularisation_terms_to_result_dict(d_result, regularization_terms)

        if "sample_z0" in d_sample:
            x_neg_z0 = self.decoder(d_sample["sample_z0"])
            d_result["negative_energy_z0"] = self.energy(x_neg_z0).mean().item()

        return d_result

    def __send_warning_disabled_parameter(self, parameter_name, step_name, additional_text=""):
        log.warning(
            f"{parameter_name} disabled for {step_name}, will be ignored. {additional_text}",
            repeat=False,
        )

    def train_step_ae(self, x, produce_negative_samples=False):

        if produce_negative_samples:
            self.set_shapes(x)
            d_sample = self.sample(x)
            positive_energy, negative_energy, pos_z, neg_z = self.get_energies_and_z(x, d_sample)
        else:
            positive_energy, pos_z = self.energy_with_z(x)
            negative_energy, neg_z = None, None

        loss = positive_energy.mean()
        energy_difference_term = 0.
        loss_function_expression = "E+"  # to print the form of the loss

        if self.temperature_trainable:
            self.__send_warning_disabled_parameter(
                "temperature_trainable",
                "standard AE training step",
                "This is an NAE feature.",
            )

        if self.use_log_cosh:
            self.__send_warning_disabled_parameter(
                "use_log_cosh",
                "standard AE training step",
                "This is an NAE feature.",
            )

        decoder_norm, encoder_norm, ae_norm, positive_latent_space_norm, negative_latent_space_norm, latent_space_norm = \
            self.get_norms(pos_z, neg_z)
        loss_function_expression, loss, regularization_terms, _ = self.regularize_loss(
            loss_function_expression, loss, positive_energy, None,
            encoder_norm, decoder_norm, ae_norm, latent_space_norm, x, None,
        )

        if produce_negative_samples:
            d_result = self.get_d_result_nae(
                d_sample, energy_difference_term, positive_energy, negative_energy,
                loss, encoder_norm, decoder_norm, ae_norm,
                positive_latent_space_norm, negative_latent_space_norm, latent_space_norm,
                pos_z, neg_z, regularization_terms,
            )
            d_result["energy_difference_term"] = (positive_energy.mean() - negative_energy.mean()).detach().item()

        else:
            d_result = self.get_d_result_ae(
                positive_energy, loss, encoder_norm, decoder_norm, ae_norm,
                positive_latent_space_norm, pos_z
            )

        return loss, d_result

    def train_step_nae(self, x):
        """Standard NAE train step.
        
        Args:
            x (torch.Tensor): Data
        """

        loss_function_expression = ""  # to print the form of the loss

        self.set_shapes(x)
        d_sample = self.sample(x)
        positive_energy, negative_energy, pos_z, neg_z = self.get_energies_and_z(x, d_sample)

        loss = positive_energy.mean() - negative_energy.mean()
        loss_function_expression += "(E+ - E-)"

        if self.temperature_trainable:
            loss = loss + loss.detach() / self.temperature
            loss_function_expression += f" + {loss_function_expression} / T"

        if self.use_log_cosh:
            loss = log_cosh(loss)
            loss_function_expression = f"log(cosh({loss_function_expression}))"

        if self.coef_energy_difference == 0:
            loss = torch.Tensor([0])
            loss_function_expression = ""
        else:
            loss = self.coef_energy_difference * loss
            loss_function_expression = f"{self.coef_energy_difference} * {loss_function_expression}"

        energy_difference_term = loss.clone()

        decoder_norm, encoder_norm, ae_norm, positive_latent_space_norm, negative_latent_space_norm, latent_space_norm = \
            self.get_norms(pos_z, neg_z)
        loss_function_expression, loss, regularization_terms = self.regularize_loss(
            loss_function_expression, loss, positive_energy, negative_energy,
            encoder_norm, decoder_norm, ae_norm, latent_space_norm, x, d_sample["sample_x"],
        )

        if loss_function_expression.startswith(" + "):
            loss_function_expression = loss_function_expression[3:]

        log.info("Loss function:", repeat=False)
        log.info(loss_function_expression, repeat=False)

        energy_difference_term = energy_difference_term.detach().item()

        d_result = self.get_d_result_nae(
            d_sample, energy_difference_term, positive_energy, negative_energy,
            loss, encoder_norm, decoder_norm, ae_norm,
            positive_latent_space_norm, negative_latent_space_norm, latent_space_norm,
            pos_z, neg_z, regularization_terms,
        )
        
        return loss, d_result

    def validation_step(self, x):
        energy = self.energy(x)
        loss = energy.mean().item()
        d_result = {
            "loss": loss,
            "loss_sample": energy.detach().cpu(),
            "positive_energy": loss,
        }
        return d_result

    def run_mcmc(self, x):
        self.set_shapes(x)
        d_sample = self.sample(x, replay=False)
        return d_sample["sample_x"]

