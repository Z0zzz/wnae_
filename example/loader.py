import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer


class Loader():
    """User defined data loader.
    
    To work with the example trainer, must implement the definition of the
    following attributes:
        * training_loader (torch.utils.data.DataLoader)
        * validation_loader (torch.utils.data.DataLoader)
        * validation_loader_no_batch (torch.utils.data.DataLoader)
        * ood_loader (torch.utils.data.DataLoader)
        * input_size (int)
    """

    def __init__(
            self,
            device,
            batch_size,
    ):

        self.device = device
        self.batch_size = batch_size
        self.data_split_seed = 12
        self.data_generation_seed = 123

        # Load and prepare the data for training and evaluation of performance
        data = self.__generate_toy_data()
        ood_data = self.__generate_toy_ood_data()

        training_data, validation_data, _ = self.__split_data(
            data=data,
            validation_fraction=0.45,
            test_fraction=0.1,
        )
        scaler = self.__fit_scaler(training_data)
        training_data_normalized = self.__normalize_data(training_data, scaler)
        validation_data_normalized = self.__normalize_data(validation_data, scaler)
        ood_data_normalized = self.__normalize_data(ood_data, scaler)

        self.training_data_normalized = training_data_normalized
        self.ood_data_normalized = ood_data_normalized

        self.input_size = training_data.shape[1]
        self.training_loader = self.__get_torch_data_loader(training_data_normalized, batched=True)
        self.validation_loader = self.__get_torch_data_loader(validation_data_normalized, batched=True)
        self.validation_loader_no_batch = self.__get_torch_data_loader(validation_data_normalized, batched=False)
        self.ood_loader = self.__get_torch_data_loader(ood_data_normalized, batched=False)

    def __generate_toy_data(self, n=10**5):
        np.random.seed(self.data_generation_seed)
        x1 = np.random.normal(0, 1, size=n)
        x2 = x1 + np.random.normal(0, 0.4, size=n)
        data = np.array([x1, x2]).T
        return data

    def __generate_toy_ood_data(self, n=10**5):
        np.random.seed(self.data_generation_seed)
        x1 = np.random.normal(2, 1, size=n)
        x2 = x1 + np.random.normal(0, 0.4, size=n)
        data = np.array([x1, x2]).T
        return data

    @staticmethod
    def plot_data(data_1, data_2, data_1_legend, data_2_legend, plot_name):

        def __sample(data, n):
            indices = np.random.choice(
                np.arange(0, len(data)),
                size=int(n),
                replace=True,
            )
            return data[indices]

        rc_params = {
            "mathtext.default": "regular",
            "font.size": 25,
            "axes.labelsize": "large",
            "axes.unicode_minus": False,
            "xtick.labelsize": "large",
            "ytick.labelsize": "large",
            "legend.handlelength": 1.5,
            "legend.borderpad": 0.5,
            "legend.frameon": False,
            "xtick.direction": "in",
            "xtick.major.size": 12,
            "xtick.minor.size": 6,
            "xtick.major.pad": 6,
            "xtick.top": True,
            "xtick.major.top": True,
            "xtick.major.bottom": True,
            "xtick.minor.top": True,
            "xtick.minor.bottom": True,
            "xtick.minor.visible": True,
            "ytick.direction": "in",
            "ytick.major.size": 12,
            "ytick.minor.size": 6.0,
            "ytick.right": True,
            "ytick.major.left": True,
            "ytick.major.right": True,
            "ytick.minor.left": True,
            "ytick.minor.right": True,
            "ytick.minor.visible": True,
            "grid.alpha": 0.8,
            "grid.linestyle": ":",
            "axes.linewidth": 2,
            "savefig.transparent": False,
            "figure.figsize": (15.0, 10.0),
            "legend.numpoints": 1,
            "lines.markersize": 8,
        }
        
        for k, v in rc_params.items():
            plt.rcParams[k] = v

        plt.plot()
        levels = [0.5, 0.68, 0.90, 0.95]

        n_data_max = 5e3
        if len(data_1) > n_data_max:
            data_1 = __sample(data_1, n=n_data_max)
        if len(data_2) > n_data_max:
            data_2 = __sample(data_2, n=n_data_max)

        sns.kdeplot(
            x=data_1[:, 0],
            y=data_1[:, 1],
            levels=levels,
            color="blue",
            label=data_1_legend,
        )
        sns.kdeplot(
            x=data_2[:, 0],
            y=data_2[:, 1],
            levels=levels,
            color="red",
            label=data_2_legend,
        )
        x_min = min(np.percentile(data_1[:, 0], 3), np.percentile(data_2[:, 0], 3))
        x_max = max(np.percentile(data_1[:, 0], 97), np.percentile(data_2[:, 0], 97))
        y_min = min(np.percentile(data_1[:, 1], 3), np.percentile(data_2[:, 1], 3))
        y_max = max(np.percentile(data_1[:, 1], 97), np.percentile(data_2[:, 1], 97))
        plt.xlim((x_min, x_max))
        plt.ylim((y_min, y_max))
        plt.legend(loc="best")

        print(f"Plot of toy data saved in {plot_name}")
        plt.savefig(plot_name)

    def __split_data(self, data, validation_fraction, test_fraction):
        training_data, validation_test_data = train_test_split(
            data,
            test_size=test_fraction + validation_fraction,
            random_state=self.data_split_seed,
            shuffle=True,
        )
        validation_data, test_data = train_test_split(
            validation_test_data,
            test_size=test_fraction / (test_fraction + validation_fraction),
            random_state=self.data_split_seed,
            shuffle=True,
        )

        return training_data, validation_data, test_data

    def __fit_scaler(self, data):
        scaler = QuantileTransformer(n_quantiles=1000, output_distribution="normal", copy=True)
        scaler.fit(data)
        return scaler

    def __normalize_data(self, data, scaler):
        return scaler.transform(data)

    def __get_torch_data_loader(self, data, batched=True):
        data = torch.tensor(data.astype(np.float32)).to(self.device)

        sampler = torch.utils.data.RandomSampler(
            data_source=data,
            num_samples=2**13, # fixing the size of an epoch
            replacement=True,
        )

        if batched:
            batch_size = self.batch_size
        else:
            batch_size = len(data)

        loader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(data),
            batch_size=batch_size,
            sampler=sampler,
        )

        return loader