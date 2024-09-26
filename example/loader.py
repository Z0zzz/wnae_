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
            validation_fraction=0.15,
            test_fraction=0.15,
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

    def __generate_toy_data(self, n=10**4):
        np.random.seed(self.data_generation_seed)
        x1 = np.random.normal(0, 1, size=n)
        x2 = x1 + np.random.normal(0, 0.2, size=n)
        data = np.array([x1, x2]).T
        return data

    def __generate_toy_ood_data(self, n=10**4):
        np.random.seed(self.data_generation_seed)
        x1 = np.random.normal(2, 1, size=n)
        x2 = x1 + np.random.normal(0, 0.2, size=n)
        data = np.array([x1, x2]).T
        return data

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
            num_samples=len(data),
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
