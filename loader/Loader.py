import numpy as np
import torch
from torch.utils import data

from SVJanalysis.MLFramework.module.DataProcessor import DataProcessor
from SVJanalysis.MLFramework.module.DataLoader import DataLoader


class Loader():


    def __init__(
            self,
            device,
            batch_size,
    ):

        self.device = device
        self.batch_size = batch_size

        data_loader_settings = {
            "max_jets": 2,
            "max_constituents": 0,
            "variables": [
                "FatJet_msoftdrop", "FatJet_tau2", "FatJet_tau3",
                "FatJet_efp1d3", "FatJet_c2b0p5",
                "FatJet_axisMajor", "FatJet_axisMinor", "FatJet_ptD",
            ],
            "variables_on_the_fly": {
                "FatJet_softDropMassLog": "np.log(FatJet_msoftdrop)", # -1 will become NaN and will be replaced by 0 by imputer
            },
            "variables_to_remove": ["FatJet_msoftdrop"],
            "jet_collection_name": "FatJet",
            "max_events_per_file": -1,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "data_seed": None,
        }

        data_processing_settings = {
            "normalization_type": "MinMaxScaler",
            "normalization_args": {
                "feature_range": (-3, 3),
                "copy": True,
            },
            "dtype": 'float32',
        }

        self.samples = {
            "backgrounds": {
                "processes": {
                    "ttbar": {
                        "processes": {
                            "ttbar_files": "/work/fleble/SVJ/store/ml/dataset1/TTJets/PFNANOAODSUPER_TTJets_part-{0..0}.root",
                        },
                        "weights": "cross_section",
                    },
                },
                "weights": "cross_section",
            },
            "weights": "none",
        }

        self.data_loader = DataLoader(**data_loader_settings)
        self.data_processor = DataProcessor(**data_processing_settings)

        # Declare attributes
        self.train_data = None
        self.validation_data = None
        self.train_data_normalized = None
        self.validation_data_normalized = None
        
        # Load and split the data
        self.__load_data()


    def __load_data(self):
        """Load background data for training.
        
        Usual data preprocessing (data imputation and normalization) as well as
        DataTable gymnatics are performed.
        """
        
        samples = self.data_loader.get_data(self.samples, step="train")
        backgrounds = samples["backgrounds"]

        backgrounds.setup_splitting()
        (self.train_data, self.validation_data, _) = self.data_processor.split_to_train_validate_test(backgrounds)

        self.train_data.setup_weights()
        self.validation_data.setup_weights()

        self.data_processor.setup_data_processor(self.train_data)
        train_data_normalized = self.data_processor.transform(self.train_data)
        validation_data_normalized = self.data_processor.transform(self.validation_data)

        # The loader must define the attribute below
        self.training_loader = self.__get_torch_data_loader(train_data_normalized)
        self.validation_loader = self.__get_torch_data_loader(validation_data_normalized)
        self.training_loader_no_batch = self.__get_torch_data_loader(train_data_normalized, batched=False)

        self.all_features = train_data_normalized.df.columns
        self.input_size = len(self.all_features)


    def __get_torch_data_loader(self, data_table, batched=True):
        features = torch.tensor(data_table.df.values.astype(np.float32)).to(self.device)
        weights = torch.tensor(data_table.weights.astype(np.float32)).to(self.device)

        sampler = data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(features),
            replacement=True,
        )

        if batched:
            batch_size = self.batch_size
        else:
            batch_size = len(features)

        loader = data.DataLoader(
            dataset=data.TensorDataset(features),
            batch_size=batch_size,
            sampler=sampler,
        )

        return loader
