import numpy as np
import torch
from torch.utils import data

from SVJanalysis.MLFramework.module.DataProcessor import DataProcessor
from SVJanalysis.MLFramework.module.DataLoader import DataLoader


class Loader():
    """User defined data loader.
    
    To work with the example trainer, must implement the definition of the
    following attributes:
        * training_loader (torch.utils.data.DataLoader)
        * validation_loader (torch.utils.data.DataLoader)
        * validation_loader_no_batch (torch.utils.data.DataLoader)
        * ood_loader_no_batch (torch.utils.data.DataLoader)
        * input_size (int)
    """

    def __init__(
            self,
            device,
            batch_size,
    ):

        self.device = device
        self.batch_size = batch_size

        data_loader_settings = {
            "max_jets": None,
            "max_constituents": None,
            "variables": [
                "JetsAK8_softDropMass", 
                "JetsAK8_NsubjettinessTau1",
                "JetsAK8_NsubjettinessTau2",
                "JetsAK8_NsubjettinessTau3",
                "JetsAK8_NsubjettinessTau4",
                "JetsAK8_NsubjettinessTau5",
                "JetsAK8_axismajor",
                "JetsAK8_axisminor",
                "JetsAK8_ptD",
                "JetsAK8_chargedHadronEnergyFraction",
                "JetsAK8_neutralHadronEnergyFraction",
                "JetsAK8_electronEnergyFraction",
                "JetsAK8_photonEnergyFraction",
                "JetsAK8_girth",
                "JetsAK8_ecfFullC2b1",
                "JetsAK8_ecfFullC2b2",
                "JetsAK8_ecfFullD2b1",
                "JetsAK8_ecfFullD2b2",
                "JetsAK8_ecfFullM2b1",
                "JetsAK8_ecfFullM2b2",
                "JetsAK8_ecfFullN2b1",
                "JetsAK8_ecfFullN2b2",
            ],
            "variables_on_the_fly": {
                "JetsAK8_softDropMassLog": "np.log(JetsAK8_softDropMass + 0.001)", # -1 will become NaN and will be replaced by 0 by imputer
            },
            "variables_to_remove": ["JetsAK8_softDropMass"],
            "jet_collection_name": "JetsAK8",
            "max_events_per_file": -1,
            "validation_fraction": 0.15,
            "test_fraction": 0.15,
            "data_seed": None,
            "cuts": [
                {"cut": "JetsAK8_isDarkJetTight == 1"},
                {"cut": "JetsAK8_isGood == 1"},
                {"cut": "JetsAK8_isTrainingJet == 1"},
                {"cut": "JetsAK8_pt > 400"},
                {"cut": "JetsAK8_pt < 500"},
            ],
        }

        data_processing_settings = {
            "normalization_type": "MinMaxScaler",
            "normalization_args": {
                "feature_range": (-3, 3),
                "copy": True,
            },
            "dtype": 'float32',
        }

        path_qcd = "/pnfs/psi.ch/cms/trivcat/store/user/fleble/MLFW/tree_maker/data_qcd_tr/dataset1/2018/EGamma/"
        path_ttjets_ele = "/pnfs/psi.ch/cms/trivcat/store/user/fleble/MLFW/tree_maker/data_top_tr/dataset1/2018/EGamma/"
        path_ttjets_mu = "/pnfs/psi.ch/cms/trivcat/store/user/fleble/MLFW/tree_maker/data_top_tr/dataset1/2018/SingleMuon/"
        self.samples = {
            "backgrounds": {
                "processes": {
                    "QCD": {
                        "processes": {
                                # "QCD": path_qcd + "part-{0..0}.root",
                                "QCD": path_qcd + "part-{0..26}.root",
                        },
                        "weights": "cross_section",
                    },
                    "ttjets": {
                        "processes": {
                            # "ttjets_ele": path_ttjets_ele + "part-{0..0}.root",
                            # "ttjets_mu": path_ttjets_mu + "part-{0..0}.root",
                            "ttjets_ele": path_ttjets_ele + "part-{0..12}.root",
                            "ttjets_mu": path_ttjets_mu + "part-{0..10}.root",
                        },
                        "weights": "cross_section",
                    },
                },
                "weights": "weights:2_8",
            }
        }

        self.ood_samples = {
            "backgrounds": {
                "processes": {
                    "ttbar": {
                        "processes": {
                            "ood_files": "/pnfs/psi.ch/cms/trivcat/store/user/fleble/MLFW/tree_maker/dataset7/2018/t-channel_mMed-2000_mDark-20_rinv-0p3_alpha-peak_yukawa-1/part-0.root",
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
        (train_data, validation_data, _) = self.data_processor.split_to_train_validate_test(backgrounds)

        train_data.setup_weights()
        validation_data.setup_weights()

        self.data_processor.setup_data_processor(train_data)
        train_data_normalized = self.data_processor.transform(train_data)
        validation_data_normalized = self.data_processor.transform(validation_data)

        # The loader must define the attribute below
        self.training_loader = self.__get_torch_data_loader(train_data_normalized)
        self.validation_loader = self.__get_torch_data_loader(validation_data_normalized)
        self.training_loader_no_batch = self.__get_torch_data_loader(train_data_normalized, batched=False)
        self.validation_loader_no_batch = self.__get_torch_data_loader(validation_data_normalized, batched=False)

        # self.all_features = train_data_normalized.df.columns
        self.input_size = len(train_data_normalized.df.columns)

        ood_data = self.data_loader.get_data(self.ood_samples, step="train")["backgrounds"]
        ood_data.setup_splitting()
        ood_data.setup_weights()
        ood_data_normalized = self.data_processor.transform(ood_data)
        self.ood_loader_no_batch = self.__get_torch_data_loader(ood_data_normalized, batched=False)


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
