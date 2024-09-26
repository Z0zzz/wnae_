import time
from pathlib import Path
import time

import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from wnae import WNAE
from wnae._logger import log


class TrainerWassersteinNormalizedAutoEncoder():
    
    def __init__(
            self,
            config,
            loader,
            encoder,
            decoder,
            device,
            output_path,
            loss_function="wnae",
        ):
        """
        Constructor of the specialized Trainer class.
        """

        self.config = config
        self.loader = loader
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.output_path = output_path
        self.loss_function = loss_function
        self.metrics_tracker = {
            "epoch": [],
            "training_loss": [],
            "validation_loss": [],
            "auc": [],
        }

        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        self.hyper_parameters = {}
        self.model = self.__get_model()

    def __train_epoch(self, training_loader, optimizer):

        self.model.train()

        # Can monitore more quantities than the loss, showing loss as example
        monitored_quantities = {
            "loss": 0.,
        }

        n_batches = 0
        bar_format = '{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        for batch in tqdm(training_loader, bar_format=bar_format):
            n_batches += 1
            x = batch[0]  # batch is a list of len 1 with the tensor inside

            optimizer.zero_grad()
            if self.loss_function == "ae":
                loss, training_dict = self.model.train_step_ae(x)
            elif self.loss_function == "nae":
                loss, training_dict = self.model.train_step_nae(x)
            elif self.loss_function == "wnae":
                loss, training_dict = self.model.train_step(x)
            loss.backward()
            optimizer.step()

            monitored_quantities["loss"] += training_dict["loss"]
            #print(training_dict["mcmc_data"]["samples"][-1][:10])

        monitored_quantities["loss"] /= n_batches

        return monitored_quantities

    def __evaluate(self, loader):
        self.model.eval()
        x = next(iter(loader))[0]
        return self.model.evaluate(x)
  
    def __validate_epoch(self, validation_loader):

        self.model.eval()

        # Can monitore more quantities than the loss, showing loss as example
        monitored_quantities = {
            "loss": 0.,
        }

        n_batches = 0
        for batch in validation_loader:
            n_batches += 1
            x = batch[0]  # x is a list of len 1 with the tensor inside

            validation_dict = self.model.validation_step(x)
            monitored_quantities["loss"] += validation_dict["loss"]

        monitored_quantities["loss"] /= n_batches

        return monitored_quantities

    def __save_model_checkpoint(self, name, state_dict_only=False):
        path = f"{self.output_path}/{name}.pt"
        if state_dict_only:
            torch.save({"model_state_dict": self.model.state_dict()}, path)
        else:
            torch.save(self.model, path)
        log.info(f"Saved model checkpoint {path}")

    def __fit(self,
              n_epochs,
              optimizer,
              lr_scheduler,
              es_patience,
        ):
        """Fit model.

        Args:
            n_epochs (int): Max number of epochs
            optimizer (torch.optim)
            lr_scheduler (torch.optim.lr_scheduler)
            es_patience (int): Number of epochs for early stop        
        """

        training_loader = self.loader.training_loader
        validation_loader = self.loader.validation_loader
        validation_loader_no_batch = self.loader.validation_loader_no_batch
        ood_loader = self.loader.ood_loader

        best_epoch = 0
        lowest_validation_loss = np.inf
        early_stopping_counter = 0
        early_stopped = False

        for i_epoch in range(n_epochs):

            # Training and evaluation
            log.info("\nEpoch %d/%d" % (i_epoch, n_epochs))

            t0 = time.time()
            training_monitored_quantities = self.__train_epoch(
                training_loader,
                optimizer,
            )

            validation_monitored_quantities = self.__validate_epoch(validation_loader)

            training_loss = training_monitored_quantities["loss"]
            validation_loss = validation_monitored_quantities["loss"]

            background_reco_errors = self.__evaluate(validation_loader_no_batch)["reco_errors"]
            signal_reco_errors = self.__evaluate(ood_loader)["reco_errors"]
            y_true = np.concatenate((np.zeros(len(background_reco_errors)), np.ones(len(signal_reco_errors))))
            y_pred = np.concatenate((background_reco_errors, signal_reco_errors))
            auc = roc_auc_score(y_true, y_pred)

            self.metrics_tracker["epoch"].append(i_epoch)
            self.metrics_tracker["training_loss"].append(training_loss)
            self.metrics_tracker["validation_loss"].append(validation_loss)
            self.metrics_tracker["auc"].append(auc)
            
            # LR scheduler step
            if lr_scheduler is not None:
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(training_loss)
                else:
                    lr_scheduler.step()

            t1 = time.time()
            elapsed = t1 - t0

            log.info(f"{elapsed:.2f} s/epoch - loss: {training_loss:.3f} "
                     f"- validation loss: {validation_loss:.3f}")

            metrics_file = f"{self.output_path}/training.csv"
            pd.DataFrame(self.metrics_tracker).to_csv(metrics_file, index=False)

            # Early stopping
            if validation_loss < lowest_validation_loss:
                log.info(f"Validation loss improved from {lowest_validation_loss:.3f} to {validation_loss:.3f}. Saving checkpoint.")
                self.__save_model_checkpoint(name="best")
                lowest_validation_loss = validation_loss
                early_stopping_counter = 0
                best_epoch = i_epoch
            else:
                early_stopping_counter += 1

            if early_stopping_counter > es_patience:
                early_stopped = False
                log.info(f"Epoch {i_epoch}: early stopping")
                break

        self.__save_model_checkpoint(name="last_epoch")

        metrics_file = f"{self.output_path}/training.csv"
        log.info(f"Saving metrics file {metrics_file}")
        pd.DataFrame(self.metrics_tracker).to_csv(metrics_file, index=False)
        with open(f"{self.output_path}/info.txt", "w") as file:
            file.write(f"Best epoch: {best_epoch}\n")
            if early_stopped:
                file.write(f"Early stopping at epoch {i_epoch}.\n")

    def train(self):
        """
        @mandatory
        Runs the training of the previously prepared model on the normalized data
        """

        log.info("Starting model fitting")

        optimizer_args = {
            "params": self.model.parameters(),
            "lr": self.config.training_params["learning_rate"],
        }
        torch_optimizer = getattr(torch.optim, self.config.training_params["optimizer"])
        optimizer = torch_optimizer(**optimizer_args)
        
        if self.config.training_params["lr_scheduler"] is not None:
            lr_scheduler = getattr(torch.optim.lr_scheduler, self.config.training_params["lr_scheduler"])(
                optimizer,
                **self.config.training_params["lr_scheduler_args"]
            )
        else:
            lr_scheduler = None

        self.__fit(
            n_epochs=self.config.training_params["n_epochs"],
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            es_patience=self.config.training_params["es_patience"],
        )

        log.info("Finished training")

    def __get_model(self):
        """
        Builds an auto-encoder model as specified in object's fields: input_size,
        intermediate_architecture and bottleneck_size
        """

        model = WNAE(
            encoder=self.encoder,
            decoder=self.decoder,
            **self.config.training_params["wnae_parameters"],
        )
        model.to(self.device)
        return model

