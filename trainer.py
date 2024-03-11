import time
from pathlib import Path
from copy import deepcopy
import time

from tqdm import tqdm
import numpy as np
import torch
from torchview import draw_graph
import ot
from array import array as py_array
#from comet_ml import Experiment

from modules.wnae import WNAE
from utils.TorchMetricsTracker import TorchMetricsTracker
from utils.TorchBestModelTracker import TorchBestModelTracker
from utils.plotStyles import set_plot_style_root
from Logger import *
import ROOT


class TrainerWassersteinNormalizedAutoEncoder():
    
    def __init__(
            self,
            config,
            loader,
            encoder,
            decoder,
            device,
        ):
        """
        Constructor of the specialized Trainer class.
        """

        self.config = config
        self.loader = loader
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        self.all_features = self.loader.all_features
        self.input_size = self.loader.input_size

        self.io_settings = {
            "training_output_path": self.config.output_path,
            "training_plots_output_path": self.config.output_path + "plot",
            "checkpoints_output_path": self.config.output_path + "checkpoints/"
        }
        for path in self.io_settings.values():
            Path(path).mkdir(parents=True, exist_ok=True)

        self.hyper_parameters = {}
        self.model = self.__get_model()
        self.step = ""
        self.training_plots_output_path_for_step = ""

        self.training_steps = [
            x for x, y in self.config.training_params.items()
            if isinstance(y, dict) and "optimizer" in y.keys()
        ]
        log.info(f"Training steps: {self.training_steps}")


        # Internal metrics are metrics returned by the NAE training / validation step
        # External metrics are metrics computed in this script
        # Scalar means that the metrics is a float  per batch
        # Array means that the metrics is an array
        # Need to tell the code if it should expect a scalar or an array
        # Non-batched internal metrics means the calculation is done on the full dataset
        # using the validation method
        self.training_internal_metrics = {
            "loss": "scalar",
            "temperature": "scalar",
            "positive_energy": "scalar",
            "negative_energy": "scalar",
            "energy_difference": "scalar",
            "encoder_norm": "scalar",
            "decoder_norm": "scalar",
            "ae_norm": "scalar",
            "positive_latent_space_norm": "scalar",
            "negative_latent_space_norm": "scalar",
            "latent_space_norm": "scalar",
            "l1_positive_energy_term": "scalar",
            "l1_negative_energy_term": "scalar",
            "l2_positive_energy_term": "scalar",
            "l2_negative_energy_term": "scalar",
            "energy_difference_term": "scalar",
        }
        self.training_internal_metrics.update({
            coef_name: "scalar"
            for coef_name in self.model.regularisation_coefficients_names
        })

        self.training_external_metrics = {
            "emd_negative_positive_samples_one_batch": "scalar",
            "emd_negative_positive_samples": "scalar",
        }

        self.validation_internal_metrics = {
            "loss": "scalar",
            "positive_energy": "scalar",
            "loss_sample": "array",
        }

        self.validation_external_metrics = {
            # "auc": "scalar",
            "emd_negative_positive_samples": "scalar",
        }

        # Same mechanism for non batched internal metric and validation so the metrics are the same
        self.training_non_batched_internal_metrics = {
            "positive_energy": "scalar",
            "loss_sample": "array",
        }


        # This is just some gymnastics to remove array-like quantities
        # while remembering if it's a training, training_non_batched or validation quantity
        all_training_metrics = {
            **self.training_internal_metrics,
            **self.training_external_metrics,
        }
        all_tracked_training_metrics = {
            k: v for k, v in all_training_metrics.items()
            if v == "scalar"
        }

        all_validation_metrics = {
            **self.validation_internal_metrics,
            **self.validation_external_metrics,
        }
        all_tracked_validation_metrics = {
            k: v for k, v in all_validation_metrics.items()
            if v == "scalar"
        }

        all_tracked_training_non_batched_metrics = {
            k: v for k, v in self.training_non_batched_internal_metrics.items()
            if v == "scalar"
        }

        self.tracked_metrics = \
            ["training_" + m for m in all_tracked_training_metrics] \
            + ["training_non_batched_" + m for m in all_tracked_training_non_batched_metrics] \
            + ["validation_" + m for m in all_tracked_validation_metrics] \

        self.metrics_tracker = TorchMetricsTracker(
            external_metric_names=self.tracked_metrics,
            automatic_epoch_numbering=True,
            automatic_batch_numbering=True,

        )
        self.metrics_tracker_file_name = self.io_settings["training_output_path"] + "metrics.csv"

        self.n_columns_merged_plot = 4
        self.n_rows_merged_plot = 3

        #if self.comet_settings is not None:
        #    if self.comet_settings["log_data"]:
        #        self.comet_experiment = Experiment(
        #            api_key=self.comet_settings["api_key"],
        #            project_name=self.comet_settings["project_name"],
        #            workspace=self.comet_settings["workspace"],
        #        )

        #        self.comet_experiment.log_parameters(self.hyper_parameters)

    def __compute_emd(
            self,
            sample_1=None,
            sample_2=None,
            loader_1=None,
            loader_2=None,
            n_samples_max=None,
            n_repetitions=1,
        ):
        """Wrapper method to handle simple and high stat EMD calculation.
        
        Args:
            sample_1 (torch.Tensor): If `None`, `loader_1` will be used.
            sample_2 (torch.Tensor): If `None`, `loader_2` will be used.
            loader_1 (torch.utils.data.DataLoader): If `sample_1` is `None`,
                will be used to get a sample.
            loader_2 (torch.utils.data.DataLoader): If `sample_2` is `None`,
                will be used to get a sample.
            n_samples_max (int or None): Limit number of samples used for EMD
                calculation. If `None`, number of samples not limited.
            n_repetitions (int): Repeat EMD calculation to improve stat. Only
                makes sense to use > 1 when using loaders.
        """

        assert sample_1 is not None or (sample_1 is None and loader_1 is not None)
        assert sample_2 is not None or (sample_2 is None and loader_1 is not None)

        if n_repetitions > 1 and sample_1 is not None and sample_2 is not None:
            log.critical("Does not makes sense to repeat > 1 times EMD calculation if samples are fixed.")
            log.critical("Exiting to avoid waste of time. Please fix the code.")
            exit(1)

        avg_emd = 0.
        for n in range(n_repetitions):
            if sample_1 is None:
                sample_1 = self.__get_sample_from_loader(loader_1).detach().numpy()
            if sample_2 is None:
                sample_2 = self.__get_sample_from_loader(loader_2).detach().numpy()

            if n_samples_max is not None:
                n_sample_1 = len(sample_1)
                n_sample_2 = len(sample_2)
                n_samples = min(n_samples_max, n_sample_1, n_sample_2)
                if n_sample_1 != n_samples:
                    sample_1 = sample_1[:n_samples]
                if n_sample_2 != n_samples:
                    sample_2 = sample_2[:n_samples]

            loss_matrix = ot.dist(sample_1, sample_2)
            weights = torch.ones(n_samples) / n_samples
            avg_emd += ot.emd2(
                weights,
                weights,
                loss_matrix,
                numItermax=1e6,
            )

        avg_emd /= n_repetitions

        return avg_emd

    def __build_negative_histograms_base_path(self, data_split, high_stat):
        path_template = "{scale}_scale/{extension}/{sub_directory}/"
        path_begin = "{training_output_path}negative_samples_1d_histograms/" + data_split + "/"
        if high_stat:
            path_begin += "high_stat/"
        else:
            path_begin += "low_stat/"
        base_path = path_begin + path_template

        return base_path

    def __make_negative_samples_1d_histograms(self, positive_samples, negative_samples, data_split="", high_stat=False):

        def __get_pads():
            pads = []

            x1, y1, x2, y2 = 0.0, 0.3, 1.0, 0.9
            pad = ROOT.TPad("pad0", "pad0", x1, y1, x2, y2)
            pad.SetBottomMargin(0.035)
            pad.SetTopMargin(0.035)
            pad.Draw()
            pads.append(pad)

            x1, y1, x2, y2 = 0.0, 0.035, 1.0, 0.3
            pad = ROOT.TPad("pad1", "pad1", x1, y1, x2, y2)
            pad.SetBottomMargin(0.4)
            pad.SetTopMargin(0.046)
            pad.Draw()
            pads.append(pad)

            return pads

        def __plot_histograms(positive_feature, negative_feature, signal_feature, signal_weights):
            """
            Args:
                positive_feature (torch.Tensor): 1D tensor
                negative_feature (torch.Tensor): 1D tensor
                signal_feature (numpy.ndarray or None)
                signal_weights (numpy.ndarray or None)
            """

            if self.hyper_parameters["x_bound"] is None:
                x_min_signal = min(signal_feature) if signal_feature is not None else np.inf
                x_max_signal = max(signal_feature) if signal_feature is not None else -np.inf
                x_min = min(min(positive_feature), min(negative_feature), x_min_signal)
                x_max = max(max(positive_feature), max(negative_feature), x_max_signal)
            else:
                x_min = self.hyper_parameters["x_bound"][0]
                x_max = self.hyper_parameters["x_bound"][1]

            user_range_x_min = 1.05 * x_min
            user_range_x_max = 1.05 * x_max

            positive_histogram = ROOT.TH1D("", "", 40, x_min, x_max)
            ROOT.SetOwnership(positive_histogram, 0)
            positive_feature = py_array('d', positive_feature)
            positive_histogram.FillN(len(positive_feature), positive_feature, np.ones_like(positive_feature))
            positive_histogram.Scale(100 / positive_histogram.Integral())
            positive_histogram.Sumw2()

            negative_histogram = ROOT.TH1D("", "", 40, x_min, x_max)
            ROOT.SetOwnership(negative_histogram, 0)
            negative_feature = py_array('d', negative_feature)
            negative_histogram.FillN(len(positive_feature), negative_feature, np.ones_like(negative_feature))
            negative_histogram.Scale(100 / negative_histogram.Integral())
            negative_histogram.Sumw2()

            positive_histogram.SetLineColor(ROOT.kRed)
            positive_histogram.SetLineWidth(2)
            positive_histogram.SetLineStyle(1)
            negative_histogram.SetLineColor(ROOT.kBlue)
            negative_histogram.SetLineWidth(2)
            negative_histogram.SetLineStyle(7)

            positive_histogram.GetXaxis().SetRangeUser(user_range_x_min, user_range_x_max)
            positive_histogram.GetXaxis().SetLabelSize(0)
            positive_histogram.GetYaxis().SetLabelSize(30)
            positive_histogram.GetXaxis().SetTitleSize(0)
            positive_histogram.GetYaxis().SetTitleSize(30)
            positive_histogram.GetXaxis().SetTitle("")
            positive_histogram.GetYaxis().SetTitle("A.U.")
            positive_histogram.GetYaxis().SetTitleOffset(1.4)

            positive_histogram.Draw("HIST E0")
            negative_histogram.Draw("HIST E0 SAME")

            y_min_legend = 0.75 if signal_feature is None else 0.7
            legend = ROOT.TLegend(0.18, y_min_legend, 0.9, 0.945)
            ROOT.SetOwnership(legend, 0)
            legend.SetNColumns(1)
            legend.SetFillStyle(0)
            legend.SetBorderSize(0)
            legend.AddEntry(positive_histogram, "Positive samples", "lep")
            legend.AddEntry(negative_histogram, "Negative samples", "lep")

            if signal_feature is not None:
                signal_histogram = ROOT.TH1D("", "", 40, x_min, x_max)
                ROOT.SetOwnership(signal_histogram, 0)
                signal_feature = py_array('d', signal_feature)
                signal_weights = py_array('d', signal_weights)
                signal_histogram.FillN(len(signal_feature), signal_feature, signal_weights)
                signal_histogram.Scale(positive_histogram.Integral() / signal_histogram.Integral())
                signal_histogram.SetLineColor(ROOT.kGreen + 2)
                signal_histogram.SetLineWidth(2)
                signal_histogram.SetLineStyle(2)
                signal_histogram.Draw("HIST E0 SAME")
                legend.AddEntry(signal_histogram, "Signal samples", "lep")
            else:
                signal_histogram = None

            legend.Draw()

            return positive_histogram, negative_histogram, signal_histogram, user_range_x_min, user_range_x_max

        def __plot_ratio(positive_histogram, negative_histogram, feature_name, x_min, x_max):

            def __get_x_min_max_from_root_object(root_object):
                if isinstance(root_object, ROOT.TGraph):
                    x_min = root_object.GetPointX(0)
                    x_max = root_object.GetPointX(root_object.GetN()-1)
                elif isinstance(root_object, ROOT.TH1):
                    x_min = root_object.GetBinLowEdge(1)
                    x_max = root_object.GetBinLowEdge(root_object.GetXaxis().GetNbins()+1)
                else:
                    log.critical("Unsupported ROOT object to find x min and max %s" % (type(root_object)))

                return x_min, x_max

            def draw_horizontal_line(
                    y_coord,
                    x_min=None,
                    x_max=None,
                    root_object=None,
                    color=ROOT.kBlack,
                    line_style=7,
                    line_width=2,
                ):
                """Draw an horizontal line in the current pad.
                
                Args:
                    y_coord (float)
                    x_min (float, optional): Required if root_object is None
                    x_max (float, optional): Required if root_object is None
                    root_object (Any): ROOT object from which to get min and max x coordinates
                    color (int): ROOT color
                    line_style (int): ROOT line style
                    line_width (int): Line style
                """

                if root_object is None:
                    assert x_min is not None and x_max is not None
                else:
                    x_min, x_max = __get_x_min_max_from_root_object(root_object)

                fun = ROOT.TF1("", str(y_coord), x_min, x_max)
                fun.SetLineColor(color)
                fun.SetLineStyle(line_style)
                fun.SetLineWidth(line_width)
                fun.DrawClone("SAME")

            ratio = negative_histogram.Clone()
            ROOT.SetOwnership(ratio, 0)

            ratio.SetMarkerStyle(8)
            ratio.SetMarkerSize(1)
            ratio.SetMarkerColor(ROOT.kBlack)

            ratio.Divide(positive_histogram)
            ratio.GetXaxis().SetRangeUser(x_min, x_max)

            ratio.GetXaxis().SetLabelSize(30)
            ratio.GetYaxis().SetLabelSize(30)
            ratio.GetXaxis().SetTitleSize(30)
            ratio.GetYaxis().SetTitleSize(30)
            x_label = self.config.training_params["features_labels"][feature_name]
            ratio.GetXaxis().SetTitle(x_label)
            ratio.GetYaxis().SetTitle("Neg. / pos.")
            ratio.GetXaxis().SetTitleOffset(1.07)
            ratio.GetYaxis().SetTitleOffset(1.4)
            ratio.GetYaxis().SetRangeUser(0, 2)
            ratio.GetYaxis().SetNdivisions(203)

            ratio.Draw("E1")
            draw_horizontal_line(1, x_min=x_min, x_max=x_max)

        def __make_plot(positive_feature, negative_feature, signal_feature, signal_weights, feature_name, templates):
            """Make negative sample 1D histogram plot.
            
            Args:
                positive_feature (torch.Tensor): 1D tensor
                negative_feature (torch.Tensor): 1D tensor
                signal_feature (numpy.ndarray or None)
                signal_weights (numpy.ndarray or None)
                feature_name (str)
                template (tuple(str)): template paths to be formatted
            """

            path_template, plot_per_feature_template, plot_per_epoch_template = templates
            set_plot_style_root(style="CMS")
            ROOT.gStyle.SetPadTopMargin(0.1)
            ROOT.gStyle.SetTextFont(43)
            ROOT.gStyle.SetTitleFont(43, "XYZ")
            ROOT.gStyle.SetLabelFont(43, "XYZ")

            canvas = ROOT.TCanvas("", "", 600, 750)

            title = "Epoch %s" % self.epoch
            title_text = ROOT.TLatex()
            title_text.SetNDC()
            title_text.SetTextFont(43)
            title_size = 30
            title_text.SetTextSize(title_size)
            title_text.SetTextAlign(21)
            title_text.DrawLatex(0.5, 0.96, title)

            pads = __get_pads()
            pads[0].cd()
            positive_histogram, negative_histogram, signal_histogram, x_min, x_max = \
                __plot_histograms(positive_feature, negative_feature, signal_feature, signal_weights)
            signal_maximum = signal_histogram.GetMaximum() if signal_histogram is not None else 0.
            y_maximum = max(positive_histogram.GetMaximum(), negative_histogram.GetMaximum(), signal_maximum)
            pads[1].cd()
            __plot_ratio(positive_histogram, negative_histogram, feature_name, x_min, x_max)

            pads[0].cd()
            for extension in ("pdf", "png", "C"):
                for scale in ("linear", "log"):
                    if scale == "linear":
                        ROOT.gPad.SetLogy(ROOT.kFALSE)
                        positive_histogram.SetMaximum(1.6 * y_maximum)
                    else:
                        ROOT.gPad.SetLogy(ROOT.kTRUE)
                        positive_histogram.SetMaximum(20 * y_maximum)

                    path = path_template.format(
                        training_output_path=self.training_plots_output_path_for_step,
                        scale=scale,
                        extension=extension,
                        sub_directory=feature_name,
                    )
                    Path(path).mkdir(parents=True, exist_ok=True)

                    plot_name = path + plot_per_epoch_template.format(
                        epoch=self.epoch,
                        batch=self.batch_number,
                        extension=extension,
                    )
                    canvas.SaveAs(plot_name)
                    log.info(f"{plot_name} has been saved.")

                    path = path_template.format(
                        training_output_path=self.training_plots_output_path_for_step,
                        scale=scale,
                        extension=extension,
                        sub_directory="epoch-%s_batch-%s" % (self.epoch, self.batch_number),
                    )
                    Path(path).mkdir(parents=True, exist_ok=True)
                    plot_name = path + plot_per_feature_template.format(
                        feature=feature_name,
                        extension=extension,
                    )
                    canvas.SaveAs(plot_name)
                    log.info(f"{plot_name} has been saved.")

            del canvas
        
        path_template = self.__build_negative_histograms_base_path(data_split, high_stat)
        plot_per_feature_template = "{feature}.{extension}"
        plot_per_epoch_template = "epoch-{epoch}_batch-{batch}.{extension}"
        templates = (path_template, plot_per_feature_template, plot_per_epoch_template)

        for idx, feature_name in enumerate(self.all_features):
            positive_feature = positive_samples[:, idx]
            negative_feature = negative_samples[:, idx]
            signal_feature = None
            signal_weights = None
            __make_plot(positive_feature, negative_feature, signal_feature, signal_weights, feature_name, templates)

    def __train_epoch(self, training_method_name, training_loader, optimizer, **kwargs):

        self.model.train()
        scalar_monitored_quantities = {
            q: 0. for q, v in self.training_internal_metrics.items()
            if v == "scalar"
        }
        scalar_monitored_quantities["emd_negative_positive_samples_one_batch"] = 0.  # "external" quantity
        array_monitored_quantities_buffer = {
            q: [] for q, v in self.training_internal_metrics.items()
            if v == "array"
        }

        n_batches = 0
        bar_format = '{l_bar}{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        for batch in tqdm(training_loader, bar_format=bar_format):
            n_batches += 1
            self.batch_number = n_batches

            if (self.batch_number > 1
                and "produce_negative_samples" in kwargs.keys()
            ):
                kwargs["produce_negative_samples"] = False

            x = batch[0]  # batch is a list of len 1 with the tensor inside...

            optimizer.zero_grad()
            loss, state = getattr(self.model, training_method_name)(x, **kwargs)
            loss.backward()
            optimizer.step()

            for q in scalar_monitored_quantities.keys():
                v = state[q] if q in state.keys() else 0.
                scalar_monitored_quantities[q] += v
            for q in array_monitored_quantities_buffer.keys():
                if q in state.keys():
                    v = state[q]
                    array_monitored_quantities_buffer[q].append(v)

            if self.make_negative_samples_1d_hist and self.batch_number == 0:
                self.__make_negative_samples_1d_histograms(x, state["negative_samples"])
            
            # This is an "external" quantity
            if self.compute_emd_one_batch and self.batch_number == 1:
                scalar_monitored_quantities["emd_negative_positive_samples_one_batch"] = self.__compute_emd(x, state["negative_samples"])

        monitored_quantities = {}
        for q in scalar_monitored_quantities.keys():
            if "one_batch" in q:
                monitored_quantities[q] = scalar_monitored_quantities[q]
            else:
                monitored_quantities[q] = scalar_monitored_quantities[q] / n_batches
            #if self.comet_settings["log_data"]:
            #    self.comet_experiment.log_metric(q, monitored_quantities[q], step=self.epoch)
        for q in array_monitored_quantities_buffer.keys():
            if len(array_monitored_quantities_buffer[q]) > 0:
                monitored_quantities[q] = torch.cat(array_monitored_quantities_buffer[q], dim=0)
            else:
                monitored_quantities[q] = None

        return monitored_quantities

    def __evaluate(
            self,
            evaluation_method_name,
            data_loader,
            monitored_quantities={},
        ):

        self.model.eval()

        scalar_monitored_quantities = {
            q: 0. for q, v in monitored_quantities.items()
            if v == "scalar"
        }
        array_monitored_quantities = {
            q: None for q, v in monitored_quantities.items()
            if v == "array"
        }

        monitored_quantities = {}
        batch = next(iter(data_loader))
        x = batch[0]  # x is a list of len 1 with the tensor inside...
        state = getattr(self.model, evaluation_method_name)(x)
        for q in scalar_monitored_quantities:
            v = state[q] if q in state.keys() else 0.
            monitored_quantities[q] = v
        for q in array_monitored_quantities:
            v = state[q] if q in state.keys() else None
            monitored_quantities[q] = v

        return monitored_quantities

    def __get_non_batched_training_monitored_quantities(self, validation_method_name, training_loader_no_batch):

        monitored_quantities = self.__evaluate(
            evaluation_method_name=validation_method_name,
            data_loader=training_loader_no_batch,
            monitored_quantities=self.training_non_batched_internal_metrics,
        )

        return monitored_quantities

    def __get_sample_from_loader(self, data_loader):
        batch = next(iter(data_loader))
        return batch[0]  # x is a list of len 1 with the tensor inside...

    def __make_non_batched_samples(self, data_loader):

        self.model.eval()
        positive_samples = self.__get_sample_from_loader(data_loader)
        negative_samples = self.model.run_mcmc(positive_samples)

        return positive_samples, negative_samples

    def __validate_epoch(self, validation_method_name, validation_loader):

        monitored_quantities = self.__evaluate(
            evaluation_method_name=validation_method_name,
            data_loader=validation_loader,
            monitored_quantities=self.validation_internal_metrics,
        )

        return monitored_quantities

    def __save_model_checkpoint(self, name, state_dict_only=False):
        path = self.io_settings["checkpoints_output_path"] + f"{name}.pt"
        if state_dict_only:
            torch.save({"model_state_dict": self.model.state_dict()}, path)
        else:
            torch.save(self.model, path)
        log.info(f"Saved model checkpoint {path}")

    def __load_model_from_checkpoint(self, name, state_dict_only=False):
        path = self.io_settings["checkpoints_output_path"] + f"{name}.pt"
        checkpoint = torch.load(path)
        if state_dict_only:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model = checkpoint

    def save_best_model_checkpoint(self):
        """Method to save best model checkpoint, must be public."""
        self.__save_model_checkpoint(name="best", state_dict_only=True)

    def __load_best_model_from_checkpoint(self):
        self.__load_model_from_checkpoint(name="best", state_dict_only=True)

    def __fit(self,
              step,
              n_epochs,
              optimizer,
              lr_scheduler,
              es_patience,
        ):
        """Fit model.

        Args:
            step (str): Fitting step, "ae" or "nae"
            n_epochs (int): Max number of epochs
            training_loader (torch.utils.data.DataLoader)
            validation_loader (torch.utils.data.DataLoader)
            optimizer (torch.optim)
            es_patience (int): Number of epochs for early stop        
        """

        training_loader = self.loader.training_loader
        validation_loader = self.loader.validation_loader
        training_loader_no_batch = self.loader.training_loader_no_batch

        self.metrics_tracker.set_n_batches_per_epoch(len(training_loader))

        self.step = step
        self.training_plots_output_path_for_step = self.io_settings["training_plots_output_path"] + "/" + self.step + "/"
        training_kwargs = {}

        if "best_model_metric_name" in self.config.training_params[step].keys():
            metric_name = self.config.training_params[step]["best_model_metric_name"]
        else:
            metric_name = self.config.training_params[step]["best_model_metric"]
        self.best_model_tracker = TorchBestModelTracker(
            rule=self.config.training_params[step]["best_model_rule"],
            metric_name=metric_name,
            saving_function=(self, "save_best_model_checkpoint"),
            es_patience=es_patience,
        )
        early_stopped = False

        train_method_name = "train_step_" + step
        validation_method_name = "validation_step"
       
        # Make negative samples histogram before the first gradient descent
        self.epoch = 0
        self.batch_number = "all"
        if self.config.training_params[step]["make_negative_samples_1d_histograms"]:
            training_positive_samples, training_negative_samples = self.__make_non_batched_samples(training_loader_no_batch)
            validation_positive_samples, validation_negative_samples = self.__make_non_batched_samples(validation_loader)
            self.__make_negative_samples_1d_histograms(
                training_positive_samples, training_negative_samples, data_split="training", high_stat=True
            )

        for i_epoch in range(n_epochs):
            self.epoch = i_epoch + 1
            training_positive_samples, training_negative_samples = None, None
            validation_positive_samples, validation_negative_samples = None, None

            ## Initialize variables used to produce plots and compute quantities to be saved
            # Check whether to run MCMC for this AE step
            if step == "ae":
                if isinstance(self.config.training_params["ae"]["mcmc_run_interval"], int):
                    mcmc_run_interval = self.config.training_params[step]["mcmc_run_interval"]
                else:
                    mcmc_run_interval = eval(self.config.training_params[step]["mcmc_run_interval"])
                training_kwargs["produce_negative_samples"] = (self.epoch % mcmc_run_interval == 0)

            # Negative samples 1D histograms saving interval
            if isinstance(self.config.training_params[step]["negative_samples_1d_histograms_saving_interval"], int):
                negative_samples_1d_histograms_saving_interval = self.config.training_params[step]["negative_samples_1d_histograms_saving_interval"]
            else:
                negative_samples_1d_histograms_saving_interval = eval(self.config.training_params[step]["negative_samples_1d_histograms_saving_interval"])

            # EMD computation saving interval
            if isinstance(self.config.training_params[step]["emd_computation_saving_interval"], int):
                emd_computation_saving_interval = self.config.training_params[step]["emd_computation_saving_interval"]
            else:
                emd_computation_saving_interval = eval(self.config.training_params[step]["emd_computation_saving_interval"])
            if isinstance(self.config.training_params[step]["emd_one_batch_computation_saving_interval"], int):
                emd_one_batch_computation_saving_interval = self.config.training_params[step]["emd_one_batch_computation_saving_interval"]
            else:
                emd_one_batch_computation_saving_interval = eval(self.config.training_params[step]["emd_one_batch_computation_saving_interval"])

            self.make_negative_samples_1d_hist = (
                (self.epoch % negative_samples_1d_histograms_saving_interval == 0)
                and self.config.training_params[step]["make_negative_samples_1d_histograms"]
            )
            self.compute_emd_one_batch = (
                (self.epoch % emd_one_batch_computation_saving_interval == 0)
                and self.config.training_params[step]["compute_emd_one_batch"]
            )
            self.compute_emd = (
                (self.epoch % emd_computation_saving_interval == 0)
                and self.config.training_params[step]["compute_emd"]
            )


            ## Training and evaluation
            log.info("\nEpoch %d/%d" % (i_epoch+1, n_epochs))

            t0 = time.time()
            training_monitored_quantities = self.__train_epoch(
                train_method_name, training_loader, optimizer, **training_kwargs)
            training_non_batched_monitored_quantities = \
                self.__get_non_batched_training_monitored_quantities(
                    validation_method_name, training_loader_no_batch)

            validation_monitored_quantities = self.__validate_epoch(
                validation_method_name, validation_loader)

            training_loss = training_monitored_quantities["loss"]
            validation_loss = validation_monitored_quantities["loss"]
            energy_difference = training_monitored_quantities["energy_difference"]
            
            ## LR scheduler step
            if lr_scheduler is not None:
                if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(training_loss)
                else:
                    lr_scheduler.step()

            t1 = time.time()
            elapsed = t1 - t0

            log.info(f"{elapsed:.2f} s/epoch - loss: {training_loss:.3f} "
                     f"- validation loss: {validation_loss:.3f} "
                     f"- energy difference: {energy_difference:.3f}")

            ## Example to comoute AUC during training, need some code developent
            # signal_monitored_quantities = self.__evaluate(
            #     evaluation_method_name=validation_method_name,
            #     data_loader=signal_loader,
            #     monitored_quantities=self.validation_internal_metrics,
            # )

            # signal_losses = signal_monitored_quantities["loss_sample"]
            # background_losses = validation_monitored_quantities["loss_sample"]
            # auc = compute_roc_auc_score(
            #     background_losses,
            #     signal_losses,
            # )
            # validation_monitored_quantities["auc"] = auc

            # if self.comet_settings["log_data"]:
            #     self.comet_experiment.log_metric("AUC", auc, step=self.epoch)


            ## Produce plots on high stats and save tracked metrics
            if self.make_negative_samples_1d_hist or self.compute_emd:
                training_positive_samples, training_negative_samples = self.__make_non_batched_samples(training_loader_no_batch)
                validation_positive_samples, validation_negative_samples = self.__make_non_batched_samples(validation_loader)

            if self.make_negative_samples_1d_hist:
                self.batch_number = "all"
                self.__make_negative_samples_1d_histograms(
                    training_positive_samples, training_negative_samples, data_split="training", high_stat=True
                )
                self.__make_negative_samples_1d_histograms(
                    validation_positive_samples, validation_negative_samples, data_split="validation", high_stat=True
                )
            if self.compute_emd:
                t0 = time.time()
                # Computing the EMD on very large dataset requires A LOT of RAM and time
                # Need to sample it down
                n_samples_max = 2000

                # Training samples
                training_monitored_quantities["emd_negative_positive_samples"] = self.__compute_emd(
                    sample_1=training_positive_samples,
                    sample_2=training_negative_samples,
                    n_samples_max=n_samples_max,
                )

                # Validation samples
                validation_monitored_quantities["emd_negative_positive_samples"] = self.__compute_emd(
                    sample_1=validation_positive_samples, 
                    sample_2=validation_negative_samples,
                    n_samples_max=n_samples_max,
                )

                t1 = time.time()
                log.info("High stat EMD calculation took %.2fs" % (t1-t0))
                
            else:
                training_monitored_quantities["emd_negative_positive_samples"] = 0.
                validation_monitored_quantities["emd_negative_positive_samples"] = 0.

            #if self.comet_settings["log_data"]:
            #    emd_names = [
            #        "emd_negative_positive_samples",
            #    ]
            #    emd_comet_names = [
            #        "emd_neg_pos",
            #    ]
            #    for emd_name, comet_emd_name in zip(emd_names, emd_comet_names):
            #        self.comet_experiment.log_metric(
            #            comet_emd_name,
            #            validation_monitored_quantities[emd_name],
            #            step=self.epoch,
            #        )

            external_metrics = {
                **{"training_" + metric: value
                    for metric, value in training_monitored_quantities.items()
                },
                **{"training_non_batched_" + metric: value
                   for metric, value in training_non_batched_monitored_quantities.items()
                },
                **{"validation_" + metric: value
                    for metric, value in validation_monitored_quantities.items()
                }
            }
            # Removing non scalar quantities
            external_metrics = {k: v for k, v in external_metrics.items() if k in self.tracked_metrics}

            self.metrics_tracker.epoch_update(external_metrics)


            ## Early stopping
            best_metric_param = self.config.training_params[step]["best_model_metric"]
            if isinstance(best_metric_param, str):
                best_model_metric = eval(best_metric_param)
            elif isinstance(best_metric_param, tuple):
                if len(best_metric_param) != 2:
                    log.critical("When best model metric is a tuple, it should be of length 2!")
                    exit(1)
                args = eval(best_metric_param[0])
                function_ = best_metric_param[1]
                best_model_metric = function_(*args)
            else:
                log.critical(f"Best model metric must be a string or a tuple. Got {type(best_metric_param)}.")
                exit(1)

            early_stop = self.best_model_tracker.update(best_model_metric)
            if early_stop:
                log.info(f"Epoch {i_epoch+1}: early stopping")
                early_stopped = True
                break

            if isinstance(self.config.training_params[step]["checkpoint_saving_interval"], int):
                checkpoint_saving_interval = self.config.training_params[step]["checkpoint_saving_interval"]
            else:
                checkpoint_saving_interval = eval(self.config.training_params[step]["checkpoint_saving_interval"])
            if self.epoch % checkpoint_saving_interval == 0:
                self.__save_model_checkpoint(name=f"epoch{self.epoch}")

        fit_info = {
            "n_epochs": self.epoch,
            "early_stopped": early_stopped,
            "validation_loss": validation_loss,
            "best_validation_loss": self.best_model_tracker.best_value,
        }
        return fit_info

    def train(self):
        """
        @mandatory
        Runs the training of the previously prepared model on the normalized data
        """

        log.info("Starting model fitting")

        for step in self.training_steps:
            if self.config.training_params[step]["n_epochs"] > 0:
                log.info(f"Training model with {step} step")

                optimizer_args = {
                    "params": self.model.parameters(),
                    "lr": self.config.training_params[step]["learning_rate"],
                }
                torch_optimizer = getattr(torch.optim, self.config.training_params[step]["optimizer"])
                optimizer = torch_optimizer(**optimizer_args)
                
                if self.config.training_params[step]["lr_scheduler"] is not None:
                    lr_scheduler = getattr(torch.optim.lr_scheduler, self.config.training_params[step]["lr_scheduler"])(
                        optimizer,
                        **self.config.training_params[step]["lr_scheduler_args"]
                    )
                else:
                    lr_scheduler = None

                training_info = self.__fit(
                    step=step,
                    n_epochs=self.config.training_params[step]["n_epochs"],
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    es_patience=self.config.training_params[step]["es_patience"],
                )

                if self.config.training_params[step]["restore_best_model_after_training"]:
                    log.info("Restoring best model")
                    self.__load_best_model_from_checkpoint()

        self.metrics_tracker.write_to_file(self.metrics_tracker_file_name)

        log.info("Finished training")

    def __get_model(self):
        """
        Builds an auto-encoder model as specified in object's fields: input_size,
        intermediate_architecture and bottleneck_size
        """

        for step in self.config.training_params:
            if (isinstance(self.config.training_params[step], dict)
                and "hyper_parameters" in self.config.training_params[step].keys()
            ):
                self.hyper_parameters = deepcopy(self.config.training_params[step]["hyper_parameters"])
                break
        
        model = WNAE(self.encoder, self.decoder, **self.hyper_parameters)

        # Visualize model
        graph = draw_graph(
            model,
            input_size=(self.config.training_params["batch_size"], self.input_size,),
            graph_name="NAE",
        ).visual_graph

        plot_path = self.io_settings["training_plots_output_path"] + "network_architecture"
        for extension in ("png", "pdf"):
            file_name = f"{plot_path}.{extension}"
            log.info(f"Saving network architecture plot in {file_name}")
            graph.render(filename=plot_path, format=extension)

        model.to(self.device)
        return model

