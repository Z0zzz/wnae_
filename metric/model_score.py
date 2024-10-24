import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import json
import torch

# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Input
# from tensorflow.keras.models import Sequential, Model
# import tensorflow as tf
# from qkeras import quantized_bits

class model_threshold_manager():
    def __init__(self,model, loss_func, config):#trimmed_model, loss_func, config):
        self.config = config
        self.model = model
        # self.trimmed_model = trimmed_model
        
        self.target_rate = self.config["target_rate"]
        self.threshold = None
        self.bc_rate_khz = self.config["bc_khz"]
        
        self.score_dict = {} #Internal variable not to be accessed directly
        self.data_file = None #Internal variable not to be accessed directly
        
        self.data_path = self.config["data_path"]
        self.HT_THRESHOLD = self.config["ht_threshold"]

        self.loss_func = loss_func
        
        # Init ...
        self._open_hdf5()
        ap_fixed = self.config["precision"]
        # self.input_quantizer = quantized_bits(ap_fixed[0],ap_fixed[1],alpha=self.config["alpha"])
        print("Calculating Threshold...")
        self.get_threshold()
        print("Evaluating Score for model...")
        self.calculate_score()
        print("DONE")
        
    def _open_hdf5(self):
        self.data_file = h5py.File(self.data_path,"r")
    
    def get_threshold(self):
        x_test = self.data_file["Background_data"]["Test"]["DATA"][:]
        x_test = torch.tensor(np.reshape(x_test,(x_test.shape[0],-1))).to(torch.float32)       

        #claculate bkg score for full model
        # x_reco, z_mean, z_log_var= self.full_model.predict(self.input_quantizer(x_test),batch_size = 120000)
        # y_loss = self.loss_func(x_test, x_reco, z_mean, z_log_var)

        prediction_outputs = self.model(x_test)
        # if isinstance(prediction_outputs, list):
        #     prediction_outputs = tuple(prediction_outputs)
        # if not isinstance(prediction_outputs, tuple):
        #         prediction_outputs = (prediction_outputs,)
        # y_loss = self.loss_func(self.input_quantizer(x_test), *prediction_outputs)
        
        y_loss = prediction_outputs.detach().numpy()
        
        # #Calculate bkg score for trimmed model
        # latent_axo_qk = self.trimmed_model.predict(self.input_quantizer(x_test),batch_size = 120000)
        # y_axo_qk = np.sum(latent_axo_qk**2, axis=1)
        
        threshold = {}
        # full_threshold = {}
        # axo_threshold = {}
        #Calculate threshold for each rate
        for target_rate in self.target_rate: 
            #converts from rate to percentile using predetermined hardware rate
            threshold[str(target_rate)] = np.percentile(y_loss, 100-(target_rate/self.bc_rate_khz)*100) 
            # threshold[str(target_rate)] = np.percentile(y_axo_qk, 100-(target_rate/self.bc_rate_khz)*100)
        # threshold['full'] = full_threshold
        # threshold['axo'] = axo_threshold
        self.threshold = threshold #Store threshold as dictionary of thresholds for the full and trimmed model

    def calculate_score(self):
        HT_THRESHOLD = self.HT_THRESHOLD
        signal_names = list(self.data_file["Signal_data"].keys())
        score = {}
        score["SIGNAL_NAMES"] = signal_names
        score["SCORE"] = {}
        for target_rate in self.target_rate:
            _raw_rate = [] #Rate/efficiency for model
            # _raw_axo_rate = [] #rate/efficiency for trimmed model
            _l1_rate = [] #rate/efficiency for l1 trigger
            _ht_rate = []
            _model_improv_rate = []
            for signal in signal_names:
                signal_data = self.data_file["Signal_data"][signal]["DATA"][:]
                signal_data = torch.tensor(np.reshape(signal_data,(signal_data.shape[0],-1))).to(torch.float32)
                signal_ET = self.data_file["Signal_data"][signal]["ET"][:]
                signal_HT = self.data_file["Signal_data"][signal]["HT"][:]
                signal_L1 = self.data_file["Signal_data"][signal]["L1bits"][:]
                signal_PU = self.data_file["Signal_data"][signal]["PU"][:]

                #Calc sig score for full model
                # signal_reco, z_mean, z_log_var = self.full_model.predict(self.input_quantizer(signal_data),batch_size = signal_data.shape[0],verbose=0)
                # signal_loss = self.loss_func(signal_data, signal_reco, z_mean, z_log_var)
                prediction_outputs = self.model(signal_data)
                # if isinstance(prediction_outputs, list):
                #     prediction_outputs = tuple(prediction_outputs)
                # if not isinstance(prediction_outputs, tuple):
                #         prediction_outputs = (prediction_outputs,)
                # signal_loss = self.loss_func(self.input_quantizer(signal_data), *prediction_outputs)
                
                signal_loss = prediction_outputs.detach().numpy()
                
                # #Calc signal score for trimmed model
                # latent_axo_qk = self.trimmed_model.predict(self.input_quantizer(signal_data),batch_size = signal_data.shape[0],verbose=0)
                # y_axo_qk = np.sum(latent_axo_qk**2, axis=1)

                nsamples = signal_data.shape[0]

                model_triggered = np.where(signal_loss > self.threshold[str(target_rate)])[0].tolist()
                # axo_triggered = np.where(y_axo_qk > self.threshold['axo'][str(target_rate)])[0].tolist()
                l1_triggered = np.where(signal_L1)[0].tolist()
                ht_triggered = np.where(signal_HT > HT_THRESHOLD)[0].tolist()

                raw_rate = len(model_triggered)/nsamples
                # raw_rate = len(axo_triggered)/nsamples
                l1_rate = len(l1_triggered)/nsamples
                ht_rate = len(ht_triggered)/nsamples

                model_improv = list(set(model_triggered)-set(l1_triggered))
                model_improv_rate = len(model_improv)/nsamples

                _raw_rate.append(raw_rate)
                _l1_rate.append(l1_rate)
                # _raw_axo_rate.append(raw_rate)
                _model_improv_rate.append(model_improv_rate)
                _ht_rate.append(ht_rate)

            score["SCORE"][str(target_rate)] = {
                "model_rate": _raw_rate,
                # "raw-axo":_raw_axo_rate,
                "L1_rate":_l1_rate,
                "HT_rate":_ht_rate,
                "model_improvement":_model_improv_rate,
            }

        self.score_dict = score ## Storing it here
        
    def get_raw_dict(self):
        return self.score_dict,self.threshold
    def get_score(self,thres):
        signal_names = self.data_file["Signal_data"].keys()
        df = pd.DataFrame()
        df["Signal Name"] = signal_names
        df['MODEL SCORE'] = self.score_dict["SCORE"][str(thres)]['model_rate']
        # df["AXO SCORE"] = self.score_dict["SCORE"][str(thres)]['raw-axo']
        df["L1 SCORE"] = self.score_dict["SCORE"][str(thres)]['L1_rate']
        df["HT SCORE"] = self.score_dict["SCORE"][str(thres)]['HT_rate']
        df["MODEL Improvement"] = self.score_dict["SCORE"][str(thres)]['model_improvement']
        
        return df