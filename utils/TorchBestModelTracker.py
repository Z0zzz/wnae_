import numpy as np
from Logger import *


class TorchBestModelTracker():
    def __init__(self, rule, saving_function, metric_name, es_patience):
        self.best_value = 0
        self.allowed_rules = ("minimum", "maximum", "minimum_positive")
        self.rule = rule
        self.saving_function = saving_function
        self.metric_name = metric_name
        self.es_patience = es_patience
        self.es_epochs = 0

        self.__do_sanity_checks()
        self.__initialize_best_value()

    def __do_sanity_checks(self):
        if self.rule not in self.allowed_rules:
            log.critical(f"Rule {self.rule} not in allowed rules:")
            log.critical(f"{self.allowed_rules}")
    
    def __initialize_best_value(self):
        if self.rule in ("minimum", "minimum_positive"):
            self.best_value = np.inf
        elif self.rule == "maximum":
            self.best_value = -np.inf

    def __print_log(self, value):
        log.info(f"{self.metric_name} improved from {self.best_value:.4f} "
                 f"to {value:.4f}. Saving checkpoint.")

    def __save_best_model(self):
        if not isinstance(self.saving_function, tuple):
            self.saving_function()
        else:
            instance = self.saving_function[0]
            method = self.saving_function[1]
            getattr(instance, method)()

    def __update(self, value):
        self.es_epochs = 0
        self.__print_log(value)
        self.best_value = value
        self.__save_best_model()

    def update(self, value):
        if self.rule == "minimum":
            if value < self.best_value:
                self.__update(value)
                return False
        elif self.rule == "minimum_positive":
            if value < self.best_value and value >= 0:
                self.__update(value)
                return False
        elif self.rule == "maximum":
            if value > self.best_value:
                self.__update(value)
                return False

        self.es_epochs += 1

        if self.es_epochs + 1 > self.es_patience:
            return True
    
    def reset(self):
        self.__initialize_best_value()
        self.es_epochs = 0

