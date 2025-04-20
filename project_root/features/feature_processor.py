from abc import ABC, abstractmethod
import numpy as np

class FeatureProcessor(ABC):
    def __init__(self, raw_data=None, processed_data=None):
        self.raw_data = raw_data
        self.processed_data = processed_data

    @abstractmethod
    def process(self):
        """ Process the raw data and prepare for use """
        pass

    @abstractmethod
    def reduce_dimensionality(self, method='autoencoder'):
        """ Reduce dimensionality with the specified method """
        pass
