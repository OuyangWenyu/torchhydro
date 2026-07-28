import logging

import numpy as np

log = logging.getLogger(__name__)

class Metrics:
    def nse(self, predictions, observations):
        mean_observed = np.mean(observations)
        numerator = np.sum(np.power(observations - predictions, 2))
        denominator = np.sum(np.power(observations - mean_observed, 2))
        return 1 - np.divide(numerator, denominator)