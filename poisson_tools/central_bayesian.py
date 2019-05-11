import numpy as np
from poisson_tools import common


def statistical_uncertainty(y, sigma=1, cl=None):
    ci = statistical_confidence_interval(y, sigma, cl)
    return np.array(ci)-y


def statistical_confidence_interval(y, sigma=1, cl=None):
    percentile_low = common.percentile_from_sigma(sigma, lower=True)
    percentile_high = common.percentile_from_sigma(sigma, lower=False)
    y_low = [common.find_limit(percentile_low, y_i) for y_i in y]
    y_high = [common.find_limit(percentile_high, y_i) for y_i in y]
    return [y_low, y_high]