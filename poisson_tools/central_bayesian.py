import numpy as np
from poisson_tools import common


def statistical_uncertainty(y, sigma=1, confidence_level=None):
    """
    Calculates the statistical uncertainty
    for an array y of observed Poisson events y_i.

    Arguments:
        y {list of int} -- List of observations (point estimates)

    Keyword Arguments:
        sigma {int} -- Number of standard deviation of the
        uncertainty estimate (default: {1})
        cl {[float]} -- Confidence level, e.g., 0.95 indicates
        a CL of 95%. If this is used, the sigma parameter
        ignored. (default: {None})

    Returns:
        [list] -- A list containing two arrays,
        the first array is the lower uncertainty bound,
        the second array is the upper uncertainty bound.
        The arrays have the same length as y.
    """
    confidence_interval = statistical_confidence_interval(
        y, sigma, confidence_level)
    return np.array(confidence_interval) - y


def statistical_confidence_interval(y, sigma=1, confidence_level=None):
    """
    Calculates the statistical confidence interval
    for an array y of observed Poisson events y_i.

    Arguments:
        y {list of int} -- List of observations (point estimates)

    Keyword Arguments:
        sigma {int} -- Number of standard deviation of the
        uncertainty estimate (default: {1})
        cl {[float]} -- Confidence level, e.g., 0.95 indicates
        a CL of 95%. If this is used, the sigma parameter
        ignored. (default: {None})

    Returns:
        [list] -- A list containing two arrays,
        the first array is the lower confidence interval bound,
        the second array is the upper confidence interval bound.
        The arrays have the same length as y.
    """
    if confidence_level:
        percentile_low = 0.5*(1-confidence_level)
        percentile_high = 0.5*(1+confidence_level)
    else:
        percentile_low = common.percentile_from_sigma(sigma, lower=True)
        percentile_high = common.percentile_from_sigma(sigma, lower=False)

    y_low = [np.min([float(y_i),
                     common.find_limit(percentile_low, y_i)]) for y_i in y]
    y_high = [np.max([float(y_i),
                      common.find_limit(percentile_high, y_i)]) for y_i in y]
    return [y_low, y_high]
