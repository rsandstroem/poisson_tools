from scipy.stats import poisson, norm
from scipy.optimize import minimize
from numpy import square


def cost_function(mu, n, target):
    """
    Calculates the squared distance between the
    survival function (1-cdf) of a poisson function
    with rate mu and n observations and a
    the target value.

    Arguments:
        mu {float} -- Event rate, a.k.a. rate parameter
        n {int} -- Number of observed events
        target {float} -- Estimated/desired survival rate

    Returns:
        float -- Squared distance between the survival rate
        and the target
    """

    return square(poisson.sf(n, mu) - target)


def find_limit(limit, n):
    """
    Finds the limit of the event rate parameter
    corresponding to the desired survival rate
    of the poisson distribution through minimization.

    Arguments:
        limit {float} -- The survival rate, i.e.,
        0.16 corresponds to the lower 1-sigma boundary
        of a two sided interval
        n {int} -- Number of observed events

    Returns:
        float -- The event rate (mu) corresponding to the limit
    """

    opt_result = minimize(
        cost_function,
        # poisson.ppf(limit, n),
        n,
        args=(n, limit),
        method='L-BFGS-B',
        tol=1e-5,
        bounds=[(0, None)])
    if opt_result.success:
        return opt_result.x[0]
    print(opt_result)
    return None


def percentile_from_sigma(sigma, lower):
    """
    Converts a limit in standard deviation into the corresponding
    percentile. This function assumes a two sided interval,
    e.g., sigma == 2 and lower == False will return 0.977.

    Arguments:
        sigma {float} -- Number of standard deviations
        lower {bool} -- Lower True returns the lower percentile,
        if False the upper percentile will be returned

    Returns:
        float -- The percentile as a value between 0 and 1
    """
    percentile = -1
    if lower:
        percentile = norm.sf(sigma)
    else:
        percentile = norm.cdf(sigma)
    return percentile
