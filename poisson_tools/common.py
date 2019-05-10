from scipy.stats import poisson
from scipy.optimize import minimize
from numpy import square


def cost_function(mu, n, target):
    '''
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
    '''

    return square(poisson.sf(n, mu)-target)


def find_limit(limit, n):
    '''
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
    '''

    opt_result = minimize(
        cost_function,
        poisson.isf(limit, n),
        args=(n, limit))
    return opt_result.x[0]
