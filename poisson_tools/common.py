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
