from scipy.stats import poisson
from scipy.optimize import minimize
from numpy import square

def cost_function(mu, n, target):
    return square(poisson.sf(n, mu)-target)
