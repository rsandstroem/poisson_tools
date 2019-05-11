from poisson_tools import common

def test_cost_function():

    # 1 sigma low limit
    assert common.cost_function(2.84,4,0.16) < 1e-4

    # 2 sigma low limit
    assert common.cost_function(1.62,4,0.025) < 1e-4

    # 1 sigma upper limit
    assert common.cost_function(7.14,4,0.84) < 1e-4

    # Outside limits
    assert common.cost_function(20,4,0.84) > 1e-2


def test_find_limit():

    # 1 sigma low limit for 4 observed events
    assert common.find_limit(0.158655, 4) - 2.84 < 1e-2

    # 1 sigma upper limit for 4 observed events
    assert common.find_limit(0.8413447, 4) - 7.16 < 1e-2

