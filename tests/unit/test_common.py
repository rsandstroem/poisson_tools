from poisson_tools import common
from numpy import testing


def test_cost_function():

    # 1 sigma low limit
    assert common.cost_function(2.84, 4, 0.16) < 1e-4

    # 2 sigma low limit
    assert common.cost_function(1.62, 4, 0.025) < 1e-4

    # 1 sigma upper limit
    assert common.cost_function(7.14, 4, 0.84) < 1e-4

    # Outside limits
    assert common.cost_function(20, 4, 0.84) > 1e-2


def test_find_limit():

    # 1 sigma low limit for 4 observed events
    assert common.find_limit(0.158655, 4) - 2.84 < 1e-2

    # 1 sigma upper limit for 4 observed events
    assert common.find_limit(0.8413447, 4) - 7.16 < 1e-2


def test_percentile_from_sigma():

    # 1 sigma lower percentile
    testing.assert_almost_equal(
        common.percentile_from_sigma(1, lower=True),
        0.158655,
        decimal=4
    )

    # 2 sigma lower percentile
    testing.assert_almost_equal(
        common.percentile_from_sigma(2, lower=True),
        0.022750,
        decimal=4
    )

    # 2 sigma upper percentile
    testing.assert_almost_equal(
        common.percentile_from_sigma(2, lower=False),
        0.97725,
        decimal=4
    )


def test_two_sided_interval_percentiles():

    testing.assert_allclose(
        common.two_sided_interval_percentiles(
            sigma=None, confidence_level=0.95),
        common.two_sided_interval_percentiles(
            sigma=1.956, confidence_level=None),
        rtol=1e-2
    )
