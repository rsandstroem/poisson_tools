from numpy import testing

from poisson_tools import central_bayesian as cb


def test_statistical_uncertainty():
    # TODO: correct these values
    testing.assert_allclose(
        cb.statistical_uncertainty([0, 1, 2, 3, 4]),
        [
            [0.,
             0.291285,
             0.63267527,
             0.91436818,
             1.15969305],
            [1.84101481,
             2.29950098,
             2.63778268,
             2.91807215,
             3.16230862]
        ],
        rtol=1e-2
    )


def test_statistical_ci():
    # TODO: correct these values
    testing.assert_allclose(
        cb.statistical_confidence_interval([0, 1, 2, 3, 4]),
        [
            [0.,
             0.70871450,
             1.36732473,
             2.08563181,
             2.84030694],
            [1.8410148107,
             3.2995009813,
             4.6377826766,
             5.9185235301,
             7.1627273838]
        ],
        rtol=1e-2
    )

    testing.assert_allclose(
        cb.statistical_confidence_interval(
            [0, 1, 2, 3, 4],
            sigma=1.96),
        cb.statistical_confidence_interval(
            [0, 1, 2, 3, 4],
            confidence_level=0.95),
        rtol=1e-2
    )
