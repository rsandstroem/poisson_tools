import numpy as np

from poisson_tools import central_bayesian as cb


def test_statistical_uncertainty():
    # TODO: correct these values
    np.testing.assert_allclose(
        cb.statistical_uncertainty([0, 1, 2, 3, 4]),
        [[0.,
        -1.,
        -0.63267527,
        -0.91436818,
        -1.15969305],
        [1.84101481,
        2.29950098,
        2.63778268,
        2.91807215,
        3.16230862]
        ],
        rtol = 1e-2
    )


def test_statistical_confidence_interval():
    # TODO: correct these values
    np.testing.assert_allclose(
        cb.statistical_confidence_interval([0, 1, 2, 3, 4]),
        [[0.0,
         0.0,
         1.36732473141,
         2.08563181612,
         2.84030694667],
         [1.8410148107,
          3.2995009813,
          4.6377826766,
          5.9185235301,
          7.1627273838]
         ],
         rtol = 1e-2
    )
