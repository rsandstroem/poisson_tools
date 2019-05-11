import numpy as np

from poisson_tools import central_bayesian as cb


def test_statistical_uncertainty():
    # TODO: correct these values
    np.testing.assert_allclose(
        cb.statistical_uncertainty([0, 1, 2, 3, 4]),
        [[0., 
        -0.29181481, 
        -0.63270422, 
        -0.91441415, 
        -1.15974518],
        [1.84101481, 
        -1.,  
        2.63803524,  
        2.91852353,  
        3.16272738]
        ]
    )


def test_statistical_confidence_interval():
    # TODO: correct these values
    np.testing.assert_allclose(
        cb.statistical_confidence_interval([0, 1, 2, 3, 4]),
        [[0.,
          0.708185187,
          1.367295778,
          2.085585851,
          2.840254815],
         [1.8410148107,
          0.0,
          4.6380352351,
          5.9185235301,
          7.1627273838]
         ]
    )
