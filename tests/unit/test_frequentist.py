from numpy import testing

from poisson_tools import frequentist as fr


def test_statistical_uncertainty():
    # TODO: correct these values
    testing.assert_allclose(
        fr.statistical_uncertainty([0, 1, 2, 3, 4]),
        [[0.,
          0.82723962,
          1.2912855,
          1.63259023,
          1.91385838],
         [1.84076324,
          2.29932892,
          2.6365641,
          2.91768889,
          3.16250245]
         ],
        rtol=1e-2
    )


def test_statistical_ci():
    # TODO: correct these values
    testing.assert_allclose(
        fr.statistical_confidence_interval([0, 1, 2, 3, 4]),
        [[0.0,
          0.17276038,
          0.70871450,
          1.36740977,
          2.08614161],
         [1.840763235,
          3.299328918,
          4.636564102,
          5.917688891,
          7.162502454]
         ],
        rtol=1e-2
    )

    testing.assert_allclose(
        fr.statistical_confidence_interval(
            [0, 1, 2, 3, 4],
            sigma=1.96),
        fr.statistical_confidence_interval(
            [0, 1, 2, 3, 4],
            confidence_level=0.95),
        rtol=1e-2
    )
