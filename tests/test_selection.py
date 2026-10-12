import unittest

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_selection import mutual_info_classif

from src.features.selection import mutual_information


class MutualInformationTests(unittest.TestCase):
    def test_known_information_values(self):
        # Perfect indicator, independent indicator, always absent, always present.
        X = csr_matrix([[0, 0, 0, 1], [0, 2, 0, 1], [3, 0, 0, 1], [7, 4, 0, 1]])
        np.testing.assert_allclose(mutual_information(X, [0, 0, 1, 1]), [1, 0, 0, 0], atol=1e-14)

    def test_matches_discrete_sklearn_reference_in_bits(self):
        rng = np.random.default_rng(42)
        X = csr_matrix(rng.integers(0, 5, size=(30, 12)))
        y = np.array([0] * 15 + [1] * 15)
        expected = mutual_info_classif(X > 0, y, discrete_features=True) / np.log(2)
        np.testing.assert_allclose(mutual_information(X, y), expected, atol=1e-14)
        np.testing.assert_allclose(mutual_information(X, y), mutual_information(X > 0, y), atol=1e-14)


if __name__ == "__main__":
    unittest.main()
