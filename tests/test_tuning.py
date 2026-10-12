import unittest
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.exceptions import FitFailedWarning

from src.config import CV_FOLDS
from src.models.naive_bayes import build_pipeline
from src.models.tuning import _one_se_threshold, cv_results, one_se_rule, selection_summary, tune


def results(folds, costs):
    folds = np.array(folds, dtype=float).T
    with np.errstate(invalid="ignore"):
        means = folds.mean(axis=0)
    return {"params": [{"cost": c} for c in costs], "mean_test_accuracy": means,
            "mean_test_f1": means, **{f"split{i}_test_accuracy": folds[i] for i in range(CV_FOLDS)}}


def complexity(params):
    return params["cost"]


class OneSETests(unittest.TestCase):
    def test_sample_standard_error_and_inclusive_boundary(self):
        r = results([[.6, .85, .85, .85, .85], [.75] * 5, [.749] * 5], [3, 2, 1])
        best, se, threshold = _one_se_threshold(r, complexity)
        self.assertEqual(best, 0)
        self.assertAlmostEqual(se, .05)
        self.assertAlmostEqual(threshold, .75)
        self.assertEqual(one_se_rule(complexity)(r), 1)

    def test_tied_best_uses_simplest_reference(self):
        r = results([[.7, .7, .8, .9, .9], [.8] * 5, [.79] * 5], [3, 2, 1])
        self.assertEqual(_one_se_threshold(r, complexity), (1, 0, .8))
        self.assertEqual(one_se_rule(complexity)(r), 1)

    def test_equal_complexity_prefers_accuracy_then_grid_order(self):
        r = results([[.6, .85, .85, .85, .85], [.76] * 5, [.77] * 5, [.77] * 5], [3, 1, 1, 1])
        self.assertEqual(one_se_rule(complexity)(r), 2)

    def test_rounding_noise_and_single_valid_candidate(self):
        r = results([[.8] * 5, [.8] * 5], [2, 1])
        r["mean_test_accuracy"][0] += 1e-12
        self.assertEqual(one_se_rule(complexity)(r), 1)
        self.assertEqual(one_se_rule(complexity)(results([[.8] * 5], [1])), 0)

    def test_nonfinite_candidates_and_partial_fold_failure_are_excluded(self):
        for invalid in (np.nan, np.inf, -np.inf):
            r = results([[.8] * 5, [invalid] * 5], [1, 0])
            self.assertEqual(one_se_rule(complexity)(r), 0)
        r = results([[.8] * 5, [.9, .9, .9, .9, np.nan], [.9] * 5], [1, 0, np.nan])
        r["mean_test_accuracy"][1] = .9  # A finite mean cannot hide an invalid fold.
        self.assertEqual(one_se_rule(complexity)(r), 0)
        search = SimpleNamespace(cv_results_=r, best_index_=0, best_params_=r["params"][0])
        table = cv_results(search, complexity)
        self.assertEqual(table.within_1se.tolist(), [True, False, False])
        self.assertEqual(table.index.isna().sum(), 2)
        self.assertEqual(selection_summary(search, complexity).loc["settings with the best accuracy", "value"], "1")

    def test_all_invalid_candidates_raise_clear_error(self):
        for r in (results([[np.nan] * 5], [1]), results([[.8] * 5], [np.inf]), results([[.8, .8, .8, .8, np.nan]], [1])):
            with self.subTest(results=r), self.assertRaisesRegex(ValueError, "No valid cross-validation candidates"):
                one_se_rule(complexity)(r)

    def test_grid_search_recovers_from_an_actual_failed_candidate(self):
        train = pd.DataFrame({"clean_text": ["human shared", "ai shared"] * 10, "label": [0, 1] * 10})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            search = tune(build_pipeline(), {"select__k": [0, 1]}, train, lambda p: p["select__k"])
        self.assertTrue(any(isinstance(w.message, FitFailedWarning) for w in caught))
        self.assertEqual(search.best_params_["select__k"], 1)
        self.assertFalse(cv_results(search, lambda p: p["select__k"]).iloc[-1].within_1se)


if __name__ == "__main__":
    unittest.main()
