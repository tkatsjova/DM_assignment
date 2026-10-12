import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import main
from src.config import RANDOM_STATE
from src.models import classification_tree as tree
from src.models.tuning import CV


def reviews():
    rng = np.random.default_rng(42)
    counts = rng.integers(0, 4, size=(40, 6))
    terms = ["apple", "broom", "clean", "desk", "easy", "floor"]
    return pd.DataFrame({
        "clean_text": [
            " ".join(term for term, count in zip(terms, row) for _ in range(count)) + f" unique{i}"
            for i, row in enumerate(counts)
        ],
        "label": (counts.sum(axis=1) > 9).astype(int),
    })


class ClassificationTreeTests(unittest.TestCase):
    def test_pruning_reduces_tree_size_and_importances_map_to_terms(self):
        train = reviews()
        pipeline = tree.build_pipeline().fit(train.clean_text, train.label)
        self.assertEqual(pipeline["model"].criterion, "gini")
        self.assertEqual(pipeline["model"].random_state, RANDOM_STATE)
        self.assertIn(0.0, tree.ALPHA_VALUES)
        self.assertLess(tree.pruning_strength({"model__ccp_alpha": .01}), tree.pruning_strength({"model__ccp_alpha": 0}))
        features = tree.top_features(pipeline, train)
        importance = dict(zip(pipeline["vectorizer"].get_feature_names_out(), pipeline["model"].feature_importances_))
        self.assertEqual(len(features), min(5, sum(value > 0 for value in importance.values())))
        self.assertTrue(features.importance.is_monotonic_decreasing)
        for row in features.itertuples():
            self.assertAlmostEqual(row.importance, importance[row.term], places=4)
        pruned = tree.build_pipeline().set_params(model__ccp_alpha=.5).fit(train.clean_text, train.label)
        self.assertEqual(pruned["model"].get_n_leaves(), 1)
        self.assertLess(pruned["model"].get_depth(), pipeline["model"].get_depth())
        self.assertTrue(tree.top_features(pruned, train).empty)  # A stump has no important features.

    def test_cv_fits_vocabulary_within_folds_and_refits_full_training_set(self):
        train = reviews()
        fits = []
        original = CountVectorizer.fit_transform

        def record_fit(vectorizer, documents, y=None):
            X = original(vectorizer, documents, y)
            fits.append((tuple(documents), set(vectorizer.get_feature_names_out())))
            return X

        grid = {"vectorizer__min_df": [1], "vectorizer__ngram_range": [(1, 1)], "model__ccp_alpha": [0, .01]}
        with patch.object(tree, "PARAM_GRID", grid), patch.object(CountVectorizer, "fit_transform", record_fit):
            pipeline, tables = tree.run_classification_tree(train)
        folds = list(CV.split(train.clean_text, train.label))
        self.assertEqual(len(fits), 2 * len(folds) + 1)
        for i, (documents, vocabulary) in enumerate(fits[:-1]):
            training, validation = folds[i % len(folds)]
            self.assertEqual(documents, tuple(train.clean_text.iloc[training]))
            self.assertTrue(all(f"unique{j}" not in vocabulary for j in validation))
        self.assertEqual(fits[-1][0], tuple(train.clean_text))
        self.assertEqual(pipeline["model"].tree_.n_node_samples[0], len(train))
        summary = next(iter(tables.values()))["value"]
        self.assertEqual(int(summary["tree depth"]), pipeline["model"].get_depth())
        self.assertEqual(int(summary["leaves"]), pipeline["model"].get_n_leaves())
        results = tables["Cross-validation results (all settings)"]
        self.assertEqual(results.selected.sum(), 1)
        self.assertTrue(results.loc[results.selected, "within_1se"].all())

    def test_main_registers_tree_without_running_other_experiments(self):
        train = reviews()
        with patch.object(main, "run_naive_bayes", return_value=("nb", {})) as nb, \
             patch.object(main, "run_logistic_regression", return_value=("lr", {})) as lr, \
             patch.object(main, "run_classification_tree", return_value=("tree", {})) as classification_tree, \
             patch.object(main, "run_random_forest", return_value=("forest", {})), \
             patch.object(main, "run_gradient_boosting", return_value=("boosting", {})), \
             patch.object(main, "report") as report:
            self.assertEqual(main.train_models(train), {"naive_bayes": "nb", "logistic_regression": "lr", "classification_tree": "tree", "random_forest": "forest", "gradient_boosting": "boosting"})
            self.assertEqual(report.call_count, 5)
            for run in (nb, lr, classification_tree):
                self.assertIs(run.call_args.args[0], train)


if __name__ == "__main__":
    unittest.main()
