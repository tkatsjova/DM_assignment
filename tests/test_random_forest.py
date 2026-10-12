import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import main
from src.config import RANDOM_STATE
from src.models import random_forest as forest
from src.models.tuning import CV, one_se_rule


def reviews():
    return pd.DataFrame({
        "clean_text": [f"shared {'human story' if i % 2 == 0 else 'generated service'} unique{i}" for i in range(40)],
        "label": [0, 1] * 20,
    })


class RandomForestTests(unittest.TestCase):
    def test_leaf_proxy_prefers_larger_leaves_and_feature_importances_map_to_terms(self):
        train = reviews()
        pipeline = forest.build_pipeline().set_params(model__n_estimators=5).fit(train.clean_text, train.label)
        model = pipeline["model"]
        self.assertEqual(model.random_state, RANDOM_STATE)
        self.assertTrue(model.bootstrap)
        self.assertFalse(model.oob_score)
        self.assertFalse(hasattr(model, "oob_score_"))
        self.assertEqual(model.criterion, "gini")
        r = {"params": [{"model__min_samples_leaf": 1}, {"model__min_samples_leaf": 3}], "mean_test_accuracy": [.9, .9]}
        r.update({f"split{i}_test_accuracy": [.9, .9] for i in range(CV.n_splits)})
        self.assertEqual(one_se_rule(forest.leaf_size)(r), 1)
        features = forest.top_features(pipeline, train)
        importance = dict(zip(pipeline["vectorizer"].get_feature_names_out(), model.feature_importances_))
        self.assertEqual(len(features), 5)
        self.assertTrue(features.importance.is_monotonic_decreasing)
        self.assertAlmostEqual(model.feature_importances_.sum(), 1)
        for row in features.itertuples():
            self.assertAlmostEqual(row.importance, importance[row.term], places=4)
        structure = forest.forest_structure(pipeline)
        self.assertAlmostEqual(structure.loc["Tree depth", "mean"], np.mean([tree.get_depth() for tree in model.estimators_]), places=2)

    def test_cv_vocabulary_is_fold_local_and_final_forest_uses_all_training_reviews(self):
        train = reviews()
        fits = []
        original = CountVectorizer.fit_transform

        def record_fit(vectorizer, documents, y=None):
            X = original(vectorizer, documents, y)
            fits.append((tuple(documents), set(vectorizer.get_feature_names_out())))
            return X

        grid = {"vectorizer__min_df": [1], "vectorizer__ngram_range": [(1, 1)],
                "model__n_estimators": [5], "model__max_features": ["sqrt"], "model__min_samples_leaf": [1, 3]}
        with patch.object(forest, "PARAM_GRID", grid), patch.object(CountVectorizer, "fit_transform", record_fit):
            pipeline, tables = forest.run_random_forest(train)
        folds = list(CV.split(train.clean_text, train.label))
        self.assertEqual(len(fits), 2 * len(folds) + 1)
        for i, (documents, vocabulary) in enumerate(fits[:-1]):
            training, validation = folds[i % len(folds)]
            self.assertEqual(documents, tuple(train.clean_text.iloc[training]))
            self.assertTrue(all(f"unique{j}" not in vocabulary for j in validation))
        self.assertEqual(fits[-1][0], tuple(train.clean_text))
        # Each final tree draws len(train) bootstrap observations, with repetitions.
        for tree in pipeline["model"].estimators_:
            self.assertEqual(tree.tree_.weighted_n_node_samples[0], len(train))
        summary = next(iter(tables.values()))["value"]
        self.assertAlmostEqual(float(summary["training accuracy"]), pipeline.score(train.clean_text, train.label), places=4)
        self.assertEqual(tables["Cross-validation results (all settings)"].selected.sum(), 1)

    def test_main_registers_forest_and_receives_only_supplied_training_data(self):
        train = reviews()
        with patch.object(main, "run_naive_bayes", return_value=("nb", {})), \
             patch.object(main, "run_logistic_regression", return_value=("lr", {})), \
             patch.object(main, "run_classification_tree", return_value=("tree", {})), \
             patch.object(main, "run_random_forest", return_value=("forest", {})) as run, \
             patch.object(main, "run_gradient_boosting", return_value=("boosting", {})), \
             patch.object(main, "report"):
            self.assertEqual(main.train_models(train)["random_forest"], "forest")
            self.assertIs(run.call_args.args[0], train)


if __name__ == "__main__":
    unittest.main()
