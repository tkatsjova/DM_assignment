import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from joblib import effective_n_jobs
from sklearn.feature_extraction.text import CountVectorizer

import main
from src.config import RANDOM_STATE
from src.models import gradient_boosting as boosting
from src.models.tuning import CV, one_se_rule


def reviews():
    rng = np.random.default_rng(42)
    counts = rng.integers(0, 4, size=(40, 6))
    terms = ["apple", "broom", "clean", "desk", "easy", "floor"]
    return pd.DataFrame({
        "clean_text": [" ".join(term for term, count in zip(terms, row) for _ in range(count)) + f" unique{i}"
                       for i, row in enumerate(counts)],
        "label": (counts.sum(axis=1) > 9).astype(int),
    })


class GradientBoostingTests(unittest.TestCase):
    def test_fixed_stage_count_depth_proxy_and_feature_importance_mapping(self):
        train = reviews()
        pipeline = boosting.build_pipeline()
        self.assertEqual(pipeline["model"].n_estimators, 100)
        self.assertEqual(pipeline["model"].random_state, RANDOM_STATE)
        self.assertEqual(pipeline["model"].loss, "log_loss")
        self.assertEqual(pipeline["model"].subsample, 1.0)
        self.assertIsNone(pipeline["model"].n_iter_no_change)
        pipeline.set_params(model__n_estimators=5, model__max_depth=2).fit(train.clean_text, train.label)
        r = {"params": [{"model__max_depth": 3}, {"model__max_depth": 1}], "mean_test_accuracy": [.9, .9]}
        r.update({f"split{i}_test_accuracy": [.9, .9] for i in range(CV.n_splits)})
        self.assertEqual(one_se_rule(boosting.tree_depth)(r), 1)
        features = boosting.top_features(pipeline, train)
        importance = dict(zip(pipeline["vectorizer"].get_feature_names_out(), pipeline["model"].feature_importances_))
        self.assertEqual(len(features), min(5, sum(value > 0 for value in importance.values())))
        self.assertTrue(features.importance.is_monotonic_decreasing)
        for row in features.itertuples():
            self.assertAlmostEqual(row.importance, importance[row.term], places=4)
        self.assertLessEqual(boosting.stage_structure(pipeline).loc["Tree depth", "max"], 2)

    def test_cv_vocabulary_is_fold_local_and_final_refit_uses_all_training_reviews(self):
        train = reviews()
        fits = []
        original = CountVectorizer.fit_transform

        def record_fit(vectorizer, documents, y=None):
            X = original(vectorizer, documents, y)
            fits.append((tuple(documents), set(vectorizer.get_feature_names_out())))
            return X

        grid = {"vectorizer__min_df": [1], "vectorizer__ngram_range": [(1, 1)],
                "model__learning_rate": [.1], "model__max_depth": [1, 2], "model__n_estimators": [5]}
        with patch.object(boosting, "PARAM_GRID", grid), patch.object(boosting, "CV_WORKERS", 1), \
             patch.object(CountVectorizer, "fit_transform", record_fit):
            pipeline, tables = boosting.run_gradient_boosting(train)
        folds = list(CV.split(train.clean_text, train.label))
        self.assertEqual(len(fits), 2 * len(folds) + 1)
        for i, (documents, vocabulary) in enumerate(fits[:-1]):
            training, validation = folds[i % len(folds)]
            self.assertEqual(documents, tuple(train.clean_text.iloc[training]))
            self.assertTrue(all(f"unique{j}" not in vocabulary for j in validation))
        self.assertEqual(fits[-1][0], tuple(train.clean_text))
        self.assertEqual(pipeline["model"].n_estimators_, 5)
        for tree in pipeline["model"].estimators_.ravel():
            self.assertEqual(tree.tree_.n_node_samples[0], len(train))
        summary = next(iter(tables.values()))["value"]
        self.assertAlmostEqual(float(summary["training accuracy"]), pipeline.score(train.clean_text, train.label), places=4)
        curve = tables["Training performance by stage (selected setting, descriptive only)"]
        self.assertAlmostEqual(curve.iloc[-1]["training accuracy"], float(summary["training accuracy"]), places=4)

    def test_parallel_context_is_effective_and_scoped_without_launching_workers(self):
        previous = effective_n_jobs(None)

        def check_context(*args):
            self.assertEqual(effective_n_jobs(None), boosting.CV_WORKERS)
            raise RuntimeError("parallelism checked")

        with patch.object(boosting, "tune", side_effect=check_context), self.assertRaisesRegex(RuntimeError, "parallelism checked"):
            boosting.run_gradient_boosting(reviews())
        self.assertEqual(effective_n_jobs(None), previous)

    def test_main_registers_boosting_and_receives_supplied_training_data(self):
        train = reviews()
        with patch.object(main, "run_naive_bayes", return_value=("nb", {})), \
             patch.object(main, "run_logistic_regression", return_value=("lr", {})), \
             patch.object(main, "run_classification_tree", return_value=("tree", {})), \
             patch.object(main, "run_random_forest", return_value=("forest", {})), \
             patch.object(main, "run_gradient_boosting", return_value=("boosting", {})) as run, \
             patch.object(main, "report"):
            self.assertEqual(main.train_models(train)["gradient_boosting"], "boosting")
            self.assertIs(run.call_args.args[0], train)


if __name__ == "__main__":
    unittest.main()
