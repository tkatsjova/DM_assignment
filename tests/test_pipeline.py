import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif

import main
from src.models.logistic_regression import build_pipeline as logistic_pipeline
from src.models.naive_bayes import build_pipeline as nb_pipeline
from src.models.tuning import CV, tune
from src.data.split import make_split, train_test_data
from test_split import reviews


class PipelineTests(unittest.TestCase):
    def test_vectorizer_and_supervised_selection_fit_only_fold_training_rows(self):
        train = pd.DataFrame({"clean_text": [f"shared class{i % 2} unique{i}" for i in range(40)], "label": [0, 1] * 20})
        original_vectorizer_fit = CountVectorizer.fit_transform
        original_selector_fit = SelectKBest.fit
        vectors, selections = [], []

        def vectorizer_fit(vectorizer, documents, y=None):
            X = original_vectorizer_fit(vectorizer, documents, y)
            vectors.append((tuple(documents), vectorizer.get_feature_names_out().copy()))
            return X

        def selector_fit(selector, X, y):
            fitted = original_selector_fit(selector, X, y)
            selections.append((X.copy(), np.array(y), selector.scores_.copy()))
            return fitted

        with patch.object(CountVectorizer, "fit_transform", vectorizer_fit), patch.object(SelectKBest, "fit", selector_fit):
            tune(nb_pipeline(), {"select__k": [1, 2]}, train, lambda p: p["select__k"])
        folds = list(CV.split(train.clean_text, train.label))
        self.assertEqual(len(vectors), 2 * len(folds) + 1)
        self.assertEqual(len(selections), len(vectors))
        for i, ((documents, terms), (X, y, scores)) in enumerate(zip(vectors[:-1], selections[:-1])):
            training, validation = folds[i % len(folds)]
            self.assertEqual(documents, tuple(train.clean_text.iloc[training]))
            np.testing.assert_array_equal(y, train.label.iloc[training])
            self.assertEqual(X.shape, (len(training), len(terms)))
            self.assertTrue(all(f"unique{j}" not in terms for j in validation))
            self.assertTrue(all(f"unique{j}" in terms for j in training))
            expected = mutual_info_classif(X > 0, y, discrete_features=True) / np.log(2)
            np.testing.assert_allclose(scores, expected, atol=1e-14)
        self.assertEqual(vectors[-1][0], tuple(train.clean_text))  # Final training-set refit.

        vectors.clear()
        with patch.object(CountVectorizer, "fit_transform", vectorizer_fit):
            tune(logistic_pipeline(), {"model__C": [1]}, train, lambda p: p["model__C"])
        self.assertEqual(len(vectors), len(folds) + 1)
        for (documents, terms), (training, validation) in zip(vectors[:-1], folds):
            self.assertEqual(documents, tuple(train.clean_text.iloc[training]))
            self.assertTrue(all(f"unique{j}" not in terms for j in validation))

    def run_main(self, evaluate_test=False, invalid_split=False):
        data = reviews()
        split = make_split(data)
        with ExitStack() as stack:
            mocks = {
                name: stack.enter_context(patch.object(main, name))
                for name in (
                    "prepare_data", "split_data", "save_cleaned_data", "describe_data",
                    "audit", "summarize_bow", "report", "train_models", "evaluate_models",
                )
            }
            partition = stack.enter_context(patch.object(main, "train_test_data", wraps=train_test_data))
            mocks["prepare_data"].return_value = data
            mocks["split_data"].return_value = split
            if invalid_split:
                mocks["split_data"].side_effect = ValueError("invalid split")
                with self.assertRaisesRegex(ValueError, "invalid split"):
                    main.main(evaluate_test=evaluate_test)
                mocks["save_cleaned_data"].assert_not_called()
                mocks["describe_data"].assert_not_called()
                mocks["train_models"].assert_not_called()
            else:
                main.main(evaluate_test=evaluate_test)
                training = mocks["train_models"].call_args.args[0]
                expected_ids = data.loc[data.id.isin(split.loc[split.split.eq("train"), "id"]), "id"]
                self.assertEqual(training.id.tolist(), expected_ids.tolist())
                if evaluate_test:
                    partition.assert_called_once()
                    mocks["evaluate_models"].assert_called_once()
                    test = mocks["evaluate_models"].call_args.args[1]
                    self.assertEqual(set(test.id), set(split.loc[split.split.eq("test"), "id"]))
                else:
                    partition.assert_not_called()
                    mocks["evaluate_models"].assert_not_called()

    def test_default_execution_does_not_evaluate_or_materialize_test_partition(self):
        self.assertFalse(main.parse_args([]).evaluate_test)
        self.run_main()

    def test_explicit_option_dispatches_evaluation_using_only_synthetic_data(self):
        self.assertTrue(main.parse_args(["--evaluate-test"]).evaluate_test)
        self.run_main(evaluate_test=True)

    def test_incompatible_input_stops_before_artifact_writes(self):
        self.run_main(invalid_split=True)


if __name__ == "__main__":
    unittest.main()
