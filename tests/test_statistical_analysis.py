import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import main
from src.evaluation import statistical_analysis as analysis


def paired_predictions(a=5, b=8, c=2, d=3):
    # Counts: both correct, first only correct, second only correct, both incorrect.
    n = a + b + c + d
    labels = np.arange(n) % 2
    first_correct = np.array([True] * (a + b) + [False] * (c + d))
    second_correct = np.array([True] * a + [False] * b + [True] * c + [False] * d)
    first = np.where(first_correct, labels, 1 - labels)
    second = np.where(second_correct, labels, 1 - labels)
    return pd.DataFrame({
        "id": np.arange(n), "y_true": labels, "sentiment": np.where(np.arange(n) % 2, "neg", "pos"),
        "pred_naive_bayes": first, "pred_logistic_regression": second,
        "pred_classification_tree": first, "pred_random_forest": second, "pred_gradient_boosting": second,
    })


class StatisticalAnalysisTests(unittest.TestCase):
    def test_mcnemar_table_and_known_two_sided_binomial_probability(self):
        predictions = paired_predictions()
        table, p_value = analysis.mcnemar(predictions, "naive_bayes", "logistic_regression")
        np.testing.assert_array_equal(table.to_numpy(), [[5, 8], [2, 3]])
        self.assertAlmostEqual(p_value, 0.109375)  # 2 * (1 + 10 + 45) / 2**10.
        reversed_table, reversed_p = analysis.mcnemar(predictions, "logistic_regression", "naive_bayes")
        np.testing.assert_array_equal(reversed_table.to_numpy(), table.to_numpy().T)
        self.assertEqual(reversed_p, p_value)
        shuffled, shuffled_p = analysis.mcnemar(predictions.sample(frac=1, random_state=42), "naive_bayes", "logistic_regression")
        pd.testing.assert_frame_equal(shuffled, table)
        self.assertEqual(shuffled_p, p_value)

    def test_no_discordance_and_empty_correctness_cells(self):
        table, p_value = analysis.mcnemar(paired_predictions(a=4, b=0, c=0, d=0), "naive_bayes", "logistic_regression")
        np.testing.assert_array_equal(table.to_numpy(), [[4, 0], [0, 0]])
        self.assertEqual(p_value, 1.0)

    def test_fisher_uses_sentiment_groups_and_known_hypergeometric_probability(self):
        predictions = paired_predictions(a=13, b=0, c=0, d=0)
        predictions["sentiment"] = ["pos"] * 8 + ["neg"] * 5
        correct = np.array([True] * 6 + [False] * 2 + [True] + [False] * 4)
        predictions["pred_random_forest"] = np.where(correct, predictions.y_true, 1 - predictions.y_true)
        table, p_value = analysis.forest_by_sentiment(predictions)
        np.testing.assert_array_equal(table[["correct", "incorrect"]].to_numpy(), [[6, 2], [1, 4]])
        self.assertEqual(table.loc["pos", "accuracy"], .75)
        self.assertEqual(table.loc["neg", "accuracy"], .2)
        self.assertAlmostEqual(p_value, 132 / 1287)  # Sum of the three tables no more likely than the observed table.

    def test_planned_tests_and_significance_decisions(self):
        tables = analysis.statistical_tables(paired_predictions(a=0, b=0, c=18, d=0))
        comparisons = tables["McNemar test results"]
        self.assertEqual(len(comparisons), 3)
        self.assertTrue(comparisons["decision at alpha=0.01"].eq("Reject equal accuracy").all())
        for p_value in comparisons["exact two-sided p-value"]:
            self.assertAlmostEqual(float(p_value), 2 / 2**18)
        tables = analysis.statistical_tables(paired_predictions())
        self.assertTrue(tables["McNemar test results"]["decision at alpha=0.01"].eq("Do not reject equal accuracy").all())

    def test_invalid_prediction_data_is_rejected(self):
        predictions = paired_predictions()
        for invalid in (predictions.assign(id=0), predictions.assign(pred_random_forest=np.nan),
                        predictions.assign(y_true=2), predictions.assign(sentiment="pos")):
            with self.subTest(), self.assertRaises(ValueError):
                analysis.statistical_tables(invalid)

    def test_prediction_consistency_matches_by_id_and_detects_changes(self):
        current = paired_predictions()
        previous = current.sample(frac=1, random_state=42).reset_index(drop=True)
        self.assertTrue(analysis.prediction_consistency(previous, current)["changed predictions"].eq(0).all())
        previous.loc[0, "pred_naive_bayes"] = 1 - previous.loc[0, "pred_naive_bayes"]
        self.assertEqual(analysis.prediction_consistency(previous, current).loc["naive_bayes", "changed predictions"], 1)
        with self.assertRaisesRegex(ValueError, "same unique review IDs"):
            analysis.prediction_consistency(previous.iloc[1:], current)

    def test_feature_summary_copies_recorded_scores_without_following_sections(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "naive_bayes.md"
            table = "| towards | term | score diff |\n|---|---|---:|\n| human | told | -3.85 |"
            source.write_text("# Model\n\n## Most important features (training set)\n\n" + table + "\n\n## Other table\nDo not copy", encoding="utf-8")
            output = Path(folder) / "feature_summary.md"
            with patch.object(analysis, "MODEL_REPORTS", {"Naive Bayes": source}):
                analysis.save_feature_summary(output)
            text = output.read_text(encoding="utf-8")
            self.assertIn(table, text)
            self.assertIn("unsigned", text)
            self.assertNotIn("Do not copy", text)

    def test_final_evaluation_saves_all_predictions_and_dispatches_analysis(self):
        predictions = paired_predictions()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "predictions.csv"
            predictions.to_csv(path, index=False)
            with patch.object(main, "evaluate", return_value=(predictions, {})), \
                 patch.object(main, "TEST_PREDICTIONS_PATH", path), \
                 patch.object(main, "report") as report, \
                 patch.object(main, "save_feature_summary") as features:
                main.evaluate_models({}, pd.DataFrame())
            pd.testing.assert_frame_equal(pd.read_csv(path), predictions)
            self.assertEqual(report.call_count, 2)
            consistency = report.call_args_list[0].args[0]["Prediction consistency with previous saved NB/LR predictions"]
            self.assertTrue(consistency["changed predictions"].eq(0).all())
            self.assertIn("McNemar test results", report.call_args_list[1].args[0])
            features.assert_called_once()


if __name__ == "__main__":
    unittest.main()
