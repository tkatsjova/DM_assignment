import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data.split import load_or_create_split, make_split, save_split, train_test_data, validate_split


def reviews():
    return pd.DataFrame({
        "id": range(40),
        "label": [0] * 20 + [1] * 20,
        "sentiment": ["pos"] * 10 + ["neg"] * 10 + ["pos"] * 10 + ["neg"] * 10,
        "clean_text": [f"review {i}" for i in range(40)],
    })


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.data = reviews()
        self.split = make_split(self.data)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "split.csv"

    def test_valid_split_is_balanced_disjoint_and_complete(self):
        train, test = train_test_data(self.data, self.split)
        self.assertEqual((len(train), len(test)), (30, 10))
        self.assertFalse(set(train.id) & set(test.id))
        self.assertEqual(set(train.id) | set(test.id), set(self.data.id))
        self.assertEqual(test.label.value_counts().to_dict(), {0: 5, 1: 5})

    def test_invalid_structure_and_partitioning_are_rejected(self):
        duplicate = pd.concat([self.split, self.split.iloc[[0]].assign(split="test")])
        extra = self.split.copy()
        extra.loc[0, "id"] = 100
        invalid = self.split.copy()
        invalid.loc[0, "split"] = "validation"
        null_id = self.split.copy()
        null_id.loc[0, "id"] = None
        for split in (duplicate, extra, invalid, null_id, self.split.iloc[1:], self.split.drop(columns="split"), self.split.assign(split="train")):
            with self.subTest(split=split.head().to_dict()), self.assertRaises(ValueError):
                train_test_data(self.data, split)
        for data in (pd.concat([self.data, self.data.iloc[[0]]]), self.data.assign(id=None), self.data.assign(label=2), self.data.assign(sentiment="unknown")):
            with self.subTest(data=data.head().to_dict()), self.assertRaises(ValueError):
                validate_split(data, self.split)

    def test_wrong_partition_size_and_class_balance_are_rejected(self):
        wrong_size = self.split.copy()
        wrong_size.loc[wrong_size.split.eq("train").idxmax(), "split"] = "test"
        with self.assertRaisesRegex(ValueError, "ratio"):
            validate_split(self.data, wrong_size)
        unbalanced = self.split.assign(split=["test"] * 10 + ["train"] * 30)
        with self.assertRaisesRegex(ValueError, "balance"):
            validate_split(self.data, unbalanced)
        # Balanced classes alone are insufficient: preserve sentiment stratification too.
        unbalanced_sentiment = self.split.assign(split="train")
        unbalanced_sentiment.loc[unbalanced_sentiment.id.isin(list(range(5)) + list(range(20, 25))), "split"] = "test"
        with self.assertRaisesRegex(ValueError, "balance"):
            validate_split(self.data, unbalanced_sentiment)

    def test_create_once_then_reuse_without_writing(self):
        first = load_or_create_split(self.data, self.path)
        pd.testing.assert_frame_equal(first, self.split)  # Same fixed-seed assignment.
        before = self.path.read_bytes()
        second = load_or_create_split(self.data, self.path)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(before, self.path.read_bytes())
        with self.assertRaises(FileExistsError):
            save_split(second, self.path)

    def test_incompatible_review_ids_do_not_regenerate_saved_split(self):
        load_or_create_split(self.data, self.path)
        changed = self.data.copy()
        changed.loc[0, "id"] = 100
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "cover every review"):
            load_or_create_split(changed, self.path)
        self.assertEqual(before, self.path.read_bytes())


if __name__ == "__main__":
    unittest.main()
