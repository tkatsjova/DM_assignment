import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, SPLIT_PATH, TEST_SIZE


def make_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    # Stratified train/test split on class x sentiment.

    strata = df["label"].astype(str) + "_" + df["sentiment"]
    train_ids, test_ids = train_test_split(
        df["id"], test_size=test_size, stratify=strata, random_state=random_state
    )
    split = pd.concat(
        [
            pd.DataFrame({"id": train_ids, "split": "train"}),
            pd.DataFrame({"id": test_ids, "split": "test"}),
        ]
    )
    return split.sort_values("id").reset_index(drop=True)


def save_split(split, path=SPLIT_PATH):
    # Save the split as CSV with columns id, split (train/test)
    path.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(path, index=False)


def load_split(path=SPLIT_PATH):
    return pd.read_csv(path)


def train_test_data(df, split):
    # Return (train, test) DataFrames of the reviews according to the saved split
    df = df.merge(split, on="id")
    return df[df["split"] == "train"], df[df["split"] == "test"]
