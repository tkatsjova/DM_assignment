import math

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
    # Create once; never overwrite an established hold-out partition.
    path.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(path, index=False, mode="x")


def load_split(path=SPLIT_PATH):
    return pd.read_csv(path)


def validate_split(df, split):
    """Check identities, coverage, disjointness, size and class x sentiment balance."""
    if not {"id", "label", "sentiment"}.issubset(df.columns):
        raise ValueError("Review data must contain id, label and sentiment columns.")
    if list(split.columns) != ["id", "split"]:
        raise ValueError("Split must contain exactly the columns id and split.")
    for name, frame in (("Review data", df), ("Split", split)):
        if frame["id"].isna().any() or not frame["id"].is_unique:
            raise ValueError(f"{name} IDs must be non-missing and unique; partitions cannot overlap.")
    if set(df["label"]) != {0, 1} or set(df["sentiment"]) != {"pos", "neg"}:
        raise ValueError("Review labels must be 0/1 and sentiments must be pos/neg.")
    if set(split["split"]) != {"train", "test"}:
        raise ValueError("Split partition labels must be train and test, with both present.")
    if set(df["id"]) != set(split["id"]):
        raise ValueError("Split IDs must cover every review exactly once, with no extra IDs.")
    joined = df[["id", "label", "sentiment"]].merge(split, on="id", validate="one_to_one")
    n_test = math.ceil(len(df) * TEST_SIZE)
    if joined["split"].eq("test").sum() != n_test:
        raise ValueError("Split size is incompatible with the configured train/test ratio.")
    for columns in (["label"], ["label", "sentiment"]):
        totals = joined.groupby(columns).size()
        for partition, size in (("test", n_test), ("train", len(df) - n_test)):
            counts = joined[joined["split"] == partition].groupby(columns).size().reindex(totals.index, fill_value=0)
            # Stratified allocation can round by at most one review per group.
            if ((counts - totals * size / len(df)).abs() >= 1).any():
                raise ValueError(f"Split {partition} class/sentiment balance is incompatible with the data.")


def load_or_create_split(df, path=SPLIT_PATH):
    """Reuse the saved split; create it with the fixed seed only if absent."""
    if path.exists():
        split = load_split(path)
        validate_split(df, split)
        return split
    split = make_split(df)
    validate_split(df, split)
    save_split(split, path)
    return split


def train_test_data(df, split):
    # Return (train, test) DataFrames of the reviews according to the saved split
    validate_split(df, split)
    df = df.merge(split, on="id", validate="one_to_one")
    return df[df["split"] == "train"], df[df["split"] == "test"]
