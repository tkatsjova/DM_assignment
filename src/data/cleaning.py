import pandas as pd

from src.config import ENGLISH_DATA_PATH, LANGUAGE, RAW_DATA_PATH

COLUMN_NAMES = {
    "Unnamed: 0": "orig_index",
    "Review_Language": "language",
    "City Name": "city",
    "Hotel Name": "hotel",
    "Upside_Review": "upside",
    "Downside_Review": "downside",
    "Review_Score": "score",
    "Sentiment": "sentiment",
    "source": "label",
    "Prompt_Language": "prompt_language",
}

def load_raw_data(path=RAW_DATA_PATH):
    return pd.read_csv(path)

def clean_data(raw, language=LANGUAGE):
    # Keep reviews in one language and build a single text per review
    data_frame = raw[raw["Review_Language"] == language].rename(columns=COLUMN_NAMES)
    data_frame = data_frame[list(COLUMN_NAMES.values())].reset_index(drop=True)

    # The na_* flags in the raw file miss some empty fields, so recompute them from the text itself
    data_frame["upside"] = data_frame["upside"].fillna("").str.strip()
    data_frame["downside"] = data_frame["downside"].fillna("").str.strip()
    data_frame["upside_missing"] = data_frame["upside"] == ""
    data_frame["downside_missing"] = data_frame["downside"] == ""

    # A review consists of an upside and a downside part but the classifier sees them as one document
    data_frame["text"] = (data_frame["upside"] + " " + data_frame["downside"]).str.strip()
    data_frame = data_frame[data_frame["text"] != ""].copy()

    data_frame["sentiment"] = data_frame["sentiment"].str.lower()
    data_frame.insert(0, "id", range(len(data_frame)))
    return data_frame

def save_cleaned_data(data_frame, path=ENGLISH_DATA_PATH):
    # Save the cleaned reviews as CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    data_frame.to_csv(path, index=False)

def load_clean_data(path=ENGLISH_DATA_PATH):
    # keep_default_na=False keeps empty upside/downside as "" instead of NaN
    return pd.read_csv(path, keep_default_na=False)
