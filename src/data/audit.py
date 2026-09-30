import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_selection import mutual_info_classif

from src.config import LABEL_NAMES, RANDOM_STATE
from src.data.preprocessing import is_empty_marker, normalize_text
from src.features.vectorizer import build_vectorizer

"""Evidence behind the preprocessing decisions (docs/preprocessing_decisions.md).

Computed on the training set only, so that the test set does not influence any preprocessing choice.
"""


STOP_WORDS_SHOWN = ["a", "the", "i", "we", "my", "it", "very", "were", "with", "was"]
MIN_DF_VALUES = [1, 2, 5, 10, 20, 75]  # 75 documents = 5% of the training set


def _share(mask, labels):
    # Share of reviews per class for which `mask` is True, in %
    return (pd.Series(mask, index=labels.index).groupby(labels.map(LABEL_NAMES)).mean() * 100).round(1)


def _contains_word(texts, words):
    return np.array([bool(w) and f" {w} " in f" {t} " for t, w in zip(texts, words)])


def placeholder_parts(train):
    # D3: parts that only say "nothing", and truly empty parts
    group = [train["label"].map(LABEL_NAMES), train["sentiment"]]
    return pd.DataFrame(
        {
            "Upside placeholder": train["upside"].map(is_empty_marker).groupby(group).sum(),
            "Downside placeholder": train["downside"].map(is_empty_marker).groupby(group).sum(),
            "Upside empty": train["upside_missing"].groupby(group).sum(),
            "Downside empty": train["downside_missing"].groupby(group).sum(),
        }
    ).rename_axis(["class", "sentiment"])


def text_properties(train):
    # D4-D7: share of reviews (%) with properties that differ between the classes
    text, clean, labels = train["text"], train["clean_text"], train["label"]
    city = train["city"].map(normalize_text)
    hotel = train["hotel"].map(normalize_text)
    raw_norm = text.map(normalize_text)
    # first word of the hotel name that is not generic, e.g. "ickale" for "Hotel Ickale"
    hotel_word = hotel.map(lambda h: next((w for w in h.split() if w not in {"hotel", "the"}), ""))

    rows = {
        "D4 city name (before)": _contains_word(raw_norm, city),
        "D4 full hotel name (before)": _contains_word(raw_norm, hotel),
        "D4 city name (after)": _contains_word(clean, city),
        "D4 hotel name word (after)": _contains_word(clean, hotel_word),
        "D4 word 'hotel' (after)": clean.str.contains(r"\bhotel\b").to_numpy(),
        "D5 written fully in lower case": (text == text.str.lower()).to_numpy(),
        "D6 curly apostrophe": text.str.contains("[’‘]").to_numpy(),
        "D6 exclamation mark": text.str.contains("!", regex=False).to_numpy(),
        "D7 contains a digit": text.str.contains(r"\d").to_numpy(),
    }
    table = pd.DataFrame({name: _share(mask, labels) for name, mask in rows.items()}).T
    return table.rename_axis("Share of reviews (%)")


def stop_word_usage(train):
    # D8: relative frequency of common stop words, and stop words among the most informative terms
    vectorizer = build_vectorizer(min_df=5)
    dtm = vectorizer.fit_transform(train["clean_text"])
    terms = vectorizer.get_feature_names_out()
    y = train["label"].to_numpy()

    freq = pd.DataFrame(index=pd.Index(STOP_WORDS_SHOWN, name="stop word"))
    for label, name in LABEL_NAMES.items():
        counts = np.asarray(dtm[y == label].sum(axis=0)).ravel()
        per_1000 = pd.Series(counts / counts.sum() * 1000, index=terms)
        freq[f"{name} per 1000 tokens"] = per_1000.reindex(STOP_WORDS_SHOWN).round(1)
    freq["ai / human"] = (freq.iloc[:, 1] / freq.iloc[:, 0]).round(2)

    mi = mutual_info_classif(dtm > 0, y, discrete_features=True, random_state=RANDOM_STATE)
    order = np.argsort(-mi)
    top = pd.DataFrame(
        {
            "Stop words among them": [sum(t in ENGLISH_STOP_WORDS for t in terms[order[:n]]) for n in (10, 30, 100)],
            "Stop words (examples)": [
                ", ".join([t for t in terms[order[:n]] if t in ENGLISH_STOP_WORDS][:12]) for n in (10, 30, 100)
            ],
        },
        index=pd.Index(["top 10", "top 30", "top 100"], name="Terms by mutual information"),
    )
    return freq, top


def sparse_terms(train):
    # D13: what sparse-term removal keeps for different min_df values
    rows = []
    for min_df in MIN_DF_VALUES:
        uni = build_vectorizer(min_df=min_df)
        dtm = uni.fit_transform(train["clean_text"])
        full_tokens = build_vectorizer().fit_transform(train["clean_text"]).sum()
        bi = build_vectorizer(min_df=min_df, ngram_range=(1, 2)).fit(train["clean_text"])
        rows.append(
            {
                "min_df (documents)": min_df,
                "Unigrams kept": dtm.shape[1],
                "Share of tokens kept (%)": round(dtm.sum() / full_tokens * 100, 1),
                "Distinct terms per review": round((dtm > 0).sum(axis=1).mean(), 1),
                "Uni+bigrams kept": len(bi.vocabulary_),
            }
        )
    table = pd.DataFrame(rows).set_index("min_df (documents)")
    doc_freq = np.asarray((build_vectorizer().fit_transform(train["clean_text"]) > 0).sum(axis=0)).ravel()
    singletons = pd.DataFrame(
        {"Value": [len(doc_freq), int((doc_freq == 1).sum()), round((doc_freq == 1).mean() * 100, 1)]},
        index=pd.Index(["Unigrams", "Occurring in one review only", "Share (%)"], name="Vocabulary"),
    )
    return table, singletons


def audit(train):
    stop_freq, stop_mi = stop_word_usage(train)
    sparse, singletons = sparse_terms(train)
    return {
        "D3 Placeholder and empty parts": placeholder_parts(train),
        "D4-D7 Text properties by class": text_properties(train),
        "D8 Stop word frequency": stop_freq,
        "D8 Stop words among the most informative terms": stop_mi,
        "D13 Sparse-term removal": sparse,
        "D13 Terms occurring in a single review": singletons,
    }
