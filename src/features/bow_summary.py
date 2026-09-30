import numpy as np
import pandas as pd

from src.config import LABEL_NAMES
from src.features.vectorizer import build_vectorizer

# Overview of the bag-of-words representation of the training set.

# Mirrors the inspection steps of the course example in R: a review before and after
# preprocessing, the size of the document-term matrix before and after sparse-term
# removal, its sparsity, and a small part of the matrix (tm's inspect()).


EXAMPLE_MIN_DF = 5  # illustrative threshold; the value used by the models is tuned by CV
N_ROWS_PER_CLASS, N_TERMS = 5, 6


def _dtm_row(name, dtm, vectorizer):
    n_docs, n_terms = dtm.shape
    nonzero = dtm.nnz
    return {
        "Matrix": name,
        "Documents": n_docs,
        "Terms": n_terms,
        "Non-sparse entries": nonzero,
        "Sparse entries": n_docs * n_terms - nonzero,
        "Sparsity": f"{1 - nonzero / (n_docs * n_terms):.2%}",
        "Maximal term length": max(len(t) for t in vectorizer.get_feature_names_out()),
    }


def _examples(train):
    rows = []
    for label, name in LABEL_NAMES.items():
        review = train[train["label"] == label].iloc[0]
        rows.append({"Review": f"{name} (id {review['id']})", "Stage": "before", "Text": review["text"]})
        rows.append({"Review": f"{name} (id {review['id']})", "Stage": "after", "Text": review["clean_text"]})
    return pd.DataFrame(rows).set_index(["Review", "Stage"])


def _matrix_part(train, dtm, vectorizer):
    # A few documents x a few mid-frequency terms, like inspect(dtm[rows, cols]) in R
    terms = vectorizer.get_feature_names_out()
    doc_freq = np.asarray((dtm > 0).sum(axis=0)).ravel()
    # skip the 100 most common terms (mostly stop words) and very rare ones; alphabetical order
    candidates = np.argsort(-doc_freq)[100:130]
    cols = np.sort(candidates[:N_TERMS])
    rows = np.concatenate(
        [np.flatnonzero(train["label"].to_numpy() == label)[:N_ROWS_PER_CLASS] for label in LABEL_NAMES]
    )
    part = pd.DataFrame(
        dtm[rows][:, cols].toarray(),
        columns=terms[cols],
        index=pd.Index(train["id"].iloc[rows], name="id"),
    )
    part.insert(0, "class", train["label"].iloc[rows].map(LABEL_NAMES).to_numpy())
    return part


def _top_terms(train, dtm, vectorizer, n=15):
    terms = vectorizer.get_feature_names_out()
    table = {}
    for label, name in LABEL_NAMES.items():
        counts = np.asarray(dtm[(train["label"] == label).to_numpy()].sum(axis=0)).ravel()
        top = np.argsort(-counts)[:n]
        table[f"{name} term"] = terms[top]
        table[f"{name} count"] = counts[top]
    return pd.DataFrame(table, index=pd.RangeIndex(1, n + 1, name="rank"))


def summarize_bow(train):
    # Tables describing the document-term matrix of the training set
    full = build_vectorizer()
    dtm_full = full.fit_transform(train["clean_text"])
    reduced = build_vectorizer(min_df=EXAMPLE_MIN_DF)
    dtm_reduced = reduced.fit_transform(train["clean_text"])

    words = dtm_full.sum(axis=1).A1
    distinct = (dtm_full > 0).sum(axis=1).A1
    per_review = pd.DataFrame(
        {"mean": [words.mean(), distinct.mean()], "median": [np.median(words), np.median(distinct)],
         "min": [words.min(), distinct.min()], "max": [words.max(), distinct.max()]},
        index=pd.Index(["Tokens per review", "Distinct terms per review"], name="Measure"),
    ).round(1)

    return {
        "Example reviews before and after preprocessing": _examples(train),
        "Document-term matrix (training set)": pd.DataFrame(
            [_dtm_row("all terms", dtm_full, full),
             _dtm_row(f"min_df = {EXAMPLE_MIN_DF}", dtm_reduced, reduced)]
        ).set_index("Matrix"),
        "Review length after preprocessing": per_review,
        f"Part of the document-term matrix ({N_ROWS_PER_CLASS} reviews per class)": _matrix_part(train, dtm_full, full),
        "Most frequent terms per class": _top_terms(train, dtm_full, full),
    }
