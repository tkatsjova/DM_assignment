import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.config import LABEL_NAMES
from src.features.selection import mutual_information
from src.features.vectorizer import VECTORIZER_GRID, build_vectorizer
from src.models.tuning import cv_results, selection_summary, tune

# Multinomial naive Bayes with top-k feature selection by mutual information (course pipeline)

# Number of terms kept by feature selection; "all" = no selection besides sparse-term removal
K_VALUES = [50, 100, 250, 500, 1000, 2000, "all"]

PARAM_GRID = {**VECTORIZER_GRID, "select__k": K_VALUES}


def build_pipeline():
    return Pipeline(
        [
            ("vectorizer", build_vectorizer()),
            ("select", SelectKBest(mutual_information)),
            ("model", MultinomialNB(alpha=1.0)),  # Laplace smoothing, as in the course
        ]
    )


def feature_count(train):
    # Complexity of a setting: the number of terms the model uses.
    # min(k, vocabulary size), with the vocabulary size of the vectorizer setting measured on the training set.

    vocabulary_sizes = {}

    def count(params):
        setting = (params["vectorizer__ngram_range"], params["vectorizer__min_df"])
        if setting not in vocabulary_sizes:
            vectorizer = build_vectorizer(ngram_range=setting[0], min_df=setting[1])
            vocabulary_sizes[setting] = len(vectorizer.fit(train["clean_text"]).vocabulary_)
        k = params["select__k"]
        return vocabulary_sizes[setting] if k == "all" else min(k, vocabulary_sizes[setting])

    return count


def top_features(pipeline, train, n=5):
    # Terms with the largest score difference log P(w|ai) - log P(w|human) (course slide 45).

    terms = pipeline["vectorizer"].get_feature_names_out()[pipeline["select"].get_support()]
    model = pipeline["model"]
    ai, human = (list(model.classes_).index(label) for label in (1, 0))
    diff = model.feature_log_prob_[ai] - model.feature_log_prob_[human]

    # the matrix the model sees: vectorizer and feature selection, without the classifier
    present = pipeline[:-1].transform(train["clean_text"]) > 0
    labels = train["label"].to_numpy()
    reviews = {label: np.asarray(present[labels == label].sum(axis=0)).ravel() for label in LABEL_NAMES}

    order = np.argsort(diff)
    rows = []
    for direction, idx in ((LABEL_NAMES[1], order[::-1][:n]), (LABEL_NAMES[0], order[:n])):
        for rank, i in enumerate(idx, start=1):
            rows.append(
                {
                    "towards": direction,
                    "rank": rank,
                    "term": terms[i],
                    "score diff": round(diff[i], 2),
                    **{f"reviews {name}": reviews[label][i] for label, name in LABEL_NAMES.items()},
                }
            )
    return pd.DataFrame(rows).set_index(["towards", "rank"])


def _by_k(results):
    # Best CV accuracy for each k, to show how accuracy depends on the number of terms
    results = results.assign(k=results["k"].astype(str))
    best = results.sort_values("accuracy", ascending=False).drop_duplicates("k")
    order = {str(k): i for i, k in enumerate(K_VALUES)}
    return best.sort_values("k", key=lambda s: s.map(order)).set_index("k")


def run_naive_bayes(train):
    complexity = feature_count(train)
    search = tune(build_pipeline(), PARAM_GRID, train, complexity)
    results = cv_results(search, complexity)
    summary = selection_summary(search, complexity)
    # k larger than the vocabulary keeps every term, i.e. no selection by mutual information
    n_terms = len(search.best_estimator_["vectorizer"].vocabulary_)
    summary.loc["feature selection by MI"] = "yes" if complexity(search.best_params_) < n_terms else "no (all terms)"
    tables = {
        "Selected hyperparameters (1-SE rule, complexity = number of terms used)": summary,
        "Cross-validation results (10 best settings)": results.head(10),
        "Settings within one standard error of the best": results[results["within_1se"]],
        "Accuracy by number of selected terms (best vectorizer setting per k)": _by_k(results),
        "Most important features (reviews = training reviews containing the term)": top_features(
            search.best_estimator_, train
        ),
    }
    return search.best_estimator_, tables
