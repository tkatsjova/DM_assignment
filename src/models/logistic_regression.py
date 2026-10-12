import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import LABEL_NAMES, RANDOM_STATE
from src.features.vectorizer import VECTORIZER_GRID, build_vectorizer
from src.models.tuning import cv_results, selection_summary, tune

# Logistic regression with L1 (lasso) penalty, as cv.glmnet in the course

# C = 1 / lambda: small C = strong penalty = few non-zero coefficients
C_VALUES = np.logspace(-2, 2, 17).round(4).tolist()

PARAM_GRID = {**VECTORIZER_GRID, "model__C": C_VALUES}


def build_pipeline():
    return Pipeline(
        [
            ("vectorizer", build_vectorizer()),
            (
                "model",
                LogisticRegression(
                    l1_ratio=1,  # pure L1 penalty (sklearn >= 1.8 replaces penalty="l1" by l1_ratio)
                    solver="liblinear",
                    # liblinear penalizes the intercept; large scaling makes this penalty
                    # negligible, approximating the course's unpenalized beta_0.
                    intercept_scaling=100,
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def penalty_strength(params):
    # Complexity for the 1-SE rule: C (a weaker penalty allows more non-zero coefficients).
    # Choosing the smallest C within one SE is the lambda.1se rule of cv.glmnet.

    return params["model__C"]


def _coefficients(pipeline):
    terms = pipeline["vectorizer"].get_feature_names_out()
    model = pipeline["model"]
    coef = model.coef_[0]
    # coef_ refers to classes_[1]; make positive values point towards the generated class
    return terms, coef if model.classes_[1] == 1 else -coef


def top_features(pipeline, train, n=5):
    # Terms with the largest positive (towards ai) and negative (towards human) coefficients.

    # exp(coefficient) is the factor by which the odds of a generated review change for every
    # additional occurrence of the term (course, slide 25).

    terms, coef = _coefficients(pipeline)
    present = pipeline["vectorizer"].transform(train["clean_text"]) > 0
    labels = train["label"].to_numpy()
    reviews = {label: np.asarray(present[labels == label].sum(axis=0)).ravel() for label in LABEL_NAMES}

    order = np.argsort(coef)
    rows = []
    for direction, idx in ((LABEL_NAMES[1], order[::-1][:n]), (LABEL_NAMES[0], order[:n])):
        for rank, i in enumerate(idx, start=1):
            rows.append(
                {
                    "towards": direction,
                    "rank": rank,
                    "term": terms[i],
                    "coefficient": round(coef[i], 3),
                    "odds factor": round(np.exp(coef[i]), 2),
                    **{f"reviews {name}": reviews[label][i] for label, name in LABEL_NAMES.items()},
                }
            )
    return pd.DataFrame(rows).set_index(["towards", "rank"])


def sparsity_path(pipeline, train):
    # Number of non-zero coefficients for every C, with the selected vectorizer setting.
    rows = []
    for c in C_VALUES:
        fitted = clone(pipeline).set_params(model__C=c).fit(train["clean_text"], train["label"])
        rows.append({
            "C": c,
            "lambda = 1/C": round(1 / c, 4),
            "non-zero coefficients": int(np.count_nonzero(fitted["model"].coef_)),
        })
    return pd.DataFrame(rows).set_index("C")


def _by_c(results):
    # Best CV accuracy for each C (over the vectorizer settings)
    best = results.sort_values("accuracy", ascending=False).drop_duplicates("C")
    return best.sort_values("C").set_index("C")


def run_logistic_regression(train):
    search = tune(build_pipeline(), PARAM_GRID, train, penalty_strength)
    pipeline = search.best_estimator_
    results = cv_results(search, penalty_strength)
    summary = selection_summary(search, penalty_strength)
    summary.loc["non-zero coefficients"] = str(np.count_nonzero(pipeline["model"].coef_))
    summary.loc["vocabulary size"] = str(len(pipeline["vectorizer"].vocabulary_))

    tables = {
        "Selected hyperparameters (1-SE rule, complexity = C)": summary,
        "Cross-validation results (10 best settings)": results.head(10),
        "Settings within one standard error of the best": results[results["within_1se"]],
        "Accuracy by C (best vectorizer setting per C)": _by_c(results),
        "Non-zero coefficients by C (selected vectorizer setting, whole training set)": sparsity_path(
            pipeline, train
        ),
        "Most important features (reviews = training reviews containing the term)": top_features(
            pipeline, train
        ),
    }
    return pipeline, tables
