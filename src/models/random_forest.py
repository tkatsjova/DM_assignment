from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.config import LABEL_NAMES, RANDOM_STATE
from src.features.vectorizer import VECTORIZER_GRID, build_vectorizer
from src.models.tuning import cv_results, selection_summary, tune

PARAM_GRID = {
    **VECTORIZER_GRID,
    "model__n_estimators": [100, 300],
    "model__max_features": ["sqrt", "log2"],
    "model__min_samples_leaf": [1, 3],
}


def build_pipeline():
    return Pipeline([
        ("vectorizer", build_vectorizer()),
        ("model", RandomForestClassifier(
            criterion="gini",
            bootstrap=True,
            oob_score=False,
            random_state=RANDOM_STATE,
            n_jobs=1,  # Sequential CV and tree fits: modest workload, no nested parallelism.
        )),
    ])


def leaf_size(params):
    # The shared selector minimizes complexity; larger leaves constrain individual trees.
    # Other parameters tie on this proxy and are resolved by CV accuracy, then grid order.
    return -params["model__min_samples_leaf"]


def top_features(pipeline, train, n=5):
    # Fitted forest's normalized impurity importances, not directional class coefficients.
    terms = pipeline["vectorizer"].get_feature_names_out()
    importance = pipeline["model"].feature_importances_
    used = np.flatnonzero(importance > 0)
    top = used[np.argsort(-importance[used], kind="stable")[:n]]
    present = pipeline["vectorizer"].transform(train["clean_text"]) > 0
    labels = train["label"].to_numpy()
    return pd.DataFrame(
        {
            "term": terms[top],
            "importance": importance[top].round(4),
            **{
                f"reviews {name}": np.asarray(present[labels == label].sum(axis=0)).ravel()[top]
                for label, name in LABEL_NAMES.items()
            },
        },
        index=pd.RangeIndex(1, len(top) + 1, name="rank"),
    )


def forest_structure(pipeline):
    trees = pipeline["model"].estimators_
    rows = {}
    for name, values in {
        "Tree depth": [tree.get_depth() for tree in trees],
        "Leaves per tree": [tree.get_n_leaves() for tree in trees],
        "Nodes per tree": [tree.tree_.node_count for tree in trees],
    }.items():
        rows[name] = {
            "min": min(values),
            "median": np.median(values),
            "mean": np.mean(values),
            "max": max(values),
        }
    return pd.DataFrame(rows).T.round(2).rename_axis("Measure")


def _by_forest_setting(results):
    parameters = ["n_estimators", "max_features", "min_samples_leaf"]
    best = results.sort_values("accuracy", ascending=False, kind="stable").drop_duplicates(parameters)
    return best.sort_values(parameters).set_index(parameters)


def run_random_forest(train):
    start = perf_counter()
    search = tune(build_pipeline(), PARAM_GRID, train, leaf_size)
    pipeline = search.best_estimator_
    results = cv_results(search, leaf_size)
    summary = selection_summary(search, leaf_size)
    model = pipeline["model"]
    training_accuracy = pipeline.score(train["clean_text"], train["label"])
    chosen_cv_accuracy = search.cv_results_["mean_test_accuracy"][search.best_index_]
    summary.loc["training accuracy"] = f"{training_accuracy:.4f}"
    summary.loc["training minus selected CV accuracy"] = f"{training_accuracy - chosen_cv_accuracy:.4f}"
    summary.loc["vocabulary size"] = str(len(pipeline["vectorizer"].vocabulary_))
    summary.loc["features used across forest"] = str(np.count_nonzero(model.feature_importances_))
    summary.loc["nominal features per split"] = str(model.estimators_[0].max_features_)
    summary.loc["total nodes across forest"] = str(sum(tree.tree_.node_count for tree in model.estimators_))
    tables = {
        "Selected hyperparameters (1-SE rule, complexity = negative min_samples_leaf)": summary,
        "Cross-validation results (all settings)": results,
        "Settings within one standard error of the best": results[results["within_1se"]],
        "Accuracy by forest setting (best vectorizer setting per setting)": _by_forest_setting(results),
        "Selected forest structure (whole training set)": forest_structure(pipeline),
        "Most important features (reviews = training reviews containing the term)": top_features(pipeline, train),
    }
    summary.loc["experiment seconds (CV, refit and summaries)"] = f"{perf_counter() - start:.2f}"
    return pipeline, tables
