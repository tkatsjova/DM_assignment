from time import perf_counter

import numpy as np
import pandas as pd
from joblib import parallel_config
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from src.config import LABEL_NAMES, RANDOM_STATE
from src.features.vectorizer import VECTORIZER_GRID, build_vectorizer
from src.models.tuning import cv_results, selection_summary, tune

CV_WORKERS = 4
PARAM_GRID = {
    **VECTORIZER_GRID,
    "model__learning_rate": [0.05, 0.1, 0.2, 0.3],
    "model__max_depth": [1, 2, 3, 4],
}


def build_pipeline():
    return Pipeline([
        ("vectorizer", build_vectorizer()),
        ("model", GradientBoostingClassifier(
            loss="log_loss",
            n_estimators=100,
            subsample=1.0,
            n_iter_no_change=None,  # Fixed stage count; no internal validation split.
            random_state=RANDOM_STATE,
        )),
    ])


def tree_depth(params):
    # Shallower regression trees limit interactions; other settings tie on this proxy.
    return params["model__max_depth"]


def top_features(pipeline, train, n=5):
    # Unsigned impurity importance aggregated over the fitted residual regression trees.
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


def stage_structure(pipeline):
    trees = pipeline["model"].estimators_.ravel()  # One regression tree per binary boosting stage.
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


def staged_training_results(pipeline, train):
    # Descriptive training curve only; it does not select the number of stages.
    model = pipeline["model"]
    X = pipeline["vectorizer"].transform(train["clean_text"])
    checkpoints = {10, 25, 50, 75, model.n_estimators_}
    rows = []
    for stage, prediction in enumerate(model.staged_predict(X), start=1):
        if stage in checkpoints:
            rows.append({
                "stage": stage,
                "training accuracy": accuracy_score(train["label"], prediction),
                "training loss": model.train_score_[stage - 1],
            })
    return pd.DataFrame(rows).set_index("stage").round(4)


def _by_boosting_setting(results):
    parameters = ["max_depth", "learning_rate"]
    best = results.sort_values("accuracy", ascending=False, kind="stable").drop_duplicates(parameters)
    return best.sort_values(parameters).set_index(parameters)


def run_gradient_boosting(train):
    start = perf_counter()
    # GridSearchCV's n_jobs=None inherits this context; existing models are unaffected.
    with parallel_config(backend="loky", n_jobs=CV_WORKERS, inner_max_num_threads=1):
        search = tune(build_pipeline(), PARAM_GRID, train, tree_depth)
    pipeline = search.best_estimator_
    results = cv_results(search, tree_depth)
    summary = selection_summary(search, tree_depth)
    model = pipeline["model"]
    training_accuracy = pipeline.score(train["clean_text"], train["label"])
    chosen_cv_accuracy = search.cv_results_["mean_test_accuracy"][search.best_index_]
    summary.loc["n_estimators (fixed)"] = str(model.n_estimators)
    summary.loc["fitted boosting stages"] = str(model.n_estimators_)
    summary.loc["maximum fitted tree depth"] = str(max(tree.get_depth() for tree in model.estimators_.ravel()))
    summary.loc["training accuracy"] = f"{training_accuracy:.4f}"
    summary.loc["training minus selected CV accuracy"] = f"{training_accuracy - chosen_cv_accuracy:.4f}"
    summary.loc["vocabulary size"] = str(len(pipeline["vectorizer"].vocabulary_))
    summary.loc["features used across stages"] = str(np.count_nonzero(model.feature_importances_))
    summary.loc["CV workers"] = str(CV_WORKERS)
    tables = {
        "Selected hyperparameters (1-SE rule, complexity = max_depth)": summary,
        "Cross-validation results (all settings)": results,
        "Settings within one standard error of the best": results[results["within_1se"]],
        "Accuracy by boosting setting (best vectorizer setting per setting)": _by_boosting_setting(results),
        "Selected boosting tree structure (whole training set)": stage_structure(pipeline),
        "Training performance by stage (selected setting, descriptive only)": staged_training_results(pipeline, train),
        "Most important features (reviews = training reviews containing the term)": top_features(pipeline, train),
    }
    summary.loc["experiment seconds (CV, refit and summaries)"] = f"{perf_counter() - start:.2f}"
    return pipeline, tables
