import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import LABEL_NAMES, RANDOM_STATE
from src.features.vectorizer import VECTORIZER_GRID, build_vectorizer
from src.models.tuning import cv_results, selection_summary, tune

# Fixed pruning grid: unpruned baseline, then light through strong pruning.
# Each CV training fold has 1200 reviews; small impurity changes are of order 1/1200.
ALPHA_VALUES = [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]
PARAM_GRID = {**VECTORIZER_GRID, "model__ccp_alpha": ALPHA_VALUES}


def build_pipeline():
    return Pipeline([
        ("vectorizer", build_vectorizer()),
        ("model", DecisionTreeClassifier(criterion="gini", random_state=RANDOM_STATE)),
    ])


def pruning_strength(params):
    # The shared selector minimizes complexity: larger alpha means stronger pruning.
    return -params["model__ccp_alpha"]


def top_features(pipeline, train, n=5):
    # Normalized, sample-weighted Gini reductions; unsigned and fitted on training data.
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


def tree_size_path(pipeline, train):
    # Descriptive fits at the selected vocabulary; these do not select hyperparameters.
    X = pipeline["vectorizer"].transform(train["clean_text"])
    rows = []
    for alpha in ALPHA_VALUES:
        model = clone(pipeline["model"]).set_params(ccp_alpha=alpha).fit(X, train["label"])
        rows.append({
            "ccp_alpha": alpha,
            "depth": model.get_depth(),
            "leaves": model.get_n_leaves(),
            "training accuracy": model.score(X, train["label"]),
        })
    return pd.DataFrame(rows).set_index("ccp_alpha").round(4)


def _by_alpha(results):
    best = results.sort_values("accuracy", ascending=False, kind="stable").drop_duplicates("ccp_alpha")
    return best.sort_values("ccp_alpha").set_index("ccp_alpha")


def run_classification_tree(train):
    search = tune(build_pipeline(), PARAM_GRID, train, pruning_strength)
    pipeline = search.best_estimator_
    results = cv_results(search, pruning_strength)
    summary = selection_summary(search, pruning_strength)
    model = pipeline["model"]
    summary.loc["tree depth"] = str(model.get_depth())
    summary.loc["leaves"] = str(model.get_n_leaves())
    summary.loc["features used"] = str(np.count_nonzero(model.feature_importances_))
    summary.loc["vocabulary size"] = str(len(pipeline["vectorizer"].vocabulary_))
    tables = {
        "Selected hyperparameters (1-SE rule, complexity = negative ccp_alpha)": summary,
        "Cross-validation results (all settings)": results,
        "Settings within one standard error of the best": results[results["within_1se"]],
        "Accuracy by alpha (best vectorizer setting per alpha)": _by_alpha(results),
        "Tree size by alpha (selected vectorizer setting, whole training set)": tree_size_path(pipeline, train),
        "Most important features (reviews = training reviews containing the term)": top_features(pipeline, train),
    }
    return pipeline, tables
