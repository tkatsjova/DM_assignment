import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.config import CV_FOLDS, RANDOM_STATE

# Hyperparameter tuning with cross-validation on the training set, shared by all models

# Selection uses the one-standard-error rule:
# among all settings whose CV accuracy is within one standard error of the best one,
# the simplest setting is chosen. Each model defines what "simple" means through a
# complexity function of its parameters (e.g. the number of features it uses).

# The same folds for every model, so that their CV results are comparable
CV = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
SCORING = {"accuracy": "accuracy", "f1": "f1"}


def _accuracy(cv_results):
    # Mean CV accuracy, rounded so that settings with the same number of correct predictions are equal (the fold means otherwise differ in the last floating-point digit)
    return np.round(np.asarray(cv_results["mean_test_accuracy"]), 10)


def _fold_std(cv_results):
    # Sample standard deviation (ddof=1) of the accuracy over the folds, for every setting.

    folds = np.array([cv_results[f"split{i}_test_accuracy"] for i in range(CV_FOLDS)])
    with np.errstate(invalid="ignore"):
        return folds.std(axis=0, ddof=1)


def _selection_values(cv_results, complexity):
    accuracy = _accuracy(cv_results)
    cost = np.array([complexity(p) for p in cv_results["params"]], dtype=float)
    folds = np.array([cv_results[f"split{i}_test_accuracy"] for i in range(CV_FOLDS)])
    valid = np.isfinite(accuracy) & np.isfinite(cost) & np.isfinite(folds).all(axis=0)
    if not valid.any():
        raise ValueError("No valid cross-validation candidates: mean accuracy, every fold accuracy and complexity must be finite.")
    return accuracy, cost, valid


def _simplest(indices, cost, accuracy):
    # Index with the lowest complexity; ties broken by the higher accuracy, then the grid order
    return indices[np.lexsort((-accuracy[indices], cost[indices]))[0]]


def _one_se_threshold(cv_results, complexity):
    # Best CV accuracy, its standard error and the 1-SE threshold.

    accuracy, cost, valid = _selection_values(cv_results, complexity)
    best = _simplest(np.flatnonzero(valid & (accuracy == accuracy[valid].max())), cost, accuracy)
    # standard error of the mean over the folds, as in cv.glmnet: sample std / sqrt(K)
    se = _fold_std(cv_results)[best] / np.sqrt(CV_FOLDS)
    return best, se, accuracy[best] - se


def one_se_rule(complexity):
    # Refit strategy for GridSearchCV: simplest setting within 1 SE of the best CV accuracy

    def select(cv_results):
        accuracy, cost, valid = _selection_values(cv_results, complexity)
        _, _, threshold = _one_se_threshold(cv_results, complexity)
        return _simplest(np.flatnonzero(valid & (accuracy >= threshold)), cost, accuracy)

    return select


def tune(pipeline, param_grid, train, complexity):
    # Grid search with CV; the setting chosen by the 1-SE rule is refitted on the whole training set
    search = GridSearchCV(pipeline, param_grid, scoring=SCORING, refit=one_se_rule(complexity), cv=CV)
    with warnings.catch_warnings():
        # SelectKBest warns when k exceeds the vocabulary size and then keeps all terms, which is intended
        warnings.filterwarnings("ignore", message="k=.* is greater than n_features")
        search.fit(train["clean_text"], train["label"])
    return search


def cv_results(search, complexity):
    # CV accuracy and F1 (mean and std over the folds) for every setting, best accuracy first.

    results = search.cv_results_
    accuracy, _, valid = _selection_values(results, complexity)
    _, _, threshold = _one_se_threshold(results, complexity)
    params = pd.DataFrame(list(results["params"]))
    params.columns = [c.split("__")[-1] for c in params.columns]
    table = params.assign(
        complexity=[complexity(p) for p in results["params"]],
        accuracy=accuracy,
        accuracy_std=_fold_std(results),
        f1=results["mean_test_f1"],
        within_1se=valid & (accuracy >= threshold),
        selected=np.arange(len(params)) == search.best_index_,
    )
    table["rank"] = pd.Series(np.where(valid, accuracy, np.nan)).rank(method="min", ascending=False).astype("Int64")
    return table.sort_values(["rank", "complexity"]).round(4).set_index("rank")


def selection_summary(search, complexity):
    # The chosen setting and how it was chosen
    results = search.cv_results_
    best, se, threshold = _one_se_threshold(results, complexity)
    accuracy, _, valid = _selection_values(results, complexity)
    chosen = search.best_index_
    rows = {k.split("__")[-1]: str(v) for k, v in search.best_params_.items()}
    rows.update(
        {
            "complexity": complexity(results["params"][chosen]),
            "CV accuracy (chosen)": f"{accuracy[chosen]:.4f}",
            "CV accuracy (best)": f"{accuracy[best]:.4f}",
            "settings with the best accuracy": int(np.sum(valid & (accuracy == accuracy[best]))),
            "standard error (best)": f"{se:.4f}",
            "1-SE threshold": f"{threshold:.4f}",
        }
    )
    return pd.Series(rows).astype(str).to_frame("value").rename_axis("parameter")
