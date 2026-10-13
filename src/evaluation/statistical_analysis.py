import pandas as pd
from scipy.stats import binomtest, fisher_exact

from src.config import (
    CLASSIFICATION_TREE_PATH,
    FEATURE_SUMMARY_PATH,
    GRADIENT_BOOSTING_PATH,
    LOGISTIC_REGRESSION_PATH,
    NAIVE_BAYES_PATH,
    RANDOM_FOREST_PATH,
    STATISTICAL_COMPARISONS_PATH,
    TEST_PREDICTIONS_PATH,
)
from src.data.statistics import print_statistics, save_statistics

ALPHA = 0.01
COMPARISONS = [
    ("Q1", "naive_bayes", "logistic_regression"),
    ("Q2", "classification_tree", "random_forest"),
    ("Q2", "classification_tree", "gradient_boosting"),
]
MODEL_REPORTS = {
    "Naive Bayes": NAIVE_BAYES_PATH,
    "L1 logistic regression": LOGISTIC_REGRESSION_PATH,
    "Classification tree": CLASSIFICATION_TREE_PATH,
    "Random forest": RANDOM_FOREST_PATH,
    "Gradient boosting": GRADIENT_BOOSTING_PATH,
}
METHOD_NOTE = (
    "McNemar: exact two-sided binomial test of discordant correctness counts, with p=0.5. "
    "Fisher: two-sided test of correct/incorrect counts in the disjoint sentiment groups. "
    "All decisions use nominal per-test alpha=0.01, without multiplicity adjustment. "
    "Nonsignificance does not establish equivalence. Tests assume independent reviews; "
    "reviews from the same hotel may be related.\n\n"
    "Methods: [exact McNemar](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html), "
    "[SciPy binomtest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html), "
    "[SciPy fisher_exact](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html)."
)


def mcnemar(predictions, first, second):
    first_correct = predictions[f"pred_{first}"] == predictions["y_true"]
    second_correct = predictions[f"pred_{second}"] == predictions["y_true"]
    table = pd.DataFrame(
        [[(first_correct & second_correct).sum(), (first_correct & ~second_correct).sum()],
         [(~first_correct & second_correct).sum(), (~first_correct & ~second_correct).sum()]],
        index=pd.Index(["correct", "incorrect"], name=first),
        columns=[f"{second} correct", f"{second} incorrect"],
    )
    b, c = int(table.iloc[0, 1]), int(table.iloc[1, 0])
    # With no discordances, there is no evidence against equal marginal accuracy.
    p_value = binomtest(b, n=b + c, p=0.5, alternative="two-sided").pvalue if b + c else 1.0
    return table, p_value


def forest_by_sentiment(predictions):
    correct = predictions["pred_random_forest"] == predictions["y_true"]
    rows = {}
    for sentiment in ("pos", "neg"):
        group = correct[predictions["sentiment"] == sentiment]
        rows[sentiment] = {"correct": int(group.sum()), "incorrect": int((~group).sum()),
                           "accuracy": group.mean()}
    table = pd.DataFrame(rows).T.rename_axis("sentiment")
    table[["correct", "incorrect"]] = table[["correct", "incorrect"]].astype(int)
    p_value = fisher_exact(table[["correct", "incorrect"]].to_numpy(), alternative="two-sided").pvalue
    return table, p_value


def statistical_tables(predictions):
    prediction_columns = {f"pred_{model}" for _, first, second in COMPARISONS for model in (first, second)}
    if predictions.empty or predictions["id"].isna().any() or not predictions["id"].is_unique:
        raise ValueError("Test predictions must have unique, non-missing review IDs and at least one review.")
    if not predictions[["y_true", *sorted(prediction_columns)]].isin([0, 1]).all().all():
        raise ValueError("True labels and model predictions must be non-missing binary values.")
    if set(predictions["sentiment"]) != {"pos", "neg"}:
        raise ValueError("Both pos and neg sentiment groups are required.")

    tables, comparisons = {}, []
    for question, first, second in COMPARISONS:
        table, p_value = mcnemar(predictions, first, second)
        tables[f"{question}: paired correctness ({first} vs {second})"] = table
        comparisons.append({
            "comparison": f"{first} vs {second}",
            "first accuracy": f"{(predictions[f'pred_{first}'] == predictions['y_true']).mean():.4f}",
            "second accuracy": f"{(predictions[f'pred_{second}'] == predictions['y_true']).mean():.4f}",
            "first only correct (b)": int(table.iloc[0, 1]),
            "second only correct (c)": int(table.iloc[1, 0]),
            "discordant reviews": int(table.iloc[0, 1] + table.iloc[1, 0]),
            "exact two-sided p-value": f"{p_value:.16g}",
            "decision at alpha=0.01": "Reject equal accuracy" if p_value < ALPHA else "Do not reject equal accuracy",
        })
    tables["McNemar test results"] = pd.DataFrame(comparisons).set_index("comparison")
    sentiment, p_value = forest_by_sentiment(predictions)
    tables["Q3: random forest correctness by review sentiment"] = sentiment.round(4)
    tables["Q3: two-sided Fisher test"] = pd.Series({
        "accuracy difference (pos minus neg)": f"{sentiment.loc['pos', 'accuracy'] - sentiment.loc['neg', 'accuracy']:.4f}",
        "exact two-sided p-value": f"{p_value:.16g}",
        "decision at alpha=0.01": "Reject equal sentiment accuracy" if p_value < ALPHA else "Do not reject equal sentiment accuracy",
    }).to_frame("value").rename_axis("quantity")
    return tables


def prediction_consistency(previous, current):
    if not previous["id"].is_unique or not current["id"].is_unique or set(previous["id"]) != set(current["id"]):
        raise ValueError("Previous and current predictions must cover the same unique review IDs.")
    previous = previous.set_index("id").loc[current["id"]].reset_index()
    if not previous[["y_true", "sentiment"]].equals(current[["y_true", "sentiment"]].reset_index(drop=True)):
        raise ValueError("Previous and current test labels or sentiments differ.")
    rows = {}
    for model in ("naive_bayes", "logistic_regression"):
        column = f"pred_{model}"
        differences = int((previous[column].to_numpy() != current[column].to_numpy()).sum())
        rows[model] = {"reviews compared": len(current), "changed predictions": differences}
    return pd.DataFrame(rows).T.rename_axis("model")


def save_feature_summary(path=FEATURE_SUMMARY_PATH):
    # Copy the recorded training-based tables; do not calculate importance on test reviews.
    sections = ["# Important features of the final models (training set)",
                "AI is the generated class. NB score differences and LR coefficients are signed; "
                "tree/ensemble importances are unsigned. Review counts describe training associations, "
                "not causal effects. Scores from different model types are not directly comparable."]
    for name, source in MODEL_REPORTS.items():
        _, heading, remaining = source.read_text(encoding="utf-8").partition("## Most important features")
        if not heading:
            raise ValueError(f"No important-feature table in {source}.")
        table = remaining.partition("\n")[2].split("\n## ", 1)[0].strip()
        sections.append(f"## {name}\n\nSource: `results/{source.name}`.\n\n{table}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")


if __name__ == "__main__":
    tables = statistical_tables(pd.read_csv(TEST_PREDICTIONS_PATH))
    print_statistics(tables, title="Final statistical comparisons")
    save_statistics(tables, STATISTICAL_COMPARISONS_PATH, title="Final statistical comparisons", note=METHOD_NOTE)
    save_feature_summary()
