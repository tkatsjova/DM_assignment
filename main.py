import argparse

from src.config import (
    AUDIT_PATH,
    BOW_SUMMARY_PATH,
    CLASSIFICATION_TREE_PATH,
    GRADIENT_BOOSTING_PATH,
    LOGISTIC_REGRESSION_PATH,
    NAIVE_BAYES_PATH,
    RANDOM_FOREST_PATH,
    TEST_EVALUATION_PATH,
    TEST_PREDICTIONS_PATH,
)
from src.data.audit import audit
from src.data.cleaning import clean_data, load_raw_data, save_cleaned_data
from src.data.preprocessing import add_clean_text
from src.data.split import load_or_create_split, train_test_data
from src.data.statistics import compute_statistics, print_statistics, save_statistics
from src.evaluation.evaluate import evaluate, save_predictions
from src.features.bow_summary import summarize_bow
from src.models.classification_tree import run_classification_tree
from src.models.gradient_boosting import run_gradient_boosting
from src.models.logistic_regression import run_logistic_regression
from src.models.naive_bayes import run_naive_bayes
from src.models.random_forest import run_random_forest


def prepare_data():
    df = add_clean_text(clean_data(load_raw_data()))
    return df


def describe_data(df):
    stats = compute_statistics(df)
    print_statistics(stats)
    save_statistics(stats)


def split_data(df):
    return load_or_create_split(df)


def report(tables, title, path, note=None):
    print_statistics(tables, title=title)
    save_statistics(tables, path=path, title=title, note=note)


def train_models(train):
    # Tune every model with CV on the training set; each returned model is refitted on the whole training set
    naive_bayes, naive_bayes_tables = run_naive_bayes(train)
    report(
        naive_bayes_tables,
        "Multinomial naive Bayes (training set, 5-fold CV)",
        NAIVE_BAYES_PATH,
        note="Decisions and interpretation: `docs/naive_bayes.md`.",
    )

    logistic_regression, logistic_regression_tables = run_logistic_regression(train)
    report(
        logistic_regression_tables,
        "Logistic regression with L1 penalty (training set, 5-fold CV)",
        LOGISTIC_REGRESSION_PATH,
        note="Decisions and interpretation: `docs/logistic_regression.md`.",
    )
    classification_tree, classification_tree_tables = run_classification_tree(train)
    report(
        classification_tree_tables,
        "Classification tree (training set, 5-fold CV)",
        CLASSIFICATION_TREE_PATH,
        note="Decisions and interpretation: `docs/classification_tree.md`.",
    )
    random_forest, random_forest_tables = run_random_forest(train)
    report(
        random_forest_tables,
        "Random forest (training set, 5-fold CV)",
        RANDOM_FOREST_PATH,
        note="Decisions and interpretation: `docs/random_forest.md`.",
    )
    gradient_boosting, gradient_boosting_tables = run_gradient_boosting(train)
    report(
        gradient_boosting_tables,
        "Gradient boosting (training set, 5-fold CV)",
        GRADIENT_BOOSTING_PATH,
        note="Decisions and interpretation: `docs/gradient_boosting.md`.",
    )
    return {
        "naive_bayes": naive_bayes,
        "logistic_regression": logistic_regression,
        "classification_tree": classification_tree,
        "random_forest": random_forest,
        "gradient_boosting": gradient_boosting,
    }


def evaluate_models(models, test):
    # Evaluate the final models on the test set and save the predictions for McNemar's test
    predictions, tables = evaluate(models, test)
    save_predictions(predictions, TEST_PREDICTIONS_PATH)
    report(tables, "Evaluation on the test set", TEST_EVALUATION_PATH)


def main(evaluate_test=False):
    # 1. Data: English reviews, cleaned text, descriptive statistics, train/test split
    data = prepare_data()
    split = split_data(data)  # Check split integrity before writing generated artifacts.
    save_cleaned_data(data)
    describe_data(data)
    train = data.merge(split[split["split"] == "train"], on="id", validate="one_to_one")

    # 2. Evidence for the preprocessing decisions and the bag-of-words representation (training set only)
    report(audit(train), "Preprocessing audit (training set)", AUDIT_PATH)
    report(summarize_bow(train), "Bag-of-words representation (training set)", BOW_SUMMARY_PATH)

    # 3. Models: hyperparameters tuned with cross-validation on the training set only
    models = train_models(train)

    # 4. Held-out predictions and outcomes require an explicit final-evaluation request.
    if evaluate_test:
        _, test = train_test_data(data, split)
        evaluate_models(models, test)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train review classifiers using training-set CV.")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Also evaluate finalized models on the held-out test set and write test results.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(evaluate_test=parse_args().evaluate_test)
