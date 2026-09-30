from src.config import (
    AUDIT_PATH,
    BOW_SUMMARY_PATH,
    LOGISTIC_REGRESSION_PATH,
    NAIVE_BAYES_PATH,
    TEST_EVALUATION_PATH,
    TEST_PREDICTIONS_PATH,
)
from src.data.audit import audit
from src.data.cleaning import clean_data, load_raw_data, save_cleaned_data
from src.data.preprocessing import add_clean_text
from src.data.split import make_split, save_split, train_test_data
from src.data.statistics import compute_statistics, print_statistics, save_statistics
from src.evaluation.evaluate import evaluate, save_predictions
from src.features.bow_summary import summarize_bow
from src.models.logistic_regression import run_logistic_regression
from src.models.naive_bayes import run_naive_bayes


def prepare_data():
    df = add_clean_text(clean_data(load_raw_data()))
    save_cleaned_data(df)
    return df


def describe_data(df):
    stats = compute_statistics(df)
    print_statistics(stats)
    save_statistics(stats)


def split_data(df):
    split = make_split(df)
    save_split(split)
    return split


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
    return {"naive_bayes": naive_bayes, "logistic_regression": logistic_regression}


def evaluate_models(models, test):
    # Evaluate the final models on the test set and save the predictions for McNemar's test
    predictions, tables = evaluate(models, test)
    save_predictions(predictions, TEST_PREDICTIONS_PATH)
    report(tables, "Evaluation on the test set", TEST_EVALUATION_PATH)


def main():
    # 1. Data: English reviews, cleaned text, descriptive statistics, train/test split
    data = prepare_data()
    describe_data(data)
    train, test = train_test_data(data, split_data(data))

    # 2. Evidence for the preprocessing decisions and the bag-of-words representation (training set only)
    report(audit(train), "Preprocessing audit (training set)", AUDIT_PATH)
    report(summarize_bow(train), "Bag-of-words representation (training set)", BOW_SUMMARY_PATH)

    # 3. Models: hyperparameters tuned with cross-validation on the training set only
    models = train_models(train)

    # 4. Evaluation on the test set: run once, after all models are final
    evaluate_models(models, test)


if __name__ == "__main__":
    main()
