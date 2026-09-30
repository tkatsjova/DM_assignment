from src.config import AUDIT_PATH, BOW_SUMMARY_PATH
from src.data.audit import audit
from src.data.cleaning import clean_data, load_raw_data, save_cleaned_data
from src.data.preprocessing import add_clean_text
from src.data.split import make_split, save_split, train_test_data
from src.data.statistics import compute_statistics, print_statistics, save_statistics
from src.features.bow_summary import summarize_bow


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


def report(tables, title, path):
    print_statistics(tables, title=title)
    save_statistics(tables, path=path, title=title)


if __name__ == "__main__":
    data = prepare_data()
    describe_data(data)
    train, _ = train_test_data(data, split_data(data))
    report(audit(train), "Preprocessing audit (training set)", AUDIT_PATH)
    report(summarize_bow(train), "Bag-of-words representation (training set)", BOW_SUMMARY_PATH)
