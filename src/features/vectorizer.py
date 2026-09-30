from sklearn.feature_extraction.text import CountVectorizer

# Bag-of-words representation shared by all models

# Default token_pattern (\w\w+) drops one-character tokens such as "i" and "a",
# which differ between the classes; keep every word token instead
TOKEN_PATTERN = r"(?u)\b\w+\b"

# Fixed settings: text is already lowercased and cleaned in src/data/preprocessing.py
VECTORIZER_PARAMS = {
    "lowercase": False,
    "token_pattern": TOKEN_PATTERN,
    "stop_words": None,  # stop words are kept, they carry writing style
}

# Settings tuned with cross-validation together with the model hyperparameters.
# min_df is an absolute document count within the training folds
VECTORIZER_GRID = {
    "vectorizer__ngram_range": [(1, 1), (1, 2)],
    "vectorizer__min_df": [1, 2, 5, 10],
}

def build_vectorizer(**overrides):
    return CountVectorizer(**{**VECTORIZER_PARAMS, **overrides})
