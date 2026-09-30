import re

# Per-document text normalization

# Parts that only say "there is nothing to report"; humans write them much more often than GPT-4, so they are treated the same as an empty part
EMPTY_MARKERS = {"nothing", "none", "nil", "n/a", "na", "-", "no", "nothing at all"}

# Curly apostrophes are mostly a keyboard/encoding difference, map them to the straight one
APOSTROPHES = str.maketrans({"’": "'", "‘": "'"})

def is_empty_marker(part):
    return part.strip().lower().rstrip(".!") in EMPTY_MARKERS

def remove_names(text, names):
    # Remove hotel and city names; they appear in the GPT-4 prompts, not in the writing style.
    for name in names:
        text = re.sub(rf"(?<!\w){re.escape(name)}(?!\w)", " ", text, flags=re.IGNORECASE)
    return text


def normalize_text(text):
    text = text.translate(APOSTROPHES).lower()
    text = text.replace("'", "")  # didn't -> didnt
    text = re.sub(r"[^\w\s]|_", " ", text)  # remaining punctuation and symbols
    return re.sub(r"\s+", " ", text).strip()


def preprocess_review(upside, downside, hotel, city, drop_empty_markers=True, drop_names=True):
    parts = [upside, downside]
    if drop_empty_markers:
        parts = ["" if is_empty_marker(p) else p for p in parts]
    text = " ".join(parts)
    if drop_names:
        text = remove_names(text, [hotel, city])
    return normalize_text(text)


def add_clean_text(df, **options):
    # Add column `clean_text` with the normalized review text used for bag-of-words
    df = df.copy()
    df["clean_text"] = [
        preprocess_review(r.upside, r.downside, r.hotel, r.city, **options)
        for r in df.itertuples()
    ]
    return df