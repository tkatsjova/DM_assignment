# Detection of Fake Hotel Reviews

Classification of genuine (Booking.com) vs. GPT-4-generated hotel reviews with
bag-of-words models. Data: MAiDE-up (Ignat et al., 2025), English reviews only.

## Start

1. Run the training pipeline (see *How to run*) — it verifies the saved split and creates training tables.
2. Read `results/data_statistics.md` and `results/bow_summary.md` to get to know the data.
3. Read the overview tables in `docs/` to see what was decided and why (see *Documentation*).
4. To add a model, follow *Adding a model* below.

## How to run

Requires Python 3.13. From the project root:

```bash
pip install -r requirements.txt
python main.py
```

The raw data file `all_data.csv` (from https://github.com/MichiganNLP/multilingual_reviews_deception)
must be placed in `data/raw/`. The default command runs data preparation and training-set
CV, and prints descriptive/training tables. It does **not** predict on the held-out test
set, report test performance, or write the test evaluation/prediction files.
Full-dataset descriptive statistics remain descriptive only; they must not guide modelling.

To deliberately reproduce the completed models' test results, or evaluate additional models
after their training-only development is final:

```bash
python main.py --evaluate-test
```

This opt-in command also writes `results/test_evaluation.md` and `results/test_predictions.csv`.
Do not use it while tuning or choosing models. Naive Bayes and logistic regression are already
finalized; their hyperparameters and preprocessing remain unchanged.

### Split protection

Normal runs reuse `data/split/train_test_split.csv` without rewriting it. Basic checks
verify unique/non-missing IDs, complete coverage, disjoint partitions, valid labels,
the 75/25 ratio, and class × sentiment balance. An invalid split stops execution before
generated files are written. If the split is absent, it is created with seed 42.

Keep the submitted split and original raw dataset, including its row order, to reproduce
the experiments. Review IDs are assigned during preparation; the checks validate split
structure rather than changes to review content.

### Automated checks

The lightweight tests use Python's standard-library `unittest`; no extra dependency is needed:

```bash
python -B -m unittest discover -s tests -v
```

They use synthetic reviews and temporary files, including a mocked final-evaluation dispatch.
They do not run the completed experiment grids, evaluate the real test set, or overwrite
recorded results.

## Pipeline

`main()` in `main.py` runs the following steps; step 7 requires `--evaluate-test`:

| Step | Function in `main.py` | Code | Output |
|---|---|---|---|
| 1. Select English reviews, clean, normalize text | `prepare_data` | `src/data/cleaning.py`, `src/data/preprocessing.py` | prepared reviews in memory |
| 2. Verify and reuse the split, then save prepared data | `split_data` | `src/data/split.py` | verified split; `data/processed/english_reviews.csv` |
| 3. Descriptive statistics (all reviews) | `describe_data` | `src/data/statistics.py` | `results/data_statistics.md` |
| 4. Evidence for preprocessing decisions (training set) | `audit` | `src/data/audit.py` | `results/preprocessing_audit.md` |
| 5. Bag-of-words overview (training set) | `summarize_bow` | `src/features/bow_summary.py` | `results/bow_summary.md` |
| 6. Tune the models with CV, refit on the whole training set | `train_models` | `src/models/` | `results/<model>.md` |
| 7. Evaluate the final models on the test set (opt-in) | `evaluate_models` | `src/evaluation/evaluate.py` | `results/test_evaluation.md`, `results/test_predictions.csv` |

### Training and evaluation of the models (steps 6–7)

```
data/processed/english_reviews.csv + data/split/train_test_split.csv
        │
        ├── training set (1500) ──► for every model (train_models):
        │                              Pipeline = vectorizer → [selection] → classifier
        │                              GridSearchCV: VECTORIZER_GRID + model grid, 5-fold CV
        │                              1-SE rule → selected setting
        │                              refit on all 1500 reviews → final model
        │                              → results/<model>.md (CV tables, important features)
        │
        └── test set (500) ───────► --evaluate-test → evaluate_models (final models):
                                       predictions → accuracy, precision, recall, F1,
                                       confusion matrices, accuracy by sentiment
                                       → results/test_evaluation.md
                                       → results/test_predictions.csv (McNemar)
```

The steps shared by every model are implemented once (see *Shared code*); a model only
defines its pipeline, its grid, its complexity function and how its important features are read.

## Documentation

| Document | Content |
|---|---|
| `docs/preprocessing_decisions.md` | data and preprocessing decisions (D1–D15), limitations |
| `docs/modelling_setup.md` | shared modelling decisions (CV, 1-SE rule, test evaluation), comparison of the models |
| `docs/naive_bayes.md` | naive Bayes: decisions, results, important features |
| `docs/logistic_regression.md` | logistic regression with L1 penalty: decisions, results, important features |

Every document states the motivation with references and ends with a checklist of what must
be mentioned in the report.

## Shared code for all models

| Module | Function / constant | What it does |
|---|---|---|
| `src/features/vectorizer.py` | `build_vectorizer()` | bag-of-words `CountVectorizer` with the fixed settings (D5–D11) |
| | `VECTORIZER_GRID` | n-grams and `min_df`, tuned together with the model |
| `src/models/tuning.py` | `CV` | 5-fold stratified CV, the same folds for every model |
| | `tune(pipeline, param_grid, train, complexity)` | grid search with CV; the setting chosen by the 1-SE rule is refitted on the whole training set |
| | `one_se_rule(complexity)` | simplest setting within one standard error of the best CV accuracy |
| | `cv_results(search, complexity)` | table of all settings: CV accuracy (mean, std), F1, within 1 SE, selected |
| | `selection_summary(search, complexity)` | chosen setting, best accuracy, SE and threshold |
| `src/evaluation/evaluate.py` | `evaluate(models, data)` | predictions and tables: metrics, confusion matrices, accuracy by sentiment |
| | `save_predictions(predictions, path)` | predictions per review for McNemar's test |
| `src/data/statistics.py` | `print_statistics`, `save_statistics` | print tables to the console and save them as Markdown |

`complexity` is a function that receives one parameter setting of the grid, as a dict with the
full parameter names (e.g. `{"model__C": 1.0, "vectorizer__min_df": 10, "vectorizer__ngram_range": (1, 1)}`),
and returns a number: larger means a more complex model. Each model defines its own. Among the
settings within one standard error of the best, the one with the lowest complexity is chosen;
ties are broken by the higher CV accuracy.

Tables are passed around as a dict `{title: DataFrame}`; `print_statistics` and `save_statistics`
show the index of every DataFrame as its first column(s).

## Models

### Multinomial naive Bayes — `src/models/naive_bayes.py`

- Pipeline: `vectorizer → SelectKBest(mutual_information, k) → MultinomialNB(alpha=1)`.
- `mutual_information` (`src/features/selection.py`): MI between term presence and class,
  as in the course.
- Grid: `k` ∈ {50, 100, 250, 500, 1000, 2000, all} × `VECTORIZER_GRID` (56 settings).
- Complexity for the 1-SE rule: number of terms used (`feature_count`).
- `run_naive_bayes(train)` returns the final model (refitted on the whole training set) and the
  tables of `results/naive_bayes.md`; `top_features` gives the most important terms per class.
- Selected: unigrams + bigrams, `min_df = 10`, all 1756 terms (no MI selection), CV accuracy 0.903.
- Decisions and interpretation: `docs/naive_bayes.md`.

### Logistic regression with L1 penalty — `src/models/logistic_regression.py`

- Pipeline: `vectorizer → LogisticRegression(l1_ratio=1, solver="liblinear", C)`;
  `intercept_scaling=100` keeps the intercept practically unpenalized, as in the course.
- Grid: `C` = 1/λ, 17 values from 0.01 to 100 × `VECTORIZER_GRID` (136 settings).
- Complexity for the 1-SE rule: `C` (`penalty_strength`), i.e. `lambda.1se` of `cv.glmnet`.
- `run_logistic_regression(train)` returns the final model and the tables of
  `results/logistic_regression.md`; `top_features` gives the largest positive and negative
  coefficients, `sparsity_path` the number of non-zero coefficients per `C`.
- Selected: `C = 1`, unigrams, `min_df = 10`, 208 of 829 coefficients non-zero, CV accuracy 0.915.
- Decisions and interpretation: `docs/logistic_regression.md`.

## Using the data in a model

```python
from src.data.cleaning import load_clean_data
from src.data.split import load_or_create_split, train_test_data
from src.features.vectorizer import VECTORIZER_GRID, build_vectorizer
from sklearn.pipeline import Pipeline

data = load_clean_data()
train, _ = train_test_data(data, load_or_create_split(data))
X_train, y_train = train["clean_text"], train["label"]

pipeline = Pipeline([("vectorizer", build_vectorizer()), ("model", ...)])
param_grid = {**VECTORIZER_GRID, "model__...": [...]}
```

- Always use the text column `clean_text` and the saved split; do not create a new split.
- The vectorizer must stay inside the Pipeline, so that it is fitted on training folds only.
- `label`: 1 = generated by GPT-4, 0 = written by a human. `sentiment`: `pos` / `neg`.
- The test set is used only once, to evaluate the final model.

## Adding a model

`src/models/logistic_regression.py` is the shortest complete example. A new model, e.g.
`src/models/classification_tree.py`, needs four parts:

```python
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE
from src.features.vectorizer import VECTORIZER_GRID, build_vectorizer
from src.models.tuning import cv_results, selection_summary, tune

# 1. The pipeline: always starts with the shared vectorizer; the classifier step is called "model"
def build_pipeline():
    return Pipeline([
        ("vectorizer", build_vectorizer()),
        ("model", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ])

# 2. The grid: the shared vectorizer grid + the hyperparameters of the model
PARAM_GRID = {**VECTORIZER_GRID, "model__ccp_alpha": [0.0, 0.001, 0.002, 0.005, 0.01]}

# 3. Complexity for the 1-SE rule: larger = more complex (here: less pruning)
def complexity(params):
    return -params["model__ccp_alpha"]

# 4. Tuning, final model and the tables for results/<model>.md
def run_classification_tree(train):
    search = tune(build_pipeline(), PARAM_GRID, train, complexity)  # 5-fold CV + 1-SE rule + refit
    tables = {
        "Selected hyperparameters (1-SE rule)": selection_summary(search, complexity),
        "Cross-validation results (10 best settings)": cv_results(search, complexity).head(10),
        # + a table with the most important features of the final model (Q4)
    }
    return search.best_estimator_, tables
```

Then:

1. Add `CLASSIFICATION_TREE_PATH = ROOT / "results" / "classification_tree.md"` to `src/config.py`.
2. In `train_models` in `main.py`, call `run_classification_tree(train)`, save its tables with
   `report(tables, title, CLASSIFICATION_TREE_PATH, note="Decisions and interpretation: ...")`
   and add the model to the returned dict (e.g. `"classification_tree": model`). The dict key is
   the model name in all test results (`pred_classification_tree`).
3. Use `RANDOM_STATE` from `src/config.py` for every random component.
4. Add the model to *Models* and *Output files* in this README and document its decisions in
   `docs/<model>.md` (only what concerns this model; shared decisions are in
   `docs/modelling_setup.md`), in the same style as `docs/naive_bayes.md`.

*Random forest with out-of-bag evaluation.* The assignment allows OOB evaluation instead of CV
for the random forest. `tune` always uses the shared 5-fold CV; using it keeps all models
comparable. If OOB is used instead, write the tuning loop in the model file (fit the pipeline
for every setting with `oob_score=True`, choose the setting, refit on the whole training set),
return `(final_model, tables)` in the same way, and describe the choice in `docs/random_forest.md`.

## Working with a single model

`python main.py` runs preparation and training without test evaluation. To work on one model interactively:

```python
from src.data.cleaning import load_clean_data
from src.data.split import load_or_create_split, train_test_data
from src.data.statistics import print_statistics
from src.models.logistic_regression import run_logistic_regression

data = load_clean_data()
train, _ = train_test_data(data, load_or_create_split(data))   # never use the test set here
model, tables = run_logistic_regression(train)
print_statistics(tables, title="Logistic regression")
```

This needs the prepared data and verified split (`data/processed/`, `data/split/`), created by one run of
`python main.py`.

## Evaluation on the test set

1. Step 7 is disabled by default. `python main.py --evaluate-test` explicitly evaluates all
   models returned by `train_models`. Naive Bayes and logistic regression are final and have
   already been evaluated; preserve their settings. Tune additional models on training data
   only, then deliberately run final evaluation once their settings are frozen. Deterministic
   reproduction of existing test results is allowed; using them to choose settings is not.
2. `results/test_evaluation.md`: accuracy, precision, recall, F1 of every model (positive class:
   generated), confusion matrices, accuracy on positive and negative reviews (Q3).
3. `results/test_predictions.csv`: one row per test review with `id, y_true, sentiment` and a column
   `pred_<model>` for every model. The 2 × 2 table for McNemar's test of two models:

```python
import pandas as pd

p = pd.read_csv("results/test_predictions.csv")
correct_nb = p["pred_naive_bayes"] == p["y_true"]
correct_lr = p["pred_logistic_regression"] == p["y_true"]
table = pd.crosstab(correct_lr, correct_nb, rownames=["LR correct"], colnames=["NB correct"])
```

The same file gives the accuracy of the random forest on positive and negative reviews (Q3):
group by `sentiment` and compare `pred_random_forest` with `y_true`.

## Output files

### `data/processed/english_reviews.csv`

Cleaned English reviews, one row per review (2000 rows).

| Column | Description |
|---|---|
| `id` | review id, 0..1999 |
| `text` | original review text: upside + downside |
| `clean_text` | normalized text used for bag-of-words |
| `label` | 1 = generated by GPT-4, 0 = written by a human |
| `sentiment` | `pos` or `neg` |
| `upside`, `downside` | the two parts of the review (empty string if missing) |
| `upside_missing`, `downside_missing` | True if the part is empty |
| `score` | review score (2–10), not used as a feature |
| `city`, `hotel` | hotel location and name |
| `orig_index`, `language`, `prompt_language` | columns kept from the raw file |

Load it with `src.data.cleaning.load_clean_data()`, so that empty text parts stay empty strings.

### `data/split/train_test_split.csv`

Columns `id` and `split` (`train` / `test`). 75% / 25% (1500 / 500 reviews),
stratified on class × sentiment (375 / 125 reviews per group), `random_state = 42`.
Existing split assignments are reused and never overwritten.

### `results/data_statistics.md`

All 2000 reviews: class balance, review scores by class and sentiment, review length,
empty parts and non-ASCII characters. Used in the *Data* section of the report.

### `results/preprocessing_audit.md`

Training set only. The evidence behind the preprocessing decisions: placeholder and empty
parts (D3), class differences in names, case, punctuation and digits (D4–D7), stop-word
usage (D8) and the effect of sparse-term removal (D13). Used in the *Preprocessing* section.

### `results/bow_summary.md`

Training set only: a review before and after preprocessing, size and sparsity of the
document-term matrix, a small part of the matrix and the most frequent terms per class.

### `results/naive_bayes.md`

Tuning of naive Bayes (training set, 5-fold CV): selected setting with the 1-SE threshold,
the 10 best settings, the settings within one SE, CV accuracy by k, and the most important
terms per class with the number of training reviews containing them.
Used in the *Results* section and the appendix.

### `results/logistic_regression.md`

Tuning of logistic regression with L1 penalty (training set, 5-fold CV): selected setting with
the 1-SE threshold and the number of non-zero coefficients, the 10 best settings, the settings
within one SE, CV accuracy by C, the number of non-zero coefficients by C, and the most important
terms per class (coefficient, odds factor, number of training reviews containing them).
Used in the *Results* section and the appendix.

### `results/test_evaluation.md`, `results/test_predictions.csv`

Performance of the final models on the test set (accuracy, precision, recall, F1),
confusion matrices, accuracy on positive and negative reviews, and the predictions per
review: `id, y_true, sentiment, pred_<model>` (input for McNemar's test).

## Requirements

Python 3.13 with pandas, numpy, scikit-learn, scipy and rich
(exact versions in `requirements.txt`).
