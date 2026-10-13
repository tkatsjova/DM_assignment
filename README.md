# Detection of Fake Hotel Reviews

Classification of genuine (Booking.com) vs. GPT-4-generated hotel reviews with
bag-of-words models. Data: MAiDE-up (Ignat et al., 2025), English reviews only.

## Start

1. Read `results/data_statistics.md` and `results/bow_summary.md` to get to know the data.
2. Read the overview tables in `docs/` to see what was decided and why (see *Documentation*).
3. Follow *How to run* to reproduce all experiments, or *Working with a single model* to inspect one.

## How to run

Requires Python 3.13. From the project root:

On Windows, use `.\.venv\Scripts\python.exe` in place of `python` in the commands
below. On a fresh checkout, create that environment first with
`py -3.13 -m venv .venv`.

```bash
python -m pip install -r requirements.txt
python main.py
```

Recorded results are already in `results/`. Running `main.py` retunes all five models
and rewrites the training reports; no rerun is needed just to read them.

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
It also produces the statistical comparisons and consolidated training-based feature tables.
All five model choices are now frozen. The command reproduces their committed training
searches and refits before evaluation; test outcomes must not guide new model choices.

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
| 7. Evaluate the final models and answer Q1–Q4 (opt-in) | `evaluate_models` | `src/evaluation/evaluate.py`, `src/evaluation/statistical_analysis.py` | test metrics/predictions, statistical comparisons, feature summary |

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
| `docs/classification_tree.md` | Gini classification tree: pruning, CV results, size, important features |
| `docs/random_forest.md` | random forest: CV tuning, leaf-size selection, forest structure, important features |
| `docs/gradient_boosting.md` | gradient boosting: fixed-stage tuning, depth selection, training curve, important features |

The documents explain the choices, results and limitations, with references and
points to include in the report.

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
| `src/evaluation/statistical_analysis.py` | `statistical_tables(predictions)` | the three planned McNemar tests and random-forest sentiment Fisher test |
| | `save_feature_summary()` | copies the important-feature tables from the training reports |
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

### Classification tree — `src/models/classification_tree.py`

- Pipeline: `vectorizer → DecisionTreeClassifier(criterion="gini", random_state=42)`.
- Grid: `ccp_alpha` ∈ {0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02} ×
  `VECTORIZER_GRID` (56 settings). Zero is the unpruned baseline.
- Complexity for the 1-SE rule: `-ccp_alpha`; larger alpha is preferred within one SE.
- `run_classification_tree(train)` returns the refitted model and tables with selected
  settings, all CV results, tree depth/leaves, and five Gini feature importances.
- Selected: `ccp_alpha=0.002`, uni+bigrams, `min_df=5`, CV accuracy 0.8040;
  depth 17 and 61 leaves after refitting on all training reviews.
- Decisions and interpretation: `docs/classification_tree.md`. Final evaluation: `results/test_evaluation.md`.

### Random forest — `src/models/random_forest.py`

- Pipeline: `vectorizer → RandomForestClassifier(criterion="gini", bootstrap=True,
  random_state=42, n_jobs=1)`; OOB scoring is disabled.
- Grid: `n_estimators` ∈ {100, 300}, `max_features` ∈ {sqrt, log2},
  `min_samples_leaf` ∈ {1, 3}, together with `VECTORIZER_GRID` (64 settings).
- Shared five-fold CV: 320 fits and 64,000 trees, plus the selected model's full-training refit.
- Complexity for the 1-SE rule: `-min_samples_leaf`. Larger leaves are preferred;
  other settings tie on this proxy and are resolved by CV accuracy, then grid order.
- `run_random_forest(train)` returns the refitted model and tables with all CV results,
  training accuracy, forest depth/leaf/node summaries, five impurity importances, and runtime.
- Selected: 300 trees, log2 features, leaf size 1, uni+bigrams, `min_df=10`;
  CV accuracy 0.9207 and training accuracy 1.0000. The full RF experiment took about 4m 22s
  on the recorded machine; runtimes vary.
- Decisions and interpretation: `docs/random_forest.md`. Final evaluation: `results/test_evaluation.md`.

### Gradient boosting — `src/models/gradient_boosting.py`

- Pipeline: `vectorizer → GradientBoostingClassifier(loss="log_loss", n_estimators=100,
  subsample=1.0, n_iter_no_change=None, random_state=42)`.
- Final grid: `learning_rate` ∈ {0.05, 0.1, 0.2, 0.3}, `max_depth` ∈ {1, 2, 3, 4},
  together with `VECTORIZER_GRID` (128 settings). The stage count is fixed, not tuned.
- Shared five-fold CV: 640 fits and 64,000 regression trees, plus the 100-tree final refit.
  A scoped joblib context uses four CPU workers and limits library threads to one per worker;
  boosting stages within each fit are sequential. Existing classifiers' execution is unchanged.
- Complexity for the 1-SE rule: `max_depth`; shallower trees are preferred. This proxy
  does not rank learning rate, vocabulary size, or the total additive model's complexity.
- `run_gradient_boosting(train)` returns the refitted model, all CV results, training
  accuracy, fitted tree sizes, descriptive training-stage metrics, five importances, and runtime.
- Final selected: learning rate 0.3, depth 2, 100 stages, unigrams, `min_df=10`;
  CV accuracy 0.9133 and training accuracy 0.9733, with 163 features used.
  The expanded experiment took 133.69 seconds. The final planned search is complete;
  the remaining learning-rate boundary is documented as a limitation.
- Decisions and interpretation: `docs/gradient_boosting.md`. Final evaluation: `results/test_evaluation.md`.

## Using the data in a model

The single-model example below shows how to load the saved training partition.

- Always use the text column `clean_text` and the saved split; do not create a new split.
- The vectorizer must stay inside the Pipeline, so that it is fitted on training folds only.
- `label`: 1 = generated by GPT-4, 0 = written by a human. `sentiment`: `pos` / `neg`.
- The test set is used only once, to evaluate the final model.

## Adding a model

Use `src/models/logistic_regression.py` or `src/models/classification_tree.py` as a complete
example. Each model defines its Pipeline (vectorizer first, classifier named `model`),
parameter grid, complexity function, and `run_<model>(train)` returning the refitted
pipeline plus a dict of report tables. Use `RANDOM_STATE` for every random component.

Add its result path to `src/config.py`, call it and save its tables in `main.py`'s
`train_models`, and register the model in the returned dict. That key becomes its name in
test results, e.g. `classification_tree` produces `pred_classification_tree` only during
explicit final evaluation. Document the model in `docs/<model>.md` and the README;
shared decisions remain in `docs/modelling_setup.md`.

*Random forest validation.* This implementation uses the shared five-fold CV rather than
the assignment's permitted OOB alternative, so the vocabulary is refitted within each fold
and model comparisons use the same validation reviews. See `docs/random_forest.md`.

## Working with a single model

`python main.py` runs preparation and training without test evaluation. To work on one model interactively:

```python
from src.data.cleaning import load_clean_data
from src.data.split import load_or_create_split
from src.data.statistics import print_statistics
from src.models.logistic_regression import run_logistic_regression

data = load_clean_data()
split = load_or_create_split(data)
train = data.merge(split[split["split"] == "train"], on="id", validate="one_to_one")
model, tables = run_logistic_regression(train)
print_statistics(tables, title="Logistic regression")
```

To run another model, change the import and function call:

| Model | Module | Function |
|---|---|---|
| Naive Bayes | `src.models.naive_bayes` | `run_naive_bayes` |
| L1 logistic regression | `src.models.logistic_regression` | `run_logistic_regression` |
| Classification tree | `src.models.classification_tree` | `run_classification_tree` |
| Random forest | `src.models.random_forest` | `run_random_forest` |
| Gradient boosting | `src.models.gradient_boosting` | `run_gradient_boosting` |

Each function runs its training/CV search and returns the refitted model and tables.
It does not evaluate the held-out test set or save reports. Gradient boosting uses
four CPU workers for CV; no GPU is used.

The example uses the prepared file already in `data/processed/`. If it is missing,
import `prepare_data` from `main` and use `data = prepare_data()` instead of
`load_clean_data()`. This runs the same preprocessing without training the other models.
The saved split is checked by `load_or_create_split` in either case.

## Evaluation on the test set

1. Step 7 is disabled by default. `python main.py --evaluate-test` reproduces all five
   frozen models and evaluates them on the same saved 500-review test partition.
   Reproducing results is allowed; using test outcomes to choose settings is not.
2. `results/test_evaluation.md`: accuracy, precision, recall, F1 of every model (positive class:
   generated), confusion matrices, accuracy on positive and negative reviews (Q3).
3. `results/test_predictions.csv`: one row per test review with `id, y_true, sentiment` and
   a `pred_<model>` column for each classifier. Before overwriting an existing file, the
   pipeline aligns reviews by ID and compares the old and new NB/LR predictions; counts
   of changed predictions are saved in the evaluation report.
4. `results/statistical_comparisons.md`: full paired correctness tables and exact two-sided
   McNemar tests for NB vs LR, tree vs forest, and tree vs boosting; Fisher's exact test
   for forest accuracy on positive vs negative sentiment reviews. Class label 1 means
   generated; positive review sentiment is a different variable.
5. `results/feature_summary.md`: five important terms per direction for NB/LR and five
   highest importances for each tree model, copied from the training reports.

The exact McNemar test uses only discordant reviews: under equal marginal accuracy,
the number correct only for the first model follows Binomial(b+c, 0.5). We use
`scipy.stats.binomtest(..., alternative="two-sided")`; no discordances gives p=1.
Fisher's test uses the disjoint sentiment groups' correct/incorrect counts and
`scipy.stats.fisher_exact(..., alternative="two-sided")`, with its default exact
probability-based definition. Both test methods are described by the
[SciPy binomial](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html)
and [Fisher](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html)
documentation.

All four planned tests use nominal per-test alpha=0.01, without multiplicity adjustment.
This does not guarantee family-wide error control. Nonsignificance is not equivalence;
signed model scores and unsigned impurity importances describe predictive associations,
not causal effects. The tests assume independent reviews, which may be imperfect when
several reviews come from the same hotel. Final inference concerns this dataset and split.
The comparisons concern complete tuned pipelines with different vocabularies, so gains
cannot be attributed solely to the ensemble method.

### Commands for reproducing the reported results

Run from the project root with the Python environment described above:

| Command | Results reproduced |
|---|---|
| `python main.py` | Data statistics, preprocessing audit, bag-of-words summary, and all five training/CV reports |
| `python main.py --evaluate-test` | Everything above, plus all final test metrics, predictions, Q1–Q3 statistical comparisons and Q4 feature summary |
| `python -m src.evaluation.statistical_analysis` | Q1–Q3 from saved test predictions and Q4 from saved training reports; no model fitting or prediction |
| `python -B -m unittest discover -s tests -v` | Lightweight automated checks using synthetic data |

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

### `results/classification_tree.md`

Training-set five-fold CV: selected pruning/vectorizer configuration and 1-SE threshold,
all 56 settings, accuracy by alpha, refitted tree depth/leaves, a training-only size
comparison across pruning levels, and the five most important terms with Gini importances.
No held-out tree evaluation is included.

### `results/random_forest.md`

Training-set five-fold CV: selected forest/vectorizer configuration and 1-SE threshold,
all 64 settings, best CV accuracy per forest setting, training accuracy, distributions
of fitted tree depth/leaves/nodes, five impurity importances, and experiment runtime.
No held-out forest evaluation is included. Runtime depends on the machine and varies between runs.

### `results/gradient_boosting.md`

Training-set five-fold CV: selected configuration and 1-SE threshold, all 128 settings,
best accuracy per depth/learning-rate pair, refit training accuracy, regression-tree sizes,
descriptive training performance by stage, five impurity importances, and experiment runtime.
No held-out boosting evaluation is included. Joblib is already a scikit-learn dependency;
no additional package installation is required.

### `results/test_evaluation.md`, `results/test_predictions.csv`

Performance of the final models on the test set (accuracy, precision, recall, F1),
confusion matrices, accuracy on positive and negative reviews, and the predictions per
review: `id, y_true, sentiment, pred_<model>` (input for McNemar's test).

### `results/statistical_comparisons.md`, `results/feature_summary.md`

The statistical report contains the three planned paired correctness tables,
discordant counts, exact p-values and decisions at alpha=0.01, plus the forest's
sentiment-group counts, accuracies and difference. The feature summary preserves
the recorded training-based scores and review counts for all five classifiers.

## Requirements

Python 3.13 with pandas, numpy, scikit-learn, scipy and rich
(exact versions in `requirements.txt`).
