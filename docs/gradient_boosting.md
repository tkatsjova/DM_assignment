# Gradient boosting

Code: `src/models/gradient_boosting.py`. Tables: `results/gradient_boosting.md`.
Shared preprocessing, vectorizer candidates and CV folds follow
`docs/preprocessing_decisions.md` and `docs/modelling_setup.md`.

## Methodology and justification

The Pipeline is `vectorizer → GradientBoostingClassifier(loss="log_loss",
n_estimators=100, subsample=1.0, n_iter_no_change=None, random_state=42)`.
Binary gradient boosting fits sequential regression trees to negative loss gradients
and adds their predictions with shrinkage controlled by the learning rate.
The model minimizes classification log-loss; its base trees are regression trees,
not Gini classification trees. See the
[sklearn classifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html).
The deprecated `criterion` argument is omitted: in the pinned sklearn 1.9.1 it has
no effect, and the implementation uses squared-error regression-tree splits.

| Parameter | Candidates | Motivation |
|---|---|---|
| `learning_rate` | 0.05, 0.1, 0.2, 0.3 | Compare conservative through faster updates at the fixed stage budget |
| `max_depth` | 1, 2, 3, 4 | Compare stumps with increasingly expressive feature interactions |
| `n_estimators` | 100, fixed | Keep the search manageable and stage budget comparable across candidates |
| `ngram_range` | (1, 1), (1, 2) | Shared unigram versus uni+bigram comparison |
| `min_df` | 1, 2, 5, 10 documents | Shared sparse-term removal candidates |

The initial 72-setting search selected rate 0.2 and depth 3, both upper boundaries.
The team authorized one final extension adding rate 0.3 and depth 4 to check those
boundaries. The complete joint search was rerun, retaining all original candidates
and every shared vectorizer setting. This extension was motivated by training-set
CV results only and specified before running the expanded experiment. Other model parameters use
sklearn defaults. All stages use the available training observations (`subsample=1`);
there is no internal validation split, early stopping or OOB scoring. The 100-stage
budget is an experimental constraint, not a claim that 100 is optimal.
Depth-one trees are additive in the selected count features; deeper trees can model
interactions between them. A bigram can already represent a phrase as one feature.

The shared five-fold stratified CV uses 1200 training and 300 validation reviews per
fold. Vocabulary and document-frequency filtering are learned inside each training
fold. Accuracy selects the model; F1 is also reported with generated reviews as the
positive class. The selected Pipeline is refitted on all 1500 training reviews.
No held-out test outcomes were used or evaluated.

The 1-SE rule defines complexity as `max_depth`: shallower trees are preferred among
candidates within one SE of the best mean CV accuracy. Equal depths are resolved by
higher CV accuracy, then grid order. This is a simple proxy for interaction complexity,
not a complete ordering of the additive model. It ignores learning rate, vocabulary,
actual leaf counts and cumulative effects of the 100 trees. The shared SE uses the
sample standard deviation of fold accuracies divided by sqrt(5). Overlapping training
folds mean it is a selection heuristic, not a confidence interval or significance test.

## Computation and reproduction

The final grid has 128 settings × 5 folds = 640 CV fits, or 64000 regression trees,
followed by a 100-tree full-training refit. The expanded workload was estimated at
2–4 minutes using the initial run's timing and allowing for deeper trees. No
candidates were changed for runtime.

A scoped joblib context gives the shared GridSearchCV four CPU workers and caps
library threads in each worker at one. Trees within a boosting fit are sequential;
there is no GPU use or nested model parallelism. Existing classifiers retain their
execution settings. GridSearchCV inherits joblib settings when `n_jobs=None`, as
described in its
[parallelism documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
and [joblib parallel_config](https://joblib.readthedocs.io/en/latest/generated/joblib.parallel_config.html).
Joblib is already a sklearn dependency.

The final run completed without warnings in **133.69 seconds**, including CV, refit and
report summaries, on the specified Ryzen machine. Timing varies between runs.
The standalone experiment loaded only records in the saved training partition,
in their original order, and updated only the boosting report. Reproduce using
`run_gradient_boosting(train)` with the README's single-model example; use
`save_statistics(tables, GRADIENT_BOOSTING_PATH, title="Gradient boosting (training set, 5-fold CV)")`
to save its tables. `python main.py` runs all five training experiments;
test evaluation remains explicitly opt-in. Keep the saved split, original data order,
seed and dependency versions to reproduce the scientific results.

## Results and selected model

| Quantity | Result |
|---|---|
| Best and selected configuration | Rate 0.3, depth 2, 100 stages, unigrams, `min_df=10` |
| Best and selected CV accuracy | 0.9133 |
| Selected CV F1 | 0.9115 |
| Sample standard deviation of best fold accuracies | 0.0118 |
| SE / 1-SE threshold | 0.0053 / 0.9081 |
| Settings within one SE | 11 |
| Full-training accuracy | 0.9733 |
| Training minus selected CV accuracy | 0.0600 (6.00 percentage points) |
| Vocabulary / features used across stages | 829 / 163 |
| Fitted stages / maximum actual tree depth | 100 / 2 |

The 11 eligible settings include depths 2, 3 and 4. The simplest eligible depth is 2;
the accuracy tie-break chooses its unique best setting, which is also the unique
overall accuracy optimum. The best stump accuracy is 0.9007, below the 0.9081 threshold.

| Best CV accuracy over vectorizer settings | Rate 0.05 | Rate 0.1 | Rate 0.2 | Rate 0.3 |
|---|---:|---:|---:|---:|
| Depth 1 | 0.8367 | 0.8673 | 0.9007 | 0.8980 |
| Depth 2 | 0.8673 | 0.8960 | 0.9027 | 0.9133 |
| Depth 3 | 0.8833 | 0.9033 | 0.9100 | 0.9113 |
| Depth 4 | 0.8927 | 0.9053 | 0.9093 | 0.9127 |

| Structure across 100 refitted regression trees | Minimum | Median | Mean | Maximum |
|---|---:|---:|---:|---:|
| Depth | 2 | 2 | 2 | 2 |
| Leaves | 3 | 4 | 3.98 | 4 |
| Nodes | 5 | 7 | 6.96 | 7 |

The report also contains a descriptive training curve: accuracy rises from 0.8593
at stage 10 to 0.9733 at stage 100, while sklearn's training loss falls from 0.8080
to 0.2726. In the pinned implementation, this binary training-loss scale is binomial
deviance (twice mean log-loss). This curve was calculated after selection and was
not used to select stages. Training improvement alone cannot establish whether more
stages would improve validation accuracy.

## Original versus expanded search

| Quantity | Original search | Final expanded search |
|---|---|---|
| Settings / CV fits | 72 / 360 | 128 / 640 |
| Selected rate / depth / min_df | 0.2 / 3 / 2 | 0.3 / 2 / 10 |
| Selected n-grams / stages | Unigrams / 100 | Unigrams / 100 |
| Best and selected CV accuracy | 0.9100 | 0.9133 |
| Training accuracy | 0.9860 | 0.9733 |
| Training–CV gap | 7.60 points | 6.00 points |
| Vocabulary / features with positive importance | 2608 / 292 | 829 / 163 |
| Mean leaves per tree | 7.46 | 3.98 |
| Runtime, including refit and summaries | 61.39 seconds | 133.69 seconds |

All 72 overlapping candidates reproduced their original reported CV accuracy,
sample standard deviation and F1 (rounded to four decimals). The net accuracy gain
is **0.33 percentage points**, or five additional correct out-of-fold predictions
across 1500 reviews. This is a modest descriptive improvement, smaller than the
best candidate's fold-based SE; that comparison is not a test of the paired difference.
No statistical significance or independent performance improvement is established.

Allowing depth 4 did not produce a higher CV optimum: its best accuracy is 0.9127,
below the selected depth-2 setting. The extension instead enabled a faster learning
rate with shallower trees and stronger sparse-term filtering. Consequently, the
selected model is simpler by our depth proxy and uses fewer features and leaves,
although a higher learning rate cannot be ranked by that proxy. The joint search
does not isolate the effect of learning rate from vectorizer changes. Additional
tree depth did not meaningfully improve the selected validation result.

## Five most important features

| Term | Importance | Human training reviews | Generated training reviews |
|---|---:|---:|---:|
| rooms | 0.1369 | 118 | 402 |
| the | 0.1091 | 542 | 703 |
| service | 0.0969 | 49 | 252 |
| we | 0.0584 | 146 | 48 |
| and | 0.0554 | 500 | 713 |

The five terms account for approximately 45.67% of normalized impurity importance.
The term set is unchanged from the original refit, with *we* now ranked above *and*.
Rooms, service and the retained function words *the* and *and* appear in more generated
training reviews; *we* appears in more human reviews. These counts describe corpus
associations, not the direction of a model coefficient. Boosting uses count thresholds
and interactions, so a term has no single signed human/generated effect.

`feature_importances_` aggregates impurity reductions across the fitted regression
trees and normalizes their sum. Scores are unsigned and based on training data.
Correlated terms can compete for credit, and features with more distinct count values
can be favoured. They are not causal effects or independent measures of generalization.

## Comparison and limitations

| Model | How trees are combined | Main distinction |
|---|---|---|
| Single classification tree | One fitted tree with class-based splits | Interpretable conditional partitions; pruning controls size |
| Random forest | Bootstrap/random-feature classification trees, averaged probabilities | Randomization and averaging aim to reduce variance |
| Gradient boosting | Sequential regression trees correcting loss gradients | Additive loss reduction with shrinkage; depth limits each stage's interactions |

These mechanisms differ even though all three use trees. Boosting need not outperform
the forest: selected CV accuracy here is 0.9133 versus the forest's 0.9207. Their
vectorizers also differ. This is a descriptive comparison of tuned pipelines, not
a controlled comparison of ensemble mechanisms or evidence of statistical significance.
The assignment's required final model comparisons remain pending.

- The 6.00-point training/CV gap indicates optimistic apparent performance and warrants
  discussion of overfitting. Selected CV accuracy also benefits from selection on these
  folds; neither score replaces final held-out evaluation.
- Learning rate and stage count interact: a lower rate may need more than 100 stages.
  This experiment compares rates only at that fixed budget.
- The selected rate 0.3 is still an upper boundary, while depth 2 is interior to the
  expanded depth grid. The result is optimal only within the evaluated grid and fixed
  stage budget. High training accuracy does not support a simple underfitting claim.
- The extension was chosen after inspecting the initial CV results. Reusing the same
  folds for this adaptive development increases selection optimism; it does not create
  independent validation evidence. The untouched test set remains reserved for the
  final evaluation after all settings are frozen.
- A fixed seed gives reproducibility, not uncertainty across seeds or samples.
  Review length, generation artifacts and the review-level split across represented
  hotels remain limitations of the shared dataset and preprocessing.

This completes the team's final planned hyperparameter search. The configuration
above is the final training-selected model for review before final evaluation.
The remaining rate boundary is recorded as a limitation; no further grid extension
is proposed or performed.

## Points for the research paper

- Justify the fixed stage budget, initial grid and CV-motivated final extension,
  fold-local vectorizer and depth-based 1-SE proxy.
- Report the selected configuration, CV and apparent training scores, tree sizes,
  unsigned importance interpretation and boundary limitations.
- Explain the distinction between classification-tree splits, forest averaging and
  sequential regression-tree boosting; reserve statistical comparisons for the
  planned final evaluation.

Reference: Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting
Machine. *The Annals of Statistics*, 29(5), 1189–1232.
