# Random forest

Code: `src/models/random_forest.py`. Tables: `results/random_forest.md`.
Shared preprocessing, vocabulary settings, CV folds and selection rules remain those
of `docs/preprocessing_decisions.md` and `docs/modelling_setup.md`.

Pipeline: `vectorizer → RandomForestClassifier(criterion="gini", bootstrap=True,
random_state=42, n_jobs=1)`. OOB scoring is disabled.

## Methodological decisions

A single classification tree builds one set of conditional splits. A random forest
combines trees grown on bootstrap samples, with random feature subsets considered at
each split. This diversity aims to reduce the variance of individual trees. Sklearn
averages the trees' class probabilities to classify a review, rather than taking a
hard majority vote. This trades whole-model interpretability for a more stable ensemble.
See the [sklearn forest overview](https://scikit-learn.org/stable/modules/ensemble.html#random-forests).

The grid is fixed as follows; the vectorizer settings are the same candidates used
by the other classifiers:

| Parameter | Candidates | Motivation |
|---|---|---|
| `n_estimators` | 100, 300 | Compare a moderate ensemble with more averaging, at a manageable cost |
| `max_features` | sqrt, log2 | Compare two standard feature-sampling budgets in a sparse vocabulary |
| `min_samples_leaf` | 1, 3 | Compare permissive trees with a modest constraint on small terminal regions |
| `ngram_range` | (1, 1), (1, 2) | Shared unigram versus uni+bigram comparison |
| `min_df` | 1, 2, 5, 10 documents | Shared sparse-term removal candidates |

Other model settings retain sklearn defaults: no depth limit, no cost-complexity
pruning (`ccp_alpha=0`), `min_samples_split=2`, and no class weights. The training
classes are balanced. Each final tree draws 1500 bootstrap observations with replacement,
so it does not see 1500 distinct reviews. The fixed seed governs both bootstrap and
feature sampling. `max_features` is a nominal sampling budget: the implementation may
inspect additional features to find a valid split.
See the [parameter documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

Five-fold stratified CV is used instead of the assignment's permitted OOB alternative.
This keeps validation reviews identical across classifiers and learns vocabulary and
sparse-term removal within each fold. Each fit uses 1200 training reviews; the selected
Pipeline is refitted using all 1500 training reviews. Held-out test outcomes were not
used for tuning, and OOB evaluation was disabled.

The 1-SE complexity function is `−min_samples_leaf`, preferring the larger minimum leaf
size among settings within one SE of the best mean CV accuracy. Other parameters tie on
this proxy and are resolved by higher CV accuracy, then grid order. This deliberately
does not penalize the number of trees: more trees change averaging stability and cost,
not the expressiveness of an individual tree. The proxy does not rank actual forest
size, depth, feature-sampling settings or different vocabularies; measured tree-size
distributions are reported separately. The 1-SE rule is a selection heuristic, not a
significance test.

## Workload and reproducibility

There are 64 settings × 5 folds = 320 CV fits, comprising 64,000 trees, plus one refit
of 100 or 300 trees. A training-fold timing pilot took 0.44 s for 100 trees with unigrams
and 0.93 s with uni+bigrams (`min_df=1`, sqrt features, leaf size 1). The full run was
budgeted at approximately 3–6 minutes. No candidates were removed or changed in response
to runtime. Sequential execution (`n_jobs=1`, shared CV defaults) avoids nested parallelism.

Reproduce with `python main.py` for all classifiers, or call `run_random_forest(train)`
using the saved training partition as shown in the README's single-model example.
The standalone experiment loaded only the 1500 saved training reviews, in their original
order, and wrote only the random-forest report. The returned tables include runtime for
CV, refitting and summaries; runtime varies between runs and machines.

## Tuning results and selected forest

| Quantity | Result |
|---|---|
| Best and selected configuration | 300 trees, log2 features, `min_samples_leaf=1`, `min_df=10`, uni+bigrams |
| Best and selected CV accuracy | 0.9207 |
| Selected CV F1 (generated class positive) | 0.9180 |
| Sample standard deviation of best fold accuracies | 0.0080 |
| SE of the best mean / 1-SE threshold | 0.0036 / 0.9171 |
| Settings within one SE | 4 |
| Full-training accuracy | 1.0000 |
| Training minus selected CV accuracy | 0.0793 (7.93 percentage points) |
| Vocabulary / features used across forest | 1756 / 1756 |
| Nominal features per split | 10, from floor(log2(1756)) |
| Total fitted nodes | 262660 |
| CV, refit and summaries runtime | 262.45 seconds (about 4 min 22 s) |

The best leaf-size-3 setting has accuracy 0.9113, below the 0.9171 threshold. Thus the
1-SE simplicity preference cannot select leaf size 3. All four eligible settings have
leaf size 1; the higher-accuracy tie-break on this complexity proxy chooses the unique
best setting. A 100-tree unigram forest also qualifies at 0.9187, but tree count is not
part of our statistical simplicity measure. Choosing it for lower runtime would require
a different selection rule, which was not introduced here.

| Structure across the 300 refitted trees | Minimum | Median | Mean | Maximum |
|---|---:|---:|---:|---:|
| Depth | 48 | 79 | 80.84 | 133 |
| Leaves | 354 | 439 | 438.27 | 518 |
| Nodes | 707 | 877 | 875.53 | 1035 |

These are deep, unpruned individual trees; perfect training accuracy is compatible with
memorization and optimistic apparent performance. Bootstrap/feature randomization and
averaging can reduce variance, but do not eliminate the need for external evaluation.
The training/CV gap warrants discussion without treating it as an independently estimated
generalization gap.

The selected forest has higher CV accuracy than the selected single tree (0.9207 vs.
0.8040). This is a descriptive comparison of independently tuned pipelines: the tree
uses `min_df=5`, whereas the forest uses 10. It is consistent with an ensemble advantage,
but does not isolate averaging from vectorizer differences or establish statistical
significance.

## Five most important features

| Term | Importance | Human training reviews | Generated training reviews |
|---|---:|---:|---:|
| rooms | 0.0173 | 118 | 402 |
| service | 0.0123 | 49 | 252 |
| the | 0.0121 | 542 | 703 |
| and | 0.0109 | 500 | 713 |
| the rooms | 0.0108 | 35 | 251 |

References to rooms and service, including the phrase *the rooms*, occur more often in
generated training reviews. The retained function words *the* and *and* also occur in
more generated reviews. These frequencies describe corpus associations; an impurity score
has no single human/generated direction, and the classifier uses word-count thresholds
and interactions. The top five account for approximately 6.34% of normalized importance,
with credit spread across all 1756 retained features. *Rooms*, *service* and *the* also
appear among the single tree's top features, but ensemble averaging changes their relative
credit and adds *and* and *the rooms* to the top five.

## Interpretation and limitations

The fitted `feature_importances_` gives normalized, averaged impurity-based credit for
features used across the forest. Scores are unsigned and sum to one for a nontrivial
forest. Class-specific review counts give descriptive context; they are not the direction
or size of a causal effect. Unlike the single tree's importance list, credit is spread
across many randomized trees and correlated terms.

- Training accuracy is an apparent-performance measure. A training/CV gap can reveal
  optimism or residual overfitting; the selected CV result is also affected by selection
  on those same folds. Neither substitutes for final held-out evaluation.
- Impurity importance uses training statistics and can favour count features with more
  distinct values. Correlated words/bigrams can share or compete for importance; these
  ranks do not establish their out-of-sample importance. See the
  [sklearn importance discussion](https://scikit-learn.org/stable/modules/ensemble.html#feature-importance-evaluation).
- A fixed seed ensures reproducibility for the specified data order and dependencies;
  it does not measure variation across seeds or independently sampled datasets.
- Review length, residual generation artifacts and the common review-level split across
  represented hotels remain limitations of the dataset/setup.
- CV comparison with the single tree is descriptive. The completed McNemar comparison
  and positive/negative sentiment analysis are in the
  [statistical comparisons](../results/statistical_comparisons.md), alongside the
  [final test evaluation](../results/test_evaluation.md). Training-based features
  for all five models are consolidated in the [feature summary](../results/feature_summary.md).

## References

- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- The linked sklearn documentation describes the implementation used here.
