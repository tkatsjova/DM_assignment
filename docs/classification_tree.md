# Single classification tree

Code: `src/models/classification_tree.py`. Tables: `results/classification_tree.md`.
Shared preprocessing, CV folds and selection rules remain those of
`docs/preprocessing_decisions.md` and `docs/modelling_setup.md`.

Pipeline: `vectorizer → DecisionTreeClassifier(criterion="gini", random_state=42)`.
Only the 1500 saved training reviews were used for tuning and refitting. The held-out
reviews were excluded from this stage and evaluated after the model settings were frozen.

## Methodological decisions

The tree greedily partitions raw word-count features using Gini impurity,
`1 − Σ p_c²`. No scaling or separate supervised feature selection is needed: splits select
terms and count thresholds directly. Other tree settings retain sklearn defaults
(`splitter="best"`, unrestricted depth, `min_samples_split=2`, `min_samples_leaf=1`,
all features considered). The fixed seed resolves randomness when split gains tie.

Cost-complexity pruning controls overfitting after growth. Its criterion is
`R(T) + α × number_of_leaves`, where sklearn's `R(T)` is the total sample-weighted
Gini impurity of the leaves, rather than the misclassification rate.
Larger `ccp_alpha` favours smaller subtrees; zero supplies the unpruned baseline.
See the [sklearn pruning documentation](https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning).

| Parameter | Candidates |
|---|---|
| `ccp_alpha` | 0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02 |
| `ngram_range` | (1, 1), (1, 2) |
| `min_df` | 1, 2, 5, 10 documents |

This is a modest fixed grid: 56 settings × 5 folds. Each CV training fold has 1200
reviews, so small improvements involving a few documents have a scale of order
`1/1200 ≈ 0.00083` in sample-weighted impurity. The candidates start below this scale,
then increase approximately geometrically to progressively stronger penalties. This is
a scale-based heuristic, not a conversion from alpha to a fixed number of reviews or leaves.
The grid was specified before the experiment and not derived from a supervised pruning
path fitted on all training labels. Each fold learns its own vocabulary and pruned tree.

Selection uses the existing mean CV accuracy and 1-SE rule. Complexity is `−ccp_alpha`:
among qualifying settings choose the largest alpha; then higher accuracy, then grid order.
Alpha is a pruning-strength proxy, not an exact ordering of leaf counts across different
vocabularies. The selected Pipeline is refitted on all 1500 training reviews.

## Tuning results

| Quantity | Result |
|---|---|
| Best and selected configuration | `ccp_alpha=0.002`, `min_df=5`, unigrams + bigrams |
| Best and selected CV accuracy | 0.8040 |
| Fold accuracy standard deviation (sample) | 0.0213 |
| SE of best mean accuracy | 0.0095 |
| 1-SE threshold | 0.7945 |
| Settings within one SE | 15 |
| Selected CV F1 (generated class positive) | 0.8045 |
| Refit depth / leaves | 17 / 61 |
| Refit vocabulary / features used in splits | 3593 / 56 |

The best accuracy for alpha 0 is 0.7960, rising to 0.8040 at 0.002. At 0.005 it
falls to 0.7893, below the 1-SE threshold; larger penalties also fall below it.
Thus the strongest eligible pruning level is 0.002, and the selected vectorizer
configuration also has the highest CV accuracy. The grid brackets this setting.

The descriptive size table uses the selected vocabulary and all training reviews:
the unpruned tree has depth 31, 160 leaves and training accuracy 1.0000;
the selected tree has depth 17, 61 leaves and training accuracy 0.9320;
alpha 0.02 produces depth 3 and 4 leaves. These additional training fits illustrate
pruning strength; they do not determine selection. The training/CV gap is consistent
with residual overfitting and must not be presented as a test estimate.

## Five most important features

Importance is the normalized sum of sample-weighted Gini decreases credited to a
feature in the refitted tree. Scores are unsigned; unlike linear coefficients, they
do not identify a universal direction towards either class. The accompanying counts
describe training reviews containing each term, not token frequencies.

| Term | Importance | Human reviews | Generated reviews |
|---|---:|---:|---:|
| rooms | 0.2064 | 118 | 402 |
| service | 0.1085 | 49 | 252 |
| room was | 0.0753 | 101 | 195 |
| the | 0.0376 | 542 | 703 |
| we | 0.0367 | 146 | 48 |

The first split is whether *rooms* occurs. References to rooms and service, and the
phrase *room was*, are more common in generated training reviews. *We* is more common
in human reviews, consistent with personal/group narration. *The* demonstrates that
a retained function word can contribute to discrimination. These are corpus-specific
associations; deeper decisions depend on combinations of terms and count thresholds.

## Limitations and report points

- A single tree is unstable and greedy; the fixed seed ensures reproducibility, not
  stability under different training samples. Even after pruning, 61 leaves limit
  whole-tree interpretability.
- Impurity importance is calculated on training data, can favour features with more
  candidate thresholds, and can assign credit unevenly among correlated words/bigrams.
  It is neither a causal effect nor a validated out-of-sample importance measure.
- Raw counts can reflect review length; residual prompt artifacts and the existing
  review-level split across shared hotels remain limitations of the common setup.
- The 1-SE rule is a simplicity heuristic. CV scores used for selection are not an
  independent final performance estimate; no significance claim follows from them.
- The grid is coarse and does not exhaust every possible pruning subtree. Its purpose
  is a reproducible, justified comparison rather than maximum attainable accuracy.
- Report Gini, the grid rationale, fold-local fitting, the 1-SE threshold, final size,
  the five importances and their limitations. The completed
  [test evaluation](../results/test_evaluation.md) and
  [statistical comparisons](../results/statistical_comparisons.md) compare the tree
  with the ensembles. All five models' training-based features are in the
  [feature summary](../results/feature_summary.md).

## References

- Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984).
  *Classification and Regression Trees*. Wadsworth.
- [sklearn DecisionTreeClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html).
