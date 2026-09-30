# Modelling setup shared by all models

Working document for the *Experimental setup* section of the report and for the comparison of
the models. Decisions that concern a single model are in its own document:
`docs/naive_bayes.md`, `docs/logistic_regression.md` (and the documents of the tree models).
Preprocessing decisions (D1–D15) are in `docs/preprocessing_decisions.md`.

Code: `src/models/tuning.py` (tuning), `src/evaluation/evaluate.py` (test evaluation),
`main.py` (`train_models`, `evaluate_models`).

---

## Overview

| # | Decision | Choice | Where |
|---|---|---|---|
| M1 | Validation for hyperparameter tuning | 5-fold stratified CV on the training set, same folds for all models | `tuning.py` |
| M2 | Tuning criterion | CV accuracy (F1 recorded as well) | `tuning.py` |
| M3 | Selection rule | one-standard-error rule | `tuning.py` |
| M4 | Final model | refitted on all 1500 training reviews with the selected setting | `tuning.py` |
| M5 | Test set | used once, for the five final models | `main.py`, `evaluate.py` |
| M6 | Test metrics | accuracy, precision, recall, F1 (positive class: generated) | `evaluate.py` |

The training and evaluation pipeline and the shared code are described in `README.md`.

---

## M1 — 5-fold stratified cross-validation on the training set

Hyperparameters are tuned on the 1500 training reviews only. The training set is split
into 5 folds, stratified on the class; each setting is trained on 4 folds and evaluated
on the fifth, five times. The folds are fixed (`random_state = 42`) and identical for
every model, so that CV results of different models are comparable.
The whole pipeline, including the vectorizer and feature selection, is refitted in every
fold, so the validation fold never influences the vocabulary or the selected terms (D14).
Each model tunes the vectorizer settings (n-grams, `min_df`) together with its own
hyperparameters, so the final models may use different vocabularies.

*Why 5 folds.* A compromise between the size of the training part in each fold
(1200 reviews) and computation time for the larger grids of the tree ensembles.
The random forest may use out-of-bag evaluation instead, as allowed by the assignment.

## M2 — Tuning criterion: accuracy

Settings are compared by mean CV accuracy. The classes are exactly balanced, so accuracy
is not dominated by one class, and it is the measure compared with McNemar's test on the
test set. The CV F1 score (positive class: generated) is recorded for information.

## M3 — One-standard-error rule

CV accuracy is an estimate with random error: the best settings often differ by less
than this error. We therefore use the one-standard-error rule (Breiman et al., 1984;
Hastie et al., 2009, §7.10):

1. find the setting with the highest mean CV accuracy; if several settings share it, take the
   simplest of them (as `cv.glmnet` takes the largest λ among equal minima);
2. compute its standard error: SE = s / √K, with s the sample standard deviation
   (divisor K − 1) of its K = 5 fold accuracies;
3. all settings with accuracy ≥ best − SE are considered equally good;
4. among these, choose the **simplest** setting; ties are broken by the higher accuracy,
   then by the order of the grid.

This is the rule the course uses for logistic regression (`cv.glmnet`, `lambda.1se`, with the
same SE formula), and the dashed line in `rpart`'s `plotcp`. The course prunes the tree of
its example at the minimum CV error instead; we apply the 1-SE rule to all models for
consistency: among statistically indistinguishable settings the simpler model is preferred,
because it is less likely to have won by chance and is easier to interpret. Each model defines
its own measure of complexity (see the model documents).

*Implementation notes.* `GridSearchCV` reports the fold standard deviation with divisor K
(ddof = 0); it is recomputed with divisor K − 1, as in `cv.glmnet`. Mean accuracies are rounded
to 10 decimals before they are compared: settings with the same number of correct predictions
otherwise differ in the last floating-point digit, which would make the choice of the reference
setting depend on rounding noise.

## M4 — Final model refitted on the whole training set

The selected setting is refitted on all 1500 training reviews (`GridSearchCV` with the
1-SE rule as refit strategy). This is the model evaluated on the test set and used for
the feature importances.

## M5 — The test set is used once

The 500 test reviews are used only to evaluate the five final models, after all tuning
is finished (step 4 in `main()`). The test set is transformed with the vocabulary and the
selected terms learned on the training set; nothing is refitted on it.

## M6 — Test metrics

Accuracy, precision, recall and F1, with the generated class (label 1) as the positive
class, plus the confusion matrix and the accuracy on positive and negative reviews
separately. The predictions of all models are saved per review
(`results/test_predictions.csv`: `id, y_true, sentiment, pred_<model>`) as input for
McNemar's test and for the positive vs. negative comparison.

---

## Comparing the models

### Q1 — Generative vs. discriminative linear model

Naive Bayes is generative: it models P(x | y) and P(y) and maximizes the joint likelihood,
assuming the terms are independent given the class. Logistic regression is discriminative: it
models P(y | x) directly and maximizes the conditional likelihood, without assumptions on the
distribution of the terms (course lecture on logistic regression, slides 2–5). The functional
form of P(y | x) is log-linear in both cases, but the estimates differ; naive Bayes corresponds
to the log-linear model in which the features are conditionally independent given the class
(course notes on graphical models, section 11). Logistic regression can therefore account for
correlated terms (e.g. the words of a common phrase), which naive Bayes counts twice (course
lecture on naive Bayes, slide 26).

The accuracy of the two final models is compared with McNemar's test on the test set
(α = 0.01), from the 2 × 2 table of correct / incorrect predictions (course lecture on
logistic regression, slide 43); precision, recall and F1 are compared qualitatively.

| | CV accuracy | Test accuracy | Precision | Recall | F1 | Setting |
|---|---:|---:|---:|---:|---:|---|
| Naive Bayes | 0.903 | 0.914 | 0.888 | 0.948 | 0.917 | uni + bigrams, `min_df` 10, 1756 terms |
| Logistic regression (L1) | 0.915 | 0.910 | 0.915 | 0.904 | 0.910 | unigrams, `min_df` 10, 208 of 829 coefficients non-zero |

Correct (1) and incorrect (0) test predictions of the two models (500 reviews, from
`results/test_predictions.csv`):

| | NB correct | NB incorrect |
|---|---:|---:|
| **LR correct** | 424 | 31 |
| **LR incorrect** | 33 | 12 |

McNemar's test only uses the 64 reviews on which the models disagree: 33 are classified
correctly by NB only, 31 by LR only. The exact (binomial) version of the test, as in the course
(slide 43), gives p ≈ 0.90, far above α = 0.01 (to be confirmed by the test script of part C).

*Interpretation.*
- **No significant difference in accuracy** between the generative and the discriminative
  linear model on this data; both agree with their CV estimates.
- **The models make different kinds of errors.** Naive Bayes over-predicts the generated class
  (30 false positives vs. 13 false negatives: high recall, lower precision), logistic regression
  is balanced (21 vs. 24). Which model is preferable depends on the cost of the errors: wrongly
  flagging a genuine review (favours LR, higher precision) or missing a fake one (favours NB,
  higher recall).
- Both models are more accurate on negative than on positive reviews (NB 0.928 vs. 0.900,
  LR 0.924 vs. 0.896).

### Q4 — Why the important features of the two linear models differ

The naive Bayes score difference favours terms that occur in only one class (*told*: 22 human
vs. 0 generated reviews); the lasso favours frequent terms that separate the classes in
combination with the other terms (*variety*: 10 vs. 109). The two lists are different views
on the same data: exclusive markers vs. frequent discriminative words. Both point in the
same direction: generated reviews use generic evaluative templates, human reviews are
concrete and narrative.

---

## Checklist for the report

**Experimental setup**
- [ ] 5-fold stratified CV on the 1500 training reviews, the same folds for all models (M1).
- [ ] Tuning criterion: accuracy; F1 recorded (M2).
- [ ] One-standard-error rule, SE = sample std / √K, the simplest of equally best settings as
      reference; motivation and references; the course's `lambda.1se`; applied to all models (M3).
- [ ] Final models refitted on all training data; test set used once (M4, M5).
- [ ] Positive class for precision / recall / F1: generated reviews (M6).
- [ ] Each model tunes the vectorizer settings itself, so the final models use different
      vocabularies (M1).
- [ ] Full CV tables of every model in the appendix (`results/<model>.md`).

**Q1**
- [ ] Theoretical difference between the generative and the discriminative model.
- [ ] McNemar's test on the test predictions, α = 0.01, with the 2 × 2 table (part C):
      33 vs. 31 discordant reviews, no significant difference.
- [ ] Qualitative comparison of precision, recall, F1: NB high recall / lower precision
      (over-predicts generated), LR balanced; which is preferable depends on the error costs.
- [ ] Test accuracy of both models agrees with CV.

**Q4**
- [ ] Why the important features of NB and LR differ (exclusive vs. frequent terms).

**Limitations**
- [ ] Single train/test split; small differences between settings are within CV noise.

---

## References

- Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). *Classification and
  Regression Trees*. Wadsworth. (one-standard-error rule)
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*
  (2nd ed.), Section 7.10: Cross-Validation. Springer.
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization Paths for Generalized
  Linear Models via Coordinate Descent. *Journal of Statistical Software*, 33(1).
  (`cv.glmnet`, `lambda.1se`)
- Dietterich, T. G. (1998). Approximate Statistical Tests for Comparing Supervised
  Classification Learning Algorithms. *Neural Computation*, 10(7), 1895–1923. (McNemar's test)
- Feelders, A. (2026). Data Mining lectures: Text Classification, Naive Bayes; Logistic
  Regression, Text Classification. Utrecht University.
- Feelders, A. Graphical Models for Discrete Data, Part 1: Undirected Graphs (course notes),
  section 11: Log-linear models and logistic regression.
