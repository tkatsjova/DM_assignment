# Logistic regression with L1 penalty

Decisions, results and interpretation for lasso logistic regression. The shared setup (CV,
1-SE rule, test evaluation) is in `docs/modelling_setup.md`.

Code: `src/models/logistic_regression.py`.
Tables: `results/logistic_regression.md` (reproduced by `python main.py`).

Pipeline: `vectorizer → LogisticRegression(L1, C)`.

---

## Overview

| # | Decision | Choice |
|---|---|---|
| LR1 | Model | logistic regression with L1 (lasso) penalty, intercept practically not penalized |
| LR2 | Features | raw term counts, not standardized |
| LR3 | Grid | C = 1/λ (17 values) × n-grams × `min_df` (136 settings) |
| LR4 | Complexity for the 1-SE rule | C (= `lambda.1se` of `cv.glmnet`) |
| LR5 | Important features | largest positive / negative coefficients |

**Selected:** C = 1 (λ = 1), unigrams, `min_df` = 10, 208 of 829 coefficients non-zero,
CV accuracy 0.915.

---

## LR1 — Lasso logistic regression

Logistic regression models P(generated | x) directly, with the log-odds linear in the term
counts. With many features, maximum likelihood overfits; the lasso penalty shrinks the
coefficients towards zero and sets many of them exactly to zero, so it also performs feature
selection (Tibshirani, 1996; course lecture on logistic regression, slide 32). As in the
course, we minimize

    E(β) = −ℓ(β) + λ Σ_{j≥1} |β_j|,   with C = 1/λ in scikit-learn.

*Implementation.* scikit-learn ≥ 1.8 specifies the L1 penalty with `l1_ratio = 1`
(`penalty = "l1"` is deprecated). The `liblinear` solver is used: it handles L1 exactly and
is fast on our data (hundredths of a second per fit, vs. up to a minute for `saga`).
`liblinear` penalizes the intercept like a coefficient, whereas the lasso objective of the
course does not; with `intercept_scaling = 100` this penalty becomes negligible.

## LR2 — Raw term counts, not standardized

`cv.glmnet`, used in the course, standardizes the features before applying the penalty by
default. We use the raw counts, as for naive Bayes: then exp(β) is the factor by which the
odds of a generated review change for every additional occurrence of a term (course,
slide 25), which makes the coefficients directly interpretable, and the same representation is
used for all models.

## LR3 — Grid

| Parameter | Values |
|---|---|
| `C` = 1/λ | 17 values, logarithmically spaced from 0.01 to 100 |
| `ngram_range` (D12) | unigrams; unigrams + bigrams |
| `min_df` (D13) | 1, 2, 5, 10 documents |

136 settings × 5 folds.

*Why `GridSearchCV` and not `LogisticRegressionCV`.* The assignment suggests
`LogisticRegressionCV`, which tunes C only. We tune C together with the vectorizer settings
(n-grams, `min_df`), which requires `GridSearchCV`; it is also used for the other models, so all
models are tuned in the same way (same folds, same selection rule).

## LR4 — Complexity: C

For the 1-SE rule the complexity is C: among all settings within one standard error of the
best CV accuracy, the one with the smallest C (strongest penalty) is chosen, ties broken by
the higher accuracy. This is the `lambda.1se` rule of `cv.glmnet` used in the course
(slides 40–42). The vectorizer setting is not part of this measure, as in the course, where
the features are fixed before `cv.glmnet`.

---

## Results of the tuning

| | Setting | CV accuracy |
|---|---|---:|
| Best mean (6 settings tie, C from 1 to 10) | simplest of them: C = 1, `min_df` = 10, unigrams | 0.9153 (SE 0.0043) |
| **Selected (1-SE rule)** | **C = 1, `min_df` = 10, unigrams** | **0.9153** |

Six settings reach the same best accuracy (1373 of 1500 reviews correct); the simplest of them is
the reference for the standard error (M3). Threshold 0.9153 − 0.0043 = 0.9110; 21 settings are
within one SE, and C = 0.56 is the first value below it (0.906). The selected setting is therefore
also one of the best settings. The selected model has a vocabulary of 829 unigrams, of which the
lasso keeps **208 non-zero coefficients** (75% of the terms are removed).

Accuracy and sparsity as a function of C (best vectorizer setting per C; non-zero coefficients
for the selected vectorizer setting, fitted on the whole training set):

| C | 0.01 | 0.03 | 0.1 | 0.32 | 1 | 3.16 | 10 | 100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CV accuracy | 0.725 | 0.816 | 0.861 | 0.895 | 0.915 | 0.915 | 0.915 | 0.907 |
| non-zero coefficients | 6 | 23 | 52 | 123 | 208 | 262 | 290 | 304 |

*Interpretation.*
- A strong penalty underfits (6 terms: 72.5%); accuracy levels off from C ≈ 1 and drops slightly
  for a weak penalty (C = 100), a sign of overfitting.
- **Unigrams are enough for the lasso.** The best unigram and uni + bigram settings differ by
  less than one SE (0.9153 vs. 0.9147), so the simpler unigram model is selected; the lasso
  selects the informative words itself. In the course example bigrams did improve the test
  accuracy (IMDB); on our data they add little for this model.

---

## LR5 — Most important features

Positive coefficients point towards a generated review, negative towards a human review;
exp(β) is the change of the odds per occurrence of the term.

| Towards | Term | Coefficient | Odds factor | Human reviews | Generated reviews |
|---|---|---:|---:|---:|---:|
| generated | limited | 2.179 | 8.83 | 8 | 62 |
| generated | properly | 1.982 | 7.25 | 10 | 42 |
| generated | downside | 1.881 | 6.56 | 1 | 50 |
| generated | variety | 1.805 | 6.08 | 10 | 109 |
| generated | unreliable | 1.785 | 5.96 | 0 | 17 |
| human | cleaning | −2.413 | 0.09 | 16 | 8 |
| human | least | −2.284 | 0.10 | 10 | 3 |
| human | us | −1.696 | 0.18 | 58 | 17 |
| human | shower | −1.228 | 0.29 | 73 | 15 |
| human | one | −1.225 | 0.29 | 73 | 22 |

*Interpretation (Q4).*
- **Generated reviews** use generic evaluative vocabulary for complaints: "limited variety",
  "did not work properly", "unreliable Wi-Fi".
- **Human reviews** mention concrete physical details ("shower", "cleaning") and narrate as a
  group ("us"), with hedges such as "at least".
- The lasso favours frequent terms that separate the classes in combination with the other
  terms (*variety*: 109 generated vs. 10 human reviews), rather than terms of one class only.
- **Artifact.** *downside* (50 generated vs. 1 human review) probably comes from the generation
  prompt, which asked for a "Downside" part ("The only downside was …").

---

## Test results

The final model evaluated once on the 500 test reviews (`results/test_evaluation.md`;
per-review predictions in `results/test_predictions.csv`). Positive class: generated.

| Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.910 | 0.915 | 0.904 | 0.910 |

Confusion matrix (rows: predicted, columns: true):

| | true human | true generated |
|---|---:|---:|
| predicted human | 229 | 24 |
| predicted generated | 21 | 226 |

Accuracy by sentiment: negative 0.924, positive 0.896 (250 reviews each).

*Interpretation.*
- **Test accuracy agrees with CV** (0.910 vs. 0.915); the difference is well within the
  uncertainty of a 500-review test set (standard error ≈ 0.013). The tuning did not overfit
  the validation folds.
- **The errors are balanced:** 21 human reviews labelled generated, 24 generated reviews missed,
  so precision and recall are almost equal.
- **Positive reviews are harder** than negative ones (0.896 vs. 0.924), as reported by
  Ignat et al. (2025). Most errors on positive reviews are missed generated reviews (18 of 26).
- **Review length plays a role.** The missed generated positive reviews are short (median 17 words
  vs. 49 for the correctly classified ones), and the human positive reviews labelled generated are
  long (median 57.5 vs. 22 words). With raw counts a long review contains more (generated-like)
  words, so the model partly relies on the amount of text; short generated reviews look human,
  long human reviews look generated. This links to the length difference between the classes
  (Data section) and belongs in the Limitations.

---

## Checklist for the report

- [ ] Lasso objective; C = 1/λ; `l1_ratio = 1`, `liblinear` (LR1).
- [ ] Intercept practically not penalized via `intercept_scaling`, as in the course (LR1).
- [ ] Raw counts instead of glmnet's standardization, with the reason (LR2).
- [ ] Grid and the `lambda.1se`-type choice of C (LR3, LR4).
- [ ] Why `GridSearchCV` instead of the suggested `LogisticRegressionCV` (LR3).
- [ ] Sparsity: 208 of 829 coefficients non-zero; accuracy and sparsity as a function of C.
- [ ] Unigrams selected; bigrams within noise for the lasso.
- [ ] Top-5 terms per direction with coefficients, odds factors and review counts; interpretation (LR5).
- [ ] Limitations: artifact *downside* (generation prompt).
- [ ] Test results: accuracy 0.910, precision 0.915, recall 0.904, F1 0.910; confusion matrix.
- [ ] Test accuracy agrees with CV (0.915): no overfitting of the tuning.
- [ ] Balanced errors (21 false positives, 24 false negatives).
- [ ] Positive reviews harder than negative (0.896 vs. 0.924), as in Ignat et al. (2025).
- [ ] Limitations: errors depend on review length (short generated reviews missed, long human
      reviews flagged).

---

## References

- Ignat, O., Xu, X., & Mihalcea, R. (2025). MAiDE-up: Multilingual Deception Detection of
  AI-generated Hotel Reviews. *Findings of NAACL 2025*, 1636–1653.
- Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso. *Journal of the
  Royal Statistical Society, Series B*, 58(1), 267–288.
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization Paths for Generalized
  Linear Models via Coordinate Descent. *Journal of Statistical Software*, 33(1).
- Feelders, A. (2026). Data Mining lecture: Logistic Regression, Text Classification.
  Utrecht University.
