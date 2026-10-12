# Multinomial naive Bayes

Decisions, results and interpretation for naive Bayes. The shared setup (CV, 1-SE rule,
test evaluation) is in `docs/modelling_setup.md`.

Code: `src/models/naive_bayes.py`, `src/features/selection.py`.
Tables: `results/naive_bayes.md` (reproduced by `python main.py`).

Pipeline: `vectorizer → SelectKBest(mutual information, k) → MultinomialNB(α = 1)`.

---

## Overview

| # | Decision | Choice |
|---|---|---|
| NB1 | Model | multinomial naive Bayes, Laplace smoothing (α = 1), not tuned |
| NB2 | Feature selection | top-k terms by mutual information, k tuned |
| NB3 | Grid | k × n-grams × `min_df` (56 settings) |
| NB4 | Complexity for the 1-SE rule | number of terms the model uses |
| NB5 | Important features | score difference log P(w\|ai) − log P(w\|human) |

**Selected:** unigrams + bigrams, `min_df` = 10, all 1756 terms (no selection by MI),
CV accuracy 0.903.

---

## NB1 — Multinomial naive Bayes with Laplace smoothing

The multinomial model uses the term counts of the bag-of-words representation (D11):
class priors are the class frequencies, and the word probabilities per class are estimated
with add-one (Laplace) smoothing, so that a term unseen in one class does not get
probability zero (course lecture; Jurafsky & Martin). The smoothing parameter α is fixed
at 1 and not tuned: the assignment names the number of selected terms k as the
hyperparameter of naive Bayes, and fixing α keeps the grid focused on it.

## NB2 — Feature selection by mutual information

The assignment notes that feature selection, in addition to sparse-term removal, may
improve naive Bayes. We select the top-k terms by mutual information between the presence
of a term and the class, as in the course (slides 28–33, 42–44): the document-term matrix
is made binary and MI is computed from the 2 × 2 table of each term. MI is written out in
`src/features/selection.py` as a vectorized implementation of the course formula in bits.
Scikit-learn also supports discrete MI: `mutual_info_classif(X > 0, y,
discrete_features=True)` computes the same estimate in nats (divide by ln(2) for bits);
sparse inputs are automatically treated as discrete. On the example of the lecture
(the word *bad*) our implementation gives 0.0557 bits, as on
slide 33. MI uses the class labels, so the selection is part of the Pipeline and is
recomputed in every CV fold (no leakage). With `k = all` no terms are removed by MI.

## NB3 — Grid

| Parameter | Values |
|---|---|
| `k` (terms kept by MI) | 50, 100, 250, 500, 1000, 2000, all |
| `ngram_range` (D12) | unigrams; unigrams + bigrams |
| `min_df` (D13) | 1, 2, 5, 10 documents |

56 settings × 5 folds. If k exceeds the vocabulary of a setting, all terms are kept.

## NB4 — Complexity: number of terms used

For the 1-SE rule, the complexity of a setting is the number of terms naive Bayes uses:
min(k, vocabulary size), with the vocabulary size of the vectorizer setting measured on
the training set (within a fold it is slightly smaller; the order of the settings is the
same).

---

## Results of the tuning

| | Setting | Terms | CV accuracy |
|---|---|---:|---:|
| Best mean | all terms, `min_df` = 2, uni + bigrams | 10583 | 0.9100 (SE 0.0071) |
| **Selected (1-SE rule)** | **all terms, `min_df` = 10, uni + bigrams** | **1756** | **0.9033** |

Threshold: 0.9100 − 0.0071 = 0.9029; 7 settings are within one SE. In the selected setting
k = 2000 exceeds the vocabulary (1756 terms), so it is equivalent to k = all:
**the final model uses no selection by mutual information**; its size is controlled by the
sparse-term threshold.

Accuracy by the number of selected terms (best vectorizer setting per k):

| k | 50 | 100 | 250 | 500 | 1000 | 2000 | all |
|---|---:|---:|---:|---:|---:|---:|---:|
| CV accuracy | 0.812 | 0.855 | 0.890 | 0.891 | 0.902 | 0.905 | 0.910 |

*Interpretation.*
- **Feature selection by MI does not improve naive Bayes.** Accuracy increases with the
  number of terms and levels off after about 1000 terms. The signal that separates generated
  from human reviews is spread over many words and word pairs, unlike the sentiment example
  of the course, where a few words (*bad*, *worst*) carry most of the information.
  Sparse-term removal alone is enough to remove noise.
- **Bigrams help.** All ten best settings use unigrams + bigrams, in line with Ott et al.
  (2011); word pairs capture phrasing (*service was*, *told us*) that single words do not.
- **`min_df` hardly matters** among the best settings (1–10 documents).
- The CV F1 score (generated = positive class) is slightly higher than the accuracy, a first
  sign that the model detects generated reviews more easily than human ones — to be checked
  with precision and recall on the test set.

---

## NB5 — Most important features

For naive Bayes the contribution of a term to the decision is the difference of its log
probabilities in the two classes, log P(w|ai) − log P(w|human) (course, slide 45): positive
values point towards a generated review, negative ones towards a human review. The table
also gives the number of training reviews of each class that contain the term.

| Towards | Term | Score diff. | Human reviews | Generated reviews |
|---|---|---:|---:|---:|
| generated | service was | 3.63 | 1 | 101 |
| generated | connection was | 3.51 | 0 | 44 |
| generated | attractions the | 3.46 | 0 | 42 |
| generated | not up | 3.44 | 0 | 41 |
| generated | the in | 3.39 | 0 | 38 |
| human | told | −3.85 | 22 | 0 |
| human | he | −3.66 | 16 | 0 |
| human | booked | −3.63 | 25 | 0 |
| human | said | −3.56 | 17 | 0 |
| human | want | −3.52 | 23 | 0 |

(*want* shares the 5th place with *com*, both −3.52.)

*Interpretation (Q4).*
- All top terms occur in only one class; the score difference rewards such exclusive
  terms, whose probability in the other class comes from smoothing only. A difference of
  3.63 means the term is e^3.63 ≈ 38 times more likely in generated reviews.
- **Generated reviews** use generic evaluative templates: "the service was …",
  "the Wi-Fi connection was …", "not up to par".
- **Human reviews** narrate concrete events with other people: "the receptionist told us",
  "he said", "we booked …".
- **Artifacts.** *the in* and *attractions the* are traces of removed hotel and city names
  ("The [hotel] in [city]"): the model still detects that a review mentions the hotel,
  which comes from the generation prompt (D4). *com* comes from "Booking.com" in human
  reviews.

---

## Test results

The final model evaluated once on the 500 test reviews (`results/test_evaluation.md`;
per-review predictions in `results/test_predictions.csv`). Positive class: generated.

| Accuracy | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.914 | 0.888 | 0.948 | 0.917 |

Confusion matrix (rows: predicted, columns: true):

| | true human | true generated |
|---|---:|---:|
| predicted human | 220 | 13 |
| predicted generated | 30 | 237 |

Accuracy by sentiment: negative 0.928, positive 0.900 (250 reviews each).

*Interpretation.*
- **Test accuracy agrees with CV** (0.914 vs. 0.903); the difference is well within the
  uncertainty of a 500-review test set (standard error √(0.914 · 0.086 / 500) ≈ 0.013).
  The tuning did not overfit the validation folds.
- **Naive Bayes over-predicts the generated class:** 30 human reviews are labelled generated,
  only 13 generated reviews are missed. Hence the high recall (0.948) and the lower precision
  (0.888). This confirms the sign seen in CV (F1 above accuracy). A plausible reason is the
  independence assumption: a generated review contains many template phrases, and naive
  Bayes adds up their evidence as if it were independent (double counting, course slide 26).
- **Positive reviews are harder** than negative ones (0.900 vs. 0.928); most errors on positive
  reviews are human reviews labelled generated (19 of 25). These are somewhat shorter than the
  correctly classified human positive reviews (median 18 vs. 25 words). The same pattern —
  positive fake reviews are harder to detect — is reported by Ignat et al. (2025).

---

## Checklist for the report

- [ ] Multinomial NB with Laplace smoothing; α fixed at 1, not tuned (NB1).
- [ ] Feature selection by mutual information (course method), k tuned (NB2, NB3).
- [ ] Result: selection by MI does **not** improve NB; the final model uses all 1756 terms.
- [ ] Accuracy increases with the number of terms and levels off after ~1000 (table by k).
- [ ] Bigrams are in all ten best settings (link to Ott et al., 2011).
- [ ] Complexity measured on the vocabulary of the whole training set (approximation) (NB4).
- [ ] Top-5 terms per direction with the score difference and review counts; interpretation (NB5).
- [ ] Limitations: artifacts *the in*, *attractions the* (removed names), *com* (Booking.com).
- [ ] Test results: accuracy 0.914, precision 0.888, recall 0.948, F1 0.917; confusion matrix.
- [ ] Test accuracy agrees with CV (0.903): no overfitting of the tuning.
- [ ] NB over-predicts the generated class (30 false positives vs. 13 false negatives); link
      to the independence assumption (double counting).
- [ ] Positive reviews harder than negative (0.900 vs. 0.928), as in Ignat et al. (2025).

---

## References

- Ignat, O., Xu, X., & Mihalcea, R. (2025). MAiDE-up: Multilingual Deception Detection of
  AI-generated Hotel Reviews. *Findings of NAACL 2025*, 1636–1653.
- Jurafsky, D., & Martin, J. H. *Speech and Language Processing* (3rd ed. draft),
  chapter "Naive Bayes and Sentiment Classification".
- Ott, M., Choi, Y., Cardie, C., & Hancock, J. T. (2011). Finding Deceptive Opinion Spam by
  Any Stretch of the Imagination. *Proceedings of ACL 2011*, 309–319.
- Feelders, A. (2026). Data Mining lecture: Text Classification, Naive Bayes.
  Utrecht University.
