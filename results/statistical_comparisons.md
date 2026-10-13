# Final statistical comparisons

McNemar: exact two-sided binomial test of discordant correctness counts, with p=0.5. Fisher: two-sided test of correct/incorrect counts in the disjoint sentiment groups. All decisions use nominal per-test alpha=0.01, without multiplicity adjustment. Nonsignificance does not establish equivalence. Tests assume independent reviews; reviews from the same hotel may be related.

Methods: [exact McNemar](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html), [SciPy binomtest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html), [SciPy fisher_exact](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html).

## Q1: paired correctness (naive_bayes vs logistic_regression)

| naive_bayes | logistic_regression correct | logistic_regression incorrect |
|---|---:|---:|
| correct | 424 | 33 |
| incorrect | 31 | 12 |

## Q2: paired correctness (classification_tree vs random_forest)

| classification_tree | random_forest correct | random_forest incorrect |
|---|---:|---:|
| correct | 395 | 13 |
| incorrect | 73 | 19 |

## Q2: paired correctness (classification_tree vs gradient_boosting)

| classification_tree | gradient_boosting correct | gradient_boosting incorrect |
|---|---:|---:|
| correct | 395 | 13 |
| incorrect | 67 | 25 |

## McNemar test results

| comparison | first accuracy | second accuracy | first only correct (b) | second only correct (c) | discordant reviews | exact two-sided p-value | decision at alpha=0.01 |
|---|---:|---:|---:|---:|---:|---:|---|
| naive_bayes vs logistic_regression | 0.9140 | 0.9100 | 33 | 31 | 64 | 0.9006532462520334 | Do not reject equal accuracy |
| classification_tree vs random_forest | 0.8160 | 0.9360 | 13 | 73 | 86 | 2.718821367078067e-11 | Reject equal accuracy |
| classification_tree vs gradient_boosting | 0.8160 | 0.9240 | 13 | 67 | 80 | 6.415141174951122e-10 | Reject equal accuracy |

## Q3: random forest correctness by review sentiment

| sentiment | correct | incorrect | accuracy |
|---|---:|---:|---:|
| pos | 228 | 22 | 0.912 |
| neg | 240 | 10 | 0.96 |

## Q3: two-sided Fisher test

| quantity | value |
|---|---|
| accuracy difference (pos minus neg) | -0.0480 |
| exact two-sided p-value | 0.0429375175743914 |
| decision at alpha=0.01 | Do not reject equal sentiment accuracy |
