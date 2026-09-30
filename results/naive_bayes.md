# Multinomial naive Bayes (training set, 5-fold CV)

Decisions and interpretation: `docs/naive_bayes.md`.

## Selected hyperparameters (1-SE rule, complexity = number of terms used)

| parameter | value |
|---|---|
| k | 2000 |
| min_df | 10 |
| ngram_range | (1, 2) |
| complexity | 1756 |
| CV accuracy (chosen) | 0.9033 |
| CV accuracy (best) | 0.9100 |
| settings with the best accuracy | 1 |
| standard error (best) | 0.0071 |
| 1-SE threshold | 0.9029 |
| feature selection by MI | no (all terms) |

## Cross-validation results (10 best settings)

| rank | k | min_df | ngram_range | complexity | accuracy | accuracy_std | f1 | within_1se | selected |
|---|---|---:|---|---:|---:|---:|---:|---|---|
| 1 | all | 2 | (1, 2) | 10583 | 0.91 | 0.0158 | 0.9161 | True | False |
| 2 | all | 5 | (1, 2) | 3593 | 0.9073 | 0.0192 | 0.9119 | True | False |
| 3 | 2000 | 5 | (1, 2) | 2000 | 0.9047 | 0.0139 | 0.9099 | True | False |
| 4 | 2000 | 1 | (1, 2) | 2000 | 0.904 | 0.0104 | 0.9096 | True | False |
| 4 | 2000 | 2 | (1, 2) | 2000 | 0.904 | 0.0104 | 0.9096 | True | False |
| 6 | 2000 | 10 | (1, 2) | 1756 | 0.9033 | 0.0186 | 0.9081 | True | True |
| 6 | all | 10 | (1, 2) | 1756 | 0.9033 | 0.0186 | 0.9081 | True | False |
| 8 | 1000 | 1 | (1, 2) | 1000 | 0.902 | 0.009 | 0.9077 | False | False |
| 8 | 1000 | 2 | (1, 2) | 1000 | 0.902 | 0.009 | 0.9077 | False | False |
| 8 | 1000 | 5 | (1, 2) | 1000 | 0.902 | 0.009 | 0.9077 | False | False |

## Settings within one standard error of the best

| rank | k | min_df | ngram_range | complexity | accuracy | accuracy_std | f1 | within_1se | selected |
|---|---|---:|---|---:|---:|---:|---:|---|---|
| 1 | all | 2 | (1, 2) | 10583 | 0.91 | 0.0158 | 0.9161 | True | False |
| 2 | all | 5 | (1, 2) | 3593 | 0.9073 | 0.0192 | 0.9119 | True | False |
| 3 | 2000 | 5 | (1, 2) | 2000 | 0.9047 | 0.0139 | 0.9099 | True | False |
| 4 | 2000 | 1 | (1, 2) | 2000 | 0.904 | 0.0104 | 0.9096 | True | False |
| 4 | 2000 | 2 | (1, 2) | 2000 | 0.904 | 0.0104 | 0.9096 | True | False |
| 6 | 2000 | 10 | (1, 2) | 1756 | 0.9033 | 0.0186 | 0.9081 | True | True |
| 6 | all | 10 | (1, 2) | 1756 | 0.9033 | 0.0186 | 0.9081 | True | False |

## Accuracy by number of selected terms (best vectorizer setting per k)

| k | min_df | ngram_range | complexity | accuracy | accuracy_std | f1 | within_1se | selected |
|---|---:|---|---:|---:|---:|---:|---|---|
| 50 | 1 | (1, 1) | 50 | 0.812 | 0.0061 | 0.8217 | False | False |
| 100 | 1 | (1, 2) | 100 | 0.8553 | 0.0141 | 0.8592 | False | False |
| 250 | 1 | (1, 2) | 250 | 0.89 | 0.0156 | 0.8943 | False | False |
| 500 | 1 | (1, 2) | 500 | 0.8913 | 0.0146 | 0.8979 | False | False |
| 1000 | 1 | (1, 2) | 1000 | 0.902 | 0.009 | 0.9077 | False | False |
| 2000 | 5 | (1, 2) | 2000 | 0.9047 | 0.0139 | 0.9099 | True | False |
| all | 2 | (1, 2) | 10583 | 0.91 | 0.0158 | 0.9161 | True | False |

## Most important features (reviews = training reviews containing the term)

| towards | rank | term | score diff | reviews human | reviews ai |
|---|---|---|---:|---:|---:|
| ai | 1 | service was | 3.63 | 1 | 101 |
| ai | 2 | connection was | 3.51 | 0 | 44 |
| ai | 3 | attractions the | 3.46 | 0 | 42 |
| ai | 4 | not up | 3.44 | 0 | 41 |
| ai | 5 | the in | 3.39 | 0 | 38 |
| human | 1 | told | -3.85 | 22 | 0 |
| human | 2 | he | -3.66 | 16 | 0 |
| human | 3 | booked | -3.63 | 25 | 0 |
| human | 4 | said | -3.56 | 17 | 0 |
| human | 5 | want | -3.52 | 23 | 0 |
