# Logistic regression with L1 penalty (training set, 5-fold CV)

Decisions and interpretation: `docs/logistic_regression.md`.

## Selected hyperparameters (1-SE rule, complexity = C)

| parameter | value |
|---|---|
| C | 1.0 |
| min_df | 10 |
| ngram_range | (1, 1) |
| complexity | 1.0 |
| CV accuracy (chosen) | 0.9153 |
| CV accuracy (best) | 0.9153 |
| settings with the best accuracy | 6 |
| standard error (best) | 0.0043 |
| 1-SE threshold | 0.9110 |
| non-zero coefficients | 208 |
| vocabulary size | 829 |

## Cross-validation results (10 best settings)

| rank | C | min_df | ngram_range | complexity | accuracy | accuracy_std | f1 | within_1se | selected |
|---|---:|---:|---|---:|---:|---:|---:|---|---|
| 1 | 1 | 10 | (1, 1) | 1 | 0.9153 | 0.0096 | 0.9137 | True | True |
| 1 | 1.7783 | 1 | (1, 1) | 1.7783 | 0.9153 | 0.0173 | 0.9143 | True | False |
| 1 | 1.7783 | 2 | (1, 1) | 1.7783 | 0.9153 | 0.0173 | 0.9143 | True | False |
| 1 | 3.1623 | 1 | (1, 1) | 3.1623 | 0.9153 | 0.0198 | 0.9141 | True | False |
| 1 | 3.1623 | 2 | (1, 1) | 3.1623 | 0.9153 | 0.0197 | 0.9142 | True | False |
| 1 | 10 | 2 | (1, 1) | 10 | 0.9153 | 0.0192 | 0.9142 | True | False |
| 7 | 1 | 5 | (1, 2) | 1 | 0.9147 | 0.0191 | 0.9125 | True | False |
| 8 | 5.6234 | 2 | (1, 1) | 5.6234 | 0.914 | 0.0202 | 0.9127 | True | False |
| 9 | 1 | 1 | (1, 1) | 1 | 0.9133 | 0.0062 | 0.9116 | True | False |
| 9 | 1 | 2 | (1, 1) | 1 | 0.9133 | 0.0062 | 0.9116 | True | False |

## Settings within one standard error of the best

| rank | C | min_df | ngram_range | complexity | accuracy | accuracy_std | f1 | within_1se | selected |
|---|---:|---:|---|---:|---:|---:|---:|---|---|
| 1 | 1 | 10 | (1, 1) | 1 | 0.9153 | 0.0096 | 0.9137 | True | True |
| 1 | 1.7783 | 1 | (1, 1) | 1.7783 | 0.9153 | 0.0173 | 0.9143 | True | False |
| 1 | 1.7783 | 2 | (1, 1) | 1.7783 | 0.9153 | 0.0173 | 0.9143 | True | False |
| 1 | 3.1623 | 1 | (1, 1) | 3.1623 | 0.9153 | 0.0198 | 0.9141 | True | False |
| 1 | 3.1623 | 2 | (1, 1) | 3.1623 | 0.9153 | 0.0197 | 0.9142 | True | False |
| 1 | 10 | 2 | (1, 1) | 10 | 0.9153 | 0.0192 | 0.9142 | True | False |
| 7 | 1 | 5 | (1, 2) | 1 | 0.9147 | 0.0191 | 0.9125 | True | False |
| 8 | 5.6234 | 2 | (1, 1) | 5.6234 | 0.914 | 0.0202 | 0.9127 | True | False |
| 9 | 1 | 1 | (1, 1) | 1 | 0.9133 | 0.0062 | 0.9116 | True | False |
| 9 | 1 | 2 | (1, 1) | 1 | 0.9133 | 0.0062 | 0.9116 | True | False |
| 9 | 1.7783 | 10 | (1, 2) | 1.7783 | 0.9133 | 0.0199 | 0.9117 | True | False |
| 12 | 1 | 5 | (1, 1) | 1 | 0.9127 | 0.006 | 0.911 | True | False |
| 12 | 1 | 10 | (1, 2) | 1 | 0.9127 | 0.0194 | 0.9107 | True | False |
| 12 | 1.7783 | 10 | (1, 1) | 1.7783 | 0.9127 | 0.0132 | 0.9118 | True | False |
| 12 | 3.1623 | 5 | (1, 1) | 3.1623 | 0.9127 | 0.0198 | 0.9114 | True | False |
| 12 | 5.6234 | 1 | (1, 1) | 5.6234 | 0.9127 | 0.0191 | 0.9115 | True | False |
| 12 | 5.6234 | 5 | (1, 1) | 5.6234 | 0.9127 | 0.0192 | 0.9116 | True | False |
| 18 | 10 | 1 | (1, 1) | 10 | 0.912 | 0.0189 | 0.9108 | True | False |
| 19 | 1 | 1 | (1, 2) | 1 | 0.9113 | 0.0185 | 0.9089 | True | False |
| 19 | 1 | 2 | (1, 2) | 1 | 0.9113 | 0.0185 | 0.9089 | True | False |
| 19 | 1.7783 | 5 | (1, 1) | 1.7783 | 0.9113 | 0.018 | 0.9105 | True | False |

## Accuracy by C (best vectorizer setting per C)

| C | min_df | ngram_range | complexity | accuracy | accuracy_std | f1 | within_1se | selected |
|---|---:|---|---:|---:|---:|---:|---|---|
| 0.01 | 1 | (1, 1) | 0.01 | 0.7247 | 0.0177 | 0.7111 | False | False |
| 0.0178 | 1 | (1, 1) | 0.0178 | 0.778 | 0.0212 | 0.7699 | False | False |
| 0.0316 | 1 | (1, 1) | 0.0316 | 0.816 | 0.0116 | 0.8123 | False | False |
| 0.0562 | 1 | (1, 1) | 0.0562 | 0.8433 | 0.0113 | 0.8405 | False | False |
| 0.1 | 1 | (1, 2) | 0.1 | 0.8607 | 0.0104 | 0.8575 | False | False |
| 0.1778 | 1 | (1, 1) | 0.1778 | 0.8787 | 0.0096 | 0.8761 | False | False |
| 0.3162 | 1 | (1, 1) | 0.3162 | 0.8947 | 0.0115 | 0.8927 | False | False |
| 0.5623 | 1 | (1, 2) | 0.5623 | 0.906 | 0.0175 | 0.9039 | False | False |
| 1 | 10 | (1, 1) | 1 | 0.9153 | 0.0096 | 0.9137 | True | True |
| 1.7783 | 1 | (1, 1) | 1.7783 | 0.9153 | 0.0173 | 0.9143 | True | False |
| 3.1623 | 1 | (1, 1) | 3.1623 | 0.9153 | 0.0198 | 0.9141 | True | False |
| 5.6234 | 2 | (1, 1) | 5.6234 | 0.914 | 0.0202 | 0.9127 | True | False |
| 10 | 2 | (1, 1) | 10 | 0.9153 | 0.0192 | 0.9142 | True | False |
| 17.7828 | 2 | (1, 1) | 17.7828 | 0.91 | 0.0243 | 0.9086 | False | False |
| 31.6228 | 1 | (1, 1) | 31.6228 | 0.9067 | 0.0249 | 0.905 | False | False |
| 56.2341 | 1 | (1, 1) | 56.2341 | 0.9067 | 0.0242 | 0.9051 | False | False |
| 100 | 1 | (1, 1) | 100 | 0.9073 | 0.0262 | 0.9058 | False | False |

## Non-zero coefficients by C (selected vectorizer setting, whole training set)

| C | lambda = 1/C | non-zero coefficients |
|---|---:|---:|
| 0.01 | 100 | 6 |
| 0.0178 | 56.1798 | 15 |
| 0.0316 | 31.6456 | 23 |
| 0.0562 | 17.7936 | 36 |
| 0.1 | 10 | 52 |
| 0.1778 | 5.6243 | 81 |
| 0.3162 | 3.1626 | 123 |
| 0.5623 | 1.7784 | 171 |
| 1 | 1 | 208 |
| 1.7783 | 0.5623 | 238 |
| 3.1623 | 0.3162 | 262 |
| 5.6234 | 0.1778 | 280 |
| 10 | 0.1 | 290 |
| 17.7828 | 0.0562 | 294 |
| 31.6228 | 0.0316 | 299 |
| 56.2341 | 0.0178 | 301 |
| 100 | 0.01 | 304 |

## Most important features (reviews = training reviews containing the term)

| towards | rank | term | coefficient | odds factor | reviews human | reviews ai |
|---|---|---|---:|---:|---:|---:|
| ai | 1 | limited | 2.179 | 8.83 | 8 | 62 |
| ai | 2 | properly | 1.982 | 7.25 | 10 | 42 |
| ai | 3 | downside | 1.881 | 6.56 | 1 | 50 |
| ai | 4 | variety | 1.805 | 6.08 | 10 | 109 |
| ai | 5 | unreliable | 1.785 | 5.96 | 0 | 17 |
| human | 1 | cleaning | -2.413 | 0.09 | 16 | 8 |
| human | 2 | least | -2.284 | 0.1 | 10 | 3 |
| human | 3 | us | -1.696 | 0.18 | 58 | 17 |
| human | 4 | shower | -1.228 | 0.29 | 73 | 15 |
| human | 5 | one | -1.225 | 0.29 | 73 | 22 |
