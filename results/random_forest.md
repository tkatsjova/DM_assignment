# Random forest (training set, 5-fold CV)

Decisions and interpretation: `docs/random_forest.md`.

## Selected hyperparameters (1-SE rule, complexity = negative min_samples_leaf)

| parameter | value |
|---|---|
| max_features | log2 |
| min_samples_leaf | 1 |
| n_estimators | 300 |
| min_df | 10 |
| ngram_range | (1, 2) |
| complexity | -1 |
| CV accuracy (chosen) | 0.9207 |
| CV accuracy (best) | 0.9207 |
| settings with the best accuracy | 1 |
| standard error (best) | 0.0036 |
| 1-SE threshold | 0.9171 |
| training accuracy | 1.0000 |
| training minus selected CV accuracy | 0.0793 |
| vocabulary size | 1756 |
| features used across forest | 1756 |
| nominal features per split | 10 |
| total nodes across forest | 262660 |
| experiment seconds (CV, refit and summaries) | 207.00 |

## Cross-validation results (all settings)

| rank | max_features | min_samples_leaf | n_estimators | min_df | ngram_range | complexity | accuracy | accuracy_std | f1 | within_1se | selected |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|---|
| 1 | log2 | 1 | 300 | 10 | (1, 2) | -1 | 0.9207 | 0.008 | 0.918 | True | True |
| 2 | log2 | 1 | 300 | 5 | (1, 1) | -1 | 0.9193 | 0.0028 | 0.9174 | True | False |
| 2 | log2 | 1 | 300 | 10 | (1, 1) | -1 | 0.9193 | 0.0064 | 0.9174 | True | False |
| 4 | log2 | 1 | 100 | 1 | (1, 1) | -1 | 0.9187 | 0.0069 | 0.9164 | True | False |
| 5 | log2 | 1 | 300 | 1 | (1, 1) | -1 | 0.916 | 0.0072 | 0.9135 | False | False |
| 6 | log2 | 1 | 100 | 5 | (1, 1) | -1 | 0.9147 | 0.0126 | 0.9124 | False | False |
| 7 | sqrt | 1 | 300 | 10 | (1, 1) | -1 | 0.914 | 0.0072 | 0.9123 | False | False |
| 8 | sqrt | 1 | 300 | 2 | (1, 1) | -1 | 0.9133 | 0.0091 | 0.9121 | False | False |
| 8 | log2 | 1 | 100 | 10 | (1, 1) | -1 | 0.9133 | 0.0133 | 0.9111 | False | False |
| 10 | sqrt | 1 | 100 | 2 | (1, 1) | -1 | 0.9127 | 0.0171 | 0.9116 | False | False |
| 10 | sqrt | 1 | 300 | 5 | (1, 1) | -1 | 0.9127 | 0.0092 | 0.9113 | False | False |
| 10 | sqrt | 1 | 300 | 5 | (1, 2) | -1 | 0.9127 | 0.0072 | 0.9109 | False | False |
| 10 | log2 | 1 | 100 | 10 | (1, 2) | -1 | 0.9127 | 0.0112 | 0.9102 | False | False |
| 10 | log2 | 1 | 300 | 5 | (1, 2) | -1 | 0.9127 | 0.008 | 0.9087 | False | False |
| 15 | log2 | 1 | 300 | 2 | (1, 1) | -1 | 0.912 | 0.0093 | 0.9092 | False | False |
| 16 | sqrt | 3 | 100 | 1 | (1, 1) | -3 | 0.9113 | 0.0077 | 0.9099 | False | False |
| 16 | sqrt | 1 | 100 | 5 | (1, 1) | -1 | 0.9113 | 0.0126 | 0.9101 | False | False |
| 16 | sqrt | 1 | 300 | 2 | (1, 2) | -1 | 0.9113 | 0.0038 | 0.9092 | False | False |
| 16 | log2 | 1 | 100 | 2 | (1, 1) | -1 | 0.9113 | 0.0112 | 0.9084 | False | False |
| 20 | sqrt | 3 | 300 | 1 | (1, 2) | -3 | 0.91 | 0.0094 | 0.9064 | False | False |
| 20 | log2 | 3 | 300 | 5 | (1, 1) | -3 | 0.91 | 0.0103 | 0.9079 | False | False |
| 22 | log2 | 3 | 300 | 10 | (1, 1) | -3 | 0.9093 | 0.0068 | 0.9072 | False | False |
| 23 | sqrt | 3 | 300 | 2 | (1, 1) | -3 | 0.9087 | 0.0069 | 0.9071 | False | False |
| 23 | log2 | 3 | 300 | 2 | (1, 1) | -3 | 0.9087 | 0.0061 | 0.9052 | False | False |
| 23 | log2 | 3 | 300 | 10 | (1, 2) | -3 | 0.9087 | 0.0102 | 0.9058 | False | False |
| 26 | sqrt | 3 | 300 | 1 | (1, 1) | -3 | 0.9073 | 0.0037 | 0.9055 | False | False |
| 27 | log2 | 3 | 300 | 1 | (1, 1) | -3 | 0.9067 | 0.0129 | 0.9016 | False | False |
| 27 | sqrt | 1 | 100 | 10 | (1, 1) | -1 | 0.9067 | 0.0082 | 0.9057 | False | False |
| 27 | sqrt | 1 | 100 | 10 | (1, 2) | -1 | 0.9067 | 0.0113 | 0.9054 | False | False |
| 27 | sqrt | 1 | 300 | 10 | (1, 2) | -1 | 0.9067 | 0.0082 | 0.9054 | False | False |
| 31 | log2 | 3 | 100 | 1 | (1, 1) | -3 | 0.906 | 0.0189 | 0.9015 | False | False |
| 31 | log2 | 3 | 300 | 2 | (1, 2) | -3 | 0.906 | 0.0196 | 0.8996 | False | False |
| 31 | sqrt | 1 | 100 | 1 | (1, 1) | -1 | 0.906 | 0.0076 | 0.9051 | False | False |
| 31 | sqrt | 1 | 100 | 2 | (1, 2) | -1 | 0.906 | 0.0086 | 0.9035 | False | False |
| 31 | sqrt | 1 | 300 | 1 | (1, 1) | -1 | 0.906 | 0.0055 | 0.9044 | False | False |
| 31 | sqrt | 1 | 300 | 1 | (1, 2) | -1 | 0.906 | 0.0072 | 0.903 | False | False |
| 37 | log2 | 3 | 100 | 10 | (1, 2) | -3 | 0.9053 | 0.0084 | 0.9027 | False | False |
| 37 | sqrt | 1 | 100 | 1 | (1, 2) | -1 | 0.9053 | 0.0069 | 0.9027 | False | False |
| 37 | sqrt | 1 | 100 | 5 | (1, 2) | -1 | 0.9053 | 0.0045 | 0.9033 | False | False |
| 40 | sqrt | 3 | 300 | 5 | (1, 1) | -3 | 0.9047 | 0.0051 | 0.9034 | False | False |
| 40 | sqrt | 3 | 300 | 10 | (1, 1) | -3 | 0.9047 | 0.0077 | 0.9032 | False | False |
| 40 | log2 | 3 | 100 | 2 | (1, 1) | -3 | 0.9047 | 0.003 | 0.902 | False | False |
| 40 | log2 | 3 | 300 | 5 | (1, 2) | -3 | 0.9047 | 0.0084 | 0.8997 | False | False |
| 40 | log2 | 1 | 300 | 2 | (1, 2) | -1 | 0.9047 | 0.0152 | 0.898 | False | False |
| 45 | log2 | 3 | 100 | 5 | (1, 1) | -3 | 0.904 | 0.0166 | 0.9013 | False | False |
| 46 | sqrt | 3 | 100 | 1 | (1, 2) | -3 | 0.9033 | 0.0113 | 0.9003 | False | False |
| 46 | sqrt | 3 | 100 | 2 | (1, 1) | -3 | 0.9033 | 0.0085 | 0.9021 | False | False |
| 46 | log2 | 1 | 100 | 2 | (1, 2) | -1 | 0.9033 | 0.0184 | 0.8968 | False | False |
| 49 | log2 | 3 | 100 | 10 | (1, 1) | -3 | 0.9027 | 0.0086 | 0.9012 | False | False |
| 50 | sqrt | 3 | 100 | 10 | (1, 1) | -3 | 0.902 | 0.011 | 0.9009 | False | False |
| 51 | sqrt | 3 | 300 | 2 | (1, 2) | -3 | 0.9007 | 0.0092 | 0.8976 | False | False |
| 51 | sqrt | 3 | 300 | 5 | (1, 2) | -3 | 0.9007 | 0.0086 | 0.899 | False | False |
| 51 | log2 | 3 | 100 | 2 | (1, 2) | -3 | 0.9007 | 0.0136 | 0.895 | False | False |
| 51 | log2 | 1 | 100 | 5 | (1, 2) | -1 | 0.9007 | 0.0132 | 0.8953 | False | False |
| 51 | log2 | 1 | 300 | 1 | (1, 2) | -1 | 0.9007 | 0.0148 | 0.8941 | False | False |
| 56 | sqrt | 3 | 100 | 5 | (1, 1) | -3 | 0.9 | 0.0053 | 0.8984 | False | False |
| 57 | sqrt | 3 | 300 | 10 | (1, 2) | -3 | 0.8993 | 0.0106 | 0.898 | False | False |
| 58 | sqrt | 3 | 100 | 5 | (1, 2) | -3 | 0.8987 | 0.0084 | 0.8966 | False | False |
| 59 | sqrt | 3 | 100 | 2 | (1, 2) | -3 | 0.8973 | 0.0076 | 0.8955 | False | False |
| 59 | log2 | 3 | 100 | 5 | (1, 2) | -3 | 0.8973 | 0.0116 | 0.893 | False | False |
| 61 | log2 | 3 | 300 | 1 | (1, 2) | -3 | 0.896 | 0.0201 | 0.8879 | False | False |
| 62 | sqrt | 3 | 100 | 10 | (1, 2) | -3 | 0.8953 | 0.0126 | 0.894 | False | False |
| 63 | log2 | 1 | 100 | 1 | (1, 2) | -1 | 0.8933 | 0.0135 | 0.8857 | False | False |
| 64 | log2 | 3 | 100 | 1 | (1, 2) | -3 | 0.866 | 0.0114 | 0.8597 | False | False |

## Settings within one standard error of the best

| rank | max_features | min_samples_leaf | n_estimators | min_df | ngram_range | complexity | accuracy | accuracy_std | f1 | within_1se | selected |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|---|
| 1 | log2 | 1 | 300 | 10 | (1, 2) | -1 | 0.9207 | 0.008 | 0.918 | True | True |
| 2 | log2 | 1 | 300 | 5 | (1, 1) | -1 | 0.9193 | 0.0028 | 0.9174 | True | False |
| 2 | log2 | 1 | 300 | 10 | (1, 1) | -1 | 0.9193 | 0.0064 | 0.9174 | True | False |
| 4 | log2 | 1 | 100 | 1 | (1, 1) | -1 | 0.9187 | 0.0069 | 0.9164 | True | False |

## Accuracy by forest setting (best vectorizer setting per setting)

| n_estimators | max_features | min_samples_leaf | min_df | ngram_range | complexity | accuracy | accuracy_std | f1 | within_1se | selected |
|---|---|---|---:|---|---:|---:|---:|---:|---|---|
| 100 | log2 | 1 | 1 | (1, 1) | -1 | 0.9187 | 0.0069 | 0.9164 | True | False |
| 100 | log2 | 3 | 1 | (1, 1) | -3 | 0.906 | 0.0189 | 0.9015 | False | False |
| 100 | sqrt | 1 | 2 | (1, 1) | -1 | 0.9127 | 0.0171 | 0.9116 | False | False |
| 100 | sqrt | 3 | 1 | (1, 1) | -3 | 0.9113 | 0.0077 | 0.9099 | False | False |
| 300 | log2 | 1 | 10 | (1, 2) | -1 | 0.9207 | 0.008 | 0.918 | True | True |
| 300 | log2 | 3 | 5 | (1, 1) | -3 | 0.91 | 0.0103 | 0.9079 | False | False |
| 300 | sqrt | 1 | 10 | (1, 1) | -1 | 0.914 | 0.0072 | 0.9123 | False | False |
| 300 | sqrt | 3 | 1 | (1, 2) | -3 | 0.91 | 0.0094 | 0.9064 | False | False |

## Selected forest structure (whole training set)

| Measure | min | median | mean | max |
|---|---:|---:|---:|---:|
| Tree depth | 48 | 79 | 80.84 | 133 |
| Leaves per tree | 354 | 439 | 438.27 | 518 |
| Nodes per tree | 707 | 877 | 875.53 | 1035 |

## Most important features (reviews = training reviews containing the term)

| rank | term | importance | reviews human | reviews ai |
|---|---|---:|---:|---:|
| 1 | rooms | 0.0173 | 118 | 402 |
| 2 | service | 0.0123 | 49 | 252 |
| 3 | the | 0.0121 | 542 | 703 |
| 4 | and | 0.0109 | 500 | 713 |
| 5 | the rooms | 0.0108 | 35 | 251 |
