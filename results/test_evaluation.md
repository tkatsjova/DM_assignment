# Evaluation on the test set

## Performance (positive class: ai)

| model | accuracy | precision | recall | f1 |
|---|---:|---:|---:|---:|
| naive_bayes | 0.914 | 0.8876 | 0.948 | 0.9168 |
| logistic_regression | 0.91 | 0.915 | 0.904 | 0.9095 |
| classification_tree | 0.816 | 0.816 | 0.816 | 0.816 |
| random_forest | 0.936 | 0.958 | 0.912 | 0.9344 |
| gradient_boosting | 0.924 | 0.9344 | 0.912 | 0.9231 |

## Confusion matrix: naive_bayes

| naive_bayes | true human | true ai |
|---|---:|---:|
| predicted human | 220 | 13 |
| predicted ai | 30 | 237 |

## Confusion matrix: logistic_regression

| logistic_regression | true human | true ai |
|---|---:|---:|
| predicted human | 229 | 24 |
| predicted ai | 21 | 226 |

## Confusion matrix: classification_tree

| classification_tree | true human | true ai |
|---|---:|---:|
| predicted human | 204 | 46 |
| predicted ai | 46 | 204 |

## Confusion matrix: random_forest

| random_forest | true human | true ai |
|---|---:|---:|
| predicted human | 240 | 22 |
| predicted ai | 10 | 228 |

## Confusion matrix: gradient_boosting

| gradient_boosting | true human | true ai |
|---|---:|---:|
| predicted human | 234 | 22 |
| predicted ai | 16 | 228 |

## Accuracy by sentiment

| sentiment | naive_bayes | logistic_regression | classification_tree | random_forest | gradient_boosting | reviews |
|---|---:|---:|---:|---:|---:|---:|
| neg | 0.928 | 0.924 | 0.848 | 0.96 | 0.944 | 250 |
| pos | 0.9 | 0.896 | 0.784 | 0.912 | 0.904 | 250 |

## Prediction consistency with previous saved NB/LR predictions

| model | reviews compared | changed predictions |
|---|---:|---:|
| naive_bayes | 500 | 0 |
| logistic_regression | 500 | 0 |
