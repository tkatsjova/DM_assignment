# Evaluation on the test set

## Performance (positive class: ai)

| model | accuracy | precision | recall | f1 |
|---|---:|---:|---:|---:|
| naive_bayes | 0.914 | 0.8876 | 0.948 | 0.9168 |
| logistic_regression | 0.91 | 0.915 | 0.904 | 0.9095 |

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

## Accuracy by sentiment

| sentiment | naive_bayes | logistic_regression | reviews |
|---|---:|---:|---:|
| neg | 0.928 | 0.924 | 250 |
| pos | 0.9 | 0.896 | 250 |
