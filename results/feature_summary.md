# Important features of the final models (training set)

AI is the generated class. NB score differences and LR coefficients are signed; tree/ensemble importances are unsigned. Review counts describe training associations, not causal effects. Scores from different model types are not directly comparable.

## Naive Bayes

Source: `results/naive_bayes.md`.

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

## L1 logistic regression

Source: `results/logistic_regression.md`.

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

## Classification tree

Source: `results/classification_tree.md`.

| rank | term | importance | reviews human | reviews ai |
|---|---|---:|---:|---:|
| 1 | rooms | 0.2064 | 118 | 402 |
| 2 | service | 0.1085 | 49 | 252 |
| 3 | room was | 0.0753 | 101 | 195 |
| 4 | the | 0.0376 | 542 | 703 |
| 5 | we | 0.0367 | 146 | 48 |

## Random forest

Source: `results/random_forest.md`.

| rank | term | importance | reviews human | reviews ai |
|---|---|---:|---:|---:|
| 1 | rooms | 0.0173 | 118 | 402 |
| 2 | service | 0.0123 | 49 | 252 |
| 3 | the | 0.0121 | 542 | 703 |
| 4 | and | 0.0109 | 500 | 713 |
| 5 | the rooms | 0.0108 | 35 | 251 |

## Gradient boosting

Source: `results/gradient_boosting.md`.

| rank | term | importance | reviews human | reviews ai |
|---|---|---:|---:|---:|
| 1 | rooms | 0.1369 | 118 | 402 |
| 2 | the | 0.1091 | 542 | 703 |
| 3 | service | 0.0969 | 49 | 252 |
| 4 | we | 0.0584 | 146 | 48 |
| 5 | and | 0.0554 | 500 | 713 |
