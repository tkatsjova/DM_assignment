# Preprocessing audit (training set)

## D3 Placeholder and empty parts

| class | sentiment | Upside placeholder | Downside placeholder | Upside empty | Downside empty |
|---|---|---:|---:|---:|---:|
| ai | neg | 0 | 0 | 6 | 0 |
| ai | pos | 0 | 3 | 0 | 35 |
| human | neg | 5 | 0 | 41 | 6 |
| human | pos | 0 | 32 | 2 | 122 |

## D4-D7 Text properties by class

| Share of reviews (%) | ai | human |
|---|---:|---:|
| D4 city name (before) | 32.5 | 7.7 |
| D4 full hotel name (before) | 25.6 | 0.3 |
| D4 city name (after) | 0 | 0 |
| D4 hotel name word (after) | 8.8 | 2.1 |
| D4 word 'hotel' (after) | 48.7 | 30.5 |
| D5 written fully in lower case | 0.4 | 8.5 |
| D6 curly apostrophe | 2.4 | 13.6 |
| D6 exclamation mark | 4.3 | 14.1 |
| D7 contains a digit | 4 | 22.8 |

## D8 Stop word frequency

| stop word | human per 1000 tokens | ai per 1000 tokens | ai / human |
|---|---:|---:|---:|
| a | 24.8 | 26.6 | 1.07 |
| the | 74 | 102.6 | 1.39 |
| i | 15.7 | 5.1 | 0.32 |
| we | 10.8 | 1.6 | 0.15 |
| my | 4.4 | 4.8 | 1.09 |
| it | 11 | 7.9 | 0.72 |
| very | 13.8 | 8.1 | 0.59 |
| were | 8.5 | 14.5 | 1.71 |
| with | 7.9 | 11.5 | 1.46 |
| was | 30.1 | 42.2 | 1.4 |

## D8 Stop words among the most informative terms

| Terms by mutual information | Stop words among them | Stop words (examples) |
|---|---:|---|
| top 10 | 3 | and, the, were |
| top 30 | 11 | and, the, were, was, a, of, with, also, we, you, no |
| top 100 | 25 | and, the, were, was, a, of, with, also, we, you, no, to |

## D13 Sparse-term removal

| min_df (documents) | Unigrams kept | Share of tokens kept (%) | Distinct terms per review | Uni+bigrams kept |
|---|---:|---:|---:|---:|
| 1 | 4748 | 100 | 38.7 | 35728 |
| 2 | 2608 | 97.1 | 37.3 | 10583 |
| 5 | 1319 | 92.4 | 35 | 3593 |
| 10 | 829 | 88 | 32.8 | 1756 |
| 20 | 495 | 81.8 | 29.7 | 881 |
| 75 | 134 | 63.3 | 20.9 | 187 |

## D13 Terms occurring in a single review

| Vocabulary | Value |
|---|---:|
| Unigrams | 4748 |
| Occurring in one review only | 2140 |
| Share (%) | 45.1 |
