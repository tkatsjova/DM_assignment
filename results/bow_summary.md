# Bag-of-words representation (training set)

## Example reviews before and after preprocessing

| Review | Stage | Text |
|---|---|---|
| human (id 0) | before | Staff. Serkan türk was a great help during our stay. |
| human (id 0) | after | staff serkan türk was a great help during our stay |
| ai (id 1000) | before | The Hotel Ickale is a splendid choice for any traveler in Ankara. With its spacious, comfortable rooms and a welcoming, helpful staff, it truly made my stay enjoyable. Breakfast was delicious with a wide variety to fit anyone's palate. Slightly further from city center, yet still within reasonable accessibility. |
| ai (id 1000) | after | the is a splendid choice for any traveler in with its spacious comfortable rooms and a welcoming helpful staff it truly made my stay enjoyable breakfast was delicious with a wide variety to fit anyones palate slightly further from city center yet still within reasonable accessibility |

## Document-term matrix (training set)

| Matrix | Documents | Terms | Non-sparse entries | Sparse entries | Sparsity | Maximal term length |
|---|---:|---:|---:|---:|---:|---:|
| all terms | 1500 | 4748 | 58049 | 7063951 | 99.18% | 17 |
| min_df = 5 | 1500 | 1319 | 52450 | 1926050 | 97.35% | 15 |

## Review length after preprocessing

| Measure | mean | median | min | max |
|---|---:|---:|---:|---:|
| Tokens per review | 51 | 43 | 2 | 417 |
| Distinct terms per review | 38.7 | 35 | 1 | 219 |

## Part of the document-term matrix (5 reviews per class)

| id | class | can | options | perfect | restaurant | station | street |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | human | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | human | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | human | 0 | 0 | 0 | 0 | 0 | 0 |
| 4 | human | 0 | 0 | 0 | 0 | 0 | 0 |
| 5 | human | 0 | 0 | 0 | 0 | 0 | 1 |
| 1000 | ai | 0 | 0 | 0 | 0 | 0 | 0 |
| 1001 | ai | 0 | 0 | 0 | 0 | 0 | 0 |
| 1006 | ai | 0 | 0 | 0 | 0 | 0 | 0 |
| 1007 | ai | 1 | 0 | 0 | 0 | 0 | 0 |
| 1008 | ai | 0 | 1 | 0 | 1 | 0 | 0 |

## Most frequent terms per class

| rank | human term | human count | ai term | ai count |
|---|---|---:|---|---:|
| 1 | the | 2395 | the | 3936 |
| 2 | and | 1256 | and | 1873 |
| 3 | was | 975 | was | 1621 |
| 4 | to | 922 | a | 1019 |
| 5 | a | 801 | to | 1003 |
| 6 | in | 595 | of | 776 |
| 7 | room | 538 | in | 654 |
| 8 | is | 509 | were | 556 |
| 9 | i | 508 | location | 546 |
| 10 | of | 460 | staff | 533 |
| 11 | very | 445 | hotel | 486 |
| 12 | location | 400 | is | 483 |
| 13 | for | 395 | not | 447 |
| 14 | not | 394 | with | 441 |
| 15 | it | 357 | rooms | 439 |
