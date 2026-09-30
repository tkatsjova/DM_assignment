# Data statistics

## Overview

| Metric | Value |
|---|---:|
| Reviews | 2000 |
| Hotels | 100 |
| Cities | 10 |
| Duplicate texts | 0 |

## Class x sentiment

| class | neg | pos | total |
|---|---:|---:|---:|
| ai | 500 | 500 | 1000 |
| human | 500 | 500 | 1000 |
| total | 1000 | 1000 | 2000 |

## Review score by class and sentiment

| class | sentiment | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ai | neg | 144 | 209 | 128 | 19 | 0 | 0 | 0 | 0 | 0 |
| ai | pos | 0 | 0 | 0 | 0 | 0 | 7 | 342 | 111 | 40 |
| human | neg | 1 | 187 | 280 | 11 | 21 | 0 | 0 | 0 | 0 |
| human | pos | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 306 | 192 |

## Review length in words

| class | sentiment | mean | median | std | min | max |
|---|---|---:|---:|---:|---:|---:|
| ai | neg | 59.6 | 57 | 28.7 | 9 | 172 |
| ai | pos | 51.3 | 48 | 27.3 | 5 | 167 |
| human | neg | 62.4 | 41.5 | 65.3 | 3 | 548 |
| human | pos | 34.8 | 24 | 34.6 | 2 | 234 |

## Missing parts and non-ASCII characters

| class | Empty upside | Empty downside | Non-ASCII chars | Non-ASCII share |
|---|---:|---:|---:|---:|
| ai | 10 | 52 | 64 | 0.064 |
| human | 67 | 174 | 177 | 0.177 |
