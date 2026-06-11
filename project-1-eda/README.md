# Project 1 — NSL-KDD Exploratory Data Analysis

## Overview

This project performs a structured exploratory analysis of the NSL-KDD dataset, a standard benchmark for network intrusion detection research. The dataset contains labeled network connection records across five traffic categories: Normal, DoS, Probe, R2L, and U2R.

The analysis answers 10 specific questions about the dataset before any modeling work begins — understanding the data is a prerequisite to building anything reliable on top of it.

## Dataset

**NSL-KDD** — an improved version of the original KDD Cup 1999 dataset that removes duplicate records and rebalances class representation.

- `KDDTrain+.txt` — 125,973 records
- `KDDTest+.txt` — 22,544 records
- 41 features + 1 label column + 1 difficulty score column

Place both files in `../data/` before running the notebook.

## 10 Questions Answered

1. What is the class distribution across attack types?
2. How severe is the class imbalance between Normal and attack traffic?
3. Which features have the highest variance, and why?
4. Are there features with near-zero variance that can be dropped?
5. What does the correlation structure look like across numeric features?
6. How do attack types differ in their feature profiles?
7. Are there duplicate or near-duplicate records in the training set?
8. What is the distribution of protocol types (TCP, UDP, ICMP) across classes?
9. Which features are most correlated with the attack label?
10. How does the test set distribution differ from training — is it representative?

## Key Findings

- The dataset is heavily skewed toward DoS attacks (~54% of training records); U2R and R2L together account for under 1%.
- Several features (`num_outbound_cmds`, `is_host_login`) have near-zero variance and are candidates for removal.
- `src_bytes`, `dst_bytes`, and `count` show strong separation between Normal and attack classes.
- The test set has a notably different class distribution than training — models that optimize for training accuracy will likely underperform on the test set.
- TCP dominates as the protocol for most attack categories; ICMP is strongly associated with DoS (smurf) attacks.

## How to Run

```bash
# From the project root
cd project-1-eda
jupyter notebook notebook.ipynb
```

Requires the NSL-KDD dataset files to be in `../data/`. See the root [README](../README.md) for download instructions.

## Files

```
project-1-eda/
├── notebook.ipynb    # Main analysis notebook
└── README.md         # This file
```
