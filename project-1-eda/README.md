# Project 1 - NSL-KDD Exploratory Data Analysis

## Dataset

**NSL-KDD** - network connection records, each row is one connection with 43 features (protocol type, duration, byte counts, error rates, etc.) plus a label and a difficulty score.

- `data/KDDTrain+.txt` - 125,973 records, 44 columns
- Labels mapped into 5 categories: Normal, DoS, Probe, R2L, U2R

## Questions Answered

1. What does the dataset look like? (shape, dtypes, basic statistics)
2. How balanced are the attack categories?
3. Which specific attack types exist and how many of each?
4. What types of features does the dataset have? (categorical, binary, numerical)
5. Are there any missing values?
6. How do key traffic features differ between normal and attack traffic?
7. Which protocols are used and how do they relate to attacks?
8. Which services are most targeted in attacks?
9. How correlated are the numerical features?
10. What is the statistical fingerprint of each attack category?

## Key Findings

- Normal traffic is 53.5% of the dataset; U2R is 0.04% (52 connections) - accuracy is a misleading metric here
- Neptune alone makes up 32.7% of all connections and 70% of all attacks
- 3 categorical features, 5 binary, 35 numerical; categorical features need encoding before modeling
- `src_bytes` ranges from 0 to 1.4 GB - log transform required for visualization
- No missing values; dataset has been pre-cleaned
- DoS connections have near-zero bytes and duration with very high connection counts, consistent with SYN flooding
- TCP dominates attack traffic; ICMP is heavily used for probe (ping sweeps) and DoS (smurf)
- `private` service and ICMP echo types (`eco_i`, `ecr_i`) dominate the attack service distribution
- `num_root` ↔ `num_compromised` correlation: 0.998; error rate features are largely redundant with each other
- U2R feature profile is nearly identical to Normal - it produces no distinct network pattern

## Files

```
project-1-eda/
├── notebook.ipynb
└── README.md
```
