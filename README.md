# ml-cybersec

A 14-day structured self-study plan applying machine learning fundamentals to cybersecurity problems. Each project targets a distinct problem domain — exploratory analysis, classification, fraud detection, and intrusion detection — using real-world datasets and progressively more complex techniques. The goal is to build a working understanding of how ML pipelines are constructed, evaluated, and interpreted in a security context.

## Projects

| # | Name | Dataset | Key Concepts | Status |
|---|------|---------|--------------|--------|
| 1 | NSL-KDD Exploratory Data Analysis | NSL-KDD | Feature distributions, class imbalance, correlation analysis, attack type breakdown | Complete |
| 2 | Network Intrusion Classifier | NSL-KDD | Binary & multiclass classification, train/test split, precision/recall, confusion matrix | In Progress |
| 3 | Fraud Detection | Credit Card Fraud (Kaggle) | Imbalanced learning, SMOTE, threshold tuning, ROC-AUC | Planned |
| 4 | Intrusion Detection System | NSL-KDD / CICIDS | Feature engineering, ensemble methods, real-time scoring pipeline | Planned |

## Getting Started

### Download the NSL-KDD Dataset

1. Go to [https://www.unb.ca/cic/datasets/nsl.html](https://www.unb.ca/cic/datasets/nsl.html)
2. Download `KDDTrain+.txt` and `KDDTest+.txt`
3. Place both files in the `data/` directory at the project root

The `data/` directory is not tracked by git. You must download the files manually before running any notebooks.

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run a Notebook

```bash
cd project-1-eda
jupyter notebook notebook.ipynb
```

## Tech Stack

- **Python 3.10+**
- **pandas** — data loading and manipulation
- **numpy** — numerical operations
- **matplotlib / seaborn** — visualization
- **scikit-learn** — preprocessing, modeling, evaluation
- **Jupyter** — interactive notebooks
