# Project 3: Catching Fraudsters — Class Imbalance in Security ML
**Goal:** Build a fraud detection classifier on a severely imbalanced real-world dataset. Demonstrate why accuracy is a broken metric in security contexts, fix it with proper imbalance handling, and evaluate using ROC and Precision-Recall curves. Produce a notebook that shows the problem, solves it, and explains the tradeoffs.

---

## Why am i doing this?
Learning ML in cyber security context, you can add notes / info about usage of what i'm doing in the cyber field

## Context for the Executor

Stan is a Python developer who has completed two ML projects. He can load data, split it, train a Decision Tree and Random Forest, and read a confusion matrix and classification report. He has no prior experience with class imbalance, scaling, ROC curves, or Precision-Recall curves. All of these are new in this project.

The narrative arc of this project has three acts — make sure the notebook tells this story clearly through markdown cells:
1. **The problem:** Extreme imbalance makes accuracy a lie
2. **The fix:** Handle imbalance properly
3. **The evaluation:** Use metrics that actually reflect security reality

Every new concept gets a markdown explanation the first time it appears. Be concrete — use numbers from the actual dataset, not abstract definitions.

---

## Environment Setup

One new package needed:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

No additional packages. Everything is in sklearn.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    ConfusionMatrixDisplay
)
```

---

## Dataset

**Name:** Credit Card Fraud Detection  
**Source:** Kaggle — https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud  
**Download:** Create a free Kaggle account, download `creditcard.csv` (144MB), place in `/data` folder next to notebook.

**What it is:** 284,807 real credit card transactions from European cardholders in September 2013. 492 of them are fraudulent. Each row is one transaction.

**Why this dataset:**  
- 0.172% fraud rate — this is what real-world security data looks like
- It breaks the naive approach you used in Project 2 completely
- The imbalance problem here is the same problem you will face in NSL-KDD in Project 4 (U2R attacks were <0.1% of the data)
- The lessons from this project transfer directly to intrusion detection

**Loading:**

```python
df = pd.read_csv('data/creditcard.csv')
print(df.shape)
print(df.head())
```

---

## Feature Reference

**30 features + 1 target = 31 columns**

- **V1–V28:** PCA-transformed features. The original transaction features (merchant, location, card number, etc.) are anonymised for privacy — PCA was applied to produce these 28 numerical components. They are already scaled. You cannot interpret what V1 or V14 means in plain English. That is intentional.
- **Time:** Seconds elapsed since first transaction in the dataset. Not scaled.
- **Amount:** Transaction amount in euros. Not scaled.
- **Class:** Target column. `0` = legitimate, `1` = fraud.

**Preprocessing note:** V1–V28 are already on a similar scale (output of PCA). Time and Amount are not — Time ranges from 0 to 172,792 seconds and Amount ranges from 0 to ~25,000 euros. These two columns need to be scaled before modelling.

---

## The 10 Tasks

---

### T1 — Load the data and quantify the imbalance

**Operations:** `df.shape`, `df.head()`, `df['Class'].value_counts()`, `df['Class'].value_counts(normalize=True) * 100`  
**Visualization:** Bar chart of class distribution with counts and percentages labelled on each bar  
**Output:** Chart + printed class balance

```python
fraud_count = df['Class'].sum()
legit_count = len(df) - fraud_count
print(f"Legitimate: {legit_count:,} ({legit_count/len(df)*100:.3f}%)")
print(f"Fraudulent: {fraud_count:,}  ({fraud_count/len(df)*100:.3f}%)")
```

**Markdown to write:** Compare this to the breast cancer dataset from Project 2 (~37/63 split). That was mildly imbalanced. This is extreme imbalance — 578 legitimate transactions for every 1 fraudulent transaction. Note that this ratio is realistic: most financial institutions see fraud rates between 0.1% and 1%. The same pattern exists in network traffic — in NSL-KDD, U2R attacks were a fraction of a percent of all connections.

---

### T2 — The naive accuracy trap (demonstrate the problem explicitly)

**Operations:** Build the dumbest possible "model" — one that predicts every single transaction as legitimate regardless of features. Calculate its accuracy.

```python
# Zero-rule classifier: always predict legitimate (0)
y_naive = np.zeros(len(df['Class']))
naive_accuracy = accuracy_score(df['Class'], y_naive)
print(f"Naive classifier accuracy: {naive_accuracy:.4f}")

# Now check recall for fraud
from sklearn.metrics import recall_score
naive_recall = recall_score(df['Class'], y_naive)
print(f"Naive classifier recall (fraud): {naive_recall:.4f}")
print(f"Fraudulent transactions caught: {int(naive_recall * fraud_count)} out of {fraud_count}")
```

**Output:** Printed accuracy and recall

**Markdown to write:** The naive classifier scores ~99.83% accuracy by predicting "legitimate" for every single transaction. It catches exactly 0 fraud cases. This is the accuracy trap — a model can be 99.83% accurate and be completely useless for its intended purpose. This is why you must never use accuracy as your primary metric when classes are imbalanced. From this project forward, accuracy is a secondary metric. Recall for the minority class is what matters.

---

### T3 — Train/test split with stratification

**Operations:** `train_test_split` with `stratify=y`  
**Output:** Print class balance in training set and test set

```python
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training set class balance:")
print(y_train.value_counts())
print(f"\nFraud ratio in training: {y_train.mean():.4f}")
print(f"Fraud ratio in test:     {y_test.mean():.4f}")
```

**Markdown to write:** Explain `stratify=y`. Without it, a random 20% sample might accidentally contain very few or very many fraud cases — because fraud is so rare, random sampling is unreliable. `stratify=y` forces the train/test split to preserve the exact class ratio from the original dataset. This is non-negotiable with imbalanced data. Check that the fraud ratio is approximately equal in both splits.

---

### T4 — Scale Time and Amount

**Operations:** `StandardScaler` on Time and Amount columns only. V1–V28 are already scaled.

```python
scaler = StandardScaler()

# Scale only the two unscaled columns
X_train = X_train.copy()
X_test = X_test.copy()

X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
X_test[['Time', 'Amount']]  = scaler.transform(X_test[['Time', 'Amount']])

# Verify
print("Time stats after scaling (train):")
print(X_train['Time'].describe())
```

**Critical note for executor:** `scaler.fit_transform()` on training data only. `scaler.transform()` (no fit) on test data. Fitting the scaler on test data is a common mistake called data leakage — the model would be using information from the test set during preparation, which invalidates the evaluation.

**Markdown to write:** Explain what StandardScaler does — it subtracts the mean and divides by the standard deviation, so each feature has mean ≈ 0 and standard deviation ≈ 1. Explain why this matters: some models (and the distance calculations that some models use) are sensitive to features being on vastly different scales. A feature ranging from 0–172,000 (Time) would dominate a feature ranging from 0–1 (V1) without scaling. Explain data leakage: the scaler is "fitted" (learns mean and std) only on training data — you then apply those same training statistics to scale the test data. You never let the test set influence any part of preparation.

---

### T5 — Train a naive Random Forest (no imbalance handling)

**Operations:** `RandomForestClassifier(random_state=42)` — no `class_weight` parameter  
**Output:** Accuracy, confusion matrix, classification report — same pattern as Project 2

```python
rf_naive = RandomForestClassifier(random_state=42)
rf_naive.fit(X_train, y_train)
y_pred_naive = rf_naive.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred_naive):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_naive))

cm_naive = confusion_matrix(y_test, y_pred_naive)
ConfusionMatrixDisplay(cm_naive, display_labels=['Legitimate', 'Fraud']).plot(cmap='Blues')
plt.title('Naive Random Forest — Confusion Matrix')
plt.show()
```

**Output:** Accuracy, report, confusion matrix

**Markdown to write:** The model will likely achieve ~99.9% accuracy — even higher than the naive zero-rule classifier from T2. But look at the confusion matrix carefully. How many fraud cases are in the bottom-left cell (predicted legitimate, actually fraud)? Those are missed fraudulent transactions. A model that misses fraud while reporting 99.9% accuracy is not a good fraud detector — it is a good legitimate-traffic detector that occasionally catches fraud by accident.

---

### T6 — Fix it: train a balanced Random Forest

**Operations:** `RandomForestClassifier(class_weight='balanced', random_state=42)` — one parameter change

```python
rf_balanced = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_balanced.fit(X_train, y_train)
y_pred_balanced = rf_balanced.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred_balanced):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_balanced))

cm_balanced = confusion_matrix(y_test, y_pred_balanced)
ConfusionMatrixDisplay(cm_balanced, display_labels=['Legitimate', 'Fraud']).plot(cmap='Oranges')
plt.title('Balanced Random Forest — Confusion Matrix')
plt.show()
```

**Output:** Accuracy, report, confusion matrix

**Markdown to write:** Explain what `class_weight='balanced'` does. sklearn calculates a weight for each class: `n_samples / (n_classes * n_samples_for_class)`. This makes the model treat each fraud sample as if it were ~578x more important than a legitimate sample — compensating for the ratio. The model now pays more attention to the rare class. Accuracy may drop slightly. That is acceptable. The question is: how many more fraud cases does the balanced model catch?

---

### T7 — Side-by-side confusion matrix comparison

**Operations:** Plot both confusion matrices in one figure. Calculate missed fraud in money terms.

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ConfusionMatrixDisplay(cm_naive,    display_labels=['Legit', 'Fraud']).plot(ax=axes[0], cmap='Blues')
ConfusionMatrixDisplay(cm_balanced, display_labels=['Legit', 'Fraud']).plot(ax=axes[1], cmap='Oranges')

axes[0].set_title('Naive Random Forest')
axes[1].set_title('Balanced Random Forest')
plt.tight_layout()
plt.show()

# Financial impact calculation
avg_fraud_amount = df[df['Class'] == 1]['Amount'].mean()
missed_naive    = cm_naive[1][0]     # FN: predicted legit, actually fraud
missed_balanced = cm_balanced[1][0]  # FN: predicted legit, actually fraud

print(f"Average fraudulent transaction: €{avg_fraud_amount:.2f}")
print(f"Naive model    — missed fraud cases: {missed_naive}  (~€{missed_naive * avg_fraud_amount:,.0f})")
print(f"Balanced model — missed fraud cases: {missed_balanced}  (~€{missed_balanced * avg_fraud_amount:,.0f})")
```

**Output:** Side-by-side confusion matrices + financial impact calculation

**Markdown to write:** This is the clearest way to demonstrate the real-world value of handling imbalance properly. The difference in missed fraud cases, multiplied by the average transaction value, translates directly to business cost. In security terms: missed intrusions have a different cost than blocked legitimate traffic — and the question "which error is worse?" always depends on the specific context.

---

### T8 — ROC curve and AUC

**Operations:** `predict_proba()` to get probability scores, then `roc_curve()` and `roc_auc_score()`

```python
# Get fraud probability scores (column 1 = probability of class 1 = fraud)
proba_naive    = rf_naive.predict_proba(X_test)[:, 1]
proba_balanced = rf_balanced.predict_proba(X_test)[:, 1]

fpr_naive,    tpr_naive,    _ = roc_curve(y_test, proba_naive)
fpr_balanced, tpr_balanced, _ = roc_curve(y_test, proba_balanced)

auc_naive    = roc_auc_score(y_test, proba_naive)
auc_balanced = roc_auc_score(y_test, proba_balanced)

plt.figure(figsize=(8, 6))
plt.plot(fpr_naive,    tpr_naive,    label=f'Naive    (AUC = {auc_naive:.4f})')
plt.plot(fpr_balanced, tpr_balanced, label=f'Balanced (AUC = {auc_balanced:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random guessing (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve — Naive vs Balanced')
plt.legend()
plt.tight_layout()
plt.show()
```

**Output:** ROC curve chart with AUC scores in legend

**Markdown to write:** Introduce `predict_proba()` — instead of a hard 0/1 prediction, this returns the model's confidence score for each class. A score of 0.9 means the model is 90% confident the transaction is fraud. The default threshold of 0.5 converts these to hard predictions — any score above 0.5 is predicted fraud.

Explain the ROC curve: each point on the curve represents a different threshold. Moving left (lower threshold) catches more fraud (higher TPR) but also triggers more false alarms (higher FPR). AUC summarises the entire curve as one number — 0.5 is random guessing, 1.0 is perfect. Note: ROC AUC can look high even when the model struggles with extreme imbalance. This is why the Precision-Recall curve in T9 is a more honest metric for this type of problem.

---

### T9 — Precision-Recall curve

**Operations:** `precision_recall_curve()`, `average_precision_score()`

```python
prec_naive,    rec_naive,    _ = precision_recall_curve(y_test, proba_naive)
prec_balanced, rec_balanced, _ = precision_recall_curve(y_test, proba_balanced)

ap_naive    = average_precision_score(y_test, proba_naive)
ap_balanced = average_precision_score(y_test, proba_balanced)

# Baseline: a random classifier's precision = fraud rate in test set
baseline = y_test.mean()

plt.figure(figsize=(8, 6))
plt.plot(rec_naive,    prec_naive,    label=f'Naive    (AP = {ap_naive:.4f})')
plt.plot(rec_balanced, prec_balanced, label=f'Balanced (AP = {ap_balanced:.4f})')
plt.axhline(y=baseline, color='k', linestyle='--', label=f'Random baseline (AP = {baseline:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve — Naive vs Balanced')
plt.legend()
plt.tight_layout()
plt.show()
```

**Output:** PR curve chart

**Markdown to write:** Explain why the PR curve is more informative than ROC for imbalanced data. ROC uses True Negative Rate in its x-axis — when negatives (legitimate transactions) massively outnumber positives (fraud), even a bad model can score well on that axis. The PR curve replaces the x-axis with Recall and the y-axis with Precision — both of which focus on the minority class performance. The random baseline on a PR curve is the fraud rate (~0.0017), not 0.5 like on ROC. A big gap between your model's PR curve and that baseline is meaningful improvement.

The upper-right corner of the PR curve is the ideal — high precision (few false alarms) AND high recall (few missed fraud cases). The curve shows you cannot usually have both at the same time — this is the precision-recall tradeoff.

---

### T10 — Threshold tuning and deployment decision

**Operations:** Loop over thresholds, calculate precision and recall at each. Plot the tradeoff.

```python
thresholds = np.arange(0.1, 0.95, 0.05)
precisions, recalls, f1s = [], [], []

for threshold in thresholds:
    y_pred_thresh = (proba_balanced >= threshold).astype(int)
    precisions.append(precision_score(y_test, y_pred_thresh, zero_division=0))
    recalls.append(recall_score(y_test, y_pred_thresh))
    f1s.append(f1_score(y_test, y_pred_thresh, zero_division=0))

plt.figure(figsize=(10, 5))
plt.plot(thresholds, precisions, label='Precision', marker='o')
plt.plot(thresholds, recalls,    label='Recall',    marker='s')
plt.plot(thresholds, f1s,        label='F1',        marker='^')
plt.xlabel('Decision Threshold')
plt.ylabel('Score')
plt.title('Precision / Recall / F1 vs Decision Threshold (Balanced RF)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Output:** Threshold tradeoff chart

**Mandatory written section — deployment decision:**  
Write a paragraph answering: *"At what threshold would you operate this fraud detection system, and why?"*

The paragraph must address:
1. What happens to precision and recall as you lower the threshold (more alerts, more caught fraud, more false alarms)
2. Who bears the cost of false positives in fraud detection (legitimate customers get declined)
3. Who bears the cost of false negatives (bank absorbs fraud losses)
4. What threshold you would choose and what that means operationally
5. How this same reasoning applies to intrusion detection in Project 4 — the question "what threshold do I operate at?" will reappear there

---

## Output Specification

The completed notebook must have:

- [ ] All 10 tasks with code that runs without errors
- [ ] Minimum 7 visualisations (class distribution, 2x confusion matrices side-by-side, ROC curve, PR curve, threshold tradeoff chart)
- [ ] Every section has a markdown cell above explaining what and why
- [ ] T2 explicitly demonstrates the naive accuracy trap with printed numbers
- [ ] T7 includes the financial impact calculation in euros
- [ ] T10 contains a written deployment decision paragraph addressing threshold choice
- [ ] Notebook runs clean top-to-bottom with `Restart Kernel and Run All`

---

## Common Pitfalls to Flag

1. **Data leakage on scaler:** `fit_transform` on X_train only. `transform` (no fit) on X_test. Fitting the scaler on test data lets test set statistics leak into preparation — the evaluation is no longer honest.
2. **`stratify=y` is non-negotiable with imbalanced data.** Without it, you might get a test set with 0 fraud cases, making evaluation meaningless.
3. **`predict_proba()[:,1]`** — the output has two columns, one per class. Column 1 is the fraud probability. Column 0 is the legitimate probability. Always index with `[:,1]` for the positive class.
4. **`zero_division=0` in metrics** — at very high thresholds, the model may predict no fraud at all, causing division-by-zero in precision. Pass `zero_division=0` to suppress the warning and return 0.
5. **Random Forest on 284k rows is slow.** Training may take 2–5 minutes depending on hardware. This is normal. If it is too slow, reduce `n_estimators=50` for exploration and set it back to 100 for the final run.
6. **Do not drop V1–V28 features.** Even though they are anonymised and uninterpretable, they contain real predictive signal. The model uses them.

---

## New Concepts Introduced (carry forward to Project 4)

| Concept | Definition | Why it matters |
|---|---|---|
| Class imbalance | Severe mismatch in samples per class | Breaks accuracy as a metric — the core challenge in security ML |
| Stratified split | Preserves class ratio in train/test split | Required with imbalanced data |
| StandardScaler | Normalises features to mean=0, std=1 | Prevents large-scale features from dominating models |
| Data leakage | Test set statistics contaminate training preparation | Invalidates evaluation — always fit scalers on training data only |
| `class_weight='balanced'` | Compensates for imbalance by upweighting minority class | Easiest fix for imbalanced classification in sklearn |
| `predict_proba()` | Returns confidence scores instead of hard 0/1 predictions | Required for ROC and PR curves, enables threshold tuning |
| ROC curve / AUC | Plots TPR vs FPR across all thresholds | Standard evaluation tool, but optimistic with extreme imbalance |
| PR curve / Average Precision | Plots Precision vs Recall across all thresholds | More honest metric for imbalanced security data |
| Decision threshold | The confidence cutoff for predicting positive class | Tuning this is an operational decision, not just a model decision |

---

## What Stan Should Be Able to Explain After This Project

- Why a 99.8% accurate model can be completely useless for fraud or attack detection
- What `class_weight='balanced'` does and when to use it
- Why you fit a scaler on training data only (data leakage)
- The difference between ROC AUC and Average Precision, and when each is appropriate
- What adjusting the decision threshold does to precision and recall
- How to frame the threshold choice as an operational decision (cost of false positives vs false negatives in context)

These are the exact questions the VU case study assessment will probe — you will be given a scenario and asked to reason about the security tradeoffs, not just run a model.
