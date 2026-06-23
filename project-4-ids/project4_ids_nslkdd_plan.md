# Project 4: Build an IDS — Full Intrusion Detection System on NSL-KDD
**Goal:** Build a complete, two-phase intrusion detection system on the NSL-KDD dataset. Phase 1 is binary (normal vs. attack). Phase 2 is multiclass (Normal / DoS / Probe / R2L / U2R). Apply everything from Projects 1–3. Produce a clean, documented notebook ready to present and defend at university.

---

## Context for the Executor

Stan has completed three projects. He can:
- Load, clean, and explore tabular data with pandas
- Train Decision Tree and Random Forest classifiers
- Evaluate with confusion matrix, precision, recall, F1, ROC, and PR curves
- Handle class imbalance with `class_weight='balanced'` and stratified splits
- Explain the difference between accuracy and recall in security contexts

He has zero experience with categorical feature encoding, multiclass classification, multiclass confusion matrices, or cross-validation. These are the new concepts in this project. Explain each one when it first appears.

This is the final project. The notebook should be polished — clear markdown cells throughout, all charts properly titled and labelled, and a written threat detection report at the end. It should be something he can open in a university presentation and walk through confidently.

The data (NSL-KDD) is already familiar from Project 1. Do not re-explain what each column means — Stan knows this. Reference Project 1 findings where relevant rather than repeating the EDA.

---

## Environment Setup

No new packages needed beyond what was installed in previous projects.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
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

**Name:** NSL-KDD (same as Project 1)  
**Files:** `KDDTrain+.txt` (training), `KDDTest+.txt` (test) — both from `/data` folder  
**Split strategy:** Train on KDDTrain+, evaluate on KDDTest+. This is the correct way to use this dataset — the test file has a genuinely different distribution and includes some attack types not present in training. That gap is intentional and educationally important.

**Loading — use the same column list and attack_map from Project 1:**

```python
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

attack_map = {
    'normal': 'Normal',
    'neptune': 'DoS', 'back': 'DoS', 'land': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'apache2': 'DoS', 'udpstorm': 'DoS',
    'processtable': 'DoS', 'mailbomb': 'DoS', 'worm': 'DoS',
    'satan': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe',
    'portsweep': 'Probe', 'mscan': 'Probe', 'saint': 'Probe',
    'guess_passwd': 'R2L', 'ftp_write': 'R2L', 'imap': 'R2L', 'phf': 'R2L',
    'multihop': 'R2L', 'warezmaster': 'R2L', 'warezclient': 'R2L', 'spy': 'R2L',
    'xlock': 'R2L', 'xsnoop': 'R2L', 'snmpguess': 'R2L', 'snmpgetattack': 'R2L',
    'httptunnel': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R',
    'perl': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R'
}

train_df = pd.read_csv('data/KDDTrain+.txt', names=columns)
test_df  = pd.read_csv('data/KDDTest+.txt',  names=columns)

train_df['attack_category'] = train_df['label'].map(attack_map)
test_df['attack_category']  = test_df['label'].map(attack_map)

# Some attack types in KDDTest+ don't appear in the attack_map
# These will produce NaN — drop them and note how many
unmapped_train = train_df['attack_category'].isnull().sum()
unmapped_test  = test_df['attack_category'].isnull().sum()
print(f"Unmapped labels — train: {unmapped_train}, test: {unmapped_test}")

train_df = train_df.dropna(subset=['attack_category'])
test_df  = test_df.dropna(subset=['attack_category'])
```

---

## Feature Reference

**Categorical features (3) — must be encoded before any model can use them:**
- `protocol_type`: tcp, udp, icmp
- `service`: ~70 unique values (http, ftp, smtp, etc.)
- `flag`: ~11 unique values (SF, S0, REJ, etc.)

**Numerical features (38):** Already in numeric format. No scaling needed for tree-based models.

**Drop these columns before modelling:**
- `label` — the raw specific attack name, replaced by `attack_category`
- `difficulty` — meta-column, not a real feature

---

## The 12 Tasks

---

### T1 — Reload and reorient

**Operations:** Load both files using the code above. Print shape of train and test sets. Print class distribution in both. Check for unmapped labels.

```python
print(f"Training set: {train_df.shape}")
print(f"Test set:     {test_df.shape}")
print("\nTraining class distribution:")
print(train_df['attack_category'].value_counts())
print("\nTest class distribution:")
print(test_df['attack_category'].value_counts())
```

**Visualization:** Two bar charts side by side — class distribution in train vs test

**Markdown to write:** Note two things explicitly. First: U2R has ~52 training samples. A model cannot reliably learn patterns from 52 examples — this is a hard constraint, not a modelling failure. Second: the test set distribution is different from training (more R2L and U2R in test relative to training). This simulates a real-world condition — attackers evolve, and the traffic you defend against in production looks different from your training data. This is why IDS systems require continuous retraining.

---

### T2 — Encode categorical features

**Operations:** Apply `LabelEncoder` to `protocol_type`, `service`, and `flag`. Fit on training data, transform both train and test.

```python
categorical_cols = ['protocol_type', 'service', 'flag']
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    # Handle unseen categories in test set
    test_df[col] = test_df[col].map(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )
    encoders[col] = le

print("Unique values after encoding:")
for col in categorical_cols:
    print(f"  {col}: {train_df[col].nunique()} values")
```

**Markdown to write:** Explain categorical encoding. sklearn models operate on numbers — you cannot pass the string "tcp" into a Random Forest. `LabelEncoder` converts each unique string to an integer: tcp=0, udp=1, icmp=2 (order depends on alphabetical sorting). For tree-based models (Decision Tree, Random Forest), this is fine — the model splits on thresholds and the integer ordering does not imply any meaningful ranking. For linear models, one-hot encoding is more appropriate, but that is a detail for later.

Explain the unseen category problem: some `service` values in the test set may not appear in the training set at all. The `.map()` approach assigns -1 to any unseen value rather than crashing. Note this explicitly in the output — how many -1 values appear in the test set's `service` column? A high number means the test set has network services the model has never seen.

---

### T3 — Build the feature matrix

**Operations:** Define X (features) and y (labels) for both phases. Prepare binary and multiclass label versions.

```python
feature_cols = [c for c in train_df.columns 
                if c not in ['label', 'difficulty', 'attack_category']]

X_train = train_df[feature_cols]
X_test  = test_df[feature_cols]

# Binary labels: Normal=0, Attack=1
y_train_binary = (train_df['attack_category'] != 'Normal').astype(int)
y_test_binary  = (test_df['attack_category']  != 'Normal').astype(int)

# Multiclass labels: encode category names as integers
le_target = LabelEncoder()
y_train_multi = le_target.fit_transform(train_df['attack_category'])
y_test_multi  = le_target.transform(test_df['attack_category'])

print("Feature matrix shape:", X_train.shape)
print("Class label mapping:")
for i, name in enumerate(le_target.classes_):
    print(f"  {i} = {name}")

print("\nBinary class balance (train):")
print(pd.Series(y_train_binary).value_counts())
```

**Markdown to write:** Explain the two modelling phases. Binary classification is a warm-up — can the model tell attack traffic from normal traffic at all? Multiclass is the real challenge — can it tell *what kind* of attack? Note the label encoding for multiclass: `le_target.classes_` shows which integer maps to which category. Print this and keep it visible — you will need it to interpret the confusion matrix in T9.

---

### T4 — Binary IDS: train and evaluate

**Operations:** Train `RandomForestClassifier(class_weight='balanced', random_state=42)` on binary labels. Full evaluation: accuracy, classification report, confusion matrix.

```python
rf_binary = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)
rf_binary.fit(X_train, y_train_binary)
y_pred_binary = rf_binary.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test_binary, y_pred_binary):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_binary, y_pred_binary, target_names=['Normal', 'Attack']))

cm = confusion_matrix(y_test_binary, y_pred_binary)
ConfusionMatrixDisplay(cm, display_labels=['Normal', 'Attack']).plot(cmap='Blues')
plt.title('Binary IDS — Random Forest')
plt.show()
```

**Output:** Accuracy, report, confusion matrix

**Markdown to write:** Interpret the confusion matrix in operational terms. The bottom-left cell (predicted Normal, actually Attack) represents undetected intrusions — the most dangerous failure mode. The top-right cell (predicted Attack, actually Normal) represents false alarms — legitimate traffic blocked. Calculate: of all actual attacks in the test set, what percentage did the model miss? That number is the model's miss rate, and it is a direct security metric.

---

### T5 — Binary IDS: ROC and PR curves

**Operations:** Same pattern as Project 3 — `predict_proba()`, plot ROC and PR curves.

```python
proba_binary = rf_binary.predict_proba(X_test)[:, 1]

# ROC
fpr, tpr, _ = roc_curve(y_test_binary, proba_binary)
auc = roc_auc_score(y_test_binary, proba_binary)

# PR
precision, recall, _ = precision_recall_curve(y_test_binary, proba_binary)
ap = average_precision_score(y_test_binary, proba_binary)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(fpr, tpr, label=f'AUC = {auc:.4f}')
axes[0].plot([0,1],[0,1],'k--', label='Random')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC — Binary IDS')
axes[0].legend()

axes[1].plot(recall, precision, label=f'AP = {ap:.4f}')
axes[1].axhline(y=y_test_binary.mean(), color='k', linestyle='--', label='Baseline')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall — Binary IDS')
axes[1].legend()

plt.tight_layout()
plt.show()
```

**Output:** ROC and PR curves side by side

**Markdown to write:** Compare these scores to the fraud detection results from Project 3. Binary IDS on NSL-KDD is typically easier than fraud detection — attacks are more common (46% of training data) and more distinctive in their feature patterns. Note the PR curve baseline — it is the attack rate in the test set, which is different from the training set. If the test attack rate differs significantly from training, this tells you something important about how the dataset is designed to simulate realistic conditions.

---

### T6 — Multiclass setup: examine the imbalance

**Operations:** Detailed breakdown of multiclass label distribution in training and test sets. Visualize the imbalance clearly.

```python
train_counts = train_df['attack_category'].value_counts()
test_counts  = test_df['attack_category'].value_counts()

comparison = pd.DataFrame({
    'Train': train_counts,
    'Test':  test_counts
}).fillna(0).astype(int)
comparison['Train %'] = (comparison['Train'] / comparison['Train'].sum() * 100).round(2)
comparison['Test %']  = (comparison['Test']  / comparison['Test'].sum()  * 100).round(2)
print(comparison)

comparison[['Train', 'Test']].plot(kind='bar', figsize=(10, 5), logy=True)
plt.title('Class Distribution: Train vs Test (log scale)')
plt.xlabel('Attack Category')
plt.ylabel('Count (log scale)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
```

**Note on log scale:** Use `logy=True` — without it, U2R (52 samples) is invisible next to DoS (45,927).

**Markdown to write:** This is the central challenge of Project 4. U2R has 52 training samples. The model is being asked to learn a pattern from 52 examples and then recognise it in new data. That is not a modelling problem — it is a data scarcity problem. A well-designed IDS in production would use anomaly detection techniques (flagging anything unusual) for rare attack types rather than supervised classification. The supervised approach works well for DoS and Probe, acceptably for R2L, and poorly for U2R — and that is an honest result, not a failure.

---

### T7 — Multiclass IDS: naive model (no imbalance handling)

**Operations:** Train `RandomForestClassifier(random_state=42)` — no `class_weight`. Full evaluation.

```python
rf_multi_naive = RandomForestClassifier(n_estimators=100, random_state=42)
rf_multi_naive.fit(X_train, y_train_multi)
y_pred_multi_naive = rf_multi_naive.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test_multi, y_pred_multi_naive):.4f}")
print("\nClassification Report:")
print(classification_report(
    y_test_multi, y_pred_multi_naive,
    target_names=le_target.classes_
))
```

**Output:** Classification report with per-class metrics

**Markdown to write:** Introduce macro vs weighted average in the classification report. Weighted average weights each class by its sample count — it is pulled heavily toward DoS (the dominant class). A model that catches all DoS attacks but misses all U2R attacks will still show a strong weighted average. Macro average treats all classes equally — DoS and U2R count the same. In security, macro average is the more honest metric: missing a U2R attack (privilege escalation) is operationally just as serious as missing a DoS attack, even though U2R is rare.

---

### T8 — Multiclass IDS: balanced model

**Operations:** `RandomForestClassifier(class_weight='balanced', random_state=42)` — same evaluation

```python
rf_multi_balanced = RandomForestClassifier(
    class_weight='balanced', n_estimators=100, random_state=42
)
rf_multi_balanced.fit(X_train, y_train_multi)
y_pred_multi_balanced = rf_multi_balanced.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test_multi, y_pred_multi_balanced):.4f}")
print("\nClassification Report:")
print(classification_report(
    y_test_multi, y_pred_multi_balanced,
    target_names=le_target.classes_
))
```

**Output:** Classification report

**Markdown to write:** Compare the macro avg recall between naive and balanced. Overall accuracy likely drops slightly — that is the cost of forcing the model to pay attention to rare classes. The question is whether recall on R2L and U2R improved. Note: even with `class_weight='balanced'`, U2R recall may remain near zero. 52 training samples is a fundamental limit that weighting alone cannot fix. Acknowledge this honestly — it is a real IDS limitation and a key point for oral Q&A.

---

### T9 — Multiclass confusion matrix: interpret the confusion patterns

**Operations:** 5x5 confusion matrix as a heatmap. Normalise by row (shows percentage of each true class that gets classified as each predicted class).

```python
cm_multi = confusion_matrix(y_test_multi, y_pred_multi_balanced)
cm_norm  = cm_multi.astype('float') / cm_multi.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_norm,
    annot=True, fmt='.2f',
    xticklabels=le_target.classes_,
    yticklabels=le_target.classes_,
    cmap='Blues'
)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Multiclass Confusion Matrix (row-normalised)\nBalanced Random Forest')
plt.tight_layout()
plt.show()

# Also print raw counts for reference
print("Raw confusion matrix:")
print(pd.DataFrame(cm_multi, index=le_target.classes_, columns=le_target.classes_))
```

**Output:** Normalised heatmap + raw count table

**Markdown to write — this is the most important analytical cell in the project:**

Read the confusion matrix out loud, row by row:
- The diagonal is correct predictions. Everything off-diagonal is an error.
- Row = true class. Column = predicted class. Cell value = fraction of true samples classified as predicted class.
- For each off-diagonal cell that is large, write one sentence explaining *why* those two classes are confused from a network traffic perspective. For example: "R2L is often confused with Normal because R2L attacks are designed to look like legitimate remote access — they intentionally mimic normal traffic patterns to avoid detection."

Write this interpretation for every notable confusion in the matrix. This is the exact type of reasoning VU's oral Q&A will test.

---

### T10 — Per-attack-type performance deep dive

**Operations:** Extract per-class precision, recall, and F1 from the classification report into a DataFrame. Visualise as a grouped bar chart.

```python
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, support = precision_recall_fscore_support(
    y_test_multi, y_pred_multi_balanced,
    labels=range(len(le_target.classes_))
)

perf_df = pd.DataFrame({
    'Attack Category': le_target.classes_,
    'Precision': precision,
    'Recall':    recall,
    'F1':        f1,
    'Support':   support
})

print(perf_df.to_string(index=False))

perf_df.set_index('Attack Category')[['Precision', 'Recall', 'F1']].plot(
    kind='bar', figsize=(12, 5)
)
plt.title('Per-Class Performance — Balanced Random Forest')
plt.ylabel('Score')
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
```

**Output:** Table + grouped bar chart

**Markdown to write:** Rank the attack categories from easiest to hardest for the model to detect. DoS will be near-perfect (highly distinctive traffic patterns — floods of identical packets). Probe will be good (scanning behaviour is distinctive). R2L will be moderate (designed to blend in). U2R will be poor (too few training samples, designed to look like normal user behaviour). Write one sentence per category explaining *why* the model struggles or succeeds — connect the metric to the real-world behaviour of the attack.

---

### T11 — Model comparison: Decision Tree vs Random Forest

**Operations:** Train a `DecisionTreeClassifier(class_weight='balanced', random_state=42)` and compare per-class recall against the Random Forest.

```python
dt_multi = DecisionTreeClassifier(class_weight='balanced', random_state=42)
dt_multi.fit(X_train, y_train_multi)
y_pred_dt_multi = dt_multi.predict(X_test)

_, recall_dt, _, _ = precision_recall_fscore_support(
    y_test_multi, y_pred_dt_multi,
    labels=range(len(le_target.classes_))
)
_, recall_rf, _, _ = precision_recall_fscore_support(
    y_test_multi, y_pred_multi_balanced,
    labels=range(len(le_target.classes_))
)

comparison_df = pd.DataFrame({
    'Attack Category': le_target.classes_,
    'DT Recall':       recall_dt,
    'RF Recall':       recall_rf
})

comparison_df.set_index('Attack Category').plot(kind='bar', figsize=(12, 5))
plt.title('Recall by Attack Category: Decision Tree vs Random Forest')
plt.ylabel('Recall')
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.legend()
plt.tight_layout()
plt.show()

print(comparison_df.to_string(index=False))
```

**Output:** Bar chart + table

**Markdown to write:** Focus on recall differences per category. For common attack types (DoS, Probe), the difference between Decision Tree and Random Forest is likely small — both learn the patterns well. For rare types (R2L, U2R), the Random Forest may be more stable due to its ensemble averaging. Note also cross-validation as a more rigorous comparison method — a single train/test split can favour one model by luck of the split. `cross_val_score` with `StratifiedKFold` averages performance across multiple splits:

```python
# Brief cross-validation demo — runs on training data only
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_rf = cross_val_score(rf_multi_balanced, X_train, y_train_multi, cv=cv, scoring='f1_macro')
print(f"RF 5-fold CV macro F1: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})")
```

Explain cross-validation: instead of one 80/20 split, the data is divided into 5 folds. The model trains on 4 folds and tests on the 5th, rotating which fold is the test set. The result is an average performance across 5 different test sets — more reliable than a single split. The standard deviation tells you how consistent the model is.

---

### T12 — Feature importance and written IDS report

**Operations:** Feature importance from the balanced Random Forest. Plot top 20 features.

```python
importances = pd.Series(rf_multi_balanced.feature_importances_, index=feature_cols)
top20 = importances.sort_values(ascending=False).head(20)

plt.figure(figsize=(10, 7))
top20.sort_values().plot(kind='barh', color='steelblue')
plt.title('Random Forest — Top 20 Feature Importances (Multiclass IDS)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()
```

**Mandatory final section — written IDS threat detection report:**

Write this as a structured markdown section in the notebook. It should answer the following questions in prose (not bullet points):

**1. What does this IDS detect reliably?**
Name the attack categories with high recall (>0.85) and explain why the model detects them well — what features distinguish them from normal traffic?

**2. What does this IDS miss, and why?**
Name the categories with low recall. Explain the data scarcity problem for U2R. Explain why R2L is designed to evade detection. Be specific — "the model misses X% of R2L attacks in the test set" using actual numbers from T10.

**3. What is the false alarm rate, and what does that mean operationally?**
Read the false positive rate from the confusion matrix (legitimate traffic predicted as attack). In a production network environment, every false positive means a legitimate connection was flagged or blocked. High precision on the Normal class means fewer disruptions; high recall on attack classes means fewer missed intrusions. State where this model sits on that tradeoff.

**4. What would you do to improve detection of rare attack types?**
Suggest at least two realistic approaches: collecting more labelled U2R data, using anomaly detection (flag anything that deviates significantly from normal) rather than supervised classification for rare classes, or using a hybrid system where supervised classification handles DoS/Probe and anomaly detection handles U2R/R2L.

**5. Ethical considerations**
One paragraph on the human and ethical implications of deploying an IDS. False positives block legitimate users — in a workplace, this means disrupted productivity; in a hospital network, potentially something more serious. Who decides the acceptable false positive rate? Who is accountable when the model is wrong? How should privacy be handled — the model processes all network traffic, including personal communications?

---

## Output Specification

The completed notebook must have:

- [ ] Both train and test files loaded correctly, unmapped labels handled and counted
- [ ] Categorical features encoded with unseen-category handling
- [ ] Phase 1 (binary): confusion matrix, ROC, PR curve
- [ ] Phase 2 (multiclass): naive model, balanced model, normalised 5x5 confusion matrix
- [ ] T9 confusion matrix interpretation — row-by-row written analysis in markdown
- [ ] T10 per-class performance bar chart
- [ ] T11 model comparison with cross-validation demo
- [ ] T12 feature importance chart
- [ ] T12 written IDS report — all 5 sections answered in prose with actual numbers
- [ ] Notebook runs clean top-to-bottom with `Restart Kernel and Run All`
- [ ] All charts have titles and axis labels

---

## Common Pitfalls to Flag

1. **Unseen categories in test set.** The `service` and `flag` columns in KDDTest+ contain values not present in KDDTrain+. The `.map()` approach in T2 assigns -1 to these. Do not use `.fit_transform()` on the test set — that would create a different encoding than the training set, making the features incompatible.
2. **`le_target.classes_` order matters.** The confusion matrix rows and columns follow the order of `le_target.classes_` — always print this mapping before plotting the confusion matrix.
3. **Weighted vs macro average.** Always read and report both in the classification report. For security contexts, argue macro average in your written report.
4. **Row normalisation on confusion matrix.** Raw counts make rare classes invisible. Always normalise by row (`/ cm.sum(axis=1)[:, np.newaxis]`) when comparing across imbalanced classes.
5. **Cross-validation is slow on 125k rows.** The 5-fold CV in T11 will take several minutes. This is expected — do not interrupt it.
6. **Do not include `label` or `difficulty` in feature_cols.** The `label` column is the raw attack name — a direct proxy for the target, which would give artificially perfect accuracy. The `difficulty` column is a meta-label, not a real feature.

---

## New Concepts Introduced in This Project

| Concept | Definition | Why it matters |
|---|---|---|
| Categorical encoding | Converting string features to integers | sklearn cannot process strings — encoding is mandatory preprocessing |
| Unseen categories | Test set contains values not in training set | Real-world data always has new values — handle gracefully |
| Multiclass classification | Predicting one of >2 classes | IDS must distinguish *what kind* of attack, not just whether |
| Macro average | Mean metric across all classes equally weighted | More honest than weighted average for imbalanced security data |
| Weighted average | Mean metric weighted by class frequency | Inflated by dominant class — use cautiously |
| Row-normalised confusion matrix | Each row sums to 1.0 — shows error rates per class | Reveals confusion patterns invisible in raw count matrices |
| Cross-validation | Average performance across multiple train/test splits | More reliable than a single split — reveals model stability |
| `StratifiedKFold` | Cross-validation that preserves class ratio in each fold | Required for imbalanced multiclass problems |

---

## What Stan Should Be Able to Explain After This Project

- Why categorical features must be encoded and how `LabelEncoder` works
- The difference between macro and weighted average, and which to use in security
- How to read a 5x5 confusion matrix and identify which attack types are confused with each other
- Why U2R is the hardest attack category to detect (data scarcity + behavioural similarity to normal)
- What cross-validation is and why it gives a more reliable performance estimate than a single split
- The operational tradeoff between false positives (disrupted legitimate traffic) and false negatives (missed attacks)
- What a real IDS deployment would need beyond this model (continuous retraining, anomaly detection for rare types, human review workflows)

These are the exact questions the VU automated threat detection assignment and case study assessments will test — both technically and in oral Q&A.
