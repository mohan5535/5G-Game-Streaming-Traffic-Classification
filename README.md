# 5G Game-Streaming Traffic: Packet Direction Classification

Machine learning project that classifies packets from a 5G game-streaming session (GeForce Now) as **uplink** or **downlink**, using only packet size and timing.

## Overview

| Item | Details |
|---|---|
| Data | Merged GeForce Now packet captures (Wireshark-style CSV) |
| Size | 7,338,316 packets, 8 columns |
| Task | Binary classification: uplink vs downlink |
| Final model | Tuned Decision Tree |
| Test macro F1 | **0.9780** |
| Uplink precision / recall | 0.9490 / 0.9734 |

## Dataset

The merged file contains these columns: `No.`, `Time`, `Source`, `Destination`, `Protocol`, `Length`, `Info`, `source_file`.

- No duplicate rows and no missing values.
- The raw data is not included in this repository because of its size. Set the path to your own copy of `GeForce_Now_merged.csv` in the notebook.

### Why packet direction is the target

The dataset has no label column and covers a single service. The `Protocol` column is not a usable target: 99.9% of packets are UDP (7,330,803 of 7,338,316), with only a few thousand TCP, TLSv1.2 and RTCP packets. Packet direction is a balanced enough, meaningful alternative:

- Client `10.215.173.1` sending = **uplink** (939,396 packets, 12.8%)
- Servers `112.217.128.200` and `112.217.128.202` sending = **downlink** (6,398,920 packets, 87.2%)

## Approach

### Features

| Feature | Description |
|---|---|
| `Length` | Packet length in bytes |
| `iat` | Inter-arrival time since the previous packet (seconds) |
| `len_roll_mean` | Rolling mean of `Length` over the last 10 packets |
| `len_roll_std` | Rolling standard deviation of `Length` over the last 10 packets |
| `iat_roll_mean` | Rolling mean of `iat` over the last 10 packets |

Excluded on purpose: IP addresses (they define the label), `Info` (leaks protocol hints), `No.`, `Time` and `source_file`.

### Train/test split

- **Time-based split:** the first 80% of the capture is the training pool, the last 20% is the test pool.
- Because of memory limits, 300,000 random packets are sampled from the training pool and 200,000 from the test pool (`random_state=42`).
- Uplink share: 13.2% in training, 11.5% in testing.
- Class imbalance is handled with `class_weight="balanced"`.

### Models

Logistic Regression (with standard scaling), Decision Tree, Random Forest (50 trees) and Histogram Gradient Boosting. The two best models were validated with 5-fold `TimeSeriesSplit` cross-validation, and the Decision Tree was tuned with `RandomizedSearchCV` (15 iterations).

Best parameters: `criterion="entropy"`, `max_depth=20`, `min_samples_leaf=1`.

## Results

Test set (200,000 packets, time-based split):

| Model | Accuracy | Macro F1 | Uplink Precision | Uplink Recall | Uplink F1 |
|---|---|---|---|---|---|
| Decision Tree (tuned) | **0.9910** | **0.9780** | **0.9490** | 0.9734 | **0.9611** |
| Decision Tree | 0.9901 | 0.9762 | 0.9287 | **0.9894** | 0.9581 |
| Random Forest | 0.9900 | 0.9760 | 0.9294 | 0.9876 | 0.9576 |
| Hist Gradient Boosting | 0.9878 | 0.9710 | 0.9139 | 0.9866 | 0.9489 |
| Logistic Regression | 0.7311 | 0.6276 | 0.2846 | 0.8905 | 0.4314 |

Time-series cross-validation (macro F1, 5 folds):

| Model | Mean | Std |
|---|---|---|
| Decision Tree | 0.9735 | 0.0045 |
| Random Forest | 0.9718 | 0.0043 |
| Decision Tree (tuned, best CV) | 0.9777 | - |

Confusion matrix of the tuned Decision Tree:

| | Predicted downlink | Predicted uplink |
|---|---|---|
| **Actual downlink** | 175,890 | 1,198 |
| **Actual uplink** | 609 | 22,303 |

Feature importance: `Length` 0.848, `iat` 0.130, `iat_roll_mean` 0.008, `len_roll_mean` 0.008, `len_roll_std` 0.005.

### Key findings

- Tree-based models perform almost identically (macro F1 0.971 to 0.978). A simple, fast Decision Tree is enough.
- Logistic Regression performs poorly, so the relationship between the features and direction is non-linear.
- Cross-validation (0.9777) and test (0.9780) scores agree, which shows no sign of overfitting.
- Tuning gives only a small gain (macro F1 0.9762 to 0.9780), which is within cross-validation variation. The tuned model should be seen as comparable to the baseline, with a precision/recall trade-off (precision up, recall down).
- Packet length dominates. The three rolling features together contribute about 0.02, so two features would likely be enough.

## Limitations

- The data covers a single service (GeForce Now), so this is direction classification, not service classification.
- The task is relatively easy because downlink video packets are large and uplink control packets are small.
- Classification is per packet, not per flow.
- Models were trained on samples (300k/200k) of a 7.3M-packet capture.

## Future work

- Merge captures from other services and classify the service itself.
- Add flow-level or window-level features.
- Try deep learning on packet sequences.

## Getting started

### Requirements

```
python 3.10+
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Run

1. Open the Jupyter notebook and set `path` to your copy of `GeForce_Now_merged.csv`.
2. Run all cells (loading takes about 45 seconds; the full 7.3M-row file needs several GB of RAM, so close other programs).

### Use the saved model

```python
import joblib
import pandas as pd

saved = joblib.load("direction_classifier.joblib")
model, features, labels = saved["model"], saved["features"], saved["labels"]

# features: Length, iat, len_roll_mean, len_roll_std, iat_roll_mean
X_new = pd.DataFrame([[1332, 0.0001, 1300.5, 45.2, 0.0002]], columns=features)
print(labels[int(model.predict(X_new)[0])])
```

## Files

| File | Description |
|---|---|
| `direction_classifier.joblib` | Saved model with feature list, client IP and class labels |
| `model_comparison.csv` | Model comparison table |
| `model_comparison.png` | Model comparison chart |
| Jupyter notebook | Full pipeline: loading, cleaning, features, training, evaluation |
