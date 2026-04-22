# Multiclass Dataset — Usage Guide

## Overview
This dataset (`dataset/multiclass_v1`) is a grouped multiclass version of CICIoT2023 designed for IoT intrusion detection.  
It enables classification of network traffic as benign or belonging to a specific attack category.

---

## Dataset Location
dataset/multiclass_v1/

Files:
- train.csv → training split (80%)
- test.csv → testing split (20%)
- full_multiclass.csv → full dataset before split

---

## Target Column
Final_Label

Classes:
- BENIGN
- DDoS
- DoS
- Mirai
- Recon
- Spoofing
- Rare (merged class for very small attack types like Web and BruteForce)

---

## Feature Columns
Defined in:
feature_list.json

Important:
- Do NOT use `Label`, `Grouped_Label`, or `Final_Label` as input features
- Always use `feature_list.json` to select features

---

## Label Encoding
Defined in:
label_mapping.json

Example:
{
  "BENIGN": 0,
  "DDoS": 1,
  "DoS": 2,
  "Mirai": 3,
  "Rare": 4,
  "Recon": 5,
  "Spoofing": 6
}

---

## Recommended Usage
Use the provided loader to ensure consistency:

```python
from data.multiclass_loader import load_multiclass_xy

X_train, y_train = load_multiclass_xy("train")
X_test, y_test = load_multiclass_xy("test")

This automatically:

- loads the correct split
- selects features
- encodes labels

Dataset Characteristics
- Attack dominance has been reduced through proportional downsampling
- Relative proportions between attack categories are preserved
- Very small classes (Web, BruteForce) are merged into Rare

Important Notes
- The dataset is still intentionally imbalanced
- The Rare class has very few samples → metrics may be unstable

Focus evaluation on:

- macro F1-score
- per-class recall
-confusion matrix

Do NOT rely only on accuracy.