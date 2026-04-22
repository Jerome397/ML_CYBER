import json
import os

import pandas as pd
# import matplotlib.pyplot as plt


# =========================================================
# 1) LOAD DATASET
# =========================================================
df = pd.read_csv("dataset/Merged01.csv")


# =========================================================
# 2) CONVERT LABELS TO BINARY
#    BENIGN stays BENIGN
#    every other class becomes ATTACK
# =========================================================
df["Label"] = df["Label"].apply(lambda x: "BENIGN" if x == "BENIGN" else "ATTACK")


# =========================================================
# 3) BASIC EDA ON ORIGINAL DATASET
# =========================================================
print("=== ORIGINAL DATASET ===")
print("Class distribution:")
print(df["Label"].value_counts())

print("\nDuplicates:", df.duplicated().sum())
print("\nMissing values per column:")
print(df.isnull().sum())

print("\nLow-unique columns:")
print(df.nunique().sort_values().head(10))


# =========================================================
# 4) PLOT 1: ORIGINAL DATASET DISTRIBUTION
#    Disabled to avoid pop-up every run
# =========================================================
# df["Label"].value_counts().plot(kind="bar", figsize=(6, 4))
# plt.title("Original Dataset Distribution")
# plt.xlabel("Class")
# plt.ylabel("Count")
# plt.show()


# =========================================================
# 5) REMOVE DUPLICATES
# =========================================================
df = df.drop_duplicates()

print("\n=== AFTER REMOVING DUPLICATES ===")
print("Shape:", df.shape)
print("Class distribution:")
print(df["Label"].value_counts())


# =========================================================
# 6) PLOT 2: DISTRIBUTION AFTER REMOVING DUPLICATES
#    Disabled to avoid pop-up every run
# =========================================================
# df["Label"].value_counts().plot(kind="bar", figsize=(6, 4))
# plt.title("Distribution After Removing Duplicates")
# plt.xlabel("Class")
# plt.ylabel("Count")
# plt.show()


# =========================================================
# 7) SEPARATE THE TWO CLASSES
# =========================================================
df_attack = df[df["Label"] == "ATTACK"]
df_benign = df[df["Label"] == "BENIGN"]


# =========================================================
# 8) CREATE A BALANCED SAMPLE
#    Take the same number of rows from each class
# =========================================================
n = 10000

df_sample = pd.concat([
    df_attack.sample(n=n, random_state=42),
    df_benign.sample(n=n, random_state=42)
])

# Shuffle so rows are mixed
df_sample = df_sample.sample(frac=1, random_state=42).reset_index(drop=True)

print("\n=== BALANCED SAMPLE ===")
print("Class distribution:")
print(df_sample["Label"].value_counts())
print("Shape:", df_sample.shape)


# =========================================================
# 9) PLOT 3: BALANCED SAMPLE DISTRIBUTION
#    Disabled to avoid pop-up every run
# =========================================================
# df_sample["Label"].value_counts().plot(kind="bar", figsize=(6, 4))
# plt.title("Balanced Sample Distribution")
# plt.xlabel("Class")
# plt.ylabel("Count")
# plt.show()


# =========================================================
# 10) SAVE OUTPUT FILES
# =========================================================
df_sample.to_csv("clean_sample.csv", index=False)

features = {
    "target": "Label",
    "features": [col for col in df_sample.columns if col != "Label"]
}

with open("feature_list.json", "w") as f:
    json.dump(features, f, indent=4)


    