from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split


RAW_DIR = Path("dataset/raw")
OUTPUT_DIR = Path("dataset/multiclass_33_v1")


def load_raw_files(raw_dir: Path) -> pd.DataFrame:
    csv_files = sorted(raw_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    print("Raw files found:")
    for file in csv_files:
        print(f" - {file.name}")

    dataframes = []
    for file in csv_files:
        print(f"\nLoading {file.name}...")
        df = pd.read_csv(file)
        print(f"Shape of {file.name}: {df.shape}")
        dataframes.append(df)

    merged_df = pd.concat(dataframes, ignore_index=True)
    return merged_df


def inspect_basic_info(df: pd.DataFrame) -> None:
    print("\n===== BASIC DATASET INFO =====")
    print(f"Total shape: {df.shape}")

    print("\nColumns:")
    print(list(df.columns))

    if "Label" not in df.columns:
        raise ValueError("Target column 'Label' was not found in the dataset.")

    print("\n===== ORIGINAL 33-CLASS DISTRIBUTION =====")
    print(df["Label"].value_counts())

    print("\n===== ORIGINAL 33-CLASS PERCENTAGES =====")
    print(df["Label"].value_counts(normalize=True) * 100)


def build_proportional_33class_dataset(
    df: pd.DataFrame,
    benign_label: str = "BENIGN",
    label_col: str = "Label",
    random_state: int = 42
) -> pd.DataFrame:
    benign_df = df[df[label_col] == benign_label].copy()
    attack_df = df[df[label_col] != benign_label].copy()

    benign_count = len(benign_df)

    sampled_attack_df = (
        attack_df
        .groupby(label_col, group_keys=False)
        .apply(
            lambda group: group.sample(
                n=max(1, round(len(group) / len(attack_df) * benign_count)),
                random_state=random_state
            )
        )
        .reset_index(drop=True)
    )

    final_df = pd.concat([benign_df, sampled_attack_df], ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return final_df


def save_33class_artifacts(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "full_multiclass_33.csv"
    df.to_csv(output_path, index=False)

    feature_cols = [col for col in df.columns if col != "Label"]
    with open(OUTPUT_DIR / "feature_list.json", "w") as f:
        json.dump(feature_cols, f, indent=4)

    unique_labels = sorted(df["Label"].unique())
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    with open(OUTPUT_DIR / "label_mapping.json", "w") as f:
        json.dump(label_mapping, f, indent=4)

    class_dist = df["Label"].value_counts()
    class_dist.to_csv(OUTPUT_DIR / "class_distribution.csv")

    metadata = {
        "dataset_name": "multiclass_33_v1",
        "num_rows": len(df),
        "num_features": len(feature_cols),
        "target_column": "Label",
        "classes": unique_labels,
        "description": "33-class CICIoT2023 dataset with proportional attack downsampling to reduce attack dominance"
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nSaved dataset to {output_path}")


def split_and_save_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> None:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["Label"]
    )

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print("\n===== TRAIN SPLIT DISTRIBUTION =====")
    print(train_df["Label"].value_counts())

    print("\n===== TEST SPLIT DISTRIBUTION =====")
    print(test_df["Label"].value_counts())

    print(f"\nSaved train split to {OUTPUT_DIR / 'train.csv'}")
    print(f"Saved test split to {OUTPUT_DIR / 'test.csv'}")


def main() -> None:
    df = load_raw_files(RAW_DIR)
    inspect_basic_info(df)

    sampled_df = build_proportional_33class_dataset(df)

    print("\n===== SAMPLED 33-CLASS DISTRIBUTION =====")
    print(sampled_df["Label"].value_counts())

    print("\n===== SAMPLED 33-CLASS PERCENTAGES =====")
    print(sampled_df["Label"].value_counts(normalize=True) * 100)

    save_33class_artifacts(sampled_df)
    split_and_save_dataset(sampled_df)


if __name__ == "__main__":
    main()