from pathlib import Path
import pandas as pd
import json
from sklearn.model_selection import train_test_split


RAW_DIR = Path("dataset/raw")
OUTPUT_DIR = Path("dataset/multiclass_v1")

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

    print("\nLabel distribution:")
    print(df["Label"].value_counts(dropna=False))

    print("\nLabel percentages:")
    print(df["Label"].value_counts(normalize=True, dropna=False) * 100)

GROUPED_LABEL_MAP = {
    "BENIGN": "BENIGN",

    "DDOS-ICMP_FLOOD": "DDoS",
    "DDOS-UDP_FLOOD": "DDoS",
    "DDOS-TCP_FLOOD": "DDoS",
    "DDOS-PSHACK_FLOOD": "DDoS",
    "DDOS-SYN_FLOOD": "DDoS",
    "DDOS-RSTFINFLOOD": "DDoS",
    "DDOS-SYNONYMOUSIP_FLOOD": "DDoS",
    "DDOS-ICMP_FRAGMENTATION": "DDoS",
    "DDOS-ACK_FRAGMENTATION": "DDoS",
    "DDOS-UDP_FRAGMENTATION": "DDoS",
    "DDOS-HTTP_FLOOD": "DDoS",
    "DDOS-SLOWLORIS": "DDoS",

    "DOS-UDP_FLOOD": "DoS",
    "DOS-TCP_FLOOD": "DoS",
    "DOS-SYN_FLOOD": "DoS",
    "DOS-HTTP_FLOOD": "DoS",

    "MIRAI-GREETH_FLOOD": "Mirai",
    "MIRAI-UDPPLAIN": "Mirai",
    "MIRAI-GREIP_FLOOD": "Mirai",

    "VULNERABILITYSCAN": "Recon",
    "RECON-HOSTDISCOVERY": "Recon",
    "RECON-OSSCAN": "Recon",
    "RECON-PORTSCAN": "Recon",
    "RECON-PINGSWEEP": "Recon",

    "MITM-ARPSPOOFING": "Spoofing",
    "DNS_SPOOFING": "Spoofing",

    "DICTIONARYBRUTEFORCE": "BruteForce",

    "BROWSERHIJACKING": "Web",
    "COMMANDINJECTION": "Web",
    "SQLINJECTION": "Web",
    "XSS": "Web",
    "BACKDOOR_MALWARE": "Web",
    "UPLOADING_ATTACK": "Web",
}

def map_to_grouped_label(label: str) -> str:
    if label not in GROUPED_LABEL_MAP:
        raise ValueError(f"Unmapped label found: {label}")
    return GROUPED_LABEL_MAP[label]


def build_proportional_grouped_dataset(
    df: pd.DataFrame,
    benign_label: str = "BENIGN",
    grouped_label_col: str = "Grouped_Label",
    random_state: int = 42
) -> pd.DataFrame:
    benign_df = df[df[grouped_label_col] == benign_label].copy()
    attack_df = df[df[grouped_label_col] != benign_label].copy()

    benign_count = len(benign_df)

    # Sample total attack rows to match benign count
    sampled_attack_df = (
        attack_df
        .groupby(grouped_label_col, group_keys=False)
        .apply(
            lambda group: group.sample(
                n=max(1, round(len(group) / len(attack_df) * benign_count)),
                random_state=random_state
            )
        )
        .reset_index(drop=True)
    )

    final_df = pd.concat([benign_df, sampled_attack_df], ignore_index=True)

    return final_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

def merge_rare_grouped_classes(
    df: pd.DataFrame,
    grouped_label_col: str = "Grouped_Label"
) -> pd.DataFrame:
    df = df.copy()

    rare_classes = {"Web", "BruteForce"}

    df["Final_Label"] = df[grouped_label_col].apply(
        lambda label: "Rare" if label in rare_classes else label
    )

    return df

def save_multiclass_artifacts(df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "full_multiclass.csv"
    df.to_csv(output_path, index=False)

    feature_cols = [
        col for col in df.columns
        if col not in ["Label", "Grouped_Label", "Final_Label"]
    ]

    with open(OUTPUT_DIR / "feature_list.json", "w") as f:
        json.dump(feature_cols, f, indent=4)

    unique_labels = sorted(df["Final_Label"].unique())
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

    with open(OUTPUT_DIR / "label_mapping.json", "w") as f:
        json.dump(label_mapping, f, indent=4)

    class_dist = df["Final_Label"].value_counts()
    class_dist.to_csv(OUTPUT_DIR / "class_distribution.csv")

    metadata = {
        "dataset_name": "multiclass_v1",
        "num_rows": len(df),
        "num_features": len(feature_cols),
        "target_column": "Final_Label",
        "intermediate_group_column": "Grouped_Label",
        "original_label_column": "Label",
        "classes": unique_labels,
        "description": "Grouped multiclass dataset built from CICIoT2023 with proportional attack downsampling and rare class merging"
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
        stratify=df["Final_Label"]
    )

    train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

    print("\n===== TRAIN SPLIT DISTRIBUTION =====")
    print(train_df["Final_Label"].value_counts())

    print("\n===== TEST SPLIT DISTRIBUTION =====")
    print(test_df["Final_Label"].value_counts())

    print(f"\nSaved train split to {OUTPUT_DIR / 'train.csv'}")
    print(f"Saved test split to {OUTPUT_DIR / 'test.csv'}")


def main() -> None:
    df = load_raw_files(RAW_DIR)
    inspect_basic_info(df)

    df["Grouped_Label"] = df["Label"].apply(map_to_grouped_label)

    print("\n===== GROUPED LABEL DISTRIBUTION =====")
    print(df["Grouped_Label"].value_counts())

    print("\n===== GROUPED LABEL PERCENTAGES =====")
    print(df["Grouped_Label"].value_counts(normalize=True) * 100)

    sampled_df = build_proportional_grouped_dataset(df)

    print("\n===== SAMPLED GROUPED DATASET DISTRIBUTION =====")
    print(sampled_df["Grouped_Label"].value_counts())

    print("\n===== SAMPLED GROUPED DATASET PERCENTAGES =====")
    print(sampled_df["Grouped_Label"].value_counts(normalize=True) * 100)
    
# Merge rare classes into "Rare"

    final_df = merge_rare_grouped_classes(sampled_df)

    print("\n===== FINAL LABEL DISTRIBUTION =====")
    print(final_df["Final_Label"].value_counts())

    print("\n===== FINAL LABEL PERCENTAGES =====")
    print(final_df["Final_Label"].value_counts(normalize=True) * 100)


    save_multiclass_artifacts(final_df)

    split_and_save_dataset(final_df)

if __name__ == "__main__":
    main()


