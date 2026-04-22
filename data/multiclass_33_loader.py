from pathlib import Path
import json
import pandas as pd


DATASET_DIR = Path("dataset/multiclass_33_v1")


def load_json(file_path: Path):
    with open(file_path, "r") as f:
        return json.load(f)


def load_split_dataframe(split: str) -> pd.DataFrame:
    if split not in {"train", "test", "full"}:
        raise ValueError("split must be one of: 'train', 'test', 'full'")

    file_map = {
        "train": DATASET_DIR / "train.csv",
        "test": DATASET_DIR / "test.csv",
        "full": DATASET_DIR / "full_multiclass_33.csv",
    }

    file_path = file_map[split]

    if not file_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {file_path}")

    return pd.read_csv(file_path)


def load_feature_list():
    file_path = DATASET_DIR / "feature_list.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Missing feature list file: {file_path}")
    return load_json(file_path)


def load_label_mapping():
    file_path = DATASET_DIR / "label_mapping.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Missing label mapping file: {file_path}")
    return load_json(file_path)


def load_multiclass_33_xy(split: str = "train"):
    df = load_split_dataframe(split)
    feature_list = load_feature_list()
    label_mapping = load_label_mapping()

    if "Label" not in df.columns:
        raise ValueError("Target column 'Label' not found in dataset.")

    X = df[feature_list].copy()
    y = df["Label"].map(label_mapping)

    if y.isnull().any():
        missing_labels = df.loc[y.isnull(), "Label"].unique()
        raise ValueError(f"Found unmapped labels: {missing_labels}")

    y = y.astype(int)

    return X, y


def load_multiclass_33_dataset(split: str = "train"):
    df = load_split_dataframe(split)
    feature_list = load_feature_list()
    label_mapping = load_label_mapping()

    if "Label" not in df.columns:
        raise ValueError("Target column 'Label' not found in dataset.")

    X = df[feature_list].copy()
    y = df["Label"].map(label_mapping)

    if y.isnull().any():
        missing_labels = df.loc[y.isnull(), "Label"].unique()
        raise ValueError(f"Found unmapped labels: {missing_labels}")

    y = y.astype(int)

    return {
        "dataframe": df,
        "X": X,
        "y": y,
        "feature_list": feature_list,
        "label_mapping": label_mapping,
    }