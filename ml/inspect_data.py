import json
import pandas as pd
import sys

# ── 1. Load config ──────────────────────────────────────────────────────────
try:
    with open("feature_list.json", "r") as f:
        config = json.load(f)
except FileNotFoundError:
    print("❌ ERROR: 'feature_list.json' not found.")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"❌ ERROR: Failed to parse 'feature_list.json': {e}")
    sys.exit(1)

target   = config.get("target")
features = config.get("features", [])

if not target:
    print("❌ ERROR: 'target' key missing from feature_list.json.")
    sys.exit(1)
if not features:
    print("⚠️  WARNING: 'features' list is empty in feature_list.json.")

# ── 2. Load dataset ─────────────────────────────────────────────────────────
try:
    df = pd.read_csv("clean_sample.csv")
except FileNotFoundError:
    print("❌ ERROR: 'clean_sample.csv' not found.")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR: Failed to read CSV: {e}")
    sys.exit(1)

# ── 3. Basic info ────────────────────────────────────────────────────────────
print("=" * 50)
print("📊 DATASET OVERVIEW")
print("=" * 50)
print(f"  Shape             : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  Target column     : {target}")
print(f"  Features declared : {len(features)}")

# ── 4. Column alignment check ────────────────────────────────────────────────
print("\n" + "=" * 50)
print("🔍 COLUMN ALIGNMENT")
print("=" * 50)

missing_features = [col for col in features if col not in df.columns]
extra_columns    = [col for col in df.columns if col not in features + [target]]

if missing_features:
    print(f"  ⚠️  Missing from CSV ({len(missing_features)}): {missing_features}")
else:
    print("  ✅ All declared features are present in the CSV.")

if target not in df.columns:
    print(f"  ❌ Target column '{target}' is MISSING from the CSV!")
    sys.exit(1)
else:
    print(f"  ✅ Target column '{target}' found.")

if extra_columns:
    print(f"  ℹ️  Extra columns not in feature list ({len(extra_columns)}): {extra_columns}")
else:
    print("  ✅ No unexpected extra columns.")

# ── 5. Class distribution ────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("🎯 CLASS DISTRIBUTION")
print("=" * 50)
class_counts = df[target].value_counts()
class_pct    = df[target].value_counts(normalize=True) * 100
class_summary = pd.DataFrame({"Count": class_counts, "Percentage (%)": class_pct.round(2)})
print(class_summary.to_string())

if class_pct.max() > 80:
    print(f"\n  ⚠️  WARNING: Class imbalance detected — dominant class is {class_pct.max():.1f}% of data.")

# ── 6. Missing values ────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("🕳️  MISSING VALUES")
print("=" * 50)
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if missing.empty:
    print("  ✅ No missing values found.")
else:
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df  = pd.DataFrame({"Missing Count": missing, "Missing (%)": missing_pct})
    print(missing_df.to_string())

# ── 7. Data types ─────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("🔢 DATA TYPES")
print("=" * 50)
print(df.dtypes.to_string())

# ── 8. Duplicate rows ─────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("♻️  DUPLICATE ROWS")
print("=" * 50)
dupe_count = df.duplicated().sum()
if dupe_count == 0:
    print("  ✅ No duplicate rows found.")
else:
    print(f"  ⚠️  WARNING: {dupe_count} duplicate row(s) detected ({dupe_count / len(df) * 100:.2f}% of data).")

# ── 9. Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("✅ VALIDATION COMPLETE")
print("=" * 50)
