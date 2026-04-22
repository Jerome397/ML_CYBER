import numpy as np
import os
import sys


def fail(message: str) -> None:
    print(f"\n❌ ERROR: {message}")
    sys.exit(1)


SPLIT_PATH = os.path.join("artifacts", "splits", "split_indices.npz")

if not os.path.exists(SPLIT_PATH):
    fail(f"Split file not found: {SPLIT_PATH}")

data = np.load(SPLIT_PATH)

train_idx = data["train_idx"]
val_idx = data["val_idx"]
test_idx = data["test_idx"]

print("=" * 60)
print("🧪 SPLIT SANITY CHECK")
print("=" * 60)

print(f"Train size      : {len(train_idx)}")
print(f"Validation size : {len(val_idx)}")
print(f"Test size       : {len(test_idx)}")

print("\nFirst 20 validation indices:")
print(val_idx[:20])

print("\nFirst 20 test indices:")
print(test_idx[:20])

overlap_train_val = np.intersect1d(train_idx, val_idx)
overlap_train_test = np.intersect1d(train_idx, test_idx)
overlap_val_test = np.intersect1d(val_idx, test_idx)

print("\nOverlap sizes:")
print(f"Train ∩ Val   : {len(overlap_train_val)}")
print(f"Train ∩ Test  : {len(overlap_train_test)}")
print(f"Val ∩ Test    : {len(overlap_val_test)}")

total_unique = len(np.unique(np.concatenate([train_idx, val_idx, test_idx])))
expected_total = len(train_idx) + len(val_idx) + len(test_idx)

print("\nCoverage check:")
print(f"Total unique indices across all splits : {total_unique}")
print(f"Expected total from split sizes        : {expected_total}")

if total_unique == expected_total:
    print("✅ Good: all rows are covered exactly once.")
else:
    print(f"❌ Problem: {expected_total - total_unique} duplicate or overlapping indices detected.")

same_exact_order = np.array_equal(val_idx, test_idx)
same_set = np.array_equal(np.sort(val_idx), np.sort(test_idx))

print("\nExtra checks:")
print(f"Validation and test arrays identical in order? {same_exact_order}")
print(f"Validation and test contain exactly the same indices? {same_set}")

if len(overlap_train_val) == 0 and len(overlap_train_test) == 0 and len(overlap_val_test) == 0:
    print("\n✅ Good: all splits are disjoint.")
else:
    print("\n❌ Problem: some splits overlap.")

if not same_exact_order and not same_set:
    print("✅ Good: validation and test are different subsets.")
else:
    print("❌ Problem: validation and test should not be the same subset.")