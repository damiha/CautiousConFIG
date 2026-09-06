import json
from pathlib import Path
import math


def make_hashable(val):
  """Recursively convert lists or dicts into hashable types,

  so they can be used as dictionary keys.
  """
  if isinstance(val, list):
    return tuple(make_hashable(x) for x in val)
  elif isinstance(val, dict):
    return tuple((str(k), make_hashable(v)) for k, v in sorted(val.items()))
  return val


def get_hyperparams_key(json_data):
  """Extract all parameters except 'best_mse' as a stable grouping key."""
  hp_dict = {k: v for k, v in json_data.items() if k.lower() != "best_mse"}
  sorted_items = sorted((k, make_hashable(v)) for k, v in hp_dict.items())
  return tuple(sorted_items)


def load_runs_from_dir(root_dir):
  """Recursively search the directory for all JSON files

  (regardless of filename in the subfolder).
  """
  root_path = Path(root_dir)
  runs = {}

  for json_path in root_path.glob("**/*.json"):
    try:
      with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

      if "best_mse" not in data:
        continue

      hp_key = get_hyperparams_key(data)
      runs[hp_key] = {
          "path": json_path,
          "best_mse": data["best_mse"],
          "seed": data.get("seed"),
      }
    except Exception as e:
      print(f"Error reading {json_path}: {e}")

  return runs


def compare_experiment_folders(dir1, dir2, tolerance=1e-10):
  print(f"Comparing:\n  1. {dir1}\n  2. {dir2}\n")

  runs1 = load_runs_from_dir(dir1)
  runs2 = load_runs_from_dir(dir2)

  print(f"Found configurations in folder 1: {len(runs1)}")
  print(f"Found configurations in folder 2: {len(runs2)}")

  keys1 = set(runs1.keys())
  keys2 = set(runs2.keys())

  only_in_1 = keys1 - keys2
  only_in_2 = keys2 - keys1
  common_keys = keys1.intersection(keys2)

  print(f"Matching configuration groups: {len(common_keys)}")

  if only_in_1:
    print(
        f"Warning: {len(only_in_1)} configuration(s) present only in the first"
        " folder."
    )
  if only_in_2:
    print(
        f"Warning: {len(only_in_2)} configuration(s) present only in the second"
        " folder."
    )

  mismatches = 0
  matches = 0

  for key in common_keys:
    mse1 = runs1[key]["best_mse"]
    mse2 = runs2[key]["best_mse"]

    # Tolerance check for best_mse
    if math.isclose(mse1, mse2, rel_tol=0, abs_tol=tolerance):
      matches += 1
    else:
      mismatches += 1
      print(f"\nDiscrepancy detected in best_mse:")
      print(f"  Path A: {runs1[key]['path']} (MSE: {mse1})")
      print(f"  Path B: {runs2[key]['path']} (MSE: {mse2})")
      print(f"  Difference: {abs(mse1 - mse2)}")

  print("\n--- Evaluation Completed ---")
  print(f"Successfully compared (Identical): {matches}")
  print(f"Mismatched (Different MSE): {mismatches}")


if __name__ == "__main__":
    compare_experiment_folders(
      "multirun/2026-08-05/burgers_cautious_config_10_seeds (IN REPORT)",
     "multirun/2026-08-07/burgers_cautious_config_update_percentage"
    )