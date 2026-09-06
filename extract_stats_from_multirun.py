import json
from pathlib import Path


def make_hashable(val):
  """Recursively convert lists or dicts into hashable types,

  so they can be used as dictionary keys.
  """
  if isinstance(val, list):
    return tuple(make_hashable(x) for x in val)
  elif isinstance(val, dict):
    return tuple((str(k), make_hashable(v)) for k, v in sorted(val.items()))
  return val


def get_grouping_key(json_data):
  """Extract all parameters except 'seed' and 'best_mse' as a grouping key."""
  # Exclude 'seed' and 'best_mse'. 'filename' is included as it is part of the JSON data.
  filtered_dict = {
      k: v for k, v in json_data.items() if k.lower() not in ["seed", "best_mse"]
  }
  sorted_items = sorted((k, make_hashable(v)) for k, v in filtered_dict.items())
  return tuple(sorted_items)


def aggregate_runs_from_dirs(multirun_dirs):
  """Scan one or more multirun directories, group runs by their hyperparameters

  (excluding seed and best_mse), and aggregate seeds and best_mses into arrays.
  """
  grouped_data = {}

  for root_dir in multirun_dirs:
    root_path = Path(root_dir)
    if not root_path.exists():
      print(f"Directory not found: {root_dir}")
      continue

    # Recursively find all JSON files in the multirun directories
    for json_path in root_path.glob("**/*.json"):
      try:
        with open(json_path, "r", encoding="utf-8") as f:
          data = json.load(f)

        if "best_mse" not in data or "seed" not in data:
          continue

        group_key = get_grouping_key(data)

        if group_key not in grouped_data:
          # Keep all base configuration fields (excluding seed and best_mse)
          base_config = {
              k: v
              for k, v in data.items()
              if k.lower() not in ["seed", "best_mse"]
          }
          grouped_data[group_key] = {
              "config": base_config,
              "seeds": [],
              "best_mses": [],
          }

        grouped_data[group_key]["seeds"].append(data["seed"])
        grouped_data[group_key]["best_mses"].append(data["best_mse"])

      except Exception as e:
        print(f"Error reading {json_path}: {e}")

  # Save each aggregated group directly to the script's execution directory
  output_dir = Path(".")
  saved_count = 0

  for group_key, info in grouped_data.items():
    config = info["config"]
    seeds = info["seeds"]
    best_mses = info["best_mses"]

    # Sort seeds and corresponding mses together based on seed value
    combined = sorted(zip(seeds, best_mses), key=lambda x: x[0])
    sorted_seeds, sorted_mses = zip(*combined) if combined else ([], [])

    # Build the final dictionary structure
    output_data = dict(config)
    output_data["seeds"] = list(sorted_seeds)
    output_data["best_mses"] = list(sorted_mses)

    # Use the 'filename' value from the JSON as the output filename, fallback if missing
    filename_val = config.get("filename", "aggregated_run.json")
    output_path = output_dir / filename_val

    with open(output_path, "w", encoding="utf-8") as f:
      json.dump(output_data, f, indent=4)

    saved_count += 1
    print(f"Saved aggregated JSON: {output_path}")

  print(f"\nSuccessfully aggregated and saved {saved_count} JSON file(s).")

if __name__ == "__main__":

    aggregate_runs_from_dirs([
         "multirun/2026-08-27/burgers grams",
    ])