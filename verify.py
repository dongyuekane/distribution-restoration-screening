"""Compare a newly generated confirmation run with the archived benchmark."""

import argparse
import csv
import json
from pathlib import Path


def classifications(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (row["id"], row["alpha"]): int(row["feasible"])
        for row in rows
    }


def operational_counts(directory):
    metrics = json.loads((directory / "confirmation_metrics.json").read_text())
    fields = (
        "topologies", "profiles", "AC_oracle_states",
        "non_downward_closed_profiles", "decision_mismatches", "false_exclusions",
    )
    return (
        {field: metrics[field] for field in fields},
        {name: values["AC_states"] for name, values in metrics["methods"].items()},
    )


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=root / "generated")
    args = parser.parse_args()
    reference = root / "archive"
    expected = classifications(reference / "confirmation_oracle.csv")
    current = classifications(args.output_dir / "confirmation_oracle.csv")
    matches_counts = operational_counts(reference) == operational_counts(args.output_dir)
    if current != expected or not matches_counts:
        raise SystemExit("Confirmation decisions or operational counts differ from the archive.")
    print(f"Matched {len(expected):,} archived AC classifications and all method state counts.")


if __name__ == "__main__":
    main()
