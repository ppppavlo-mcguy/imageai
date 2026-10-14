# instruction: python that makes a report from a csv file
import csv
from collections import defaultdict


def totals(path: str) -> dict:
    sums = defaultdict(float)
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sums[row["category"]] += float(row["amount"])
    return dict(sums)


if __name__ == "__main__":
    for name, total in totals("sales.csv").items():
        print(f"{name}: {total:.2f}")
