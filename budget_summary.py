#!/usr/bin/env python3
"""Summarize the construction budget tracker CSV.

Reads construction_budget_tracker.csv and prints:
  - total spend by vendor
  - total spend by job
  - the five largest individual line items

Usage:
    python3 budget_summary.py [path/to/construction_budget_tracker.csv]
"""

import csv
import os
import sys
from collections import defaultdict

CSV_NAME = "construction_budget_tracker.csv"

VENDOR_ALERT_THRESHOLD = 100.0
JOB_ALERT_THRESHOLD = 500.0


def find_csv(explicit_path=None):
    """Locate the CSV, preferring an explicit path, then common locations."""
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(here, CSV_NAME))           # alongside script
    candidates.append(os.path.join(here, "..", CSV_NAME))      # parent directory
    candidates.append(os.path.join(os.getcwd(), CSV_NAME))     # current directory
    for path in candidates:
        if path and os.path.isfile(path):
            return os.path.abspath(path)
    return None


def load_rows(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                row["amount"] = float(row["amount"])
            except (TypeError, ValueError):
                continue  # skip rows without a usable amount
            rows.append(row)
    return rows


def money(value):
    return "${:,.2f}".format(value)


def build_alerts(by_vendor, by_job):
    """Return a list of (category, name, amount, threshold) over-budget alerts."""
    alerts = []
    for vendor, amount in by_vendor.items():
        if amount > VENDOR_ALERT_THRESHOLD:
            alerts.append(("VENDOR", vendor, amount, VENDOR_ALERT_THRESHOLD))
    for job, amount in by_job.items():
        if amount > JOB_ALERT_THRESHOLD:
            alerts.append(("JOB", job, amount, JOB_ALERT_THRESHOLD))
    # Largest overage first.
    alerts.sort(key=lambda a: a[2] - a[3], reverse=True)
    return alerts


def print_alerts(alerts):
    bar = "!" * 60
    print(bar)
    if not alerts:
        print("  BUDGET ALERTS: none -- all vendors and jobs within budget.")
        print(bar)
        print()
        return
    print("  *** BUDGET ALERTS ({} over budget) ***".format(len(alerts)))
    print(bar)
    for category, name, amount, threshold in alerts:
        print("  [!] {:<6} {:<30} {:>11}  (over {} by {})".format(
            category, name, money(amount), money(threshold), money(amount - threshold)))
    print(bar)
    print()


def print_totals(title, totals):
    print(title)
    print("-" * len(title))
    width = max((len(k) for k in totals), default=0)
    grand_total = 0.0
    for key, amount in sorted(totals.items(), key=lambda kv: kv[1], reverse=True):
        print("  {:<{w}}  {:>12}".format(key, money(amount), w=width))
        grand_total += amount
    print("  {:<{w}}  {:>12}".format("TOTAL", money(grand_total), w=width))
    print()


def main():
    explicit = sys.argv[1] if len(sys.argv) > 1 else None
    path = find_csv(explicit)
    if not path:
        sys.exit("Could not find {}. Pass its path as an argument.".format(CSV_NAME))

    rows = load_rows(path)
    if not rows:
        sys.exit("No usable rows found in {}.".format(path))

    by_vendor = defaultdict(float)
    by_job = defaultdict(float)
    for row in rows:
        by_vendor[row.get("vendor", "Unknown")] += row["amount"]
        by_job[row.get("job_name", "Unknown")] += row["amount"]

    alerts = build_alerts(by_vendor, by_job)
    print()
    print_alerts(alerts)

    print("=" * 60)
    print("CONSTRUCTION BUDGET SUMMARY")
    print("Source: {}".format(path))
    print("Line items: {}".format(len(rows)))
    print("=" * 60)
    print()

    print_totals("TOTAL SPEND BY VENDOR", by_vendor)
    print_totals("TOTAL SPEND BY JOB", by_job)

    print("FIVE LARGEST LINE ITEMS")
    print("-----------------------")
    top = sorted(rows, key=lambda r: r["amount"], reverse=True)[:5]
    for i, row in enumerate(top, 1):
        print("  {}. {:>11}  {}".format(i, money(row["amount"]), row.get("description", "")))
        print("       {} | {} | {}".format(
            row.get("job_name", ""), row.get("vendor", ""), row.get("date", "")))
    print()


if __name__ == "__main__":
    main()
