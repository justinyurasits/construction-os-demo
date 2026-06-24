#!/usr/bin/env python3
"""Generate a one-page cost report for a single construction job.

Reads construction_budget_tracker.csv, filters to the job named on the command
line, and writes a clean text report to [job_name]_cost_report.txt showing:
  - all receipts for that job, sorted by date, with a running total
  - spend by category (only if the CSV has a 'category' column)
  - whether the job is on track against a hardcoded $5,000 budget

Usage:
    python3 job_cost_report.py "Beacon Hill Cedar Closet"
"""

import csv
import os
import re
import sys
from collections import defaultdict

CSV_NAME = "construction_budget_tracker.csv"
JOB_BUDGET = 5000.0


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
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = []
        for row in reader:
            try:
                row["amount"] = float(row["amount"])
            except (TypeError, ValueError):
                continue  # skip rows without a usable amount
            rows.append(row)
    return rows, fieldnames


def match_job(rows, requested):
    """Find the job name in the data matching the request (case-insensitive)."""
    jobs = sorted({r.get("job_name", "") for r in rows if r.get("job_name")})
    # Exact case-insensitive match first.
    for job in jobs:
        if job.lower() == requested.lower():
            return job, jobs
    # Fall back to substring match.
    partial = [job for job in jobs if requested.lower() in job.lower()]
    if len(partial) == 1:
        return partial[0], jobs
    return None, jobs


def money(value):
    return "${:,.2f}".format(value)


def safe_filename(job_name):
    slug = re.sub(r"[^\w\s-]", "", job_name).strip()
    slug = re.sub(r"\s+", "_", slug)
    return "{}_cost_report.txt".format(slug or "job")


def build_report(job_name, rows, fieldnames):
    has_category = "category" in fieldnames
    job_rows = [r for r in rows if r.get("job_name") == job_name]
    job_rows.sort(key=lambda r: r.get("date", ""))

    lines = []
    sep = "=" * 70
    thin = "-" * 70
    lines.append(sep)
    lines.append("JOB COST REPORT".center(70))
    lines.append(job_name.center(70))
    lines.append(sep)
    lines.append("Budget: {}    Line items: {}".format(money(JOB_BUDGET), len(job_rows)))
    lines.append("")

    # --- Receipts with running total ---
    lines.append("RECEIPTS (sorted by date)")
    lines.append(thin)
    header = "{:<11} {:<16} {:<26} {:>9} {:>10}".format(
        "Date", "Vendor", "Description", "Amount", "Running")
    lines.append(header)
    lines.append(thin)
    running = 0.0
    for r in job_rows:
        running += r["amount"]
        lines.append("{:<11} {:<16} {:<26} {:>9} {:>10}".format(
            (r.get("date", "") or "")[:11],
            (r.get("vendor", "") or "")[:16],
            (r.get("description", "") or "")[:26],
            money(r["amount"]),
            money(running)))
    lines.append(thin)
    total = running
    lines.append("{:<54} {:>10}".format("TOTAL SPEND", money(total)))
    lines.append("")

    # --- Spend by category (only if the column exists) ---
    lines.append("SPEND BY CATEGORY")
    lines.append(thin)
    if has_category:
        by_cat = defaultdict(float)
        for r in job_rows:
            by_cat[r.get("category") or "Uncategorized"] += r["amount"]
        width = max((len(k) for k in by_cat), default=0)
        for cat, amt in sorted(by_cat.items(), key=lambda kv: kv[1], reverse=True):
            lines.append("  {:<{w}}  {:>11}".format(cat, money(amt), w=width))
    else:
        lines.append("  (no 'category' column in the source data -- not available)")
    lines.append("")

    # --- Budget status ---
    remaining = JOB_BUDGET - total
    pct = (total / JOB_BUDGET * 100) if JOB_BUDGET else 0.0
    status = "ON TRACK" if total <= JOB_BUDGET else "OVER BUDGET"
    lines.append("BUDGET STATUS")
    lines.append(thin)
    lines.append("  Spent:      {:>12}  ({:.1f}% of budget)".format(money(total), pct))
    lines.append("  Budget:     {:>12}".format(money(JOB_BUDGET)))
    lines.append("  Remaining:  {:>12}".format(money(remaining)))
    lines.append("  Status:     {}".format(status))
    lines.append(sep)

    return "\n".join(lines) + "\n"


def main():
    if len(sys.argv) < 2:
        sys.exit('Usage: python3 job_cost_report.py "<job name>"')
    requested = sys.argv[1]

    path = find_csv()
    if not path:
        sys.exit("Could not find {}.".format(CSV_NAME))

    rows, fieldnames = load_rows(path)
    if not rows:
        sys.exit("No usable rows found in {}.".format(path))

    job_name, all_jobs = match_job(rows, requested)
    if not job_name:
        msg = ["No job matching '{}'.".format(requested), "Available jobs:"]
        msg += ["  - {}".format(j) for j in all_jobs]
        sys.exit("\n".join(msg))

    report = build_report(job_name, rows, fieldnames)
    out_path = os.path.join(os.getcwd(), safe_filename(job_name))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print("Report saved to: {}".format(out_path))


if __name__ == "__main__":
    main()
