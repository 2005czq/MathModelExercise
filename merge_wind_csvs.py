#!/usr/bin/env python3
"""
Merge all NASA POWER daily CSVs in ./data into a single CSV.

Each input file contains a header block between -BEGIN HEADER- and -END HEADER-,
followed by a CSV header row:
    LAT,LON,YEAR,MO,DY,WS50M

This script:
  - Scans ./data for all *.csv files
  - Skips the verbose header blocks
  - Writes a single header once
  - Deduplicates overlapping rows across files by (LAT,LON,YEAR,MO,DY)
  - Saves to ./data/merged_wind_50m_daily.csv

Usage:
  python3 merge_wind_csvs.py [--out PATH]

If --out is not provided, defaults to ./data/merged_wind_50m_daily.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, List, Tuple


HEADER_ROW = ["LAT", "LON", "YEAR", "MO", "DY", "WS50M"]


def find_header_index(lines: List[str]) -> int:
    """Return the index of the CSV header row in the file lines.

    The file has a descriptive header block; we locate the row that starts with the
    expected CSV header columns.
    """
    for i, line in enumerate(lines):
        if line.strip().upper().startswith(",".join(HEADER_ROW)):
            return i
    raise ValueError("Could not find CSV header row (LAT,LON,YEAR,MO,DY,WS50M)")


def read_data_rows(file_path: Path) -> Iterable[List[str]]:
    """Yield rows from a POWER CSV file as lists of strings, including data only.

    Skips the descriptive header block and the header row itself, yielding only data rows.
    """
    text = file_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    start = find_header_index(lines)
    # Include header in csv parsing, then skip header in iteration for consistent parsing
    from io import StringIO

    sio = StringIO("\n".join(lines[start:]) + "\n")
    reader = csv.reader(sio)
    header = next(reader, None)
    if header is None:
        return  # empty file after header block
    # Normalize header for safety
    if [h.strip().upper() for h in header] != HEADER_ROW:
        # Allow minor variations such as extra whitespace/casing
        pass
    for row in reader:
        if not row:
            continue
        yield row


def parse_key(row: List[str]) -> Tuple[str, str, str, str, str]:
    """Create a deduplication key: (LAT, LON, YEAR, MO, DY) as strings.

    We keep as strings to avoid float formatting issues; they come directly from source CSVs.
    """
    # Expect exactly 6 columns; ignore extra columns if present defensively
    lat, lon, year, mo, dy = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip(), row[4].strip()
    return (lat, lon, year, mo, dy)


def merge_csvs(data_dir: Path, out_path: Path) -> Tuple[int, int, int]:
    """Merge all CSVs under data_dir to out_path.

    Returns a tuple: (files_processed, rows_written, duplicates_skipped)
    """
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    files_processed = 0
    rows_written = 0
    duplicates = 0

    with out_path.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(HEADER_ROW)

        for fp in csv_files:
            files_processed += 1
            for row in read_data_rows(fp):
                if len(row) < 6:
                    # Skip malformed
                    continue
                key = parse_key(row)
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                # Only write the first six expected columns to keep a clean schema
                writer.writerow([row[0], row[1], row[2], row[3], row[4], row[5]])
                rows_written += 1

    return files_processed, rows_written, duplicates


def main():
    parser = argparse.ArgumentParser(description="Merge NASA POWER daily CSVs in ./data")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/merged_wind_50m_daily.csv"),
        help="Output CSV path (default: data/merged_wind_50m_daily.csv)",
    )
    args = parser.parse_args()

    data_dir = Path("data")
    files, rows, dups = merge_csvs(data_dir, args.out)
    print(f"Merged {files} file(s) into {args.out}")
    print(f"Rows written: {rows}; Duplicates skipped: {dups}")


if __name__ == "__main__":
    main()
