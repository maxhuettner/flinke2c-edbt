#!/usr/bin/env python3
"""
Combine CSV files in pairs under a root folder.

For each directory, files that match:
  (events|latency|time)*_<index>.csv
are grouped by their base prefix (everything up to the trailing underscore).

Then:
  - index 1 is appended into 0, 3 into 2, 5 into 4, ...
  - the second file in each pair is deleted
  - remaining files are renumbered to 0..N-1 in index order

Example: events_test_0..events_test_9 -> events_test_0..events_test_4
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from typing import Dict, List, Tuple


FILE_RE = re.compile(r"^(?P<base>(events|latency|time).*_)(?P<idx>\d+)\.csv$")


@dataclass(frozen=True)
class FileEntry:
    idx: int
    path: str
    name: str


def _file_ends_with_newline(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size == 0:
                return True
            f.seek(-1, os.SEEK_END)
            return f.read(1) == b"\n"
    except OSError:
        return True


def _append_csv(primary: str, secondary: str) -> None:
    # Read header from primary
    with open(primary, "rb") as f:
        primary_header = f.readline()

    # Stream append from secondary, skipping header if it matches
    with open(secondary, "rb") as src, open(primary, "ab") as dst:
        secondary_header = src.readline()
        if primary_header != secondary_header:
            if _file_ends_with_newline(primary) is False:
                dst.write(b"\n")
            dst.write(secondary_header)
        else:
            if _file_ends_with_newline(primary) is False:
                dst.write(b"\n")
        shutil.copyfileobj(src, dst)


def _collect_groups(root: str) -> Dict[Tuple[str, str], List[FileEntry]]:
    groups: Dict[Tuple[str, str], List[FileEntry]] = {}
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            m = FILE_RE.match(name)
            if not m:
                continue
            base = m.group("base")
            idx = int(m.group("idx"))
            path = os.path.join(dirpath, name)
            key = (dirpath, base)
            groups.setdefault(key, []).append(FileEntry(idx=idx, path=path, name=name))
    return groups


def _combine_group(entries: List[FileEntry], dry_run: bool) -> List[FileEntry]:
    entries_sorted = sorted(entries, key=lambda e: e.idx)
    kept: List[FileEntry] = []
    i = 0
    while i < len(entries_sorted):
        primary = entries_sorted[i]
        kept.append(primary)
        if i + 1 < len(entries_sorted):
            secondary = entries_sorted[i + 1]
            if dry_run:
                print(f"  append {os.path.basename(secondary.path)} -> {os.path.basename(primary.path)}")
            else:
                _append_csv(primary.path, secondary.path)
                os.remove(secondary.path)
        i += 2
    return kept


def _renumber_group(dirpath: str, base: str, kept: List[FileEntry], dry_run: bool) -> None:
    kept_sorted = sorted(kept, key=lambda e: e.idx)
    temp_paths: List[str] = []
    for order, entry in enumerate(kept_sorted):
        tmp_path = f"{entry.path}.tmp_rename_{order}"
        temp_paths.append(tmp_path)
        if dry_run:
            print(f"  rename {os.path.basename(entry.path)} -> {os.path.basename(tmp_path)}")
        else:
            os.rename(entry.path, tmp_path)

    for order, tmp_path in enumerate(temp_paths):
        final_name = f"{base}{order}.csv"
        final_path = os.path.join(dirpath, final_name)
        if dry_run:
            print(f"  rename {os.path.basename(tmp_path)} -> {final_name}")
        else:
            os.rename(tmp_path, final_path)


def combine_update_data(root: str, dry_run: bool) -> None:
    if not os.path.isdir(root):
        raise SystemExit(f"Root folder not found: {root}")

    groups = _collect_groups(root)
    if not groups:
        print("No matching files found.")
        return

    for (dirpath, base), entries in sorted(groups.items()):
        print(f"{dirpath} :: {base}")
        kept = _combine_group(entries, dry_run=dry_run)
        _renumber_group(dirpath, base, kept, dry_run=dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine CSV files in pairs and renumber.")
    parser.add_argument("root", help="Root folder to scan (e.g., data/flink_q1f)")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without modifying files")
    args = parser.parse_args()
    combine_update_data(args.root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
