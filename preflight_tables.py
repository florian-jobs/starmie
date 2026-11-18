"""Check every table the way Starmie reads it, before a long training run.

Finds tables that would crash pretraining/embedding:
  - pd.read_csv(..., lineterminator='\\n') fails (empty file, malformed rows)
  - 0 columns (augment drop_col: random.choice on empty columns)
  - too many columns: every column gets >= 1 token and nothing truncates the
    whole sequence, so > MAX_COLS columns exceed RoBERTa's 512 positions

    python preflight_tables.py data/open_data/tables data/web_tables/tables
"""
import os
import sys

import pandas as pd

MAX_COLS = 510  # 512 positions, minus margin


def main():
    bad = 0
    for table_dir in sys.argv[1:]:
        files = sorted(f for f in os.listdir(table_dir) if f.endswith(".csv"))
        max_cols = 0
        for fn in files:
            path = os.path.join(table_dir, fn)
            try:
                df = pd.read_csv(path, lineterminator='\n', low_memory=False)
            except Exception as e:  # noqa: BLE001 - report every failure
                print("READ ERROR  %s: %s: %s" % (path, type(e).__name__, str(e)[:200]))
                bad += 1
                continue
            n = len(df.columns)
            max_cols = max(max_cols, n)
            if n == 0:
                print("NO COLUMNS  %s" % path)
                bad += 1
            elif n > MAX_COLS:
                print("TOO WIDE    %s: %d columns" % (path, n))
                bad += 1
        print("== %s: %d tables, widest %d columns" % (table_dir, len(files), max_cols))
    print("PROBLEMS: %d" % bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
