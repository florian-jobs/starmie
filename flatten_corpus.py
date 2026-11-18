"""Make a flat directory of symlinks to every *.csv under a (nested) corpus.

Starmie's PretrainTableDataset only lists *.csv directly inside one directory.
Link names are the path relative to the corpus root with '/' -> '__', so they
are unique; nothing is copied.

    python flatten_corpus.py ~/data/beluga/data/corpora/open_data data/open_data/tables
"""
import argparse
import os
from collections import Counter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("corpus", help="corpus root (searched recursively)")
    p.add_argument("out", help="flat output directory (created)")
    args = p.parse_args()

    corpus = os.path.abspath(os.path.expanduser(args.corpus))
    out = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(out, exist_ok=True)
    if os.listdir(out):
        raise SystemExit("output directory is not empty: %s" % out)

    # Same table definition as beluga (src/beluga/offline/run.py): a table is a
    # leaf directory (no subdirectories). CSVs in non-leaf directories, e.g.
    # <corpus>/join_paths.csv or fragments/<base>/reconstructed_table.csv, are
    # not corpus tables and are skipped.
    csvs, other, skipped_non_leaf = [], Counter(), []
    for root, dirs, files in os.walk(corpus, followlinks=True):
        for fn in files:
            if not fn.endswith(".csv"):
                other[os.path.splitext(fn)[1] or fn] += 1
            elif dirs or fn != "table.csv":  # beluga reads only <leaf>/table.csv
                skipped_non_leaf.append(os.path.relpath(os.path.join(root, fn), corpus))
            else:
                csvs.append(os.path.join(root, fn))

    for src in sorted(csvs):
        name = os.path.relpath(src, corpus).replace(os.sep, "__")
        os.symlink(src, os.path.join(out, name))

    print("linked %d csv files into %s" % (len(csvs), out))
    print("skipped %d csv that are not <leaf dir>/table.csv (not corpus tables):" % len(skipped_non_leaf))
    for path in sorted(skipped_non_leaf):
        print("  ", path)
    if other:
        print("skipped non-csv files:", dict(other.most_common()))


if __name__ == "__main__":
    main()
