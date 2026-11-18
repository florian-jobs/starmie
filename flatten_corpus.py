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

    csvs, other = [], Counter()
    for root, _, files in os.walk(corpus, followlinks=True):
        for fn in files:
            if fn.endswith(".csv"):
                csvs.append(os.path.join(root, fn))
            else:
                other[os.path.splitext(fn)[1] or fn] += 1

    for src in sorted(csvs):
        name = os.path.relpath(src, corpus).replace(os.sep, "__")
        os.symlink(src, os.path.join(out, name))

    print("linked %d csv files into %s" % (len(csvs), out))
    if other:
        print("skipped non-csv files:", dict(other.most_common()))


if __name__ == "__main__":
    main()
