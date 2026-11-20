"""Produce one embedding vector per column with a pretrained Starmie model.

Usage (from tools/starmie):
    python embed_columns.py --model results/open_data/model_drop_col_head_column_0.pt \
                            --tables path/to/csv_dir --out embeddings.pkl

Output: a pickle with {table_file_name: {column_name: np.ndarray}}.

Differences from upstream extractVectors.py / load_checkpoint (on purpose):
  - the model is put in eval() mode, so dropout is off and embeddings are
    deterministic (upstream never calls eval()).
  - no hard-coded benchmark paths: the dataset object is only needed for its
    tokenizer/serialisation, not for its list of tables.
"""
import argparse
import glob
import os
import pickle
import tempfile
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sdd.dataset import PretrainTableDataset
from sdd.model import BarlowTwinsSimCLR
from sdd.pretrain import inference_on_tables


def load_model(ckpt_path: str,
               device: Optional[str] = None) -> Tuple[BarlowTwinsSimCLR, PretrainTableDataset]:
    """Load a checkpoint written by run_pretrain.py --save_model.

    Args:
        ckpt_path: path to the .pt file ({'model': state_dict, 'hp': Namespace}).
        device: 'cuda' / 'cpu'; None picks cuda if available.

    Returns:
        (model, dataset): the model in eval mode on ``device``, and a
        PretrainTableDataset built from the checkpoint's hp (used only for
        _tokenize / pad / tokenizer during inference).
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # weights_only=False: the checkpoint also stores the argparse Namespace hp
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hp = ckpt['hp']

    model = BarlowTwinsSimCLR(hp, device=device, lm=hp.lm)
    model = model.to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()  # dropout off -> deterministic embeddings (upstream omits this)

    # only the tokenizer / _tokenize / pad are used at inference time, so the
    # table list can be empty; avoids upstream's hard-coded benchmark paths
    dataset = PretrainTableDataset.from_hp(tempfile.mkdtemp(), hp)

    return model, dataset


def embed_table(df: pd.DataFrame,
                model: BarlowTwinsSimCLR,
                dataset: PretrainTableDataset) -> Dict[str, np.ndarray]:
    """Embed every column of one table.

    Returns:
        {column_name: vector} in the column order of ``df``.
    """
    vectors = inference_on_tables([df], model, dataset)[0]
    # Starmie finds columns by counting <s> tokens; a cell containing "<s>"
    # would yield extra vectors and zip would silently misalign them
    assert len(vectors) == len(df.columns), \
        "got %d vectors for %d columns" % (len(vectors), len(df.columns))
    column_to_vector = {column_name: vector
                        for column_name, vector in zip(df.columns, vectors)}

    return column_to_vector


def read_tables(table_dir: str, max_rows: int = 1000) -> Dict[str, pd.DataFrame]:
    """Read all *.csv in table_dir (sorted, so the order is reproducible).

    Keeps the first ``max_rows`` rows like upstream extractVectors.get_df.
    """
    tables = {}
    for fn in sorted(glob.glob(os.path.join(table_dir, "*.csv"))):
        # nrows instead of read-then-head: some corpus tables are several GB
        tables[os.path.basename(fn)] = pd.read_csv(fn, lineterminator='\n', nrows=max_rows)
    return tables


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="checkpoint .pt from run_pretrain.py")
    parser.add_argument("--tables", required=True, help="directory with *.csv tables")
    parser.add_argument("--out", required=True, help="output .pkl")
    parser.add_argument("--max_rows", type=int, default=1000)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    model, dataset = load_model(args.model, args.device)
    tables = read_tables(args.tables, args.max_rows)

    # one bad table (e.g. > 512 tokens, or a cell containing "<s>") must not
    # discard hours of embeddings: skip it and report it at the end
    embeddings, failed = {}, {}
    for name, df in tqdm(tables.items(), total=len(tables)):
        try:
            embeddings[name] = embed_table(df, model, dataset)
        except Exception as e:  # noqa: BLE001
            failed[name] = "%s: %s" % (type(e).__name__, str(e)[:200])

    with open(args.out, "wb") as f:
        pickle.dump(embeddings, f)
    print("wrote %d tables to %s" % (len(embeddings), args.out))
    if failed:
        print("FAILED %d tables (not in output):" % len(failed))
        for name, err in sorted(failed.items()):
            print("  ", name, err)


if __name__ == "__main__":
    main()
