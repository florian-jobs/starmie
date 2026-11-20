"""Starmie: list of values -> embedding.  Works in beluga's .venv (torch, transformers, pandas).

    import sys; sys.path.insert(0, "tools/starmie")   # path to this folder
    from starmie_encoder import load_starmie

    starmie = load_starmie()             # load the pretrained model once
    starmie([1, 2, 3])                   # -> np.ndarray, shape (768,)
    starmie(["Berlin", "Paris", "Rome"])
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sdd.dataset import PretrainTableDataset  # noqa: E402
from sdd.model import BarlowTwinsSimCLR  # noqa: E402

WEB_TABLES_MODEL = ("/mnt/data2/florian/beluga/tools/starmie/models/web_tables_allrows/"
                    "model_drop_col_tfidf_entity_column_0.pt")


def load_starmie(model_path=WEB_TABLES_MODEL, device="cpu"):
    """Load the model once; returns a function: list of values -> np.ndarray (768,)."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    hp = ckpt["hp"]
    model = BarlowTwinsSimCLR(hp, device=device, lm=hp.lm).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()                                                # deterministic
    serializer = PretrainTableDataset.from_hp(tempfile.mkdtemp(), hp)   # Starmie's tokenization

    def starmie(values):
        column = pd.DataFrame({"col": list(values)[:1000]})    # the values as one column
        tokens, _ = serializer._tokenize(column)               # "<s> v1 v2 v3 ..."
        with torch.no_grad():
            vectors = model.inference(torch.LongTensor([tokens]))   # one vector per <s>
        return vectors[0].cpu().numpy()

    return starmie
