"""Run with:  python -m unittest test_embed_columns -v   (from tools/starmie)

Needs torch + transformers (run on herkules). Downloads distilbert-base-uncased
once (~250 MB) to build a tiny untrained checkpoint; no real training needed.
"""
import os
import tempfile
import unittest
from argparse import Namespace

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class TestEmbedColumns(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from sdd.model import BarlowTwinsSimCLR

        cls.tmp = tempfile.TemporaryDirectory()
        hp = Namespace(task="unit_test", lm="distilbert", projector=768, max_len=128,
                       size=None, augment_op="drop_col", single_column=False,
                       sample_meth="head", table_order="column", run_id=0)
        model = BarlowTwinsSimCLR(hp, device="cpu", lm=hp.lm)
        cls.ckpt_path = os.path.join(cls.tmp.name, "model.pt")
        torch.save({"model": model.state_dict(), "hp": hp}, cls.ckpt_path)

        cls.df = pd.DataFrame({
            "city": ["Berlin", "Paris", "Rome"],
            "country": ["Germany", "France", "Italy"],
            "population": [3_600_000, 2_100_000, 2_800_000],
        })

        # run from a directory without data/ to prove no benchmark path is needed
        cls.old_cwd = os.getcwd()
        os.chdir(cls.tmp.name)
        from embed_columns import load_model
        cls.model, cls.dataset = load_model(cls.ckpt_path, device="cpu")

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.old_cwd)
        cls.tmp.cleanup()

    def test_model_in_eval_mode(self):
        self.assertFalse(self.model.training, "call model.eval() so dropout is off")

    def test_one_vector_per_column_in_order(self):
        from embed_columns import embed_table
        emb = embed_table(self.df, self.model, self.dataset)
        self.assertEqual(list(emb.keys()), list(self.df.columns))

    def test_vector_shape(self):
        from embed_columns import embed_table
        emb = embed_table(self.df, self.model, self.dataset)
        for col, vec in emb.items():
            self.assertIsInstance(vec, np.ndarray, col)
            self.assertEqual(vec.shape, (768,), col)

    def test_deterministic(self):
        from embed_columns import embed_table
        a = embed_table(self.df, self.model, self.dataset)
        b = embed_table(self.df, self.model, self.dataset)
        for col in self.df.columns:
            np.testing.assert_allclose(a[col], b[col], rtol=0, atol=1e-6)

    def test_matches_upstream_inference(self):
        """Same vectors as Starmie's own inference_on_tables (with eval mode)."""
        from embed_columns import embed_table
        from sdd.pretrain import inference_on_tables
        emb = embed_table(self.df, self.model, self.dataset)
        upstream = inference_on_tables([self.df], self.model, self.dataset)[0]
        for col, vec in zip(self.df.columns, upstream):
            np.testing.assert_allclose(emb[col], vec, rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
