"""Run with:  python -m unittest test_starmie_encoder -v   (from tools/starmie, beluga's .venv is enough)"""
import os
import tempfile
import unittest
from argparse import Namespace

import numpy as np
import pandas as pd
import torch


class TestStarmieEncoder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from sdd.model import BarlowTwinsSimCLR
        from starmie_encoder import load_starmie

        cls.tmp = tempfile.TemporaryDirectory()
        hp = Namespace(task="unit_test", lm="distilbert", projector=768, max_len=128, size=None,
                       augment_op="drop_col", single_column=False, sample_meth="tfidf_entity",
                       table_order="column", run_id=0)
        cls.path = os.path.join(cls.tmp.name, "model.pt")
        torch.save({"model": BarlowTwinsSimCLR(hp, device="cpu", lm=hp.lm).state_dict(), "hp": hp}, cls.path)
        cls.starmie = load_starmie(cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_shape_and_types(self):
        for values in ([1, 2, 3], ["Berlin", "Paris"], [1.5, None, "x"]):
            v = self.starmie(values)
            self.assertEqual(v.shape, (768,))
            self.assertFalse(np.isnan(v).any())

    def test_deterministic(self):
        np.testing.assert_array_equal(self.starmie([1, 2, 3]), self.starmie([1, 2, 3]))

    def test_same_as_embed_columns(self):
        """Same vector as the table pipeline (embed_columns.py) for a one-column table."""
        try:
            from embed_columns import load_model, embed_table   # needs mlflow (starmie venv)
        except ImportError:
            self.skipTest("embed_columns needs the starmie venv")
        model, ds = load_model(self.path, device="cpu")
        expected = embed_table(pd.DataFrame({"col": [1, 2, 3]}), model, ds)["col"]
        np.testing.assert_allclose(self.starmie([1, 2, 3]), expected, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
