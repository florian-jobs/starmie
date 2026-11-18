"""Run with:  python -m unittest test_pretrain_cpu -v   (from tools/starmie)

Trains a tiny model (distilbert, 4 toy tables, 1 epoch) on CPU through the real
sdd.pretrain.train and checks:
  - the optimizer is torch.optim.AdamW with upstream's transformers.AdamW
    defaults (weight_decay=0.0, eps=1e-6) -> training stays faithful to Starmie
  - training runs without a GPU and writes a checkpoint
  - that checkpoint loads with embed_columns.load_model and embeds columns
Takes ~1 min on herkules.
"""
import os
import tempfile
import unittest
from argparse import Namespace
from unittest import mock

import pandas as pd
import torch

ORIGINAL_AdamW_init = torch.optim.AdamW.__init__


class TestPretrainCPU(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from sdd.dataset import PretrainTableDataset
        from sdd.pretrain import train

        cls.tmp = tempfile.TemporaryDirectory()
        table_dir = os.path.join(cls.tmp.name, "tables")
        os.makedirs(table_dir)
        toy = {
            "cities": {"city": ["Berlin", "Paris", "Rome", "Madrid"],
                       "country": ["Germany", "France", "Italy", "Spain"]},
            "people": {"name": ["Ada", "Alan", "Grace", "Linus"],
                       "born": [1815, 1912, 1906, 1969]},
            "fruit": {"fruit": ["apple", "pear", "plum", "fig"],
                      "colour": ["red", "green", "purple", "brown"]},
            "cars": {"make": ["VW", "BMW", "Fiat", "Seat"],
                     "hp": [90, 150, 70, 110]},
        }
        for name, cols in toy.items():
            pd.DataFrame(cols).to_csv(os.path.join(table_dir, name + ".csv"), index=False)

        cls.hp = Namespace(task="unit_test", logdir=os.path.join(cls.tmp.name, "results"),
                           run_id=0, batch_size=2, max_len=64, size=None, lr=5e-5,
                           n_epochs=1, lm="distilbert", projector=768,
                           augment_op="drop_col", save_model=True, fp16=False,
                           single_column=False, table_order="column", sample_meth="head")

        cls.optimizer_kwargs = []

        def spy(self, params, *args, **kwargs):
            cls.optimizer_kwargs.append((args, kwargs))
            ORIGINAL_AdamW_init(self, params, *args, **kwargs)

        trainset = PretrainTableDataset.from_hp(table_dir, cls.hp)
        # force CPU even on a GPU machine, so the test means the same everywhere
        with mock.patch("torch.cuda.is_available", return_value=False), \
             mock.patch.object(torch.optim.AdamW, "__init__", spy):
            train(trainset, cls.hp)

        cls.ckpt_path = os.path.join(cls.hp.logdir, "unit_test",
                                     "model_drop_col_head_column_0.pt")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_uses_torch_adamw(self):
        self.assertEqual(len(self.optimizer_kwargs), 1,
                         "train() must create exactly one torch.optim.AdamW")

    def test_adamw_hyperparameters_match_upstream(self):
        _, kwargs = self.optimizer_kwargs[0]
        self.assertEqual(kwargs.get("lr"), self.hp.lr)
        self.assertEqual(kwargs.get("weight_decay"), 0.0)
        self.assertEqual(kwargs.get("eps"), 1e-6)

    def test_checkpoint_written(self):
        self.assertTrue(os.path.exists(self.ckpt_path), self.ckpt_path)

    def test_checkpoint_embeds_columns(self):
        from embed_columns import load_model, embed_table
        model, dataset = load_model(self.ckpt_path, device="cpu")
        df = pd.DataFrame({"a": ["x", "y"], "b": [1, 2]})
        emb = embed_table(df, model, dataset)
        self.assertEqual(list(emb), ["a", "b"])
        self.assertEqual(emb["a"].shape, (768,))


if __name__ == "__main__":
    unittest.main()
