"""Run with:  python -m unittest test_data_path -v   (from tools/starmie)"""
import unittest

from data_path import resolve_data_path


class TestOriginalBehaviour(unittest.TestCase):
    """Without --data the path must be exactly what upstream Starmie used."""

    def test_santos(self):
        self.assertEqual(resolve_data_path("santos"), "data/santos/datalake")

    def test_santos_large(self):
        self.assertEqual(resolve_data_path("santosLarge"),
                         "data/santos-benchmark/real-benchmark/datalake")

    def test_tus(self):
        self.assertEqual(resolve_data_path("tus"),
                         "data/table-union-search-benchmark/small/benchmark")

    def test_tus_large(self):
        self.assertEqual(resolve_data_path("tusLarge"),
                         "data/table-union-search-benchmark/large/benchmark")

    def test_other_task(self):
        self.assertEqual(resolve_data_path("small"), "data/small/tables")


class TestDataOverride(unittest.TestCase):
    """With --data that directory is used, whatever the task is."""

    def test_override_plain_task(self):
        self.assertEqual(resolve_data_path("open_data", "/home/x/data/open_data"),
                         "/home/x/data/open_data")

    def test_override_beats_santos(self):
        self.assertEqual(resolve_data_path("santos", "my/tables"), "my/tables")


if __name__ == "__main__":
    unittest.main()
