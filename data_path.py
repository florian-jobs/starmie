"""Resolve the table directory that run_pretrain.py trains on.

Kept free of heavy imports (torch, mlflow, transformers) so it can be
tested without the training environment.
"""
from typing import Optional


def resolve_data_path(task: str, data: Optional[str] = None) -> str:
    """Return the directory of CSV tables to pretrain on.

    Args:
        task: the --task name (also used for the output folder results/<task>/).
        data: the --data argument; None if it was not passed.

    Returns:
        ``data`` if it was given; otherwise the path the original Starmie code
        derives from ``task``:
          - task contains "santos": 'data/<task>/datalake',
            except "santosLarge": 'data/santos-benchmark/real-benchmark/datalake'
          - task contains "tus": 'data/table-union-search-benchmark/small/benchmark',
            except "tusLarge": 'data/table-union-search-benchmark/large/benchmark'
          - anything else: 'data/<task>/tables'
    """
    if data:
        return data
    elif "santos" in task:
        path = 'data/%s/datalake' % task
        if task == "santosLarge":
            path = 'data/santos-benchmark/real-benchmark/datalake'
        return path
    elif "tus" in task:
        path = 'data/table-union-search-benchmark/small/benchmark'
        if task == "tusLarge":
            path = 'data/table-union-search-benchmark/large/benchmark'
        return path
    path = 'data/%s/tables' % task
    return path
