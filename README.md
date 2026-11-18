# Starmie column embeddings (beluga fork)

Pretrained [Starmie](https://www.vldb.org/pvldb/vol16/p1726-fan.pdf) column encoders for
**OpenData** and **WebTables (small)**. Original code and README: [`README_starmie.md`](README_starmie.md).

## 1. Setup (Python 3.12, CPU)
```bash
python3 -m venv ~/data/venvs/starmie && source ~/data/venvs/starmie/bin/activate
pip install -r requirements-starmie.lock.txt   # exact env used for training
```

## 2. Embed columns with a pretrained model
```bash
python embed_columns.py --model <model.pt> --tables <dir with *.csv> --out embeddings.pkl
```
Output: `{table_file: {column_name: np.ndarray(768,)}}`. In Python:
```python
from embed_columns import load_model, embed_table
model, ds = load_model("model.pt")      # cpu/cuda picked automatically
vectors = embed_table(df, model, ds)    # {column_name: vector}
```

## 3. Reproduce the models
```bash
python flatten_corpus.py <corpus_root> data/<name>/tables   # symlinks to the tables beluga indexes (leaf dirs)
python run_pretrain.py --task <name> --data data/<name>/tables --size 100000000 \
  --n_epochs 3 --batch_size 64 --lr 5e-5 --lm roberta --max_len 128 --projector 768 \
  --augment_op drop_col --sample_meth tfidf_entity --table_order column --run_id 0 --save_model
```
Model: `results/<name>/model_drop_col_tfidf_entity_column_0.pt`.
Hyperparameters: paper defaults; `drop_col` + `tfidf_entity` are the best operators in the paper's appendix (A.1, A.2).

## 4. Deviations from upstream
- `torch.optim.AdamW(weight_decay=0.0, eps=1e-6)` replaces the removed `transformers.AdamW` (same defaults).
- Runs on CPU (`model.to(device)`); trained without `--fp16`.
- `embed_columns.py` sets `model.eval()` (dropout off, deterministic); upstream does not.
- `--data` sets the training table directory; `--size` ≥ #tables, so all tables are used.
- Training tables = exactly beluga's corpus tables (`<leaf dir>/table.csv`, incl. `fragments/`;
  `join_paths.csv` and `reconstructed_table.csv` excluded): open_data 2,804, web_tables 4,678.
- Starmie always reads row 1 as header; 84 web_tables tables have `has_header: false`, so their first
  data row is used as column names (Starmie embeds cell values only, so 1 row lost per such table).
- `embed_columns.py` reads tables in sorted order, first 1000 rows (as upstream `extractVectors.py`).

## 5. Tests
```bash
python -m unittest test_data_path test_embed_columns test_pretrain_cpu -v
```
