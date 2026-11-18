# Starmie column embeddings (beluga fork)

Pretrained [Starmie](https://www.vldb.org/pvldb/vol16/p1726-fan.pdf) column encoders for
**OpenData** and **WebTables**. Original code and README: [`README_original.md`](README_original.md).

## 1. Setup (Python 3.12, CPU)
```bash
python3 -m venv ~/data/venvs/starmie && source ~/data/venvs/starmie/bin/activate
pip install -r requirements-starmie.lock.txt   # exact env used for training
```

## 2. Flatten the corpus (required before steps 3 and 4)
Starmie (training and embedding) only reads `*.csv` files **directly inside one folder**.
beluga's corpora are nested (`<corpus>/.../<table_id>/table.csv`), so first create a flat folder:
```bash
python flatten_corpus.py <corpus_root> data/<name>/tables
```
- Links exactly the tables beluga indexes (`<leaf dir>/table.csv`, incl. `fragments/`); skips
  `join_paths.csv`, `reconstructed_table.csv` and non-csv files.
- Creates **symlinks only**: the corpus is not copied or changed; `rm -r data/<name>` removes just the links.

## 3. Embed columns with a pretrained model
The model is a checkpoint file (`.pt`) written by the pretraining in step 4, one per corpus:

| Corpus | Model file |
|---|---|
| open_data | `results/open_data/model_drop_col_tfidf_entity_column_0.pt` |
| web_tables | `results/web_tables/model_drop_col_tfidf_entity_column_0.pt` |

```bash
python embed_columns.py --model results/open_data/model_drop_col_tfidf_entity_column_0.pt \
  --tables data/open_data/tables --out embeddings.pkl
```
Output: `{table_file: {column_name: np.ndarray(768,)}}`. In Python:
```python
from embed_columns import load_model, embed_table
model, ds = load_model("results/open_data/model_drop_col_tfidf_entity_column_0.pt")  # cpu/cuda automatic
vectors = embed_table(df, model, ds)    # df: pandas DataFrame -> {column_name: vector}
```

## 4. Reproduce the models
```bash
python run_pretrain.py --task <name> --data data/<name>/tables --size 100000000 \
  --n_epochs 3 --batch_size 64 --lr 5e-5 --lm roberta --max_len 128 --projector 768 \
  --augment_op drop_col --sample_meth tfidf_entity --table_order column --run_id 0 --save_model
```
Writes the model file used in step 3: `results/<name>/model_drop_col_tfidf_entity_column_0.pt`.
Hyperparameters: paper defaults; `drop_col` + `tfidf_entity` are the best operators in the paper's appendix (A.1, A.2).

**Example (web_tables, end to end):**
```bash
python flatten_corpus.py ~/data/beluga/data/corpora/web_tables data/web_tables/tables   # step 2
python run_pretrain.py --task web_tables --data data/web_tables/tables --size 100000000 \
  --n_epochs 3 --batch_size 64 --lr 5e-5 --lm roberta --max_len 128 --projector 768 \
  --augment_op drop_col --sample_meth tfidf_entity --table_order column --run_id 0 --save_model
# -> results/web_tables/model_drop_col_tfidf_entity_column_0.pt
python embed_columns.py --model results/web_tables/model_drop_col_tfidf_entity_column_0.pt \
  --tables data/web_tables/tables --out results/web_tables/embeddings.pkl                 # step 3
```

## 5. Deviations from upstream
- `torch.optim.AdamW(weight_decay=0.0, eps=1e-6)` replaces the removed `transformers.AdamW` (same defaults).
- Runs on CPU (`model.to(device)`); trained without `--fp16`.
- `embed_columns.py` sets `model.eval()` (dropout off, deterministic); upstream does not.
- `--data` sets the training table directory; `--size` ≥ #tables, so all tables are used.
- Training tables = exactly beluga's corpus tables (step 2): open_data 2,804, web_tables 4,678.
- Starmie always reads row 1 as header; 84 web_tables tables have `has_header: false`, so their first
  data row is used as column names (Starmie embeds cell values only, so 1 row lost per such table).
- `embed_columns.py` reads tables in sorted order, first 1000 rows (as upstream `extractVectors.py`).
