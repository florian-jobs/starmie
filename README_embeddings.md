# Starmie column embeddings (beluga fork)

Pretrained [Starmie](https://www.vldb.org/pvldb/vol16/p1726-fan.pdf) column encoders for **OpenData** and
**WebTables**. Original code and README: [`README_original.md`](README_original.md).

## Quick start on herkules (web_tables model)
```bash
git checkout starmie && git submodule update --init tools/starmie && cd tools/starmie
python3 -m venv ~/starmie-venv && source ~/starmie-venv/bin/activate
pip install -r requirements-starmie.lock.txt

python flatten_corpus.py /mnt/data2/luca/projects/beluga/data/corpora/web_tables data/web_tables/tables
python embed_columns.py \
  --model /mnt/data2/florian/beluga/tools/starmie/models/web_tables_allrows/model_drop_col_tfidf_entity_column_0.pt \
  --tables data/web_tables/tables --out web_tables_embeddings.pkl      # ~10 min, 4,678 tables
```
```python
import pickle
E = pickle.load(open("web_tables_embeddings.pkl", "rb"))   # {table_file: {column_name: np.ndarray(768,)}}
```
Already computed: `/mnt/data2/florian/beluga/tools/starmie/models/web_tables_allrows/embeddings.pkl`.

## 1. Setup (Python 3.12, CPU)
```bash
python3 -m venv ~/data/venvs/starmie && source ~/data/venvs/starmie/bin/activate
pip install -r requirements-starmie.lock.txt
```

## 2. Flatten the corpus (required)
Starmie only reads `*.csv` directly inside one folder; beluga's corpora are nested. This creates symlinks to
exactly the tables beluga indexes (`<leaf dir>/table.csv`, incl. `fragments/`); the corpus is not changed.
```bash
python flatten_corpus.py <corpus_root> data/<name>/tables
```

## 3. Embed columns
| Corpus | Model (herkules) |
|---|---|
| web_tables | `/mnt/data2/florian/beluga/tools/starmie/models/web_tables_allrows/model_drop_col_tfidf_entity_column_0.pt` |
| open_data | follows |

```bash
python embed_columns.py --model <model.pt> --tables data/<name>/tables --out embeddings.pkl
# -> {table_file: {column_name: np.ndarray(768,)}}
```
```python
import pandas as pd
from embed_columns import load_model, embed_table
model, ds = load_model("<model.pt>")
vectors = embed_table(pd.read_csv("table.csv", nrows=1000), model, ds)   # {column_name: vector}
```
Precomputed web_tables embeddings: `embeddings.pkl` next to the model.

## 4. Train a model
```bash
python run_pretrain.py --task <name> --data data/<name>/tables --size 100000000 --max_rows 1000 \
  --n_epochs 3 --batch_size 64 --lr 5e-5 --lm roberta --max_len 128 --projector 768 \
  --augment_op drop_col --sample_meth tfidf_entity --table_order column --run_id 0 --save_model
# -> results/<name>/model_drop_col_tfidf_entity_column_0.pt
```
Hyperparameters as in the paper (`drop_col` + `tfidf_entity`: best operators, appendix A.1/A.2).

## 5. Deviations from upstream
- `torch.optim.AdamW(weight_decay=0.0, eps=1e-6)` replaces the removed `transformers.AdamW` (same defaults).
- CPU training (`model.to(device)`), no `--fp16`; `embed_columns.py` uses `model.eval()` (deterministic).
- `--max_rows 1000`: training reads the first 1000 rows per table, as upstream does for embedding
  (`extractVectors.py`). Needed for open_data (76 GB CSV, tables up to 9 GB, ~375 GB RAM without it).
  The web_tables model above was trained on all rows (as upstream).
- Training tables = beluga's corpus tables: open_data 2,804, web_tables 4,678. 84 web_tables tables have no
  header; Starmie reads row 1 as header, so 1 data row is lost per such table.
