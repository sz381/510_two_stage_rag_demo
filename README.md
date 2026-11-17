---
title: Semantic Search Engine
emoji: 🔎
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.67.0
app_file: app.py
pinned: false
---

# Semantic Search Engine

AIPI 510 · Data Sourcing for Analytics · Week 13 · Interactive #2

A semantic (meaning-based) search app built from scratch: a dataset is chunked,
embedded with a sentence-transformer, stored in a Chroma vector database, and
queried through a Streamlit UI deployed on Hugging Face Spaces.

> If you build the UI with Gradio instead, change `sdk: streamlit` to
> `sdk: gradio` and `sdk_version` to `5.1.0` in the front matter above.

---

## What it does

Type a query in natural language — not keywords — and the app returns the most
semantically similar passages from the dataset, with a similarity score.

**Pipeline:** `documents → chunks → embeddings → Chroma → top-k search → UI`

## Dataset

- **Source:** TODO — where the raw data came from (URL / API / Kaggle / HF dataset)
- **Size:** TODO — number of documents, average length
- **Why it's interesting:** TODO — what a user would actually want to search for

## Chunking strategy

- **Method:** TODO — e.g. fixed-size sliding window / by paragraph / by row field
- **Params:** TODO — chunk size (characters or tokens), overlap
- **Rationale:** TODO — why this unit matches how people query (a chunk should be
  one answer-sized idea, not half a sentence and not a whole document)

## Embedding model

`sentence-transformers/all-MiniLM-L6-v2` — 384-dim, fast on CPU, good default for
short passages. Every document and every query must use the **same** model.

## Repo layout

```
.
├── app.py                     # Streamlit app: load Chroma, embed query, render top-k
├── chromadb/                  # pre-built vector DB (committed — the app needs it)
│   └── chroma.sqlite3         # documents + embeddings + metadata
├── requirements.txt
├── README.md                  # this file (also the HF Space config card)
├── notebooks/
│   └── 01_build_index.ipynb   # pick data → chunk → embed → persist to chromadb/
└── data/                      # raw source data (git-ignored)
```

`chromadb/` is committed **on purpose**: the Space has no build step, it just
serves a pre-computed index. If the UUID folder holding the HNSW `.bin` files is
missing, Chroma rebuilds that index from `chroma.sqlite3` on first load — correct
results, slightly slower cold start.

## Run locally

```bash
python3.12 -m venv .venv          # 3.12: chromadb 0.6.1 has no 3.13/3.14 wheels
source .venv/bin/activate
pip install -r requirements.txt

jupyter lab notebooks/01_build_index.ipynb   # step 1–3: build chromadb/
streamlit run app.py                          # step 4: use it
```

## Deploy

```bash
# one-time
pip install -U huggingface_hub
hf auth login

# create the Space (once), then push
git remote add space https://huggingface.co/spaces/<username>/<space-name>
git push space main
```

The Space reads the front matter above, installs `requirements.txt`, and runs
`python -m streamlit run app.py`. First load also downloads the MiniLM weights
(~90 MB), so expect a slow first request and a cold start of ~30–60 s after the
Space has been idle.

## Limitations / next steps

- TODO — e.g. no reranking, English-only model, fixed top-k, no metadata filters
