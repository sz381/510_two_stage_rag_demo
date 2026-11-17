---
title: arXiv Two-Stage Semantic Search
emoji: 🔎
colorFrom: indigo
colorTo: purple
sdk: streamlit
sdk_version: 1.65.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# arXiv Two-Stage Semantic Search

AIPI 510 · Week 13 · Interactive #2

Semantic search over arXiv papers about retrieval-augmented generation, in two
retrieval stages, plus a DeepSeek answer grounded in the retrieved chunks.

Corpus: 120 arXiv papers matching `all:"retrieval augmented generation"`
(2021–2026, relevance-ranked), full text from their PDFs.

| stage | Chroma collection | unit | index |
|---|---|---|---|
| 1 · coarse | `papers_abstracts` | title + abstract | 120 papers |
| 2 · fine | `paper_chunks` | 1000-char PDF chunk, 150 overlap | 120 papers |

Both stages cover the same corpus, so every query can be refined at chunk level.
Stage 2 is restricted to the papers stage 1 returned
(`where={"arxiv_id": {"$in": …}}`), so full text is only searched inside a small
candidate set instead of the whole corpus.

Chunks are 1000 characters because `all-MiniLM-L6-v2` truncates at 256 tokens
(~1024 characters) — longer chunks would have their tail silently ignored.
Reference sections are cut before chunking: a bibliography embeds like prose but
can never answer a question, and it was the top source of junk hits.

## Structure

```
app.py               Streamlit UI
src/config.py        paths, models, credentials
src/data.py          arXiv metadata + PDF -> text -> chunks
src/index.py         builds both collections (CLI)
src/search.py        coarse / fine retrievers
src/llm.py           DeepSeek client (OpenAI-compatible)
src/rag.py           two-stage retrieval + answer
chromadb/            pre-built index (committed)
data/                metadata jsonl + PDFs (git-ignored)
```

## Build the index

```bash
python -m src.index --fetch --coarse          # stage 1
python -m src.index --fine --fine-limit 120   # stage 2
```

## Run

```bash
streamlit run app.py
```

## Env

`.env` holds `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL_NAME`
(see `.env.example`). On Hugging Face Spaces add the same keys as Space secrets.
