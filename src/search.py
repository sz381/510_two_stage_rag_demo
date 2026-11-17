"""The two retrievers: coarse (abstracts) and fine (full-text chunks)."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Sequence

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from . import config


@lru_cache(maxsize=1)
def embedder() -> SentenceTransformer:
    return SentenceTransformer(config.EMBED_MODEL)


@lru_cache(maxsize=1)
def client() -> "chromadb.api.ClientAPI":
    return chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def collection(name: str):
    return client().get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def _safe(text: str) -> str:
    """Strip lone surrogates that the tokenizer cannot encode."""
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")


def embed(texts: Sequence[str]) -> List[List[float]]:
    return embedder().encode(
        [_safe(text) for text in texts],
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()


def counts() -> Dict[str, int]:
    return {
        config.COARSE_COLLECTION: collection(config.COARSE_COLLECTION).count(),
        config.FINE_COLLECTION: collection(config.FINE_COLLECTION).count(),
    }


# --------------------------------------------------------------------------- #
# stage 1 — coarse retrieval over title + abstract
# --------------------------------------------------------------------------- #
def search_coarse(query: str, k: int = 5) -> List[dict]:
    result = collection(config.COARSE_COLLECTION).query(
        query_embeddings=embed([query]),
        n_results=max(1, k),
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append(
            {
                "arxiv_id": meta.get("arxiv_id"),
                "title": meta.get("title"),
                "published": meta.get("published"),
                "categories": meta.get("categories"),
                "authors": meta.get("authors"),
                "text": doc,
                "score": round(1 - distance, 4),
            }
        )
    return hits


# --------------------------------------------------------------------------- #
# stage 2 — fine retrieval over full-text chunks, restricted to stage-1 papers
# --------------------------------------------------------------------------- #
def search_fine(query: str, arxiv_ids: Sequence[str], k: int = 8) -> List[dict]:
    """Search chunks of the given papers only. Empty list when none are indexed."""
    ids = [i for i in arxiv_ids if i]
    if not ids:
        return []
    where = {"arxiv_id": ids[0]} if len(ids) == 1 else {"arxiv_id": {"$in": ids}}
    try:
        result = collection(config.FINE_COLLECTION).query(
            query_embeddings=embed([query]),
            n_results=max(1, k),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:  # empty collection / no match for the filter
        return []
    hits = []
    for doc, meta, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append(
            {
                "arxiv_id": meta.get("arxiv_id"),
                "chunk_index": meta.get("chunk_index"),
                "title": meta.get("title"),
                "text": doc,
                "score": round(1 - distance, 4),
            }
        )
    return hits


def indexed_ids() -> set:
    """arxiv_ids that already have full-text chunks (used to skip work on rebuild)."""
    data = collection(config.FINE_COLLECTION).get(include=["metadatas"])
    return {m.get("arxiv_id") for m in data["metadatas"] if m.get("arxiv_id")}
