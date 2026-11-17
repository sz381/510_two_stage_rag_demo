"""Two-stage retrieval (coarse -> fine) with an optional DeepSeek answer.

Stage 1  papers_abstracts : title + abstract of every paper  -> top-k papers
Stage 2  paper_chunks     : 1500-char full-text chunks of *those* papers -> top-k chunks
Generation is a separate, optional step on top of the retrieved chunks.
"""

from __future__ import annotations

import time
from typing import Dict, List

from . import llm, search

SYSTEM_PROMPT = (
    "You are a research assistant. Answer only from the numbered sources provided. "
    "Cite them as [1], [2]. If the sources do not cover the question, say so plainly."
)

USER_TEMPLATE = """Sources:
{context}

Question: {question}

Answer concisely, citing the sources you used."""


def retrieve(query: str, k_papers: int = 5, k_chunks: int = 8) -> Dict:
    """Run both retrieval stages and return everything the UI needs."""
    started = time.perf_counter()
    papers = search.search_coarse(query, k=k_papers)
    after_coarse = time.perf_counter()

    arxiv_ids = [p["arxiv_id"] for p in papers]
    chunks = search.search_fine(query, arxiv_ids, k=k_chunks)
    finished = time.perf_counter()

    return {
        "query": query,
        "papers": papers,
        "chunks": chunks,
        "with_fulltext": sorted({c["arxiv_id"] for c in chunks}),
        "timings": {
            "coarse_ms": round((after_coarse - started) * 1000),
            "fine_ms": round((finished - after_coarse) * 1000),
        },
    }


def build_context(result: Dict, max_sources: int = 8) -> str:
    """Fine chunks when available, otherwise the coarse abstracts as a fallback."""
    blocks: List[str] = []
    for number, chunk in enumerate(result["chunks"][:max_sources], start=1):
        blocks.append(
            f"[{number}] arXiv:{chunk['arxiv_id']} — {chunk.get('title', '')}\n{chunk['text']}"
        )
    if not blocks:
        for number, paper in enumerate(result["papers"][:max_sources], start=1):
            blocks.append(
                f"[{number}] arXiv:{paper['arxiv_id']} — {paper.get('title', '')}\n{paper['text']}"
            )
    return "\n\n".join(blocks)


def answer(query: str, k_papers: int = 5, k_chunks: int = 8) -> Dict:
    """Retrieve, then let DeepSeek write an answer over the retrieved context."""
    result = retrieve(query, k_papers=k_papers, k_chunks=k_chunks)
    result["answer"] = None
    result["error"] = None

    if not llm.available():
        result["error"] = "DEEPSEEK_API_KEY not configured — showing retrieval only."
        return result

    prompt = USER_TEMPLATE.format(context=build_context(result), question=query)
    started = time.perf_counter()
    try:
        result["answer"] = llm.generate(SYSTEM_PROMPT, prompt)
    except llm.LLMError as exc:
        result["error"] = str(exc)
    result["timings"]["llm_ms"] = round((time.perf_counter() - started) * 1000)
    return result
