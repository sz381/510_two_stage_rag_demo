"""Streamlit UI — two-stage semantic search over arXiv papers."""

from __future__ import annotations

import streamlit as st

from src import config, llm, rag, search

st.set_page_config(page_title="arXiv Two-Stage Search", page_icon="🔎", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model…")
def _warm_up() -> bool:
    search.embedder()
    return True


@st.cache_data(show_spinner=False)
def _index_counts() -> dict:
    return search.counts()


def _abstract_only(document: str) -> str:
    return document.split("Abstract:", 1)[-1].strip()


def _timings(timings: dict) -> None:
    parts = [f"{key.replace('_ms', '')} {value} ms" for key, value in timings.items()]
    st.caption(" · ".join(parts))


def _render_papers(result: dict) -> None:
    st.subheader(f"Stage 1 · coarse retrieval — {len(result['papers'])} papers")
    for rank, paper in enumerate(result["papers"], start=1):
        with st.expander(f"{paper['score']:.3f} · {paper['title']}", expanded=rank == 1):
            st.caption(
                f"arXiv:{paper['arxiv_id']} · {paper.get('published', '')} · "
                f"{paper.get('categories', '')}"
            )
            if paper.get("authors"):
                st.caption(f"Authors: {paper['authors']}")
            st.write(_abstract_only(paper["text"]))
            st.markdown(
                f"[arxiv.org/abs/{paper['arxiv_id']}](https://arxiv.org/abs/{paper['arxiv_id']})"
            )


def _render_chunks(result: dict) -> None:
    st.subheader(f"Stage 2 · fine retrieval — {len(result['chunks'])} chunks")
    if not result["chunks"]:
        st.info(
            f"No full-text chunks for these {len(result['papers'])} papers — "
            "build stage 2 with `python -m src.index --fine --fine-limit 120`."
        )
        return
    for rank, chunk in enumerate(result["chunks"], start=1):
        with st.expander(
            f"{chunk['score']:.3f} · {chunk.get('title', '')} · chunk {chunk.get('chunk_index')}",
            expanded=rank == 1,
        ):
            st.caption(f"arXiv:{chunk['arxiv_id']}")
            st.write(chunk["text"])


def main() -> None:
    counts = _index_counts()
    coarse_n = counts[config.COARSE_COLLECTION]
    fine_n = counts[config.FINE_COLLECTION]

    with st.sidebar:
        st.header("Settings")
        k_papers = st.slider("Stage 1 · papers", 1, 10, 5)
        k_chunks = st.slider("Stage 2 · chunks", 1, 20, 8)
        use_llm = st.toggle("Generate answer (DeepSeek)", value=True)
        st.divider()
        st.metric("Papers indexed (stage 1)", coarse_n)
        st.metric("Chunks indexed (stage 2)", fine_n)
        st.caption(f"Embeddings: `{config.EMBED_MODEL.split('/')[-1]}`")
        st.caption(f"LLM: `{config.DEEPSEEK_MODEL}`" if llm.available() else "LLM: no API key")

    st.title("arXiv two-stage semantic search")
    st.caption(
        "Stage 1 embeds titles + abstracts to pick candidate papers; "
        "stage 2 embeds 1500-character full-text chunks of those papers only."
    )

    if coarse_n == 0:
        st.warning(
            "The index is empty. Build it first:\n\n"
            "```bash\n"
            "python -m src.index --fetch --coarse\n"
            "python -m src.index --fine --fine-limit 120\n"
            "```"
        )
        return

    _warm_up()
    query = st.text_input(
        "Ask a research question",
        placeholder="e.g. how do people reduce hallucination in retrieval-augmented generation?",
    )

    if query.strip():
        with st.spinner("Retrieving…"):
            if use_llm:
                result = rag.answer(query, k_papers=k_papers, k_chunks=k_chunks)
            else:
                result = rag.retrieve(query, k_papers=k_papers, k_chunks=k_chunks)
        st.session_state["result"] = result

    result = st.session_state.get("result")
    if not result:
        return

    st.divider()
    if result.get("answer"):
        st.markdown("### Answer")
        st.markdown(result["answer"])
    elif result.get("error"):
        st.info(result["error"])

    _timings(result["timings"])
    st.divider()
    _render_papers(result)
    st.divider()
    _render_chunks(result)


main()
