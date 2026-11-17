"""Build both Chroma collections.

  python -m src.index --fetch --coarse            # stage 1 over abstracts
  python -m src.index --fine --limit 250          # stage 2 over full-text chunks
  python -m src.index --coarse --fine --reset     # rebuild from scratch
"""

from __future__ import annotations

import argparse
import time

from . import config, data, search

EMBED_BATCH = 64


def _fresh(name: str, reset: bool):
    client = search.client()
    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


# --------------------------------------------------------------------------- #
# stage 1
# --------------------------------------------------------------------------- #
def build_coarse(limit: int | None = None, reset: bool = False) -> int:
    papers = data.load_papers()
    if limit:
        papers = papers[:limit]

    documents = [f"Title: {p['title']}\n\nAbstract: {p['abstract']}" for p in papers]
    metadatas = [
        {
            "arxiv_id": p["id"],
            "title": p["title"],
            "published": p["published"],
            "categories": ", ".join(p["categories"]),
            "authors": ", ".join(p["authors"][:4]),
        }
        for p in papers
    ]
    ids = [p["id"] for p in papers]

    col = _fresh(config.COARSE_COLLECTION, reset)
    for start in range(0, len(ids), EMBED_BATCH):
        stop = start + EMBED_BATCH
        col.add(
            ids=ids[start:stop],
            documents=documents[start:stop],
            metadatas=metadatas[start:stop],
            embeddings=search.embed(documents[start:stop]),
        )
        print(f"  coarse: {min(stop, len(ids))}/{len(ids)}")
    return col.count()


# --------------------------------------------------------------------------- #
# stage 2
# --------------------------------------------------------------------------- #
def build_fine(limit: int = 250, reset: bool = False, pause: float = 1.0) -> int:
    papers = data.load_papers()[:limit]
    col = _fresh(config.FINE_COLLECTION, reset)
    done = set() if reset else search.indexed_ids()

    added_chunks = 0
    failed: list[str] = []
    for number, paper in enumerate(papers, start=1):
        arxiv_id = paper["id"]
        if arxiv_id in done:
            continue

        chunks = data.paper_chunks(arxiv_id)
        if not chunks:
            failed.append(arxiv_id)
            continue

        for start in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[start : start + EMBED_BATCH]
            col.add(
                ids=[f"{arxiv_id}::{i}" for i in range(start, start + len(batch))],
                documents=batch,
                metadatas=[
                    {
                        "arxiv_id": arxiv_id,
                        "chunk_index": i,
                        "title": paper["title"],
                        "published": paper["published"],
                    }
                    for i in range(start, start + len(batch))
                ],
                embeddings=search.embed(batch),
            )

        added_chunks += len(chunks)
        print(f"  fine: {number}/{len(papers)} {arxiv_id} -> {len(chunks)} chunks")
        time.sleep(pause)

    print(f"  fine: {added_chunks} chunks added, {len(failed)} PDFs failed")
    return col.count()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the retrieval indexes.")
    parser.add_argument("--fetch", action="store_true", help="download paper metadata first")
    parser.add_argument("--query", default='all:"retrieval augmented generation"')
    parser.add_argument("--target", type=int, default=120, help="papers to download")
    parser.add_argument("--coarse", action="store_true", help="build stage-1 index")
    parser.add_argument("--fine", action="store_true", help="build stage-2 index")
    parser.add_argument("--limit", type=int, default=120, help="papers to index")
    parser.add_argument("--fine-limit", type=int, default=120, help="papers with full text")
    parser.add_argument("--reset", action="store_true", help="drop existing collections")
    args = parser.parse_args()

    if args.fetch:
        papers = data.fetch_metadata(query=args.query, target=args.target)
        data.save_papers(papers)
        print(f"saved {len(papers)} papers -> {config.PAPERS_JSONL}")

    if args.coarse:
        total = build_coarse(limit=args.limit, reset=args.reset)
        print(f"coarse collection: {total} papers")

    if args.fine:
        total = build_fine(limit=args.fine_limit, reset=args.reset)
        print(f"fine collection: {total} chunks")


if __name__ == "__main__":
    main()
