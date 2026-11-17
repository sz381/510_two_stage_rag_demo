"""arXiv plumbing: metadata from the official API, full text from PDFs."""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

import requests
from pypdf import PdfReader

from . import config

API_URL = "https://export.arxiv.org/api/query"
PDF_URL = "https://arxiv.org/pdf/{arxiv_id}.pdf"
NS = {"a": "http://www.w3.org/2005/Atom"}
HEADERS = {"User-Agent": "aipi510-semantic-search/1.0 (course project)"}


# --------------------------------------------------------------------------- #
# metadata
# --------------------------------------------------------------------------- #
def fetch_metadata(
    query: str = 'all:"retrieval augmented generation"',
    target: int = 120,
    page: int = 100,
    pause: float = 3.0,
    sort_by: str = "relevance",
) -> List[dict]:
    """Pull `target` papers matching an arXiv query string as flat dicts."""
    papers: List[dict] = []
    start = 0
    while len(papers) < target:
        n = min(page, target - len(papers))
        params = urllib.parse.urlencode(
            {
                "search_query": query,
                "start": start,
                "max_results": n,
                "sortBy": sort_by,
                "sortOrder": "descending",
            }
        )
        request = urllib.request.Request(f"{API_URL}?{params}", headers=HEADERS)
        with urllib.request.urlopen(request, timeout=60) as response:
            root = ET.fromstring(response.read())

        for entry in root.findall("a:entry", NS):
            papers.append(
                {
                    "id": entry.find("a:id", NS).text.rsplit("/", 1)[-1],
                    "title": " ".join((entry.find("a:title", NS).text or "").split()),
                    "abstract": " ".join((entry.find("a:summary", NS).text or "").split()),
                    "published": (entry.find("a:published", NS).text or "")[:10],
                    "authors": [a.find("a:name", NS).text for a in entry.findall("a:author", NS)],
                    "categories": [c.get("term") for c in entry.findall("a:category", NS)],
                }
            )

        start += n
        print(f"  metadata: {len(papers)}/{target}")
        if len(papers) < target:
            time.sleep(pause)
    return papers


def save_papers(papers: List[dict], path: Path = config.PAPERS_JSONL) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for paper in papers:
            handle.write(json.dumps(paper, ensure_ascii=False) + "\n")


def load_papers(path: Path = config.PAPERS_JSONL) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `python -m src.index --fetch` first.")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# --------------------------------------------------------------------------- #
# full text
# --------------------------------------------------------------------------- #
def download_pdf(arxiv_id: str, dest_dir: Path = config.PDF_DIR) -> Path:
    """Download an arXiv PDF once and return the local path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{arxiv_id}.pdf"
    if path.exists() and path.stat().st_size > 10_000:
        return path

    response = requests.get(
        PDF_URL.format(arxiv_id=arxiv_id), headers=HEADERS, timeout=60, stream=True
    )
    response.raise_for_status()
    with path.open("wb") as handle:
        for block in response.iter_content(chunk_size=8192):
            handle.write(block)
    return path


def extract_text(pdf_path: Path) -> str:
    """Concatenate every page of a PDF into one string."""
    reader = PdfReader(str(pdf_path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def clean_text(text: str) -> str:
    """Drop citation markers and inline LaTeX, collapse whitespace.

    PDF extraction can leave lone surrogates (e.g. from math glyphs such as
    U+1D400 range characters), which the tokenizer refuses to encode — they are
    stripped here by round-tripping through UTF-8.
    """
    text = re.sub(r"\[[0-9]{1,3}\]", "", text)
    text = re.sub(r"\$[^$]{1,200}\$", "", text)
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")


def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> List[str]:
    """Fixed-size sliding window over characters."""
    step = chunk_size - overlap
    return [text[start : start + chunk_size] for start in range(0, len(text), step)]


REFERENCE_HEADING = re.compile(r"(?im)^[ \t]*(?:\d+[.)]?\s*)?(references|bibliography)\s*$")


def strip_references(text: str) -> str:
    """Cut the bibliography off a paper.

    Reference lists embed with the surrounding prose but never answer a question,
    so they are the single biggest source of junk retrieval hits. The *last*
    heading wins because papers also contain the word in running text.
    """
    matches = list(REFERENCE_HEADING.finditer(text))
    if not matches:
        return text
    cut = matches[-1].start()
    return text[:cut] if cut > len(text) * 0.3 else text


def paper_chunks(arxiv_id: str) -> List[str]:
    """PDF -> text -> drop references -> clean -> chunks. [] when the PDF is unusable."""
    try:
        raw = extract_text(download_pdf(arxiv_id))
    except Exception as exc:  # noqa: BLE001 - PDFs fail in many creative ways
        print(f"  ! {arxiv_id}: {exc}")
        return []
    body = strip_references(raw)
    return [c for c in chunk_text(clean_text(body)) if c.strip()]
