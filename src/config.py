"""Single source of truth for paths, models and credentials."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# ---------- paths ----------
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
PAPERS_JSONL = DATA_DIR / "arxiv_papers.jsonl"
CHROMA_DIR = ROOT / "chromadb"

# ---------- collections ----------
COARSE_COLLECTION = "papers_abstracts"  # stage 1: title + abstract
FINE_COLLECTION = "paper_chunks"  # stage 2: full-text chunks

# ---------- models ----------
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ---------- chunking ----------
# all-MiniLM-L6-v2 truncates at 256 tokens (~1024 chars), so chunks must fit
# inside that window or their tail never influences retrieval.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# ---------- LLM (DeepSeek, OpenAI-compatible) ----------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-flash").strip()
