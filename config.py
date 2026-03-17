"""
⚙️ CONFIG — Change anything here. One file controls the entire RAG system.

Open this file, tweak a number, restart the app. That's it.
"""

# ── LLM ──────────────────────────────────────────────────────────────
LLM_MODEL = "gpt-4.1-mini"       # Options: "gpt-4.1-mini", "gpt-4.1", "gpt-4o", "gpt-4o-mini"
LLM_TEMPERATURE = 0.1             # 0 = deterministic, 1 = creative
LLM_MAX_TOKENS = 1024             # Max response length

# ── Embeddings ───────────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"   # Options: "text-embedding-3-small" (fast/cheap), "text-embedding-3-large" (accurate)
EMBEDDING_DIMENSIONS = 1536                   # 1536 for small, 3072 for large

# ── Chunking ─────────────────────────────────────────────────────────
CHUNK_SIZE = 512                  # Characters per chunk (sweet spot: 256-1024)
CHUNK_OVERLAP = 50                # Overlap between chunks (prevents cutting sentences)
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]   # Split priority: paragraphs → lines → sentences → words

# ── Search ───────────────────────────────────────────────────────────
TOP_K = 5                         # Number of final chunks sent to LLM
BM25_WEIGHT = 0.4                 # Keyword search weight (0.0 to 1.0)
SEMANTIC_WEIGHT = 0.6             # Vector search weight (0.0 to 1.0) — should sum to 1.0 with BM25
MMR_FETCH_K = 20                  # Candidates pool for MMR diversity re-ranking
MMR_LAMBDA = 0.7                  # 1.0 = pure relevance, 0.0 = pure diversity

# ── Semantic Cache ───────────────────────────────────────────────────
CACHE_ENABLED = True              # Toggle cache on/off
CACHE_THRESHOLD = 0.88            # Similarity needed for cache hit (0.0 to 1.0, higher = stricter)

# ── Database ─────────────────────────────────────────────────────────
CHROMA_PATH = "./chroma_db"       # Where ChromaDB stores vectors (local folder)
COLLECTION_DOCS = "documents"     # Collection name for document chunks
COLLECTION_CACHE = "semantic_cache"  # Collection name for cached answers

# ── File Processing ──────────────────────────────────────────────────
SUPPORTED_TYPES = ["pdf", "docx", "doc", "txt", "png", "jpg", "jpeg", "webp"]
IMAGE_DESCRIPTION_PROMPT = (
    "Describe this image in detail for a knowledge base. "
    "Cover all visible text, data, charts, diagrams, and key visual elements."
)
