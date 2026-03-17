# 🔮 Multimodal RAG — "Hi, I'm Bing!"

A production-grade **Multimodal RAG** system. Upload PDFs, images, Word docs, or text — then ask questions and get accurate answers. Built with LangChain, ChromaDB, and GPT-4.1-mini.

![Python 3.12](https://img.shields.io/badge/Python-3.12-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-red) ![LangChain](https://img.shields.io/badge/LangChain-0.3-green)

---

## How It Works (Plain English)

1. **You drag-drop files** into the sidebar → text is extracted from every page, image, and paragraph.
2. **Text is chopped into chunks** → 512-character pieces with 50-char overlap so nothing gets cut mid-sentence.
3. **Each chunk becomes a vector** → a 1536-number array where similar meanings sit close together in space.
4. **Vectors go into ChromaDB** → a local database on your machine. Nothing leaves your system except API calls.
5. **You ask a question** → it's searched two ways at once:
   - **Semantic search** — finds chunks with the closest *meaning* (even if different words are used)
   - **BM25 keyword search** — finds chunks with the best *exact word overlap* (like classic Google)
   - Both scores are fused: `0.6 × semantic + 0.4 × bm25` → best of both worlds.
6. **GPT-4.1-mini reads the top chunks** and writes a clear, grounded answer.
7. **The answer is cached** → ask something similar later, and it's returned instantly (no API cost).

---

## Quick Start

```bash
git clone https://github.com/your-username/multimodal-rag.git
cd multimodal-rag
pip install -r requirements.txt
streamlit run app.py
```

Paste your OpenAI API key in the sidebar. Upload files. Ask questions. Done.

---

## ⚙️ Want to Change Something? Open `config.py`

**Every single setting lives in one file.** Open `config.py` and change whatever you want:

| What | Setting | Default | Options |
|------|---------|---------|---------|
| LLM model | `LLM_MODEL` | `gpt-4.1-mini` | `gpt-4.1`, `gpt-4o`, `gpt-4o-mini` |
| Creativity | `LLM_TEMPERATURE` | `0.1` | `0.0` (strict) to `1.0` (creative) |
| Embedding model | `EMBEDDING_MODEL` | `text-embedding-3-small` | `text-embedding-3-large` |
| Chunk size | `CHUNK_SIZE` | `512` | `256` (precise) to `1024` (broad) |
| Chunk overlap | `CHUNK_OVERLAP` | `50` | `0` to `chunk_size/2` |
| Results count | `TOP_K` | `5` | `3` (fast) to `10` (thorough) |
| BM25 weight | `BM25_WEIGHT` | `0.4` | `0.0` to `1.0` |
| Semantic weight | `SEMANTIC_WEIGHT` | `0.6` | `0.0` to `1.0` |
| Cache strictness | `CACHE_THRESHOLD` | `0.88` | `0.80` (loose) to `0.95` (strict) |
| Cache on/off | `CACHE_ENABLED` | `True` | `True` / `False` |

Change a number, restart the app. That's it.

---

## File Structure

```
multimodal-rag/
├── config.py           ← ⚙️ ALL settings (edit this)
├── rag_engine.py       ← 🧠 Core logic (chunking, search, cache)
├── app.py              ← 🖥️ Streamlit UI (chat + BTS sidebar)
├── requirements.txt    ← 📦 Dependencies
├── .streamlit/
│   └── config.toml     ← 🎨 Dark theme
├── .env.example        ← 🔑 API key template
├── .gitignore          ← 🧹 Keeps repo clean
└── README.md
```

Three Python files. `config.py` is the control panel, `rag_engine.py` is the brain, `app.py` is the face.

---

## The BTS Toggle

Flip **"Curious about BTS? Click to explore!"** in the sidebar. For every query, you'll see the full pipeline breakdown — real numbers, not AI explanations:

- How your query was embedded (model, dimensions)
- Cache hit or miss (with similarity %)
- Semantic vs BM25 candidate counts
- Each retrieved chunk with its fused score breakdown
- How many characters were sent to the LLM
- Output length

This updates live with every new question.

---

## Architecture

| Layer | Choice | Why |
|-------|--------|-----|
| LLM | GPT-4.1-mini | Fast, cheap, multimodal (reads images natively) |
| Embeddings | text-embedding-3-small | Best cost/accuracy ratio at 1536 dims |
| Vector DB | ChromaDB | Zero config, local, cosine distance |
| Keyword Search | BM25 (built-in) | No extra dependency, handles exact matches |
| Hybrid Fusion | Weighted sum | Simple, predictable, tunable via config |
| Cache | ChromaDB (same DB) | Semantic matching, no extra infrastructure |
| Chunking | RecursiveCharacterTextSplitter | Smart paragraph → sentence → word fallback |

---

## Supported Files

| Type | Extensions | Processing |
|------|-----------|------------|
| PDF | `.pdf` | Text extracted + embedded images described by GPT vision |
| Word | `.docx` `.doc` | Paragraph text extracted |
| Image | `.png` `.jpg` `.jpeg` `.webp` | Full visual description by GPT-4.1-mini |
| Text | `.txt` | Read directly |

---

## Cost

Typical session (10 docs, 20 questions): **~$0.03–0.05**

- Embeddings: ~$0.002 (text-embedding-3-small = $0.02 per 1M tokens)
- LLM: ~$0.01–0.03 (gpt-4.1-mini is very affordable)
- Cache hits: $0.00

---

## Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Point to `app.py`
4. Add `OPENAI_API_KEY` in Secrets
5. Deploy

---

## License

MIT
