# Hybrid Search & Retrieval

Build a hybrid search engine from scratch that combines **dense embeddings** (BGE — semantic understanding) with **sparse embeddings** (SPLADE — precise lexical matching), fused using **Reciprocal Rank Fusion (RRF)**.

## What You'll Learn

| # | Topic | Key Concept |
|---|-------|-------------|
| 1 | The Search Problem | Why keyword search isn't enough |
| 2 | Dense Embeddings | Semantic similarity with neural nets |
| 3 | Sparse Embeddings & SPLADE | Learned lexical matching + term expansion |
| 4 | Why Hybrid? | When dense fails, when sparse fails |
| 5 | Data Preparation | Loading & exploring the Amazon ESCI dataset |
| 6 | Generating Embeddings | Creating both dense and sparse representations |
| 7 | Indexing with Qdrant | Storing vectors for fast retrieval |
| 8 | Hybrid Search | Running parallel dense + sparse queries |
| 9 | Reciprocal Rank Fusion | Merging ranked lists from different scoring systems |
| 10 | Evaluation | Measuring search quality with ground truth labels |
| 11 | Agentic RAG | LangChain agent that reasons over hybrid search results |

## Tech Stack

- **FastEmbed** — ONNX-based embedding generation (no PyTorch needed)
- **Qdrant** — vector database (in-memory mode for demo)
- **BAAI/bge-small-en-v1.5** — dense embeddings (384-dim)
- **SPLADE++ (prithivida/Splade_PP_en_v1)** — sparse embeddings with term expansion
- **LangChain + LangGraph** — agentic RAG pipeline (Part 11)
- **Amazon ESCI Dataset** — real-world product search relevance labels

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running

**Jupyter Notebook (recommended for learning):**
```bash
jupyter notebook "Hybrid_Search_&_Retrieval_Notebook.ipynb"
```
Or open the notebook in VS Code and select the `venv` kernel.

**Python script (Parts 1–10 only):**
```bash
python hybrid_search.py
```

## Notes

- Parts 1–10 run without any API keys
- Part 11 (Agentic RAG) requires an `OPENAI_API_KEY` environment variable
- First run downloads embedding models (~600MB total)
