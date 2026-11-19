# Corrective RAG (CRAG)

A simple implementation of **Corrective Retrieval-Augmented Generation (CRAG)** using
[LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain),
[ChromaDB](https://www.trychroma.com/), and [Groq](https://groq.com/).

Normal RAG blindly trusts whatever the retriever returns. CRAG adds a **self-correction
step**: it grades the retrieved documents, and depending on how good they are, decides
whether to answer directly, fall back to a live web search, or use a mix of both.

---

## How it works

The pipeline is built as a graph (`corrective.ipynb`) with the following stages:

```
            ┌───────────┐
            │  retrieve │   (similarity search over PDF chunks)
            └─────┬─────┘
                  ▼
          ┌────────────────┐
          │  eval_each_doc  │   (LLM scores each chunk 0.0 - 1.0)
          └─────┬───────────┘
                │
   ┌────────────┼─────────────────────┐
   │                                  │
CORRECT                      INCORRECT / AMBIGUOUS
(a chunk scored > 0.7)       (no chunk > 0.7)
   │                                  │
   │                          ┌───────▼────────┐
   │                          │ rewrite_query  │  (LLM rewrites question into
   │                          └───────┬────────┘   a short web search query)
   │                                  ▼
   │                          ┌───────────────┐
   │                          │  web_search    │  (Tavily search API)
   │                          └───────┬────────┘
   │                                  │
   └──────────────┬───────────────────┘
                  ▼
            ┌───────────┐
            │  refine    │   (split context into sentences,
            └─────┬──────┘    keep only relevant ones)
                  ▼
            ┌───────────┐
            │ generate   │   (LLM answers using ONLY the refined context)
            └───────────┘
```

### 1. Retrieve
The source PDF (`source/MEMORY_MECHANISM_LLM-4.pdf`) is split into chunks and embedded
with `sentence-transformers/all-MiniLM-L6-v2`, stored in a local Chroma vector store.
For every question, the top-k most similar chunks are retrieved.

### 2. Evaluate (Grading)
Each retrieved chunk is scored independently by an LLM evaluator on a 0.0–1.0 relevance
scale, with a short reason. Based on the scores, a verdict is assigned:

| Verdict      | Condition                                   | Meaning                          |
|--------------|----------------------------------------------|-----------------------------------|
| `CORRECT`    | at least one chunk scored > `UPPER_TH` (0.7) | retrieved docs are good enough    |
| `INCORRECT`  | all chunks scored < `LOWER_TH` (0.3)         | retrieved docs are useless        |
| `AMBIGUOUS`  | otherwise                                    | retrieved docs are only partially useful |

### 3. Route
- **CORRECT** → go straight to refinement using the good internal docs.
- **INCORRECT** → rewrite the query and search the web instead (internal docs discarded).
- **AMBIGUOUS** → rewrite the query, search the web, and combine both internal + web docs.

### 4. Knowledge Refinement ("strip" filtering)
This is the key CRAG idea: instead of dumping entire documents into the prompt, the
context is decomposed into individual sentences ("strips"), and an LLM filter decides
`keep: true/false` for each one. Only the relevant sentences are kept and joined back
into a clean, noise-free context.

### 5. Generate
A final LLM call answers the original question using **only** the refined context. If
the context is empty, it responds with "I don't know" instead of hallucinating.

---

## Project structure

```
corrective-rag/
├── corrective.ipynb      # Main notebook: builds and runs the CRAG graph
├── testing_api.py        # Small standalone script to test an LLM API (OpenRouter)
├── requirements.txt       # Python dependencies
├── source/                # Input PDF(s) used to build the vector store
└── .env                   # API keys (not committed)
```

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Create a `.env` file in the project root with the API keys you need, e.g.:
   ```
   GROQ_API_KEY=your_key_here
   TAVILY_API_KEY=your_key_here
   OPENAI_API_KEY=your_key_here
   ```
   Only `GROQ_API_KEY` (LLM) and `TAVILY_API_KEY` (web search) are required to run the
   notebook as-is.

3. Place a PDF in the `source/` folder (or point `PyPDFLoader` to your own file).

4. Open `corrective.ipynb` and run the cells top to bottom.

## Key components

| Component            | Library/Tool                                   |
|-----------------------|-------------------------------------------------|
| Vector store           | ChromaDB                                        |
| Embeddings             | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| LLM                    | Groq (`ChatGroq`)                                |
| Web search             | Tavily                                          |
| Orchestration          | LangGraph `StateGraph`                           |

## Notes
- Thresholds `UPPER_TH` / `LOWER_TH` control how strict the grading is and can be tuned
  per use case.
- Structured outputs (scores, keep/drop decisions, rewritten queries) are enforced with
  Pydantic models via `with_structured_output`.
