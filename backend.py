"""
Corrective RAG (CRAG) backend.


This is the single source of truth for the pipeline logic. Any front end
(Streamlit, CLI, API, etc.) should import from this module instead of
re-implementing the graph.
"""

import os
import re
import tempfile
from typing import List, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

load_dotenv()

# -----------------------------
# Config
# -----------------------------
UPPER_TH = 0.7
LOWER_TH = 0.3
LLM_MODEL = "qwen/qwen3.8-27b"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 4


# -----------------------------
# State & structured-output schemas
# -----------------------------
class State(TypedDict):
    question: str

    docs: List[Document]
    good_docs: List[Document]

    verdict: str
    reason: str

    strips: List[str]
    kept_strips: List[str]
    refined_context: str

    web_query: str

    web_docs: List[Document]

    answer: str


class DocEvalScore(BaseModel):
    score: float
    reason: str


class KeepOrDrop(BaseModel):
    keep: bool


class WebQuery(BaseModel):
    query: str


def empty_state(question: str) -> State:
    """Builds a fresh State dict ready to pass into app.invoke()."""
    return {
        "question": question,
        "docs": [],
        "good_docs": [],
        "verdict": "",
        "reason": "",
        "strips": [],
        "kept_strips": [],
        "refined_context": "",
        "web_query": "",
        "web_docs": [],
        "answer": "",
    }


# -----------------------------
# Shared resources
# -----------------------------
def get_llm() -> ChatGroq:
    return ChatGroq(model=LLM_MODEL)


def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def get_tavily() -> TavilySearchResults:
    return TavilySearchResults(max_results=5)


def decompose_to_sentences(text: str) -> List[str]:
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


# -----------------------------
# Vector store
# -----------------------------
def build_vector_store(
    file_bytes: bytes,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embeddings: HuggingFaceEmbeddings | None = None,
) -> tuple[Chroma, int, int]:
    """Loads a PDF (given as raw bytes), splits it, and builds a fresh Chroma store.

    Returns (vector_store, n_pages, n_chunks).
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        docs = PyPDFLoader(tmp_path).load()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        ).split_documents(docs)
        for d in chunks:
            d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")

        embeddings = embeddings or get_embeddings()
        vector_store = Chroma.from_documents(chunks, embeddings)
        return vector_store, len(docs), len(chunks)
    finally:
        os.unlink(tmp_path)


def build_vector_store_from_path(
    pdf_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    embeddings: HuggingFaceEmbeddings | None = None,
) -> tuple[Chroma, int, int]:
    """Same as build_vector_store, but reads directly from a file path on disk."""
    with open(pdf_path, "rb") as f:
        return build_vector_store(f.read(), chunk_size, chunk_overlap, embeddings)


# -----------------------------
# Graph construction
# -----------------------------
def build_graph(retriever, llm: ChatGroq | None = None, tavily: TavilySearchResults | None = None) -> CompiledStateGraph:
    """Builds and compiles the CRAG LangGraph app."""
    llm = llm or get_llm()
    tavily = tavily or get_tavily()

    # ---- Retrieve ----
    def retrieve_node(state: State) -> State:
        q = state["question"]
        return {"docs": retriever.invoke(q)}

    # ---- Score-based doc evaluator ----
    doc_eval_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a strict retrieval evaluator for RAG.\n"
                "You will be given ONE retrieved chunk and a question.\n"
                "Return a relevance score in [0.0, 1.0].\n"
                "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
                "- 0.0: chunk is irrelevant\n"
                "Be conservative with high scores.\n"
                "Also return a short reason.\n"
                "Output JSON only.",
            ),
            ("human", "Question: {question}\n\nChunk:\n{chunk}"),
        ]
    )
    doc_eval_chain = doc_eval_prompt | llm.with_structured_output(DocEvalScore, method="json_mode")

    def eval_each_doc_node(state: State) -> State:
        q = state["question"]
        scores: List[float] = []
        good: List[Document] = []

        for d in state["docs"]:
            out = doc_eval_chain.invoke({"question": q, "chunk": d.page_content})
            scores.append(out.score)
            if out.score > LOWER_TH:
                good.append(d)

        if any(s > UPPER_TH for s in scores):
            return {
                "good_docs": good,
                "verdict": "CORRECT",
                "reason": f"At least one retrieved chunk scored > {UPPER_TH}.",
            }

        if len(scores) > 0 and all(s < LOWER_TH for s in scores):
            return {
                "good_docs": [],
                "verdict": "INCORRECT",
                "reason": f"All retrieved chunks scored < {LOWER_TH}.",
            }

        return {
            "good_docs": good,
            "verdict": "AMBIGUOUS",
            "reason": f"No chunk scored > {UPPER_TH}, but not all were < {LOWER_TH}.",
        }

    # ---- Filter (LLM judge) + knowledge refinement ----
    filter_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a relevance filter for RAG.\n"
                "Return keep=true if the sentence contains information relevant to answering the question, even if only partially.\n"
                "Return keep=false only if the sentence is clearly irrelevant or pure noise.\n"
                "Use ONLY the sentence. Output JSON only, exactly like {{\"keep\": true}} or {{\"keep\": false}}.\n",
            ),
            ("human", "Question: {question}\n\nSentence:\n{sentence}"),
        ]
    )
    filter_chain = filter_prompt | llm.with_structured_output(KeepOrDrop, method="json_mode")

    def refine(state: State) -> State:
        q = state["question"]

        if state.get("verdict") == "CORRECT":
            docs_to_use = state["good_docs"]
        elif state.get("verdict") == "INCORRECT":
            docs_to_use = state["web_docs"]
        else:  # AMBIGUOUS
            docs_to_use = state["good_docs"] + state["web_docs"]

        context = "\n\n".join(d.page_content for d in docs_to_use).strip()
        strips = decompose_to_sentences(context)

        kept: List[str] = []
        for s in strips:
            if filter_chain.invoke({"question": q, "sentence": s}).keep:
                kept.append(s)

        refined_context = "\n".join(kept).strip()
        if not refined_context:
            refined_context = context

        return {
            "strips": strips,
            "kept_strips": kept,
            "refined_context": refined_context,
        }

    # ---- Query rewrite for web search ----
    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Rewrite the user question into a web search query composed of keywords.\n"
                "Rules:\n"
                "- Keep it short (6-14 words).\n"
                "- If the question implies recency (e.g., recent/latest/last week/last month), add a constraint like (last 30 days).\n"
                "- Do NOT answer the question.\n"
                "- Return JSON with a single key: query",
            ),
            ("human", "Question: {question}"),
        ]
    )
    rewrite_chain = rewrite_prompt | llm.with_structured_output(WebQuery, method="json_mode")

    def rewrite_query_node(state: State) -> State:
        out = rewrite_chain.invoke({"question": state["question"]})
        return {"web_query": out.query}

    def web_search_node(state: State) -> State:
        q = state.get("web_query") or state["question"]
        results = tavily.invoke({"query": q})

        web_docs: List[Document] = []
        for r in results or []:
            title = r.get("title", "")
            url = r.get("url", "")
            content = r.get("content", "") or r.get("snippet", "")
            text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"
            web_docs.append(Document(page_content=text, metadata={"url": url, "title": title}))

        return {"web_docs": web_docs}

    # ---- Generate ----
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful ML tutor. Answer using ONLY the provided context.\n"
                "If the context is empty, say: 'I don't know.' Otherwise, answer as fully as possible from the context.",
            ),
            ("human", "Question: {question}\n\nContext:\n{context}"),
        ]
    )

    def generate(state: State) -> State:
        out = (answer_prompt | llm).invoke(
            {"question": state["question"], "context": state["refined_context"]}
        )
        return {"answer": out.content}

    # ---- Routing ----
    def route_after_eval(state: State) -> str:
        if state["verdict"] == "CORRECT":
            return "refine"
        return "rewrite_query"

    # ---- Build graph ----
    g = StateGraph(State)
    g.add_node("retrieve", retrieve_node)
    g.add_node("eval_each_doc", eval_each_doc_node)
    g.add_node("rewrite_query", rewrite_query_node)
    g.add_node("web_search", web_search_node)
    g.add_node("refine", refine)
    g.add_node("generate", generate)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "eval_each_doc")
    g.add_conditional_edges(
        "eval_each_doc",
        route_after_eval,
        {"refine": "refine", "rewrite_query": "rewrite_query"},
    )
    g.add_edge("rewrite_query", "web_search")
    g.add_edge("web_search", "refine")
    g.add_edge("refine", "generate")
    g.add_edge("generate", END)

    return g.compile()


def run_crag(question: str, retriever, llm: ChatGroq | None = None, tavily: TavilySearchResults | None = None) -> State:
    """Convenience one-shot entry point: builds the graph and runs a single question."""
    app = build_graph(retriever, llm=llm, tavily=tavily)
    return app.invoke(empty_state(question))
