"""
Streamlit front end for the Corrective RAG (CRAG) pipeline.

All pipeline logic (vector store, graph, nodes, prompts) lives in `backend.py`.
This file only handles the UI: file upload, inputs, and rendering results.
"""

import os

import streamlit as st

import backend

# -----------------------------
# Cached resources (UI-layer caching over backend functions)
# -----------------------------
get_llm = st.cache_resource(show_spinner=False)(backend.get_llm)
get_embeddings = st.cache_resource(show_spinner=False)(backend.get_embeddings)
get_tavily = st.cache_resource(show_spinner=False)(backend.get_tavily)


@st.cache_resource(show_spinner="Building knowledge base from PDF...")
def build_vector_store(
    file_bytes: bytes, chunk_size: int, chunk_overlap: int, _embeddings
):
    # Leading underscore tells Streamlit not to hash `_embeddings` (it's unhashable).
    return backend.build_vector_store(
        file_bytes, chunk_size, chunk_overlap, _embeddings
    )


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Corrective RAG", page_icon="☠️", layout="wide")
st.title("☠️ Corrective RAG (CRAG)")
st.caption(
    "Upload a PDF, ask a question, and watch the pipeline grade its own retrieval, "
    "fall back to web search when needed, and refine the context before answering."
)

with st.sidebar:
    st.header("1. Knowledge base")
    uploaded_file = st.file_uploader("Upload a source PDF", type=["pdf"])

    with st.expander("Chunking settings"):
        chunk_size = st.number_input(
            "Chunk size", value=backend.DEFAULT_CHUNK_SIZE, min_value=100, step=50
        )
        chunk_overlap = st.number_input(
            "Chunk overlap", value=backend.DEFAULT_CHUNK_OVERLAP, min_value=0, step=10
        )

    with st.expander("Retrieval settings"):
        k = st.number_input(
            "Top-k chunks retrieved",
            value=backend.DEFAULT_TOP_K,
            min_value=1,
            max_value=20,
        )

    st.divider()
    st.caption(f"LLM: `{backend.LLM_MODEL}` (Groq)")
    st.caption(f"Embeddings: `{backend.EMBED_MODEL}`")
    st.caption(
        f"Thresholds: CORRECT > {backend.UPPER_TH}, INCORRECT < {backend.LOWER_TH}"
    )

if not uploaded_file:
    st.info("👈 Upload a PDF in the sidebar to build the knowledge base.")
    st.stop()

if not os.environ.get("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is not set. Add it to your .env file and restart the app.")
    st.stop()

embeddings = get_embeddings()
file_bytes = uploaded_file.getvalue()
vector_store, n_pages, n_chunks = build_vector_store(
    file_bytes, chunk_size, chunk_overlap, embeddings
)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
st.sidebar.success(f"Indexed {n_pages} page(s) -> {n_chunks} chunks.")

if not os.environ.get("TAVILY_API_KEY"):
    st.sidebar.warning(
        "TAVILY_API_KEY not set — web search fallback will fail on INCORRECT/AMBIGUOUS verdicts."
    )

st.header("2. Ask a question")
question = st.text_input("Your question", placeholder="e.g. What is the HNSW graph?")
run = st.button("Run CRAG", type="primary", disabled=not question)

if run:
    llm = get_llm()
    tavily = get_tavily()
    with st.spinner(
        "Running retrieve -> evaluate -> (web search) -> refine -> generate ..."
    ):
        res = backend.run_crag(question, retriever, llm=llm, tavily=tavily)

    verdict = res["verdict"]
    badge_color = {"CORRECT": "green", "INCORRECT": "red", "AMBIGUOUS": "orange"}.get(
        verdict, "gray"
    )

    st.header("3. Result")
    st.markdown(
        f"**Verdict:** :{badge_color}[{verdict}]  \n**Reason:** {res['reason']}"
    )

    if res.get("web_query"):
        st.markdown(f"**Rewritten web query:** `{res['web_query']}`")

    st.subheader("Answer")
    st.write(res["answer"])

    with st.expander(f"Retrieved chunks ({len(res['docs'])})"):
        for i, d in enumerate(res["docs"]):
            st.markdown(f"**Chunk {i + 1}**")
            st.text(d.page_content)

    with st.expander(f"Good docs kept after grading ({len(res['good_docs'])})"):
        for i, d in enumerate(res["good_docs"]):
            st.markdown(f"**Good doc {i + 1}**")
            st.text(d.page_content)

    if res.get("web_docs"):
        with st.expander(f"Web search results ({len(res['web_docs'])})"):
            for i, d in enumerate(res["web_docs"]):
                st.markdown(f"**Web doc {i + 1}**")
                st.text(d.page_content)

    with st.expander(
        f"Sentence-level refinement ({len(res['kept_strips'])}/{len(res['strips'])} kept)"
    ):
        st.markdown("**Kept sentences:**")
        for s in res["kept_strips"]:
            st.markdown(f"- {s}")
        st.markdown("**Dropped sentences:**")
        dropped = [s for s in res["strips"] if s not in res["kept_strips"]]
        for s in dropped:
            st.markdown(f"- ~~{s}~~")

    with st.expander("Final refined context sent to the generator"):
        st.text(res["refined_context"])
