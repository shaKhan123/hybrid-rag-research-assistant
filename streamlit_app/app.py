"""
Streamlit frontend for the Hybrid RAG Research Assistant.

Thin client only — all retrieval/generation/groundedness logic lives in
the FastAPI backend (api/main.py). This app just collects a query, calls
POST /query over HTTP, and renders the response. Deployed as a separate
Render service from rag-api; RAG_API_URL points at it (see render.yaml).
"""

import os

import requests
import streamlit as st

_DEFAULT_API_URL = "http://localhost:8000"


def _resolve_api_base_url() -> str:
    """RAG_API_URL is either a full URL (local dev, e.g. http://localhost:8000)
    or a bare host:port (Render's private-network `hostport` service
    reference, which has no scheme) — normalize to a full URL either way."""
    raw = os.environ.get("RAG_API_URL", _DEFAULT_API_URL)
    if "://" not in raw:
        raw = f"http://{raw}"
    return raw.rstrip("/")


API_BASE_URL = _resolve_api_base_url()

st.set_page_config(page_title="Hybrid RAG Research Assistant", page_icon="📚")
st.title("📚 Hybrid RAG Research Assistant")
st.caption("Ask a question about retrieval-augmented generation research (arXiv cs.CL / cs.LG).")

with st.form("query_form"):
    query = st.text_area(
        "Your question",
        placeholder="e.g. How does hybrid retrieval combine dense and sparse search?",
        height=100,
    )
    use_hyde = st.checkbox(
        "Use HyDE query rewriting",
        help="Rewrite the question into a hypothetical answer paragraph before retrieval — "
             "can improve recall for short/underspecified questions, at the cost of one extra LLM call.",
    )
    submitted = st.form_submit_button("Ask", type="primary")

if submitted:
    stripped = query.strip()
    if len(stripped) < 3:
        st.error("Question must be at least 3 characters.")
    elif len(stripped) > 1000:
        st.error("Question must be at most 1000 characters.")
    else:
        with st.spinner("Retrieving sources and generating an answer..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/query",
                    json={"query": stripped, "use_hyde": use_hyde},
                    timeout=120,
                )
            except requests.exceptions.RequestException as e:
                response = None
                st.error(f"Couldn't reach the backend at {API_BASE_URL}: {e}")

        if response is not None:
            if response.status_code == 200:
                data = response.json()

                st.subheader("Answer")
                if data.get("from_cache"):
                    st.caption("⚡ served from cache (a close-enough question was already answered)")
                st.markdown(data["answer"])

                if data["is_grounded"]:
                    st.success(f"✅ Grounded (verified after {data['retry_count']} check(s))")
                else:
                    st.warning(
                        f"⚠️ Not fully grounded after {data['retry_count']} retry attempt(s) "
                        "— the answer may contain claims the sources don't fully support."
                    )
                    if data.get("groundedness_report"):
                        with st.expander("Show groundedness report"):
                            st.text(data["groundedness_report"])

                st.subheader("Sources")
                for i, source in enumerate(data["sources"], start=1):
                    score = source.get("rerank_score")
                    score_str = f"{score:.4f}" if score is not None else "n/a"
                    st.markdown(
                        f"**[{i}]** [{source['arxiv_id']}](https://arxiv.org/abs/{source['arxiv_id']}) "
                        f"— chunk {source['chunk_index']} (rerank score: {score_str})"
                    )

            elif response.status_code == 429:
                st.error(
                    "This demo is rate-limited to protect the free-tier LLM quota. "
                    "Please wait a bit and try again."
                )
            elif response.status_code == 503:
                st.warning(
                    "⏳ The AI model is temporarily rate-limited on the backend's end "
                    "(not yours) — please try again in a minute."
                )
            else:
                try:
                    detail = response.json().get("detail", response.text)
                except ValueError:
                    detail = response.text
                st.error(f"Request failed ({response.status_code}): {detail}")

st.caption(f"Backend: {API_BASE_URL}")
