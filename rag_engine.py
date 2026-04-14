"""
PDF RAG Engine with Confidence-Aware Answers & Source Transparency
Two-pass generation: retrieve → score → answer with per-sentence attribution
"""

import os
import re
import json
import hashlib
from typing import Optional
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ─────────────────────────────────────────────
# Embeddings (local, no API key needed)
# ─────────────────────────────────────────────
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

# ─────────────────────────────────────────────
# PDF Processing
# ─────────────────────────────────────────────
def load_and_chunk_pdf(pdf_path: str, chunk_size: int = 800, chunk_overlap: int = 120):
    """Load a PDF and split into overlapping chunks."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    # Attach chunk index for traceability
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["char_count"] = len(chunk.page_content)

    return chunks, pages

def build_vectorstore(chunks, embeddings):
    """Build a FAISS vector store from document chunks."""
    return FAISS.from_documents(chunks, embeddings)

def pdf_fingerprint(pdf_bytes: bytes) -> str:
    return hashlib.md5(pdf_bytes).hexdigest()

# ─────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────
def retrieve_chunks(vectorstore, query: str, k: int = 5):
    """Retrieve top-k relevant chunks with similarity scores."""
    results = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    return results  # list of (Document, score)

# ─────────────────────────────────────────────
# Pass 1 – Faithfulness Scoring
# ─────────────────────────────────────────────
FAITHFULNESS_SYSTEM = """You are a strict retrieval-quality judge.

Given a user question and retrieved document chunks, evaluate:
1. RELEVANCE: How relevant are the chunks to the question? (0.0 – 1.0)
2. COVERAGE: Does the combined chunk text actually answer the question? (0.0 – 1.0)
3. GAPS: List specific aspects of the question that the chunks do NOT cover.

Respond ONLY in valid JSON with this exact schema:
{
  "relevance_score": <float 0-1>,
  "coverage_score": <float 0-1>,
  "overall_faithfulness": <float 0-1>,
  "gaps": ["gap1", "gap2"],
  "grounded_topics": ["topic1", "topic2"],
  "assessment": "<one sentence plain-English summary>"
}"""

def score_faithfulness(llm: ChatGroq, question: str, chunks: list) -> dict:
    """LLM-based faithfulness scoring of retrieved chunks vs question."""
    chunk_texts = "\n\n---\n\n".join(
        [f"[Chunk {i+1}] {doc.page_content[:600]}" for i, (doc, _) in enumerate(chunks)]
    )

    prompt = f"""QUESTION: {question}

RETRIEVED CHUNKS:
{chunk_texts}

Evaluate how faithfully these chunks can answer the question."""

    try:
        response = llm.invoke([
            SystemMessage(content=FAITHFULNESS_SYSTEM),
            HumanMessage(content=prompt),
        ])
        raw = response.content.strip()
        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        scores = json.loads(raw)
        # Clamp values
        for key in ["relevance_score", "coverage_score", "overall_faithfulness"]:
            scores[key] = max(0.0, min(1.0, float(scores.get(key, 0.5))))
        return scores
    except Exception as e:
        return {
            "relevance_score": 0.5,
            "coverage_score": 0.5,
            "overall_faithfulness": 0.5,
            "gaps": [],
            "grounded_topics": [],
            "assessment": f"Scoring unavailable: {str(e)}",
        }

# ─────────────────────────────────────────────
# Pass 2 – Confidence-Calibrated Answer Generation
# ─────────────────────────────────────────────
ANSWER_SYSTEM = """You are an expert document analyst with a commitment to epistemic honesty.

Your task: Answer the user's question using ONLY the provided document chunks.

STRICT RULES:
1. For every sentence in your answer, append a tag:
   - [DOC] if the sentence is directly grounded in the retrieved document chunks
   - [INF] if you are inferring, extrapolating, or using general knowledge not in the chunks
2. Be explicit when the document is silent on something: say so clearly.
3. Use calibrated language: "The document clearly states...", "Based on context, it appears...", "The document does not address..."
4. Structure your answer with:
   - A direct answer first
   - Supporting evidence with [DOC]/[INF] tags on EVERY sentence
   - A brief confidence note at the end

FORMAT EXAMPLE:
"The document states that X is Y. [DOC] This suggests that Z is likely true in related cases, though not explicitly mentioned. [INF] The document does not cover aspect W. [DOC]"

FAITHFULNESS CONTEXT (use this to calibrate your confidence):
{faithfulness_json}

RETRIEVED DOCUMENT CHUNKS:
{chunks}"""

def generate_confident_answer(
    llm: ChatGroq,
    question: str,
    chunks: list,
    faithfulness: dict,
) -> dict:
    """Generate a confidence-calibrated answer with per-sentence attribution."""

    chunk_texts = "\n\n---\n\n".join(
        [
            f"[Chunk {i+1} | Page {doc.metadata.get('page', '?')+1}]\n{doc.page_content}"
            for i, (doc, score) in enumerate(chunks)
        ]
    )

    system_prompt = ANSWER_SYSTEM.format(
        faithfulness_json=json.dumps(faithfulness, indent=2),
        chunks=chunk_texts,
    )

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Question: {question}"),
        ])
        raw_answer = response.content.strip()
    except Exception as e:
        raw_answer = f"Error generating answer: {str(e)} [INF]"

    # Parse sentences with tags
    sentences = parse_attributed_sentences(raw_answer)
    confidence_level = get_confidence_level(faithfulness)

    return {
        "raw_answer": raw_answer,
        "sentences": sentences,
        "faithfulness": faithfulness,
        "confidence_level": confidence_level,
        "sources": [
            {
                "chunk_id": doc.metadata.get("chunk_id", i),
                "page": doc.metadata.get("page", 0) + 1,
                "preview": doc.page_content[:200].strip(),
                "similarity": round(float(score), 3),
            }
            for i, (doc, score) in enumerate(chunks)
        ],
    }

# ─────────────────────────────────────────────
# Sentence Attribution Parser
# ─────────────────────────────────────────────
def parse_attributed_sentences(text: str) -> list:
    """
    Parse answer text into sentences with [DOC]/[INF] labels.
    Returns list of {text, label, clean_text}
    """
    # Split by sentence-ending punctuation, keeping delimiters
    pattern = r'(?<=[.!?])\s+'
    raw_parts = re.split(pattern, text.strip())

    sentences = []
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue

        label = "UNKNOWN"
        if "[DOC]" in part:
            label = "DOC"
        elif "[INF]" in part:
            label = "INF"

        clean = part.replace("[DOC]", "").replace("[INF]", "").strip()
        if clean:
            sentences.append({
                "text": part,
                "clean_text": clean,
                "label": label,
            })

    return sentences

# ─────────────────────────────────────────────
# Confidence Level Classifier
# ─────────────────────────────────────────────
def get_confidence_level(faithfulness: dict) -> dict:
    score = faithfulness.get("overall_faithfulness", 0.5)

    if score >= 0.80:
        return {
            "level": "HIGH",
            "label": "High Confidence",
            "description": "The document strongly supports this answer.",
            "color": "#10b981",
            "bg": "#d1fae5",
            "icon": "✅",
        }
    elif score >= 0.55:
        return {
            "level": "MEDIUM",
            "label": "Moderate Confidence",
            "description": "The document partially supports this answer; some inference involved.",
            "color": "#f59e0b",
            "bg": "#fef3c7",
            "icon": "⚠️",
        }
    else:
        return {
            "level": "LOW",
            "label": "Low Confidence",
            "description": "Limited document support. Treat this answer with caution.",
            "color": "#ef4444",
            "bg": "#fee2e2",
            "icon": "🔴",
        }

# ─────────────────────────────────────────────
# Main Query Pipeline
# ─────────────────────────────────────────────
def query_pdf(
    question: str,
    vectorstore,
    top_k: int = 5,
) -> dict:
    """
    Full two-pass RAG pipeline:
      1. Retrieve chunks
      2. Score faithfulness (Pass 1)
      3. Generate confidence-calibrated answer (Pass 2)
    """
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

    # Step 1: Retrieve
    chunks = retrieve_chunks(vectorstore, question, k=top_k)

    if not chunks:
        return {
            "raw_answer": "No relevant content found in the document for this question.",
            "sentences": [{"clean_text": "No relevant content found.", "label": "DOC"}],
            "faithfulness": {"overall_faithfulness": 0.0, "gaps": ["entire question"], "grounded_topics": []},
            "confidence_level": get_confidence_level({"overall_faithfulness": 0.0}),
            "sources": [],
        }

    # Step 2: Score faithfulness
    faithfulness = score_faithfulness(llm, question, chunks)

    # Step 3: Generate answer
    result = generate_confident_answer(llm, question, chunks, faithfulness)

    return result