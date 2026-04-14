# PDF-Intelligence-Confidence-Aware-RAG-Engine
 A high-precision, confidence-aware RAG engine that eliminates hallucinations through two-pass verification and per-sentence source attribution.
Standard RAG systems often present hallucinations as facts. This project implements a Two-Pass Generation Pipeline to ensure every answer is verified for faithfulness. Instead of just providing an answer, this engine tells you how it knows what it knows, providing confidence scores and sentence-level grounding.

✨ Key Features
Two-Pass Architecture:
Pass 1 (Scoring): An LLM-based judge evaluates the relevance and coverage of retrieved chunks before answering.
Pass 2 (Generation): A confidence-calibrated response generator that uses the scoring results to adjust its tone.
Per-Sentence Attribution: Every sentence is tagged with [DOC] (grounded in text) or [INF] (inferred/general knowledge) to provide total transparency.
Confidence Scoring: Automatically categorizes responses as High, Medium, or Low confidence based on document support.
Local Embeddings: Uses all-MiniLM-L6-v2 via HuggingFace for privacy-conscious, local vectorization.
High-Speed Inference: Powered by Groq (Llama 3.3) for near-instantaneous reasoning.
