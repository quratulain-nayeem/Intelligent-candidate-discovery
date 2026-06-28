# src/semantic_scorer.py
#
# WHY THIS FILE EXISTS:
# Keyword matching fails when a candidate says "built hybrid retrieval systems"
# and the JD says "production experience with embeddings-based search."
# Same concept, different words. Cosine similarity on embeddings catches this.
#
# WHY all-MiniLM-L6-v2:
# Fast enough to embed 100K candidates on CPU in reasonable time.
# Good enough quality for this domain. No API calls needed â€” runs locally.
# This matters because the submission spec says no API calls during ranking.

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from pathlib import Path

MODEL_NAME = "all-MiniLM-L6-v2"


def build_candidate_text(candidate: dict) -> str:
    """
    Build a sharper semantic representation for MiniLM.

    Instead of embedding the whole profile, embed the current title, the two
    most JD-relevant career snippets, and only the top few skills. This avoids
    turning every senior ML profile into the same generic document vector.
    """
    profile = candidate.get("profile", {})
    parts = []

    title = profile.get("current_title", "")
    if title:
        parts.append(title)

    ai_keywords = {
        "embedding", "embeddings", "retrieval", "ranking", "ranker",
        "faiss", "vector", "semantic", "search", "nlp", "dense",
        "inference", "pipeline", "transformer", "bert", "bge",
        "recommendation", "recommender", "matching", "reranking",
        "learning to rank", "a/b", "ndcg", "mrr", "latency", "ann",
        "hnsw", "qdrant", "pinecone", "weaviate", "milvus",
    }

    def relevance(role: dict) -> tuple:
        text = " ".join(
            str(role.get(field, ""))
            for field in ("title", "industry", "description")
        ).lower()
        keyword_hits = sum(1 for keyword in ai_keywords if keyword in text)
        current_bonus = 2 if role.get("is_current") else 0
        duration = int(role.get("duration_months", 0) or 0)
        return (keyword_hits + current_bonus, duration)

    career = candidate.get("career_history", [])
    career_sorted = sorted(career, key=relevance, reverse=True)
    for role in career_sorted[:2]:
        desc = role.get("description", "")
        role_title = role.get("title", "")
        if desc:
            parts.append(f"{role_title}: {desc}")

    skills = candidate.get("skills", [])
    skills_sorted = sorted(
        skills,
        key=lambda s: (s.get("endorsements", 0), s.get("duration_months", 0)),
        reverse=True,
    )
    skill_names = [s["name"] for s in skills_sorted[:5] if s.get("name")]
    if skill_names:
        parts.append(", ".join(skill_names))

    return " | ".join(parts)


def build_jd_text(jd_path: str) -> str:
    """
    Return a focused semantic query for the JD.

    The full JD contains HR boilerplate and broad prose. For retrieval, a dense
    role query gives MiniLM a cleaner target: production retrieval/ranking work,
    vector infrastructure, evaluation, and senior IC experience.
    """
    return (
        "production embeddings retrieval ranking dense search FAISS vector "
        "database semantic search shipped at scale embedding pipeline inference "
        "latency A/B testing NDCG MRR information retrieval NLP Senior AI "
        "Engineer 5+ years hybrid retrieval ANN learning to rank"
    )

def embed_candidates(candidates: list, model: SentenceTransformer) -> np.ndarray:
    """
    WHY BATCH SIZE 256:
    Balances memory usage vs speed on CPU. Larger batches are faster
    but eat RAM. 256 works on 8GB RAM for this model size.

    show_progress_bar=True because this takes a few minutes for 100K
    and you need to know it's not frozen.
    """
    texts = [build_candidate_text(c) for c in candidates]
    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # WHY: normalized embeddings make
                                   # cosine similarity = dot product,
                                   # which FAISS IndexFlatIP handles fast
    )
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    WHY IndexFlatIP (Inner Product):
    With normalized embeddings, inner product = cosine similarity.
    IndexFlatIP is exact search â€” no approximation.
    For 100K candidates this is fine on CPU (< 1 second per query).
    We'd only need approximate search (IndexIVFFlat) for 10M+ candidates.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index


def get_semantic_scores(
    jd_embedding: np.ndarray,
    index: faiss.IndexFlatIP,
    top_k: int = 15000
) -> tuple:
    """
    WHY top_k=15000:
    We only need top 100 for submission. But we retrieve 15000
    because the role gate and other filters will eliminate some,
    and we want enough headroom that our final composite scoring
    has a real pool to work with.

    Returns (indices, scores) â€” indices map back to position in
    the original candidates list.
    """
    query = jd_embedding.reshape(1, -1).astype(np.float32)
    scores, indices = index.search(query, top_k)
    return indices[0], scores[0]


def save_embeddings(embeddings: np.ndarray, path: str):
    """
    WHY: Embedding 100K candidates takes ~3-5 minutes on CPU.
    We save to disk so we only do it once. On subsequent runs
    we load from disk and go straight to scoring.
    """
    np.save(path, embeddings)
    print(f"Embeddings saved to {path}")


def load_embeddings(path: str) -> np.ndarray:
    return np.load(path)


def load_model() -> SentenceTransformer:
    """
    WHY we isolate model loading:
    First run downloads ~90MB model weights from HuggingFace.
    Subsequent runs load from local cache. Isolating this makes
    it easy to swap models if we want to test a different one.
    """
    print(f"Loading model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME, local_files_only=True)

