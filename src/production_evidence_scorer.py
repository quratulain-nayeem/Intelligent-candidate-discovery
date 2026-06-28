# src/production_evidence_scorer.py
#
# WHY THIS FILE EXISTS:
# The Redrob JD is not asking for generic AI keyword overlap. It asks for
# evidence that a candidate has shipped retrieval, ranking, search, or
# recommendation systems to real users and knows how to evaluate them.
#
# This scorer reads career history, not just skills. That is the main defense
# against keyword-stuffed profiles that list modern AI terms without proof.

import re

ACTION_TERMS = {
    "built", "build", "deployed", "deployed", "shipped", "ship", "owned",
    "led", "launched", "implemented", "designed", "architected", "operated",
    "maintained", "scaled", "improved", "optimized", "productionized"
}

DOMAIN_TERMS = {
    "retrieval", "ranking", "ranker", "search", "semantic search",
    "vector search", "hybrid search", "recommendation", "recommender",
    "matching", "candidate matching", "talent matching", "information retrieval",
    "embeddings", "embedding", "dense retrieval", "reranking", "learning to rank",
    "discovery", "personalization", "relevance", "results", "intent"
}

INFRA_TERMS = {
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "vector database", "vector index", "index refresh",
    "ann", "nearest neighbor", "sentence transformer", "bge", "e5",
    "data infrastructure", "feature-engineering pipeline", "feature monitoring",
    "drift detection", "retraining cadence", "offline experimentation",
    "index versioning", "embedding versioning", "rollback paths", "dashboards"
}

EVAL_TERMS = {
    "ndcg", "mrr", "map", "precision", "recall", "a/b", "ab test",
    "offline evaluation", "online evaluation", "relevance", "quality regression",
    "retrieval quality", "click-through", "ctr", "conversion", "feedback loop"
}

PRODUCTION_TERMS = {
    "production", "real users", "users", "scale", "latency", "millions",
    "high traffic", "marketplace", "on-call", "monitoring", "refresh",
    "drift", "sla", "pipeline", "serving", "api", "microservice"
}

ANTI_PATTERN_TERMS = {
    "tutorial", "course project", "toy project", "demo", "chatbot demo",
    "langchain tutorial", "openai wrapper", "prompt engineering", "kaggle",
    "bootcamp", "certificate", "workshop"
}

RESEARCH_ONLY_TERMS = {
    "academic lab", "research lab", "paper", "publication", "thesis",
    "simulation", "prototype"
}

PRODUCT_INDUSTRY_HINTS = {
    "software", "internet", "e-commerce", "marketplace", "saas", "fintech",
    "hr tech", "recruiting", "talent", "product", "consumer", "platform"
}

SCALE_TERMS = {
    "50m", "35m", "30m", "10m", "millions", "million", "large-scale",
    "high traffic", "serving", "queries", "users", "corpus", "scale",
    "large dataset", "engagement metrics", "time-to-shortlist"
}

EVAL_RIGOR_TERMS = {
    "ndcg", "mrr", "map", "offline-online correlation", "offline evaluation",
    "online evaluation", "a/b", "ab test", "relevance labeling", "human judgments",
    "retrieval quality", "quality regression", "feedback loop",
    "evaluation methodology", "offline metrics", "online engagement"
}


def _career_roles(candidate: dict) -> list:
    return sorted(
        candidate.get("career_history", []),
        key=lambda role: role.get("start_date", ""),
        reverse=True,
    )


def _contains_any(text: str, terms: set) -> list:
    hits = []
    for term in terms:
        if any(ch in term for ch in " /+-"):
            found = term in text
        else:
            found = re.search(r"\b" + re.escape(term) + r"\b", text) is not None
        if found:
            hits.append(term)
    return sorted(hits)


def score_production_evidence(candidate: dict) -> dict:
    """
    Scores proof of shipped retrieval/ranking/search/recommendation work.

    This intentionally measures distinct evidence quality, not raw keyword
    presence. A candidate needs several kinds of proof to approach 1.0:
    domain fit, infra, evaluation, scale, ownership, production context, and
    recent work. That keeps strong profiles from all collapsing to 1.0.
    """
    roles = _career_roles(candidate)

    evidence_phrases = []
    matched_domains = set()
    matched_eval = set()
    matched_infra = set()
    anti_hits = set()
    research_only_hits = set()
    product_context_hits = set()
    production_hits = set()
    scale_hits_all = set()
    ownership_hits_all = set()

    strong_ownership_terms = {
        "built", "deployed", "shipped", "owned", "led", "launched",
        "designed", "architected", "operated", "scaled", "productionized"
    }

    current_role_has_proof = False
    best_role_depth = 0.0

    for idx, role in enumerate(roles):
        text = " ".join(
            str(role.get(field, ""))
            for field in ("title", "company", "industry", "description")
        ).lower()

        action_hits = _contains_any(text, ACTION_TERMS)
        ownership_hits = _contains_any(text, strong_ownership_terms)
        domain_hits = _contains_any(text, DOMAIN_TERMS)
        infra_hits = _contains_any(text, INFRA_TERMS)
        eval_hits = _contains_any(text, EVAL_TERMS)
        prod_hits = _contains_any(text, PRODUCTION_TERMS)
        anti = _contains_any(text, ANTI_PATTERN_TERMS)
        research = _contains_any(text, RESEARCH_ONLY_TERMS)
        product = _contains_any(text, PRODUCT_INDUSTRY_HINTS)
        scale_hits = _contains_any(text, SCALE_TERMS)
        eval_rigor_hits = _contains_any(text, EVAL_RIGOR_TERMS)

        matched_domains.update(domain_hits)
        matched_eval.update(eval_hits)
        matched_infra.update(infra_hits)
        anti_hits.update(anti)
        research_only_hits.update(research)
        product_context_hits.update(product)
        production_hits.update(prod_hits)
        scale_hits_all.update(scale_hits)
        ownership_hits_all.update(ownership_hits)

        if idx == 0 and domain_hits and (ownership_hits or action_hits):
            current_role_has_proof = True

        if domain_hits and action_hits:
            evidence_phrases.append(f"{role.get('title', 'Role')} mentions {domain_hits[0]}")

        role_depth = 0.0
        if domain_hits:
            role_depth += 0.18
        if ownership_hits:
            role_depth += 0.18
        if infra_hits:
            role_depth += min(0.18, 0.09 * len(infra_hits))
        if eval_rigor_hits:
            role_depth += min(0.22, 0.11 * len(eval_rigor_hits))
        if scale_hits:
            role_depth += min(0.14, 0.07 * len(scale_hits))
        if prod_hits:
            role_depth += min(0.10, 0.05 * len(prod_hits))
        if product:
            role_depth += 0.05
        if idx == 0:
            role_depth *= 1.08
        best_role_depth = max(best_role_depth, role_depth)

    domain_quality = min(len(matched_domains) / 4.0, 1.0)
    infra_quality = min(len(matched_infra) / 4.0, 1.0)
    eval_quality = min(len(matched_eval) / 4.0, 1.0)
    scale_quality = min(len(scale_hits_all) / 3.0, 1.0)
    ownership_quality = min(len(ownership_hits_all) / 4.0, 1.0)
    production_quality = min(len(production_hits) / 4.0, 1.0)
    recency_quality = 1.0 if current_role_has_proof else 0.0

    score = (
        domain_quality * 0.15
        + infra_quality * 0.17
        + eval_quality * 0.20
        + scale_quality * 0.15
        + ownership_quality * 0.15
        + production_quality * 0.13
        + recency_quality * 0.05
    )

    if anti_hits and not production_hits:
        score -= 0.12
    if research_only_hits and not production_hits:
        score -= 0.08

    score = max(0.0, min(score, 1.0))
    depth_score = max(0.0, min(best_role_depth, 1.0))

    return {
        "candidate_id": candidate.get("candidate_id"),
        "production_evidence_score": round(score, 4),
        "production_depth_score": round(depth_score, 4),
        "has_strong_production_evidence": score >= 0.55,
        "matched_domains": sorted(matched_domains)[:8],
        "matched_infra": sorted(matched_infra)[:8],
        "matched_eval": sorted(matched_eval)[:8],
        "production_hits": sorted(production_hits)[:8],
        "product_context_hits": sorted(product_context_hits)[:8],
        "anti_pattern_hits": sorted(anti_hits)[:8],
        "research_only_hits": sorted(research_only_hits)[:8],
        "evidence_phrases": evidence_phrases[:3],
    }





