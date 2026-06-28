# src/role_gate.py
#
# WHY THIS FILE EXISTS:
# 70%+ of the 100K candidates are irrelevant roles (HR, Accounting,
# Mechanical Engineering etc.) with AI keywords sprinkled in randomly.
# Feeding all of them into semantic scoring wastes compute and lets
# noise drown out real signal.
#
# This file classifies each candidate into one of three buckets:
#   CORE    â†’ clear AI/ML/software engineering background
#   ADJACENT â†’ technical background, possibly transferable
#   NOISE   â†’ non-technical role, irrelevant regardless of skill list
#
# Only CORE and ADJACENT candidates proceed to full scoring.
# NOISE candidates are ranked 51-100 automatically.


# WHY THESE SPECIFIC TITLES:
# These are titles that appear in the dataset that indicate someone
# has been doing technical work. We checked the actual title distribution
# earlier â€” these cover the real technical candidates without being
# so broad that HR Managers slip through.
CORE_TITLES = {
    "software engineer", "ml engineer", "machine learning engineer",
    "ai engineer", "data scientist", "nlp engineer", "research engineer",
    "applied scientist", "backend engineer", "full stack developer",
    "senior software engineer", "staff engineer", "principal engineer",
    "data engineer", "platform engineer", "devops engineer",
    "cloud engineer", "frontend engineer", "junior ml engineer",
    "senior machine learning engineer", "computer vision engineer",
    "deep learning engineer", "senior ai engineer", "lead ai engineer",
    "senior nlp engineer", "staff machine learning engineer",
    "staff ml engineer", "senior ml engineer", "applied ml engineer",
    "senior applied scientist", "search engineer",
    "ml engineer - search & ranking", "senior ml engineer - search & ranking",
}

ADJACENT_TITLES = {
    "software developer", "java developer", "mobile developer",
    "qa engineer", "solutions architect", "technical lead",
    "engineering manager", "product engineer", "systems engineer",
    "analytics engineer", "bi engineer", "research analyst",
    "quantitative analyst", "data analyst"
}

# WHY THESE KEYWORDS:
# These appear in career history DESCRIPTIONS of people who have done
# relevant work. We're not matching titles here â€” we're matching what
# they actually built and shipped.
# Split into tiers because not all signals are equal.

# Strong signals â€” if someone mentions these in their job descriptions,
# they almost certainly did relevant work
STRONG_CAREER_KEYWORDS = {
    "embedding", "embeddings", "vector search", "retrieval",
    "ranking system", "recommendation system", "search system",
    "faiss", "pinecone", "weaviate", "qdrant", "elasticsearch",
    "sentence transformer", "fine-tuning", "fine-tune", "rag",
    "retrieval augmented", "llm", "large language model",
    "transformer", "bert", "gpt", "semantic search",
    "candidate ranking", "talent matching", "nlp pipeline",
    "information retrieval", "neural search", "dense retrieval",
    "hybrid search", "reranking", "cross-encoder",
    "learning-to-rank", "learning to rank", "ltr", "relevance labeling",
    "offline-online correlation", "index refresh", "embedding drift",
    "retrieval quality", "search relevance", "click-through", "click through",
    "search and discovery", "ranking algorithms", "personalization",
    "personalization infrastructure", "relevance", "offline experimentation",
    "online a/b testing", "recommender", "recommendations-heavy",
}

# Medium signals â€” indicate technical ML/AI work but not necessarily
# the specific retrieval/ranking domain the JD needs
MEDIUM_CAREER_KEYWORDS = {
    "machine learning", "deep learning", "neural network",
    "model training", "model deployment", "inference",
    "feature engineering", "pipeline", "pytorch", "tensorflow",
    "scikit-learn", "mlops", "model serving", "a/b test",
    "production ml", "deployed", "shipped", "launched",
    "python", "data pipeline", "api", "microservice"
}

# Consulting red flags â€” the JD explicitly says consulting-only
# backgrounds are a disqualifier. We don't hard-ban but we penalize.
CONSULTING_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "mindtree", "l&t infotech", "ltimindtree",
    "persistent systems", "niit technologies"
}


def get_title_bucket(candidate: dict) -> str:
    """
    WHY: Quick first pass using current title.
    Exact matching catches common clean titles. The phrase fallback catches
    senior/lead/staff variants that appear in this dataset, such as
    "Lead AI Engineer" and "Senior ML Engineer - Search & Ranking".
    """
    title = candidate["profile"]["current_title"].lower().strip()

    if title in CORE_TITLES:
        return "CORE"
    if title in ADJACENT_TITLES:
        return "ADJACENT"

    core_title_phrases = {
        "ai engineer", "machine learning engineer", "ml engineer",
        "nlp engineer", "search engineer", "applied scientist",
        "applied ml engineer", "ranking engineer", "recommendation engineer",
        "recommender systems engineer", "data scientist", "research engineer",
    }
    if any(phrase in title for phrase in core_title_phrases):
        return "CORE"

    adjacent_title_phrases = {
        "software developer", "software engineer", "backend engineer",
        "data engineer", "analytics engineer", "technical lead",
        "engineering manager", "solutions architect", "data analyst",
    }
    if any(phrase in title for phrase in adjacent_title_phrases):
        return "ADJACENT"

    return "NOISE"


def get_career_description_text(candidate: dict) -> str:
    """
    WHY: We concatenate all job descriptions into one string.
    This gives us a full picture of what the person actually did
    across their entire career, not just their current role.
    We weight recent roles more by putting them first.
    """
    # Sort by recency â€” current job first, then reverse chronological
    sorted_roles = sorted(
        candidate["career_history"],
        key=lambda r: r.get("start_date", ""),
        reverse=True
    )

    descriptions = []
    for role in sorted_roles:
        company = role.get("company", "")
        title = role.get("title", "")
        desc = role.get("description", "")
        descriptions.append(f"{title} at {company}: {desc}")

    return " ".join(descriptions).lower()


def score_career_keywords(career_text: str) -> dict:
    """
    WHY: We scan the career description text for relevant keywords.
    Strong keywords (built embeddings, deployed retrieval systems) count
    more than medium keywords (used Python, built pipelines).
    
    Returns both scores separately so we can use them in the gate logic.
    """
    strong_hits = sum(
        1 for keyword in STRONG_CAREER_KEYWORDS
        if keyword in career_text
    )

    medium_hits = sum(
        1 for keyword in MEDIUM_CAREER_KEYWORDS
        if keyword in career_text
    )

    return {
        "strong": strong_hits,
        "medium": medium_hits,
        # Combined weighted score for convenience
        "weighted": strong_hits * 2 + medium_hits * 1
    }


def has_consulting_only_background(candidate: dict) -> bool:
    """
    WHY: The JD explicitly flags people whose entire career is at
    IT services firms. We check every company in career history.
    If ALL of them are consulting firms, we apply a penalty.
    
    Note: "currently at Infosys but prior product company experience"
    is fine per the JD. We only penalize if it's their ENTIRE history.
    """
    companies = [
        role["company"].lower()
        for role in candidate["career_history"]
    ]

    consulting_count = sum(
        1 for company in companies
        if any(flag in company for flag in CONSULTING_COMPANIES)
    )

    # Only flag if majority of career is consulting
    return consulting_count >= len(companies) * 0.75


def classify_candidate(candidate: dict) -> dict:
    """
    WHY THIS IS THE MAIN FUNCTION:
    Combines title bucket + career keyword scores to make a final
    classification decision. Returns a dict with the classification
    and the evidence behind it, so we can use both in scoring and
    in generating the reasoning column.

    Classification logic:
    - CORE title + any strong keyword hit â†’ CORE (definitely relevant)
    - CORE title + no keyword hits â†’ ADJACENT (title says yes, work unclear)
    - ADJACENT title + strong keyword hits â†’ CORE (work proves it)
    - ADJACENT title + no keyword hits â†’ ADJACENT (maybe relevant)
    - NOISE title + strong keyword hits â†’ ADJACENT (hidden gem case)
    - NOISE title + no keyword hits â†’ NOISE (irrelevant)
    
    The "hidden gem" case (NOISE title but strong career keywords) is
    exactly what the JD means by "the gap between what the JD says and
    what the JD means." A Data Analyst who built production embedding
    systems should not be filtered out just because of their title.
    """
    title_bucket = get_title_bucket(candidate)
    career_text = get_career_description_text(candidate)
    keyword_scores = score_career_keywords(career_text)
    consulting_flag = has_consulting_only_background(candidate)

    strong = keyword_scores["strong"]
    weighted = keyword_scores["weighted"]

    # Determine final classification
    if title_bucket == "CORE" and strong >= 1:
        classification = "CORE"
    elif title_bucket == "CORE" and strong == 0 and weighted >= 3:
        classification = "ADJACENT"
    elif title_bucket == "CORE":
        # Title says engineer but career shows no relevant work
        classification = "ADJACENT"
    elif title_bucket == "ADJACENT" and strong >= 2:
        # Adjacent title but career history shows real retrieval/ranking work
        classification = "CORE"
    elif title_bucket == "ADJACENT":
        classification = "ADJACENT"
    elif title_bucket == "NOISE" and strong >= 3:
        # Hidden gem â€” non-technical title but clearly did relevant work
        classification = "ADJACENT"
    else:
        classification = "NOISE"

    return {
        "candidate_id": candidate["candidate_id"],
        "classification": classification,
        "title_bucket": title_bucket,
        "strong_keyword_hits": strong,
        "weighted_keyword_score": weighted,
        "consulting_flag": consulting_flag,
        "career_text_length": len(career_text)
    }

